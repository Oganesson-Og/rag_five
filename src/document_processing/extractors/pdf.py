"""
Enhanced PDF Document Extractor Module
----------------------------------

Advanced PDF processing system with layout recognition and structural analysis.

Key Features:
- Layout-aware extraction
- Hierarchical content detection
- Cross-page content handling
- Smart text merging
- Title/subtitle recognition
- Table structure detection
- Image extraction with context
- Zoom-based processing

Technical Details:
- PyMuPDF integration
- Layout recognition
- Structure preservation
- Content hierarchy
- Smart merging
- (Optional) OCR capabilities (now fully delegated to `ocr.py`)
- Error handling
- Performance optimization

Dependencies:
- PyMuPDF>=1.18.0
- xgboost>=1.7.0
- torch>=2.0.0
- camelot-py>=0.10.1
- tabula-py>=2.7.0
- pdfplumber>=0.7.0
- pandas>=1.5.0
- (Optional) tesseract>=5.3.0 / pytesseract>=0.3.10 for OCR

Author: Keith Satuku
Version: 2.1.0
Created: 2025
License: MIT
"""

import fitz
import re
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import io
import os
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Generator, Union
from dataclasses import dataclass
from .models import Document
from .base import BaseExtractor, ExtractorResult
from enum import Enum
import cv2  # For diagram / image analysis
from ..core.vision.table_structure_recognizer import TableStructureRecognizer
from src.utils.file_utils import get_project_base_directory
from PIL import Image
import subprocess

# Check if OpenCV is available
try:
    import cv2  # For diagram / image analysis
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# Table extraction libraries with checks
try:
    import camelot
    HAS_CAMELOT = True
except ImportError:
    HAS_CAMELOT = False

try:
    import tabula
    HAS_TABULA = True
except ImportError:
    HAS_TABULA = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# Vision-related models (e.g. for diagram classification)
try:
    from torchvision import models, transforms
    HAS_VISION_MODELS = True
except ImportError:
    HAS_VISION_MODELS = False

# If OCR is requested, we will import our local OCR class at runtime (see __init__).
# from .ocr import OCR  # <-- We do a lazy import in __init__ if use_ocr is True.


@dataclass
class LayoutElement:
    """Structure for layout elements."""
    type: str
    text: str
    bbox: Tuple[float, float, float, float]
    font_size: float
    font_name: str
    is_bold: bool
    in_row: int = 1
    row_height: float = 0
    is_row_header: bool = False
    confidence: float = 1.0


class DiagramType(Enum):
    """Types of diagrams commonly found in educational materials."""
    UNKNOWN = "unknown"
    CHART = "chart"
    GRAPH = "graph"
    SCIENTIFIC_DIAGRAM = "scientific_diagram"
    FLOWCHART = "flowchart"
    CONCEPT_MAP = "concept_map"
    GEOMETRIC_FIGURE = "geometric_figure"
    CIRCUIT_DIAGRAM = "circuit_diagram"
    CHEMICAL_STRUCTURE = "chemical_structure"
    MATHEMATICAL_PLOT = "mathematical_plot"
    ANATOMICAL_DIAGRAM = "anatomical_diagram"
    ARCHITECTURAL_DRAWING = "architectural_drawing"
    HISTORICAL_MAP = "historical_map"


class PDFExtractor(BaseExtractor):
    """Enhanced PDF extractor with layout awareness."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PDF extractor.

        Args:
            config: Optional configuration dictionary
        """
        default_config = {
            'zoom_factor': 2.0,
            'title_font_scale': 1.2,
            'merge_tolerance': 0.5,
            'min_confidence': 0.7,
            'line_spacing': 1.2,
            'row_tolerance': 5,
            'char_spacing': 0.1,
            'table_extraction': {
                'method': 'auto',
                'strategy': 'adaptive',
                'flavor': 'lattice',
                'line_scale': 40,
                'min_confidence': 80,
                'header_extraction': True,
                'fallback_to_heuristic': True,
                'table_types': {
                    'bordered': 'camelot',
                    'borderless': 'tabula',
                    'complex': 'camelot',
                    'scanned': 'tabula'
                }
            },
            'use_ocr': False,  # If True, we will initialize an OCR instance.
            'enhance_resolution': False,
            'preserve_layout': True,
            'diagram_detection': {
                'enabled': True,
                'model_path': None,
                'min_width': 100,
                'min_height': 100,
            },
            'acceleration': {
                'num_threads': 8,
                'device': 'mps'  # 'auto', 'cuda', 'mps', 'cpu'
            }
        }

        if config:
            default_config.update(config)

        super().__init__(config=default_config)
        self._init_components()
        self.base_font_size = 12.0  # Updated during processing

        # Determine device for acceleration
        self.device = self._get_device_from_config()
        
        # Initialize OCR if requested
        self.ocr = self._init_ocr()
        
        # Check for Ghostscript if Camelot is available
        if HAS_CAMELOT:
            if not self._check_ghostscript_installed():
                self.logger.warning("Ghostscript is not installed, which is required for Camelot table extraction. "
                                   "You can install it using the instructions here: "
                                   "https://camelot-py.readthedocs.io/en/master/user/install-deps.html")
                self.logger.warning("Table extraction will fall back to alternative methods.")

        # Initialize Docling extractor with acceleration and picture annotation support
        try:
            from ..extractors.docling_extractor import DoclingExtractor
            picture_config = default_config.get('picture_annotation', {})
            
            # Determine if remote services should be enabled
            enable_remote = picture_config.get('enable_remote_services', True)
            
            self.docling = DoclingExtractor(
                offline_mode=config.get('offline_mode', False),
                artifacts_path=config.get('artifacts_path'),
                model_type=picture_config.get('model_type', 'local'),
                model_name=picture_config.get('model_name', 
                    'ibm-granite/granite-vision-3.1-2b-preview'),
                image_scale=picture_config.get('image_scale', 2.0),
                picture_prompt=picture_config.get('prompt', 
                    "Describe the image in three sentences. Be concise and accurate."),
                api_config=picture_config.get('api_config'),
                num_threads=default_config['acceleration'].get('num_threads', 8),
                enable_remote_services=enable_remote  # Explicitly enable remote services
            )
            self.has_docling = True
            self.logger.info(f"Docling extraction initialized successfully on {self.device} with remote services {'enabled' if enable_remote else 'disabled'}")
        except ImportError:
            self.has_docling = False
            self.logger.warning("Docling not available, falling back to default extraction")

    def _init_components(self):
        """Initialize sub-components for layout recognition and device settings."""
        self._init_layout_recognizer()
        self._init_device()

    def _init_layout_recognizer(self):
        """Initialize layout recognition capabilities."""
        try:
            from ..core.vision.recognizer import Recognizer

            # Get layout config from PDF section
            pdf_config = self.config
            layout_config = pdf_config.get('layout', {})
            
            # If API key not in layout config, try to get from model config
            if not layout_config.get('api_key'):
                model_config = self.config.get('model', {})
                # Try Gemini API key first, then fallback to other options
                layout_config['api_key'] = (
                    model_config.get('gemini_api_key') or 
                    model_config.get('api_key')
                )

            if not layout_config.get('api_key'):
                raise ValueError("No API key found in configuration for layout recognition")

            # Get device from config or use system default
            device = layout_config.get('device') or self.config.get('acceleration', {}).get('device', 'cpu')

            self.layout_recognizer = Recognizer(
                model_type=layout_config.get('model_type', 'gemini'),  # Default to Gemini
                model_name=layout_config.get('model_name', 'gemini-pro-vision'),  # Use Gemini Pro Vision by default
                api_key=layout_config['api_key'],
                device=device,
                batch_size=layout_config.get('batch_size', 32),
                cache_dir=layout_config.get('cache_dir'),
                confidence=layout_config.get('confidence', 0.5),
                merge_boxes=layout_config.get('merge_boxes', True),
                label_list=layout_config.get('label_list', [
                    "title", "text", "list", "table", "figure",
                    "header", "footer", "sidebar", "caption"
                ]),
                task_name=layout_config.get('task_name', 'document_layout'),
                ollama_host=layout_config.get('ollama_host', 'http://localhost:11434')
            )
            self.has_layout_recognition = True
            self.logger.info("Layout recognition initialized successfully with Gemini Vision")

        except Exception as e:
            self.logger.warning(f"Layout recognition not available: {str(e)}")
            self.has_layout_recognition = False

    def _init_device(self):
        """Initialize device settings for torch if available."""
        try:
            if self.device == 'cuda':
                self.torch_device = torch.device('cuda')
            elif self.device == 'mps':
                self.torch_device = torch.device('mps')
            else:
                self.torch_device = torch.device('cpu')
        except Exception:
            self.torch_device = torch.device('cpu')

    def _init_ocr(self):
        """
        Initialize OCR component if requested in config.
        
        Returns:
            OCR instance or None if OCR is not enabled
        """
        try:
            if self.config.get('use_ocr', False):
                from ..core.vision.ocr import OCR
                
                self.logger.info(f"Initializing OCR with engine: {self.config.get('ocr_engine', 'tesseract')}")
                
                # Initialize OCR with the entire config dictionary
                # The OCR class will extract what it needs
                self.ocr = OCR(
                    config=self.config,
                    languages=self.config.get('ocr_languages', ['en']),
                    preserve_layout=self.config.get('preserve_layout', True),
                    enhance_resolution=self.config.get('enhance_resolution', True),
                    use_paligemma=self.config.get('use_paligemma', False)
                )
                
                return self.ocr
            else:
                self.logger.info("OCR is disabled in configuration")
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize OCR: {str(e)}")
            return None

    def _get_device_from_config(self) -> str:
        """
        Get the device setting from configuration.
        
        Returns:
            str: Device name ('cpu', 'cuda', 'mps', or 'auto')
        """
        device = self.config.get('acceleration', {}).get('device', 'cpu')
        
        # If auto, try to determine the best device
        if device == 'auto':
            try:
                import torch
                if torch.cuda.is_available():
                    return 'cuda'
                elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return 'mps'
                else:
                    return 'cpu'
            except ImportError:
                return 'cpu'
                
        return device

    async def extract(self, document: 'Document') -> 'Document':
        """
        Main extraction method that orchestrates the entire PDF extraction process.
        
        Args:
            document: Document object containing PDF content or path
            
        Returns:
            Document: Processed document with extracted content and metadata
        """
        try:
            # Convert document content to bytes if it's a file path
            content = document.content
            if isinstance(content, str):
                with open(content, 'rb') as f:
                    content = f.read()
            
            # Open PDF from binary content
            doc = fitz.open(stream=content, filetype="pdf")
            
            # Extract text using Docling if available
            if self.has_docling:
                try:
                    docling_result = self.docling.process_file(content)
                    if docling_result.success:
                        document.content = docling_result.text
                    else:
                        self.logger.warning(f"Docling extraction failed: {docling_result.error}")
                        # Fall back to default text extraction
                        document.content = self._extract_structured_text(doc, {})
                except Exception as e:
                    self.logger.warning(f"Docling extraction failed with exception: {str(e)}")
                    # Fall back to default text extraction
                    document.content = self._extract_structured_text(doc, {})
            else:
                # Use default text extraction
                document.content = self._extract_structured_text(doc, {})
            
            # Extract tables using voting/adaptive approach
            try:
                tables = self._extract_tables_with_structure(doc)
                if tables:
                    document.doc_info['tables'] = tables
                    
                    # Add table context and references
                    for table in tables:
                        if 'region' in table:
                            table['context'] = self._get_table_context(doc[table.get('page', 1) - 1], table['region'])
            except Exception as e:
                self.logger.warning(f"Table extraction failed: {str(e)}")
                document.doc_info['tables'] = []
            
            # Extract and analyze images/diagrams with enhanced context
            try:
                images = self._extract_images_with_context(doc)
                if images:
                    document.doc_info['images'] = images
                    # Separate diagrams from regular images
                    diagrams = [img for img in images if img['metadata'].get('is_diagram')]
                    if diagrams:
                        document.doc_info['diagrams'] = diagrams
            except Exception as e:
                self.logger.warning(f"Image extraction failed: {str(e)}")
                document.doc_info['images'] = []
            
            # Add metadata
            document.doc_info['metadata'] = self._extract_pdf_metadata(doc)
            
            # Add cross-references
            self._add_cross_references(document)
            
            # Check if document is scanned
            document.doc_info['is_scanned'] = self._check_if_scanned(doc)
            
            # Close the document
            doc.close()
            
            return document
            
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {str(e)}")
            raise

    def _process_document_structure(self, doc: fitz.Document) -> Dict[str, Any]:
        """Process document structure and hierarchy for headings, layout info, etc."""
        structure = {
            'hierarchy': [],
            'page_layouts': [],
            'content_map': {}
        }

        for page_num, page in enumerate(doc):
            layout = self._analyze_page_layout(page)
            structure['page_layouts'].append(layout)

            # Build content hierarchy from recognized headings / titles
            for element in layout:
                if element.type in ['title', 'subtitle', 'heading']:
                    structure['hierarchy'].append({
                        'text': element.text,
                        'level': self._determine_heading_level(element),
                        'page': page_num + 1
                    })
        return structure

    def _determine_heading_level(self, element: LayoutElement) -> int:
        """
        Dummy function: infer heading level from font size, boldness, or heuristics.
        For real usage, expand logic as needed.
        """
        # Example: large font => level 1, medium => level 2, else => 3+ etc.
        if element.font_size >= self.base_font_size * 1.4:
            return 1
        elif element.font_size >= self.base_font_size * 1.2:
            return 2
        else:
            return 3

    def _analyze_page_layout(self, page: fitz.Page) -> List[LayoutElement]:
        """Analyze page layout to detect text, heading, title, etc."""
        elements = []
        page_dict = page.get_text("dict")
        base_font_size = self._get_base_font_size(page_dict)

        for block in page_dict.get('blocks', []):
            if block.get('type') == 0:  # text block
                element = self._process_text_block(block, base_font_size)
                if element:
                    elements.append(element)

        # Possibly merge or further refine elements. E.g. self._merge_related_elements(...)
        elements = self._merge_related_elements(elements)
        return elements

    def _process_text_block(
        self,
        block: Dict[str, Any],
        base_font_size: float
    ) -> Optional[LayoutElement]:
        """Extract textual info from a text block and classify its type."""
        try:
            text = ' '.join(span['text'] for span in block['spans'])
            font_info = block['spans'][0]  # Use first span's style as reference.
            font_size = font_info.get('size', 0)
            is_bold = 'bold' in font_info.get('font', '').lower()

            # Determine element type (heading, title, normal text...)
            element_type = self._determine_element_type(font_size, base_font_size, is_bold)

            return LayoutElement(
                type=element_type,
                text=text,
                bbox=block['bbox'],
                font_size=font_size,
                font_name=font_info.get('font', ''),
                is_bold=is_bold
            )
        except Exception as e:
            self.logger.warning(f"Error processing text block: {str(e)}")
            return None

    def _determine_element_type(
        self, font_size: float, base_font_size: float, is_bold: bool
    ) -> str:
        if font_size >= base_font_size * 1.5:
            return 'title'
        elif font_size >= base_font_size * 1.2 or (font_size > base_font_size and is_bold):
            return 'subtitle'
        elif is_bold:
            return 'heading'
        else:
            return 'text'

    def _get_base_font_size(self, page_dict: Dict[str, Any]) -> float:
        """Heuristic: median of the encountered font sizes on the page."""
        font_sizes = []
        for block in page_dict.get('blocks', []):
            if block.get('type') == 0:  # text block
                for span in block.get('spans', []):
                    if size := span.get('size'):
                        font_sizes.append(size)
        return float(np.median(font_sizes)) if font_sizes else 12.0

    def _merge_related_elements(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """Heuristic to combine adjacent text blocks of the same type, etc."""
        merged = []
        if not elements:
            return merged

        current = elements[0]
        for elem in elements[1:]:
            if self._should_merge_elements(current, elem):
                # Concatenate text, update bounding box if needed, etc.
                combined_text = current.text + " " + elem.text
                new_bbox = (
                    min(current.bbox[0], elem.bbox[0]),
                    min(current.bbox[1], elem.bbox[1]),
                    max(current.bbox[2], elem.bbox[2]),
                    max(current.bbox[3], elem.bbox[3]),
                )
                current = LayoutElement(
                    type=current.type,
                    text=combined_text,
                    bbox=new_bbox,
                    font_size=(current.font_size + elem.font_size) / 2.0,
                    font_name=current.font_name,
                    is_bold=(current.is_bold or elem.is_bold)
                )
            else:
                merged.append(current)
                current = elem
        merged.append(current)
        return merged

    def _should_merge_elements(self, e1: LayoutElement, e2: LayoutElement) -> bool:
        """Example heuristic: same type, overlap, close in vertical space, etc."""
        if e1.type != e2.type:
            return False
        vertical_gap = e2.bbox[1] - e1.bbox[3]
        if abs(vertical_gap) < self.config['merge_tolerance'] * 10:
            return True
        return False

    def _extract_structured_text(self, doc: fitz.Document, structure: Dict[str, Any]) -> str:
        """
        Simple aggregator that joins recognized text blocks from the structure.
        A real implementation might produce a more advanced layout/JSON, etc.
        
        If structure is empty or doesn't contain page_layouts, falls back to basic text extraction.
        """
        # Check if structure has page_layouts
        if not structure or 'page_layouts' not in structure:
            # Fall back to basic text extraction
            combined_text = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text().strip()
                if page_text:
                    combined_text.append(f"--- Page {page_num+1} ---\n{page_text}")
            return "\n\n".join(combined_text)
        
        # Process structured layout if available
        combined_text = []
        for page_num, page_layout in enumerate(structure['page_layouts']):
            page_text = []
            for elem in page_layout:
                if elem.type in ['text', 'heading', 'title', 'subtitle']:
                    page_text.append(elem.text.strip())
            if page_text:
                combined_text.append(f"--- Page {page_num+1} ---\n" + "\n".join(page_text))
        return "\n\n".join(combined_text)

    def _extract_tables_with_structure(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Extract tables using the unified extraction strategy."""
        tables = []
        
        # Create a temporary file for the PDF
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_path = temp_file.name
                doc.save(temp_path)
                
                # Use the unified extraction method
                tables = self.extract_tables(doc, temp_path)
                
        except Exception as e:
            self.logger.error(f"Table extraction error: {str(e)}")
            # Return empty list on failure
            tables = []
            
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    self.logger.warning(f"Failed to delete temporary PDF: {str(e)}")
                    
        return tables

    def _regions_overlap(self, region1, region2):
        """Check if two regions (bounding boxes) overlap."""
        if not region1 or not region2:
            return False
            
        x01, y01, x11, y11 = region1
        x02, y02, x12, y12 = region2
        
        # Check if one rectangle is to the left of the other
        if x11 < x02 or x12 < x01:
            return False
            
        # Check if one rectangle is above the other
        if y11 < y02 or y12 < y01:
            return False
            
        return True

    def _check_ghostscript_installed(self) -> bool:
        """
        Check if Ghostscript is installed on the system.
        
        Ghostscript is required for Camelot to extract tables from PDFs.
        This method checks for Ghostscript in the same way that Camelot does.
        
        Returns:
            True if Ghostscript is installed, False otherwise
        """
        try:
            # First try the Camelot way of checking for Ghostscript
            from ctypes.util import find_library
            import sys
            
            self.logger.debug("Checking for Ghostscript using ctypes.util.find_library")
            
            # Check based on platform (same as Camelot does)
            if sys.platform in ["linux", "darwin"]:
                # For Linux and macOS
                library = find_library("gs")
                result = library is not None
                self.logger.debug(f"Ghostscript library check result: {result}, library path: {library}")
                if result:
                    return True
            elif sys.platform == "win32":
                # For Windows
                import ctypes
                library = find_library(
                    "".join(("gsdll", str(ctypes.sizeof(ctypes.c_voidp) * 8), ".dll"))
                )
                result = library is not None
                self.logger.debug(f"Ghostscript library check result: {result}, library path: {library}")
                if result:
                    return True
            
            # If the library check failed, fall back to checking for the executable
            self.logger.debug("Library check failed, falling back to executable check")
            
            # Try to find the Ghostscript executable
            gs_command = "gs"
            
            # Check if we're on Windows
            if sys.platform == "win32":
                gs_command = "gswin64c"  # 64-bit Ghostscript on Windows
            
            self.logger.debug(f"Checking for Ghostscript using command: {gs_command}")
                
            # Try to run Ghostscript with version flag
            result = subprocess.run(
                [gs_command, "--version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=5
            )
            
            # Log the result for debugging
            self.logger.debug(f"Ghostscript check result: returncode={result.returncode}, stdout={result.stdout.decode().strip()}, stderr={result.stderr.decode().strip()}")
            
            # If the command succeeded, Ghostscript is installed
            return result.returncode == 0
            
        except (subprocess.SubprocessError, FileNotFoundError, ImportError) as e:
            # Log the specific error
            self.logger.debug(f"Ghostscript check failed with error: {str(e)}")
            # If the command failed or the executable wasn't found
            return False

    def _detect_table_borders(self, page: fitz.Page) -> bool:
        """
        Analyze a page to determine if it contains tables with visible borders.
        
        This method counts horizontal and vertical lines on the page to determine
        if there are likely to be bordered tables present.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            True if the page likely contains bordered tables, False otherwise
        """
        try:
            # Get page dimensions
            page_width = page.rect.width
            page_height = page.rect.height
            
            # Extract paths (lines, rectangles, etc.)
            paths = page.get_drawings()
            
            # Count horizontal and vertical lines
            h_lines = 0
            v_lines = 0
            
            for path in paths:
                # Check each item in the path
                for item in path["items"]:
                    if item[0] == "l":  # Line segment
                        x0, y0 = item[1]  # Start point
                        x1, y1 = item[2]  # End point
                        
                        # Calculate line length
                        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                        
                        # Skip very short lines
                        if length < 10:
                            continue
                            
                        # Check if horizontal (y coordinates are similar)
                        if abs(y1 - y0) < 3:
                            h_lines += 1
                            
                        # Check if vertical (x coordinates are similar)
                        elif abs(x1 - x0) < 3:
                            v_lines += 1
            
            # Also check for rectangles which might be table cells
            rectangles = 0
            for path in paths:
                if path["type"] == "rectangle":
                    rectangles += 1
            
            # Determine if the page has enough lines to indicate tables with borders
            # Thresholds can be adjusted based on experience
            if (h_lines >= 5 and v_lines >= 3) or rectangles >= 10:
                return True
                
            # Check for explicit table markup in the page structure
            # This can catch tables that are semantically marked but don't have visible lines
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("type") == 1:  # Image block, might be a table
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.warning(f"Table border detection failed: {str(e)}")
            return False

    def _extract_tables_unified(self, pdf_path: str, page_number: int, page: fitz.Page, 
                             has_borders: bool) -> List[Dict[str, Any]]:
        """
        Extract tables using a unified approach that combines multiple extraction methods.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (0-indexed)
            page: PyMuPDF page object
            has_borders: Whether the page has visible table borders
            
        Returns:
            List of extracted tables
        """
        tables = []
        extraction_method = self.config.get('table_extraction', {}).get('method', 'auto')
        
        # Get the appropriate extraction method based on table type
        if extraction_method == 'auto':
            if has_borders:
                method = self.config.get('table_extraction', {}).get('table_types', {}).get('bordered', 'camelot')
            else:
                method = self.config.get('table_extraction', {}).get('table_types', {}).get('borderless', 'tabula')
        else:
            method = extraction_method
            
        self.logger.info(f"Using {method} for table extraction on page {page_number+1}")
        
        # Try the primary extraction method
        if method == 'camelot':
            try:
                if has_borders:
                    tables = self._extract_with_camelot(pdf_path, page_number+1, flavor='lattice')
                else:
                    tables = self._extract_with_camelot(pdf_path, page_number+1, flavor='stream')
            except Exception as e:
                self.logger.warning(f"Table extraction with {method} failed: {str(e)}")
                
        elif method == 'tabula':
            try:
                if has_borders:
                    tables = self._extract_with_tabula(pdf_path, page_number+1, lattice=True)
                else:
                    tables = self._extract_with_tabula(pdf_path, page_number+1, lattice=False, guess=True)
            except Exception as e:
                self.logger.warning(f"Table extraction with {method} failed: {str(e)}")
                
        elif method == 'pdfplumber':
            try:
                tables = self._extract_with_pdfplumber(pdf_path, page_number+1)
            except Exception as e:
                self.logger.warning(f"Table extraction with {method} failed: {str(e)}")
        
        # If primary method failed, try fallback methods
        if not tables and self.config.get('table_extraction', {}).get('fallback_to_heuristic', True):
            self.logger.info(f"Primary extraction method failed, trying fallback methods")
            
            # Try camelot if not already tried
            if method != 'camelot':
                try:
                    if has_borders:
                        tables = self._extract_with_camelot(pdf_path, page_number+1, flavor='lattice')
                    else:
                        tables = self._extract_with_camelot(pdf_path, page_number+1, flavor='stream')
                except Exception as e:
                    self.logger.warning(f"Fallback to camelot failed: {str(e)}")
            
            # Try tabula if not already tried and camelot fallback failed
            if not tables and method != 'tabula':
                try:
                    if has_borders:
                        tables = self._extract_with_tabula(pdf_path, page_number+1, lattice=True)
                    else:
                        tables = self._extract_with_tabula(pdf_path, page_number+1, lattice=False, guess=True)
                except Exception as e:
                    self.logger.warning(f"Fallback to tabula failed: {str(e)}")
            
            # Try pdfplumber as last resort
            if not tables and method != 'pdfplumber':
                try:
                    tables = self._extract_with_pdfplumber(pdf_path, page_number+1)
                except Exception as e:
                    self.logger.warning(f"Fallback to pdfplumber failed: {str(e)}")
        
        # Get context for each table
        for table in tables:
            try:
                table['context'] = self._get_table_context(page, table['bbox'])
            except Exception as e:
                self.logger.warning(f"Failed to get table context on page {page_number+1}: {str(e)}")
                table['context'] = ""
        
        return tables

    def _extract_with_camelot(self, pdf_path: str, page_number: int, flavor: str = 'lattice') -> List[Dict[str, Any]]:
        """
        Extract tables using Camelot.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (1-indexed for Camelot)
            flavor: 'lattice' for bordered tables, 'stream' for borderless tables
            
        Returns:
            List of extracted tables
        """
        try:
            import camelot
            
            # Check if Ghostscript is installed
            if not self._check_ghostscript_installed():
                self.logger.warning("Ghostscript is not installed. Camelot requires Ghostscript for PDF processing.")
                return []
            
            # Configure Camelot options
            line_scale = self.config.get('table_extraction', {}).get('line_scale', 40)
            
            # Extract tables
            tables = camelot.read_pdf(
                pdf_path,
                pages=str(page_number),
                flavor=flavor,
                line_scale=line_scale
            )
            
            if len(tables) == 0:
                self.logger.info(f"No tables found on page {page_number} using Camelot with {flavor} flavor")
                return []
            
            # Convert to standard format
            result = []
            for i, table in enumerate(tables):
                # Get table accuracy
                accuracy = table.accuracy
                
                # Skip tables with low accuracy
                min_confidence = self.config.get('table_extraction', {}).get('min_confidence', 80)
                if accuracy < min_confidence:
                    self.logger.info(f"Skipping table with low accuracy: {accuracy:.2f}% (threshold: {min_confidence}%)")
                    continue
                
                # Convert to DataFrame and then to dict
                df = table.df
                
                # Get table bounding box
                bbox = table._bbox
                
                # Create standardized table structure
                table_dict = {
                    'id': f"table_{page_number}_{i+1}",
                    'page': page_number,
                    'extraction_method': f"camelot_{flavor}",
                    'confidence': accuracy / 100.0,
                    'bbox': bbox,
                    'headers': df.iloc[0].tolist() if not df.empty else [],
                    'rows': df.values.tolist() if not df.empty else [],
                    'num_rows': len(df) if not df.empty else 0,
                    'num_cols': len(df.columns) if not df.empty else 0
                }
                
                # Extract header if configured
                if self.config.get('table_extraction', {}).get('header_extraction', True) and not df.empty:
                    table_dict['headers'] = df.iloc[0].tolist()
                    table_dict['rows'] = df.iloc[1:].values.tolist()
                
                result.append(table_dict)
            
            return result
            
        except ImportError:
            self.logger.warning("Camelot is not installed. Install with: pip install camelot-py")
            return []
        except Exception as e:
            self.logger.warning(f"Camelot extraction failed: {str(e)}")
            return []

    def _get_table_context(self, page: fitz.Page, bbox: List[float], context_range: int = 3) -> str:
        """
        Get the text context around a table.
        
        Args:
            page: PyMuPDF page object
            bbox: Table bounding box [x0, y0, x1, y1]
            context_range: Number of lines to include before and after the table
            
        Returns:
            Text context around the table
        """
        try:
            # Get all text blocks on the page
            blocks = page.get_text("dict")["blocks"]
            
            # Convert table bbox to fitz.Rect
            table_rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            
            # Find blocks above and below the table
            blocks_above = []
            blocks_below = []
            
            for block in blocks:
                if block["type"] == 0:  # Text block
                    block_rect = fitz.Rect(block["bbox"])
                    
                    # Check if block is above the table
                    if block_rect.y1 < table_rect.y0:
                        blocks_above.append((block, block_rect.y1))
                    
                    # Check if block is below the table
                    if block_rect.y0 > table_rect.y1:
                        blocks_below.append((block, block_rect.y0))
            
            # Sort blocks by vertical position
            blocks_above.sort(key=lambda x: x[1], reverse=True)  # Closest to table first
            blocks_below.sort(key=lambda x: x[1])  # Closest to table first
            
            # Get context text
            context_above = ""
            for i, (block, _) in enumerate(blocks_above[:context_range]):
                lines = []
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    lines.append(line_text)
                context_above = "\n".join(lines) + "\n" + context_above
            
            context_below = ""
            for i, (block, _) in enumerate(blocks_below[:context_range]):
                lines = []
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    lines.append(line_text)
                context_below += "\n" + "\n".join(lines)
            
            return context_above.strip() + "\n" + context_below.strip()
            
        except Exception as e:
            self.logger.warning(f"Failed to get table context: {str(e)}")
            return ""

    def _refine_table_structure(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine the table structure by cleaning up headers and rows.
        
        Args:
            table: Table dictionary
            
        Returns:
            Refined table dictionary
        """
        try:
            # Skip empty tables
            if not table.get('rows') or len(table['rows']) == 0:
                return table
            
            # Clean up headers
            headers = table.get('headers', [])
            if headers:
                # Remove empty headers
                headers = [h.strip() if isinstance(h, str) else str(h).strip() for h in headers]
                
                # Generate generic headers if all are empty
                if all(not h for h in headers):
                    headers = [f"Column {i+1}" for i in range(len(headers))]
                
                table['headers'] = headers
            
            # Clean up rows
            rows = table.get('rows', [])
            if rows:
                # Remove completely empty rows
                rows = [row for row in rows if any(cell and str(cell).strip() for cell in row)]
                
                # Clean cell values
                cleaned_rows = []
                for row in rows:
                    cleaned_row = []
                    for cell in row:
                        if isinstance(cell, str):
                            cell = cell.strip()
                        elif cell is None:
                            cell = ""
                        else:
                            cell = str(cell).strip()
                        cleaned_row.append(cell)
                    cleaned_rows.append(cleaned_row)
                
                table['rows'] = cleaned_rows
                table['num_rows'] = len(cleaned_rows)
            
            return table
            
        except Exception as e:
            self.logger.warning(f"Failed to refine table structure: {str(e)}")
            return table

    def _handle_cross_page_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect and merge tables that span across multiple pages.
        
        Args:
            tables: List of tables from all pages
            
        Returns:
            List of tables with cross-page tables merged
        """
        if not tables or len(tables) < 2:
            return tables
        
        try:
            # Sort tables by page number
            tables.sort(key=lambda t: (t['page'], t['bbox'][1]))
            
            merged_tables = []
            i = 0
            while i < len(tables):
                current_table = tables[i]
                
                # Check if there's a next table that might be related
                if i + 1 < len(tables):
                    next_table = tables[i + 1]
                    
                    # Check if tables are on consecutive pages and have similar structure
                    if next_table['page'] == current_table['page'] + 1 and self._are_tables_related(current_table, next_table):
                        # Merge the tables
                        merged_table = self._merge_tables(current_table, next_table)
                        merged_tables.append(merged_table)
                        i += 2  # Skip both tables
                        continue
                
                # If no merge happened, add the current table
                merged_tables.append(current_table)
                i += 1
            
            return merged_tables
            
        except Exception as e:
            self.logger.warning(f"Failed to handle cross-page tables: {str(e)}")
            return tables

    def _are_tables_related(self, table1: Dict[str, Any], table2: Dict[str, Any]) -> bool:
        """
        Check if two tables are related and might be parts of the same table.
        
        Args:
            table1: First table
            table2: Second table
            
        Returns:
            True if tables are related, False otherwise
        """
        # Check if tables have similar structure
        if table1['num_cols'] != table2['num_cols']:
            return False
        
        # Check if headers are similar
        headers1 = table1.get('headers', [])
        headers2 = table2.get('headers', [])
        
        # If both have headers and they're different, tables are likely not related
        if headers1 and headers2 and not self._are_headers_similar(headers1, headers2):
            return False
        
        # Check context for continuity indicators
        context1 = table1.get('context', '').lower()
        context2 = table2.get('context', '').lower()
        
        continuity_indicators = ['continued', 'continuation', 'cont.', 'cont\'d', 'continued from previous page']
        if any(indicator in context2 for indicator in continuity_indicators):
            return True
        
        return True  # Default to assuming they're related if structure matches

    def _are_headers_similar(self, headers1: List[str], headers2: List[str]) -> bool:
        """
        Check if two sets of headers are similar.
        
        Args:
            headers1: First set of headers
            headers2: Second set of headers
            
        Returns:
            True if headers are similar, False otherwise
        """
        if len(headers1) != len(headers2):
            return False
        
        similarity_count = 0
        for h1, h2 in zip(headers1, headers2):
            if h1 == h2 or self._string_similarity(h1, h2) > 0.8:
                similarity_count += 1
        
        # Consider headers similar if at least 70% match
        return (similarity_count / len(headers1) >= 0.7) if headers1 else False

    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate the similarity between two strings using Levenshtein distance.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple implementation of Levenshtein distance
        if len(s1) < len(s2):
            return self._string_similarity(s2, s1)
            
        if len(s2) == 0:
            return 0.0
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        # Convert distance to similarity score
        max_len = max(len(s1), len(s2))
        distance = previous_row[-1]
        return 1.0 - (distance / max_len) if max_len > 0 else 1.0

    def _merge_tables(self, table1: Dict[str, Any], table2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two related tables into one.
        
        Args:
            table1: First table
            table2: Second table
            
        Returns:
            Merged table
        """
        # Create a new merged table
        merged_table = table1.copy()
        
        # Use headers from the first table
        merged_table['headers'] = table1.get('headers', [])
        
        # Merge rows
        rows1 = table1.get('rows', [])
        rows2 = table2.get('rows', [])
        
        # Skip header row in the second table if it matches the first table's headers
        if table2.get('headers') and self._are_headers_similar(table1.get('headers', []), table2.get('headers', [])):
            rows2 = rows2[1:] if rows2 else []
        
        merged_table['rows'] = rows1 + rows2
        merged_table['num_rows'] = len(merged_table['rows'])
        
        # Update metadata
        merged_table['id'] = f"{table1['id']}_merged"
        merged_table['cross_page'] = True
        merged_table['pages'] = [table1['page'], table2['page']]
        
        # Average the confidence scores
        merged_table['confidence'] = (table1.get('confidence', 0) + table2.get('confidence', 0)) / 2
        
        return merged_table

    def _extract_with_tabula(self, pdf_path: str, page_number: int, lattice: bool = False, 
                           guess: bool = True) -> List[Dict[str, Any]]:
        """
        Extract tables using Tabula with specific parameters.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (1-indexed for Tabula)
            lattice: Whether to use lattice mode for bordered tables
            guess: Whether to use guess mode for borderless tables
            
        Returns:
            List of extracted tables
        """
        try:
            # Import tabula within the function to avoid import errors
            import tabula
            
            # Extract tables with specified parameters
            tabula_tables = tabula.read_pdf(
                pdf_path,
                pages=page_number,
                lattice=lattice,
                guess=guess,
                multiple_tables=True,
                pandas_options={'header': None}
            )
            
            # Process each table
            result = []
            for i, df in enumerate(tabula_tables):
                # Skip empty tables
                if df.empty:
                    continue
                
                # Convert to standard format
                table_dict = {
                    'id': f"table_{page_number}_{i+1}",
                    'page': page_number,
                    'extraction_method': f"tabula_{'lattice' if lattice else 'guess'}",
                    'confidence': 0.7,  # Tabula doesn't provide confidence metrics
                    'bbox': [0, 0, 0, 0],  # Tabula doesn't provide bbox information
                    'headers': df.iloc[0].tolist() if not df.empty else [],
                    'rows': df.values.tolist() if not df.empty else [],
                    'num_rows': len(df) if not df.empty else 0,
                    'num_cols': len(df.columns) if not df.empty else 0
                }
                
                # Extract header if configured
                if self.config.get('table_extraction', {}).get('header_extraction', True) and not df.empty:
                    table_dict['headers'] = df.iloc[0].tolist()
                    table_dict['rows'] = df.iloc[1:].values.tolist()
                
                result.append(table_dict)
            
            return result
            
        except ImportError:
            self.logger.warning("Tabula is not installed. Install with: pip install tabula-py")
            return []
        except Exception as e:
            self.logger.warning(f"Tabula extraction failed: {str(e)}")
            return []

    def _extract_with_pdfplumber(self, pdf_path: str, page_number: int, vertical_strategy: str = 'text', 
                               horizontal_strategy: str = 'text') -> List[Dict[str, Any]]:
        """
        Extract tables using pdfplumber with specific parameters.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (1-indexed for pdfplumber)
            vertical_strategy: Strategy for vertical lines ('text', 'lines', or 'explicit')
            horizontal_strategy: Strategy for horizontal lines ('text', 'lines', or 'explicit')
            
        Returns:
            List of extracted tables
        """
        try:
            # Import pdfplumber within the function to avoid import errors
            import pdfplumber
            
            # Open the PDF and get the specified page
            with pdfplumber.open(pdf_path) as pdf:
                if page_number <= len(pdf.pages):
                    plumber_page = pdf.pages[page_number - 1]  # Convert to 0-based index
                    
                    # Extract tables with specified parameters
                    plumber_tables = plumber_page.extract_tables(
                        table_settings={
                            'vertical_strategy': vertical_strategy,
                            'horizontal_strategy': horizontal_strategy,
                            'intersection_tolerance': 5,
                            'snap_tolerance': 3,
                            'join_tolerance': 3,
                            'edge_min_length': 3,
                            'min_words_vertical': 3,
                            'min_words_horizontal': 1
                        }
                    )
                    
                    # Process each table
                    result = []
                    for i, table_data in enumerate(plumber_tables):
                        # Skip empty tables
                        if not table_data or len(table_data) == 0:
                            continue
                        
                        # Clean up rows (remove None values)
                        rows = []
                        for row in table_data:
                            cleaned_row = ['' if cell is None else str(cell).strip() for cell in row]
                            rows.append(cleaned_row)
                        
                        # Create table entry
                        table_dict = {
                            'id': f"table_{page_number}_{i+1}",
                            'page': page_number,
                            'extraction_method': f"pdfplumber_{vertical_strategy}_{horizontal_strategy}",
                            'confidence': 0.6,  # pdfplumber doesn't provide confidence metrics
                            'bbox': [0, 0, 0, 0],  # We could calculate this from the table cells if needed
                            'headers': rows[0] if rows else [],
                            'rows': rows[1:] if len(rows) > 1 else [],
                            'num_rows': len(rows) - 1 if len(rows) > 1 else 0,
                            'num_cols': len(rows[0]) if rows else 0
                        }
                        
                        # Extract header if configured
                        if not self.config.get('table_extraction', {}).get('header_extraction', True) and rows:
                            table_dict['headers'] = []
                            table_dict['rows'] = rows
                            table_dict['num_rows'] = len(rows)
                        
                        result.append(table_dict)
                    
                    return result
                else:
                    self.logger.warning(f"Page {page_number} is out of range for the PDF with {len(pdf.pages)} pages")
                    return []
            
        except ImportError:
            self.logger.warning("pdfplumber is not installed. Install with: pip install pdfplumber")
            return []
        except Exception as e:
            self.logger.warning(f"pdfplumber extraction failed: {str(e)}")
            return []

    def extract_tables(self, doc: fitz.Document, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Unified table extraction strategy that combines the strengths of multiple libraries.
        
        This method implements a comprehensive approach to table extraction:
        1. First attempts extraction with Camelot (both lattice and stream modes)
        2. Then tries Tabula for tables Camelot might have missed
        3. Finally uses pdfplumber as a fallback
        4. Merges and deduplicates results based on spatial overlap
        5. Ranks and selects the best extraction for each detected table region
        6. Applies advanced structure analysis to refine the table structure
        7. Detects and handles cross-page tables
        
        Args:
            doc: PyMuPDF document object
            pdf_path: Path to the PDF file
            
        Returns:
            List of extracted tables with metadata and quality metrics
        """
        all_tables = []
        
        # Check if any table extraction libraries are available
        if not any([HAS_CAMELOT, HAS_TABULA, HAS_PDFPLUMBER]):
            self.logger.warning("No table extraction libraries available. "
                               "Please install at least one of: camelot-py, tabula-py, or pdfplumber.")
            return []
        
        # Initialize TableStructureRecognizer if available
        table_structure_recognizer = None
        try:
            # Pass the config path to TableStructureRecognizer
            config_path = os.path.join(get_project_base_directory(), "config/rag_config.yaml")
            
            # Check if config file exists before initializing
            if os.path.exists(config_path):
                try:
                    from ..core.vision.table_structure_recognizer import TableStructureRecognizer
                    table_structure_recognizer = TableStructureRecognizer(config_path)
                    self.logger.info("TableStructureRecognizer initialized successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize TableStructureRecognizer: {str(e)}")
            else:
                self.logger.warning(f"Config file not found at {config_path}, skipping TableStructureRecognizer initialization")
        except Exception as e:
            self.logger.warning(f"Failed to initialize TableStructureRecognizer: {str(e)}")
            
        # Process each page
        for page_num, page in enumerate(doc):
            page_number = page_num + 1  # Convert to 1-based page numbering
            
            try:
                # Step 1: Analyze page to determine table characteristics
                has_borders = self._detect_table_borders(page)
                
                # Step 2: Extract tables using the unified strategy
                page_tables = self._extract_tables_unified(pdf_path, page_number, page, has_borders)
                
                # Step 3: Apply advanced structure analysis to refine the tables
                if table_structure_recognizer and page_tables:
                    try:
                        page_tables = self._refine_table_structure(page_tables, page, table_structure_recognizer)
                    except Exception as e:
                        self.logger.warning(f"Table structure refinement failed on page {page_number}: {str(e)}")
                
                # Step 4: Add page information to each table
                for table in page_tables:
                    table['page'] = page_number
                    
                    # Add table context if region is available
                    if 'region' in table:
                        try:
                            table['context'] = self._get_table_context(page, table['region'])
                        except Exception as e:
                            self.logger.warning(f"Failed to get table context on page {page_number}: {str(e)}")
                
                all_tables.extend(page_tables)
                
            except Exception as e:
                self.logger.error(f"Table extraction failed for page {page_number}: {str(e)}")
                # Continue with next page instead of failing completely
                continue
        
        # Step 5: Detect and handle cross-page tables
        if len(all_tables) > 1:
            try:
                all_tables = self._handle_cross_page_tables(all_tables, doc)
            except Exception as e:
                self.logger.warning(f"Cross-page table handling failed: {str(e)}")
            
        return all_tables

# Test function to verify Ghostscript detection
def test_ghostscript_detection():
    import subprocess
    import os
    import logging
    
    logger = logging.getLogger("GhostscriptTest")
    logger.setLevel(logging.DEBUG)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(ch)
    
    # Print environment PATH
    logger.debug(f"PATH environment: {os.environ.get('PATH', '')}")
    
    try:
        # Try to find the Ghostscript executable
        gs_command = "gs"
        
        # Check if we're on Windows
        if os.name == 'nt':
            gs_command = "gswin64c"  # 64-bit Ghostscript on Windows
        
        logger.debug(f"Checking for Ghostscript using command: {gs_command}")
            
        # Try to run Ghostscript with version flag
        result = subprocess.run(
            [gs_command, "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            timeout=5
        )
        
        # Log the result for debugging
        logger.debug(f"Ghostscript check result: returncode={result.returncode}, stdout={result.stdout.decode().strip()}, stderr={result.stderr.decode().strip()}")
        
        # If the command succeeded, Ghostscript is installed
        return result.returncode == 0
        
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        # Log the specific error
        logger.debug(f"Ghostscript check failed with error: {str(e)}")
        # If the command failed or the executable wasn't found
        return False

# Run the test when the module is imported
test_result = test_ghostscript_detection()
print(f"Ghostscript detection test result: {test_result}")