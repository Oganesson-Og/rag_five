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
import logging
import base64
import subprocess
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from PIL import Image
from .models import Document
from .base import BaseExtractor, ExtractorResult
import cv2  # For diagram / image analysis
from src.utils.file_utils import get_project_base_directory
import statistics
import time

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
        
        # # Check for Ghostscript if Camelot is available
        # if HAS_CAMELOT:
        #     if not self._check_ghostscript_installed():
        #         self.logger.warning("Ghostscript is not installed, which is required for Camelot table extraction. "
        #                            "You can install it using the instructions here: "
        #                            "https://camelot-py.readthedocs.io/en/master/user/install-deps.html")
        #         self.logger.warning("Table extraction will fall back to alternative methods.")

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
        self._init_ocr()
        self._init_gemini_image_processor()
        self._init_docling_extractor()
        
    def _init_docling_extractor(self):
        """Initialize DoclingExtractor with fallback support."""
        try:
            from ..extractors.docling_extractor import DoclingExtractor
            
            # Get configuration
            pdf_config = self.config
            picture_config = pdf_config.get('picture_annotation', {})
            
            # Get primary API configuration
            primary_api_config = picture_config.get('api_config', {})
            if not primary_api_config and 'api_key' in picture_config:
                # Create API config from individual settings
                primary_api_config = {
                    'url': picture_config.get('url', 'https://generativelanguage.googleapis.com/v1/models/gemini-2.0-pro-exp-02-05:generateContent'),
                    'api_key': picture_config.get('api_key'),
                    'params': {
                        'model': picture_config.get('model_name', 'gemini-2.0-pro-exp-02-05'),
                        'temperature': picture_config.get('temperature', 0.7),
                        'max_output_tokens': picture_config.get('max_tokens', 1024),
                        'top_p': picture_config.get('top_p', 0.9)
                    },
                    'headers': {'Content-Type': 'application/json'},
                    'timeout': picture_config.get('timeout', 90)
                }
            
            # Get secondary API configuration
            secondary_api_config = None
            if 'second_api_key' in picture_config:
                secondary_api_config = {
                    'url': picture_config.get('url', 'https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent'),
                    'api_key': picture_config.get('second_api_key'),
                    'params': {
                        'model': picture_config.get('second_model_name', 'gemini-2.0-flash'),
                        'temperature': picture_config.get('temperature', 0.7),
                        'max_output_tokens': picture_config.get('max_tokens', 1024),
                        'top_p': picture_config.get('top_p', 0.9)
                    },
                    'headers': {'Content-Type': 'application/json'},
                    'timeout': picture_config.get('timeout', 90)
                }
            
            # Initialize DoclingExtractor with both primary and secondary configurations
            self.docling_extractor = DoclingExtractor(
                model_type='remote',
                model_name=picture_config.get('model_name', 'gemini-2.0-pro-exp-02-05'),
                image_scale=picture_config.get('image_scale', 2.0),
                picture_prompt=picture_config.get('prompt', 'Describe the image in three sentences. Be concise and accurate.'),
                api_config=primary_api_config,
                secondary_api_config=secondary_api_config,
                num_threads=pdf_config.get('acceleration', {}).get('num_threads', 8),
                enable_remote_services=True
            )
            
            self.logger.info("DoclingExtractor initialized with fallback support")
            
        except ImportError:
            self.logger.warning("DoclingExtractor not available")
            self.docling_extractor = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize DoclingExtractor: {str(e)}")
            self.docling_extractor = None

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

    def _init_gemini_image_processor(self):
        """Initialize Gemini image processor for image description and classification."""
        try:
            import google.generativeai as genai
            
            # Get configuration
            pdf_config = self.config
            picture_config = pdf_config.get('picture_annotation', {})
            
            self.logger.info(f"Picture annotation config: {picture_config}")
            
            # Check if Gemini is enabled for image processing
            if picture_config.get('enabled', True) and picture_config.get('model_type') == 'gemini':
                self.logger.info("Gemini image processing is enabled in configuration")
                
                # Get primary API key
                model_config = self.config.get('model', {})
                primary_api_key = (
                    picture_config.get('api_key') or
                    model_config.get('gemini_api_key') or 
                    model_config.get('api_key')
                )
                
                # Get secondary API key
                secondary_api_key = (
                    picture_config.get('second_api_key') or
                    pdf_config.get('layout', {}).get('second_api_key') or
                    model_config.get('second_api_key')
                )
                
                self.logger.info(f"Primary API key found: {bool(primary_api_key)}")
                self.logger.info(f"Secondary API key found: {bool(secondary_api_key)}")
                
                # Store both API keys for potential fallback
                self.primary_api_key = primary_api_key
                self.secondary_api_key = secondary_api_key
                
                if not primary_api_key and not secondary_api_key:
                    self.logger.warning("No API keys found for Gemini image processing")
                    self.gemini_image_processor = None
                    return
                
                # Get model names
                self.primary_model_name = picture_config.get('model_name', 'gemini-2.0-pro-exp-02-05')
                self.secondary_model_name = picture_config.get('second_model_name', 'gemini-2.0-flash')
                
                # Initialize with primary API key and model
                self.current_api_key = primary_api_key or secondary_api_key
                self.current_model_name = self.primary_model_name
                
                # Configure Gemini with current API key
                genai.configure(api_key=self.current_api_key)
                
                # Initialize Gemini model
                self.gemini_image_processor = genai.GenerativeModel(self.current_model_name)
                
                # Set generation config
                self.gemini_generation_config = {
                    'temperature': picture_config.get('temperature', 0.7),
                    'top_p': picture_config.get('top_p', 0.9),
                    'max_output_tokens': picture_config.get('max_tokens', 1024),
                }
                
                # Track rate limit status
                self.rate_limited = False
                self.last_rate_limit_time = 0
                self.rate_limit_cooldown = picture_config.get('rate_limit_cooldown', 60)  # seconds
                
                self.logger.info(f"Gemini image processor initialized with model: {self.current_model_name}")
            else:
                self.logger.info(f"Gemini image processing not enabled in configuration. Enabled: {picture_config.get('enabled', True)}, Model type: {picture_config.get('model_type')}")
                self.gemini_image_processor = None
        except ImportError:
            self.logger.warning("Google Generative AI package not available, Gemini image processing disabled")
            self.gemini_image_processor = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize Gemini image processor: {str(e)}")
            self.gemini_image_processor = None

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
        """
        Process a text block from PyMuPDF.
        
        Args:
            block: Text block dictionary from PyMuPDF
            base_font_size: Base font size for determining element type
            
        Returns:
            LayoutElement or None if processing fails
        """
        try:
            # Check if 'spans' key exists directly in the block
            if 'spans' in block:
                spans = block['spans']
            # Check if 'lines' exists and contains 'spans'
            elif 'lines' in block and block['lines'] and 'spans' in block['lines'][0]:
                # Use the spans from the first line
                spans = block['lines'][0]['spans']
            else:
                self.logger.warning(f"Text block missing 'spans' key: {block}")
                return None
                
            # Get the first span for font information
            if not spans:
                return None
                
            # Extract text from all spans
            text = ""
            for s in spans:
                text += s.get('text', '')
            
            # Get font information
            font_info = spans[0]
            font_size = font_info.get('size', 0)
            
            # Check if font is bold
            font_name = font_info.get('font', '')
            is_bold = False
            
            # Check if font_name is a string before using lower()
            if isinstance(font_name, str):
                is_bold = 'bold' in font_name.lower() or '-b' in font_name.lower()
            
            # Check if flags is iterable before checking if 'bold' is in it
            flags = font_info.get('flags', 0)
            if isinstance(flags, int):
                # For integer flags, we can't use 'in' operator
                # This might be a bitmask or other numeric representation
                pass  # Handle according to your specific flag encoding
            elif hasattr(flags, '__iter__'):  # Check if it's iterable
                is_bold = is_bold or 'bold' in flags
            
            # Determine element type (heading, title, normal text...)
            element_type = self._determine_element_type(font_size, base_font_size, is_bold)
            
            # Create layout element
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
        """
        Calculate the median font size on a page to establish a baseline for
        determining element types.
        """
        font_sizes = []
        for block in page_dict.get('blocks', []):
            if block.get('type') == 0:  # text block
                # Check if 'spans' key exists directly in the block
                if 'spans' in block and block['spans']:
                    for span in block['spans']:
                        if 'size' in span:
                            font_sizes.append(span['size'])
                # Check if 'lines' exists and contains 'spans'
                elif 'lines' in block and block['lines']:
                    for line in block['lines']:
                        if 'spans' in line and line['spans']:
                            for span in line['spans']:
                                if 'size' in span:
                                    font_sizes.append(span['size'])
        
        if not font_sizes:
            return 11.0  # Default font size if none found
        
        return statistics.median(font_sizes)

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
        """
        Extract tables from the document using the PDF path.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            List of extracted tables with metadata
        """
        # Get the temporary file path
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            pdf_path = temp_file.name
            doc.save(pdf_path)
        
        try:
            # Extract tables using the unified approach
            return self.extract_tables(doc, pdf_path)
        finally:
            # Clean up the temporary file
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
                
    def _extract_images_with_context(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """
        Extract images from the document with contextual information.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            List of extracted images with metadata and context
        """
        images = []
        
        try:
            for page_num, page in enumerate(doc):
                # Extract images from the page
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        
                        if base_image:
                            # Get image data
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            
                            # Get image position on page
                            for img_rect in page.get_image_rects(xref):
                                # Create image metadata
                                image_data = {
                                    'page': page_num + 1,
                                    'index': img_index,
                                    'bbox': [img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1],
                                    'width': img_rect.width,
                                    'height': img_rect.height,
                                    'format': image_ext,
                                    'size_bytes': len(image_bytes),
                                    'context': self._get_image_context(page, img_rect),
                                }
                                
                                # Process with Gemini if available
                                if hasattr(self, 'gemini_image_processor') and self.gemini_image_processor:
                                    try:
                                        description, diagram_type = self._process_image_with_gemini(image_bytes)
                                        image_data['description'] = description
                                        image_data['metadata'] = {
                                            'is_diagram': diagram_type != DiagramType.UNKNOWN.value,
                                            'diagram_type': diagram_type
                                        }
                                    except Exception as e:
                                        self.logger.warning(f"Gemini image processing failed: {str(e)}")
                                        # Fall back to basic classification
                                        image_data['metadata'] = {
                                            'is_diagram': self._is_likely_diagram(image_bytes, image_ext),
                                            'diagram_type': self._classify_diagram_type(image_bytes, image_ext) if self._is_likely_diagram(image_bytes, image_ext) else DiagramType.UNKNOWN.value
                                        }
                                else:
                                    # Use basic classification
                                    image_data['metadata'] = {
                                        'is_diagram': self._is_likely_diagram(image_bytes, image_ext),
                                        'diagram_type': self._classify_diagram_type(image_bytes, image_ext) if self._is_likely_diagram(image_bytes, image_ext) else DiagramType.UNKNOWN.value
                                    }
                                
                                # Add base64 encoded image
                                image_data['data'] = base64.b64encode(image_bytes).decode('utf-8')
                                
                                images.append(image_data)
                    except Exception as e:
                        self.logger.warning(f"Failed to extract image {img_index} on page {page_num+1}: {str(e)}")
                        continue
        except Exception as e:
            self.logger.warning(f"Image extraction process failed: {str(e)}")
            
        return images
    
    def _get_image_context(self, page: fitz.Page, rect: fitz.Rect, context_range: int = 200) -> str:
        """
        Get the text context around an image.
        
        Args:
            page: PyMuPDF page object
            rect: Rectangle defining the image position
            context_range: Range in points to consider for context
            
        Returns:
            Text context around the image
        """
        # Expand the rectangle to include surrounding text
        context_rect = fitz.Rect(
            max(0, rect.x0 - context_range),
            max(0, rect.y0 - context_range),
            min(page.rect.width, rect.x1 + context_range),
            min(page.rect.height, rect.y1 + context_range)
        )
        
        # Get text in the expanded rectangle
        return page.get_text("text", clip=context_rect).strip()
    
    def _is_likely_diagram(self, image_bytes: bytes, image_ext: str) -> bool:
        """
        Determine if an image is likely to be a diagram based on simple heuristics.
        
        Args:
            image_bytes: Raw image data
            image_ext: Image format extension
            
        Returns:
            True if the image is likely a diagram, False otherwise
        """
        try:
            # Convert image bytes to numpy array
            image = Image.open(io.BytesIO(image_bytes))
            img_array = np.array(image.convert('RGB'))
            
            # Simple heuristics for diagrams:
            # 1. High percentage of white/light background
            # 2. Limited color palette
            # 3. Presence of straight lines/geometric shapes
            
            # Check for white/light background
            is_light_background = np.mean(img_array) > 200
            
            # Check for limited color palette
            colors = np.unique(img_array.reshape(-1, 3), axis=0)
            limited_palette = len(colors) < 50
            
            return is_light_background and limited_palette
        except Exception as e:
            self.logger.warning(f"Diagram detection failed: {str(e)}")
            return False
    
    def _classify_diagram_type(self, image_bytes: bytes, image_ext: str) -> str:
        """
        Classify the type of diagram if possible.
        
        Args:
            image_bytes: Raw image data
            image_ext: Image format extension
            
        Returns:
            Diagram type as string
        """
        # If Gemini is available, use it for classification
        if hasattr(self, 'gemini_image_processor') and self.gemini_image_processor:
            try:
                description, diagram_type = self._process_image_with_gemini(image_bytes)
                if diagram_type:
                    return diagram_type
            except Exception as e:
                self.logger.warning(f"Gemini diagram classification failed: {str(e)}")
        
        # This would ideally use a ML model for classification
        # For now, return a default type
        return DiagramType.UNKNOWN.value
    
    def _process_image_with_gemini(self, image_bytes: bytes) -> Tuple[str, str]:
        """
        Process an image using Gemini to get a description and classify the image type.
        
        Args:
            image_bytes: Raw image data
            
        Returns:
            Tuple of (description, diagram_type)
        """
        if not hasattr(self, 'gemini_image_processor') or self.gemini_image_processor is None:
            self.logger.warning("Gemini image processor not initialized")
            return "", DiagramType.UNKNOWN.value
            
        try:
            import google.generativeai as genai
            from PIL import Image
            import time
            
            # Check if we should try switching back to primary API key after cooldown
            current_time = time.time()
            if (self.rate_limited and 
                hasattr(self, 'last_rate_limit_time') and 
                current_time - self.last_rate_limit_time > self.rate_limit_cooldown):
                self.logger.info("Cooldown period elapsed, attempting to switch back to primary API key")
                self._switch_to_primary_api()
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Prepare the prompt for image description
            description_prompt = """
            Describe this image in detail. If it's a diagram, chart, or technical illustration, 
            explain what it represents and its key components. Be concise but thorough.
            """
            
            # Prepare the prompt for image classification
            classification_prompt = """
            Classify this image into one of the following categories:
            - chart
            - graph
            - scientific_diagram
            - flowchart
            - concept_map
            - geometric_figure
            - circuit_diagram
            - chemical_structure
            - mathematical_plot
            - anatomical_diagram
            - architectural_drawing
            - historical_map
            - unknown
            
            Return only the category name, nothing else.
            """
            
            # Get description
            try:
                description_response = self.gemini_image_processor.generate_content(
                    [description_prompt, image],
                    generation_config=self.gemini_generation_config if hasattr(self, 'gemini_generation_config') else None
                )
                description = description_response.text.strip()
            except Exception as e:
                if "429" in str(e) or "Resource has been exhausted" in str(e):
                    self.logger.warning(f"Rate limit hit for description generation: {str(e)}")
                    # Try fallback
                    if self._switch_to_fallback_api():
                        # Retry with new API key
                        try:
                            description_response = self.gemini_image_processor.generate_content(
                                [description_prompt, image],
                                generation_config=self.gemini_generation_config
                            )
                            description = description_response.text.strip()
                        except Exception as retry_e:
                            self.logger.error(f"Fallback also failed for description: {str(retry_e)}")
                            description = ""
                    else:
                        description = ""
                else:
                    self.logger.warning(f"Error generating description: {str(e)}")
                    description = ""
            
            # Get classification
            try:
                classification_response = self.gemini_image_processor.generate_content(
                    [classification_prompt, image],
                    generation_config=self.gemini_generation_config if hasattr(self, 'gemini_generation_config') else None
                )
                classification = classification_response.text.strip().lower()
            except Exception as e:
                if "429" in str(e) or "Resource has been exhausted" in str(e):
                    self.logger.warning(f"Rate limit hit for classification: {str(e)}")
                    # Try fallback
                    if self._switch_to_fallback_api():
                        # Retry with new API key
                        try:
                            classification_response = self.gemini_image_processor.generate_content(
                                [classification_prompt, image],
                                generation_config=self.gemini_generation_config
                            )
                            classification = classification_response.text.strip().lower()
                        except Exception as retry_e:
                            self.logger.error(f"Fallback also failed for classification: {str(retry_e)}")
                            classification = DiagramType.UNKNOWN.value
                    else:
                        classification = DiagramType.UNKNOWN.value
                else:
                    self.logger.warning(f"Error generating classification: {str(e)}")
                    classification = DiagramType.UNKNOWN.value
            
            # Validate classification against DiagramType enum
            valid_types = [t.value for t in DiagramType]
            if classification not in valid_types:
                classification = DiagramType.UNKNOWN.value
                
            return description, classification
            
        except Exception as e:
            self.logger.warning(f"Image processing with Gemini failed: {str(e)}")
            return "", DiagramType.UNKNOWN.value
    
    def _switch_to_fallback_api(self) -> bool:
        """
        Switch to the fallback API key and model when rate limited.
        
        Returns:
            bool: True if successfully switched to fallback, False otherwise
        """
        import google.generativeai as genai
        import time
        
        # If we're already using the secondary API key, no fallback available
        if self.current_api_key == self.secondary_api_key:
            self.logger.warning("Already using secondary API key, no further fallback available")
            return False
            
        # Check if we have a secondary API key
        if not hasattr(self, 'secondary_api_key') or not self.secondary_api_key:
            self.logger.warning("No secondary API key available for fallback")
            return False
            
        try:
            self.logger.info("Switching to secondary API key and model due to rate limiting")
            self.current_api_key = self.secondary_api_key
            self.current_model_name = self.secondary_model_name
            
            # Configure Gemini with secondary API key
            genai.configure(api_key=self.current_api_key)
            
            # Initialize new Gemini model with secondary model
            self.gemini_image_processor = genai.GenerativeModel(self.current_model_name)
            
            # Update rate limit tracking
            self.rate_limited = True
            self.last_rate_limit_time = time.time()
            
            self.logger.info(f"Successfully switched to secondary model: {self.current_model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to switch to secondary API key: {str(e)}")
            return False
    
    def _switch_to_primary_api(self) -> bool:
        """
        Switch back to the primary API key and model after cooldown.
        
        Returns:
            bool: True if successfully switched back to primary, False otherwise
        """
        import google.generativeai as genai
        
        # If we're already using the primary API key, nothing to do
        if self.current_api_key == self.primary_api_key:
            return True
            
        # Check if we have a primary API key
        if not hasattr(self, 'primary_api_key') or not self.primary_api_key:
            self.logger.warning("No primary API key available to switch back to")
            return False
            
        try:
            self.logger.info("Switching back to primary API key and model after cooldown")
            self.current_api_key = self.primary_api_key
            self.current_model_name = self.primary_model_name
            
            # Configure Gemini with primary API key
            genai.configure(api_key=self.current_api_key)
            
            # Initialize new Gemini model with primary model
            self.gemini_image_processor = genai.GenerativeModel(self.current_model_name)
            
            # Reset rate limit tracking
            self.rate_limited = False
            
            self.logger.info(f"Successfully switched back to primary model: {self.current_model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to switch back to primary API key: {str(e)}")
            return False

    def _extract_pdf_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """
        Extract metadata from the PDF document.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        try:
            # Extract basic metadata
            meta = doc.metadata
            if meta:
                for key, value in meta.items():
                    if value:
                        metadata[key.lower()] = value
            
            # Add document structure information
            metadata['page_count'] = len(doc)
            metadata['has_toc'] = len(doc.get_toc()) > 0
            
            # Extract document structure
            structure = self._process_document_structure(doc)
            if structure:
                metadata['structure'] = structure
                
            # Add language detection
            text_sample = ""
            for i in range(min(3, len(doc))):
                text_sample += doc[i].get_text("text")
                if len(text_sample) > 1000:
                    break
                    
            if text_sample:
                try:
                    from langdetect import detect
                    metadata['language'] = detect(text_sample)
                except Exception:
                    metadata['language'] = "unknown"
                    
        except Exception as e:
            self.logger.warning(f"Metadata extraction failed: {str(e)}")
            
        return metadata
    
    def _add_cross_references(self, document: 'Document') -> None:
        """
        Add cross-references between different elements in the document.
        
        Args:
            document: Document object to update
        """
        try:
            # Initialize cross-references
            document.doc_info['cross_references'] = []
            
            # Get tables and images
            tables = document.doc_info.get('tables', [])
            images = document.doc_info.get('images', [])
            
            # Find references to tables in text
            text = document.content
            
            # Simple pattern matching for table references
            table_patterns = [
                r'table\s+(\d+)',
                r'tab\.\s*(\d+)',
                r'tab\s+(\d+)'
            ]
            
            for pattern in table_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    table_num = int(match.group(1))
                    # Find corresponding table
                    for table in tables:
                        if table.get('table_number') == table_num:
                            document.doc_info['cross_references'].append({
                                'type': 'table_reference',
                                'reference_id': table.get('id', ''),
                                'text_position': match.start(),
                                'text_context': text[max(0, match.start()-50):min(len(text), match.end()+50)]
                            })
                            
            # Similar for figures/images
            figure_patterns = [
                r'figure\s+(\d+)',
                r'fig\.\s*(\d+)',
                r'fig\s+(\d+)'
            ]
            
            for pattern in figure_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    figure_num = int(match.group(1))
                    # Find corresponding image
                    for image in images:
                        if image.get('figure_number') == figure_num:
                            document.doc_info['cross_references'].append({
                                'type': 'figure_reference',
                                'reference_id': image.get('id', ''),
                                'text_position': match.start(),
                                'text_context': text[max(0, match.start()-50):min(len(text), match.end()+50)]
                            })
                            
        except Exception as e:
            self.logger.warning(f"Cross-reference extraction failed: {str(e)}")
    
    def _check_if_scanned(self, doc: fitz.Document) -> bool:
        """
        Check if the PDF appears to be a scanned document.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            True if the document appears to be scanned, False otherwise
        """
        # Sample a few pages
        pages_to_check = min(3, len(doc))
        text_density = []
        image_coverage = []
        
        for i in range(pages_to_check):
            page = doc[i]
            
            # Check text density
            text = page.get_text("text")
            text_length = len(text)
            page_area = page.rect.width * page.rect.height
            text_density.append(text_length / page_area)
            
            # Check image coverage
            images = page.get_images(full=True)
            total_image_area = 0
            
            for img in images:
                for rect in page.get_image_rects(img[0]):
                    total_image_area += rect.width * rect.height
                    
            image_coverage.append(total_image_area / page_area)
        
        # Heuristics for scanned document:
        # 1. Low text density
        # 2. High image coverage
        avg_text_density = sum(text_density) / len(text_density) if text_density else 0
        avg_image_coverage = sum(image_coverage) / len(image_coverage) if image_coverage else 0
        
        return avg_text_density < 0.01 and avg_image_coverage > 0.5

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
                    # Import inside the method to avoid circular imports
                    import importlib
                    table_structure_recognizer_module = importlib.import_module("..core.vision.table_structure_recognizer", package=__package__)
                    TableStructureRecognizer = getattr(table_structure_recognizer_module, "TableStructureRecognizer")
                    table_structure_recognizer = TableStructureRecognizer(config_path)
                    self.logger.info("TableStructureRecognizer initialized successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize TableStructureRecognizer: {str(e)}")
            else:
                self.logger.warning(f"Config file not found at {config_path}, skipping TableStructureRecognizer initialization")
        except Exception as e:
            self.logger.warning(f"Failed to initialize TableStructureRecognizer: {str(e)}")
            
        # Get total number of pages
        total_pages = len(doc)
            
        # Process each page
        for page_num, page in enumerate(doc):
            page_number = page_num + 1  # Convert to 0-based to 1-based page numbering
            
            # Safety check to ensure we don't process pages beyond the document
            if page_number > total_pages:
                self.logger.warning(f"Skipping page {page_number} as it exceeds the document's total pages ({total_pages})")
                continue
                
            try:
                # Step 1: Analyze page to determine table characteristics
                has_borders = self._detect_table_borders(page)
                
                # Step 2: Extract tables using the unified strategy
                page_tables = self._extract_tables_unified(pdf_path, page_number, page, has_borders)
                
                # Step 3: Apply advanced structure analysis to refine the tables
                if table_structure_recognizer and page_tables:
                    try:
                        for i, table in enumerate(page_tables):
                            page_tables[i] = self._refine_table_structure(table)
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
                all_tables = self._handle_cross_page_tables(all_tables)
            except Exception as e:
                self.logger.warning(f"Cross-page table handling failed: {str(e)}")
            
        return all_tables

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
            page_number: Page number (1-indexed)
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
            
        self.logger.info(f"Using {method} for table extraction on page {page_number}")
        
        # Try the primary extraction method
        if method == 'camelot':
            try:
                if has_borders:
                    tables = self._extract_with_camelot(pdf_path, page_number, flavor='lattice')
                else:
                    tables = self._extract_with_camelot(pdf_path, page_number, flavor='stream')
            except Exception as e:
                self.logger.warning(f"Table extraction with {method} failed: {str(e)}")
                
        elif method == 'tabula':
            try:
                if has_borders:
                    tables = self._extract_with_tabula(pdf_path, page_number, lattice=True)
                else:
                    tables = self._extract_with_tabula(pdf_path, page_number, lattice=False, guess=True)
            except Exception as e:
                self.logger.warning(f"Table extraction with {method} failed: {str(e)}")
                
        elif method == 'pdfplumber':
            try:
                tables = self._extract_with_pdfplumber(pdf_path, page_number)
            except Exception as e:
                self.logger.warning(f"Table extraction with {method} failed: {str(e)}")
        
        # If primary method failed, try fallback methods
        if not tables and self.config.get('table_extraction', {}).get('fallback_to_heuristic', True):
            self.logger.info(f"Primary extraction method failed, trying fallback methods")
            
            # Try camelot if not already tried
            if method != 'camelot':
                try:
                    if has_borders:
                        tables = self._extract_with_camelot(pdf_path, page_number, flavor='lattice')
                    else:
                        tables = self._extract_with_camelot(pdf_path, page_number, flavor='stream')
                except Exception as e:
                    self.logger.warning(f"Fallback to camelot failed: {str(e)}")
            
            # Try tabula if not already tried and camelot fallback failed
            if not tables and method != 'tabula':
                try:
                    if has_borders:
                        tables = self._extract_with_tabula(pdf_path, page_number, lattice=True)
                    else:
                        tables = self._extract_with_tabula(pdf_path, page_number, lattice=False, guess=True)
                except Exception as e:
                    self.logger.warning(f"Fallback to tabula failed: {str(e)}")
            
            # Try pdfplumber as last resort
            if not tables and method != 'pdfplumber':
                try:
                    tables = self._extract_with_pdfplumber(pdf_path, page_number)
                except Exception as e:
                    self.logger.warning(f"Fallback to pdfplumber failed: {str(e)}")
        
        # Get context for each table
        for table in tables:
            try:
                if 'bbox' in table:
                    table['context'] = self._get_table_context(page, table['bbox'])
                    # Store the bbox as region for consistency
                    table['region'] = table['bbox']
            except Exception as e:
                self.logger.warning(f"Failed to get table context on page {page_number}: {str(e)}")
                table['context'] = ""
        
        return tables

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
            tables.sort(key=lambda t: (t['page'], t.get('bbox', [0, 0, 0, 0])[1]))
            
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
        if table1.get('num_cols', 0) != table2.get('num_cols', 0):
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
        if not isinstance(s1, str) or not isinstance(s2, str):
            s1 = str(s1) if s1 is not None else ""
            s2 = str(s2) if s2 is not None else ""
            
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
        # Create a new table with merged data
        merged_table = {
            'data': table1['data'] + table2['data'],
            'headers': table1['headers'],  # Keep headers from first table
            'page': table1['page'],  # Keep page from first table
            'bbox': table1['bbox'],  # Keep bbox from first table
            'confidence': (table1.get('confidence', 0.5) + table2.get('confidence', 0.5)) / 2,  # Average confidence
            'extraction_method': table1.get('extraction_method', 'merged'),
            'is_merged': True,
            'merged_from': [table1.get('page', 0), table2.get('page', 0)]
        }
        
        return merged_table
        
    def _refine_table_structure(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine the structure of an extracted table.
        
        This method improves table structure by:
        1. Removing empty rows and columns
        2. Identifying and fixing merged cells
        3. Normalizing header rows
        4. Improving data type detection
        
        Args:
            table: The table dictionary containing data and metadata
            
        Returns:
            Refined table dictionary
        """
        if not table or 'data' not in table or not table['data']:
            return table
            
        # Get table data
        data = table['data']
        
        # 1. Remove completely empty rows
        data = [row for row in data if any(cell.strip() if isinstance(cell, str) else cell for cell in row)]
        
        # 2. Remove completely empty columns
        if data:
            # Transpose to work with columns
            transposed = list(zip(*data))
            # Filter empty columns
            transposed = [col for col in transposed if any(cell.strip() if isinstance(cell, str) else cell for cell in col)]
            # Transpose back to rows
            if transposed:
                data = list(zip(*transposed))
        
        # 3. Normalize header row if present
        headers = table.get('headers', [])
        if not headers and len(data) > 0:
            # Use first row as header if not explicitly defined
            headers = data[0]
            data = data[1:]
        
        # 4. Clean and normalize headers
        cleaned_headers = []
        for header in headers:
            if isinstance(header, str):
                # Remove extra whitespace and normalize
                cleaned_header = ' '.join(header.strip().split())
                cleaned_headers.append(cleaned_header)
            else:
                cleaned_headers.append(header)
        
        # 5. Clean data cells
        cleaned_data = []
        for row in data:
            cleaned_row = []
            for cell in row:
                if isinstance(cell, str):
                    # Remove extra whitespace and normalize
                    cleaned_cell = ' '.join(cell.strip().split())
                    cleaned_row.append(cleaned_cell)
                else:
                    cleaned_row.append(cell)
            cleaned_data.append(cleaned_row)
        
        # Update table with refined data
        refined_table = table.copy()
        refined_table['data'] = cleaned_data
        refined_table['headers'] = cleaned_headers
        refined_table['refined'] = True
        
        return refined_table

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

# # Run the test when the module is imported
# test_result = test_ghostscript_detection()
# print(f"Ghostscript detection test result: {test_result}")