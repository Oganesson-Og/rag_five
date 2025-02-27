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
from typing import Dict, Any, List, Optional, Tuple, Generator
from dataclasses import dataclass
from .models import Document
from .base import BaseExtractor, ExtractorResult
from enum import Enum
import cv2  # For diagram / image analysis

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
                'device': 'auto'  # 'auto', 'cuda', 'mps', 'cpu'
            }
        }

        if config:
            default_config.update(config)

        super().__init__(config=default_config)
        self._init_components()
        self.base_font_size = 12.0  # Updated during processing

        # Determine device for acceleration
        self.device = self._determine_device()
        
        # Initialize OCR if requested
        self.ocr = self._init_ocr()

        # Initialize Docling extractor with acceleration and picture annotation support
        try:
            from ..extractors.docling_extractor import DoclingExtractor
            picture_config = default_config.get('picture_annotation', {})
            
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
                num_threads=default_config['acceleration'].get('num_threads', 8)
            )
            self.has_docling = True
            self.logger.info(f"Docling extraction initialized successfully on {self.device}")
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
            from ..core.vision.layout_recognizer import LayoutRecognizer

            layout_config = self.config.get('layout', {})
            self.layout_recognizer = LayoutRecognizer(
                model_dir=layout_config.get('model_dir', ""),
                device=layout_config.get('device', "cuda"),
                batch_size=layout_config.get('batch_size', 32),
                cache_dir=layout_config.get('cache_dir'),
                model_type=layout_config.get('model_type', 'yolov10'),
                confidence=layout_config.get('confidence', 0.5),
                merge_boxes=layout_config.get('merge_boxes', True),
                label_list=layout_config.get('label_list', [
                    "title", "text", "list", "table", "figure",
                    "header", "footer", "sidebar", "caption"
                ]),
                task_name=layout_config.get('task_name', 'document_layout')
            )
            self.has_layout_recognition = True
            self.logger.info("Layout recognition initialized successfully")

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
                docling_result = self.docling.process_file(content)
                if docling_result.success:
                    document.content = docling_result.text
                else:
                    self.logger.warning(f"Docling extraction failed: {docling_result.error}")
                    # Fall back to default text extraction
                    document.content = self._extract_structured_text(doc, {})
            else:
                # Use default text extraction
                document.content = self._extract_structured_text(doc, {})
            
            # Extract tables using voting/adaptive approach
            tables = self._extract_tables_with_structure(doc)
            if tables:
                document.doc_info['tables'] = tables
                
                # Add table context and references
                for table in tables:
                    if 'region' in table:
                        table['context'] = self._get_table_context(doc[table.get('page', 1) - 1], table['region'])
            
            # Extract and analyze images/diagrams with enhanced context
            images = self._extract_images_with_context(doc)
            if images:
                document.doc_info['images'] = images
                # Separate diagrams from regular images
                diagrams = [img for img in images if img['metadata'].get('is_diagram')]
                if diagrams:
                    document.doc_info['diagrams'] = diagrams
            
            # Add metadata
            document.doc_info.update(self._extract_pdf_metadata(doc))
            
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
        """
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
        """Extract tables using Camelot, tabula, or pdfplumber. Falls back to heuristics as needed."""
        # Re-uses existing code from your original snippet, with references removed for OCR.
        tables = []
        table_config = self.config.get('table_extraction', {})
        method = table_config.get('method', 'auto')
        strategy = table_config.get('strategy', 'adaptive')

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = temp_file.name
            doc.save(temp_path)

            try:
                for page_num, page in enumerate(doc):
                    page_tables = self._extract_page_tables(page_num, page, temp_path, table_config, method, strategy)
                    tables.extend(page_tables)
            except Exception as e:
                self.logger.error(f"Table extraction error: {str(e)}")
            finally:
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    self.logger.warning(f"Failed to delete temporary PDF: {str(e)}")

        return tables

    def _extract_page_tables(self, page_num, page, pdf_path, table_config, method, strategy):
        """Helper that attempts to extract tables from a single page."""
        page_tables = []
        table_candidates = []

        if strategy == 'vote':
            # Try all methods, gather results, pick best.
            if HAS_CAMELOT:
                c_tables = self._extract_tables_camelot(pdf_path, page_num+1, table_config)
                table_candidates.append(('camelot', c_tables))
            if HAS_TABULA:
                t_tables = self._extract_tables_tabula(pdf_path, page_num+1, table_config)
                table_candidates.append(('tabula', t_tables))
            if HAS_PDFPLUMBER:
                p_tables = self._extract_tables_pdfplumber(pdf_path, page_num+1, table_config)
                table_candidates.append(('pdfplumber', p_tables))

            page_tables = self._select_best_tables_by_voting(table_candidates)

        else:
            # 'adaptive' or 'sequential' approach.
            # Decide table type, prefer a particular library, else fallback.
            table_type = self._detect_table_type(page) if strategy == 'adaptive' else None
            preferred_extractor = table_config.get('table_types', {}).get(table_type)

            if strategy == 'adaptive' and preferred_extractor:
                page_tables = self._try_preferred_extractor(
                    preferred_extractor, pdf_path, page_num+1, table_config
                )

            if not page_tables:
                # Fallback to method or auto sequence.
                if method in ['auto', 'camelot'] and HAS_CAMELOT and not page_tables:
                    page_tables = self._extract_tables_camelot(pdf_path, page_num+1, table_config)
                if method in ['auto', 'tabula'] and HAS_TABULA and not page_tables:
                    page_tables = self._extract_tables_tabula(pdf_path, page_num+1, table_config)
                if method in ['auto', 'pdfplumber'] and HAS_PDFPLUMBER and not page_tables:
                    page_tables = self._extract_tables_pdfplumber(pdf_path, page_num+1, table_config)

        # If still no tables found, try simple heuristic.
        if not page_tables and table_config.get('fallback_to_heuristic', True):
            self.logger.info(f"Using heuristic table detection on page {page_num+1}")
            tables_heuristic = self._heuristic_table_detection(page)
            page_tables.extend(tables_heuristic)

        # Add page info, etc.
        for tb in page_tables:
            tb['page'] = page_num + 1

        return page_tables

    def _try_preferred_extractor(self, extractor_name, pdf_path, page_number, config):
        """Attempt to use the user-preferred extractor for this page/table type."""
        if extractor_name == 'camelot' and HAS_CAMELOT:
            return self._extract_tables_camelot(pdf_path, page_number, config)
        elif extractor_name == 'tabula' and HAS_TABULA:
            return self._extract_tables_tabula(pdf_path, page_number, config)
        elif extractor_name == 'pdfplumber' and HAS_PDFPLUMBER:
            return self._extract_tables_pdfplumber(pdf_path, page_number, config)
        return []

    def _extract_tables_camelot(
        self, 
        pdf_path: str, 
        page_number: int, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract tables using Camelot library."""
        if not HAS_CAMELOT:
            return []
        
        tables = []
        try:
            # Extract tables using specified flavor
            flavor = config.get('flavor', 'lattice')
            line_scale = config.get('line_scale', 40)
            
            camelot_tables = camelot.read_pdf(
                pdf_path,
                pages=str(page_number),
                flavor=flavor,
                line_scale=line_scale
            )
            
            # Process each table
            for table in camelot_tables:
                # Skip empty tables
                if table.df.empty:
                    continue
                
                # Get confidence score
                accuracy = table.accuracy
                if accuracy < config.get('min_confidence', 80):
                    continue
                
                # Convert to standard format
                df = table.df
                header = None
                rows = df.values.tolist()
                
                # Extract header if configured
                if config.get('header_extraction', True) and len(rows) > 0:
                    header = rows[0]
                    rows = rows[1:]
                
                data = {
                    'header': header,
                    'rows': rows,
                    'shape': (len(rows), len(rows[0]) if rows and rows[0] else 0)
                }
                
                # Get table region
                region = None
                if hasattr(table, 'coords'):
                    x1, y1, x2, y2 = table.coords
                    region = (x1, y1, x2, y2)
                
                tables.append({
                    'data': data,
                    'region': region,
                    'confidence': accuracy,
                    'detection_method': f'camelot_{flavor}',
                    'source': 'camelot'
                })
                
        except Exception as e:
            self.logger.warning(f"Camelot table extraction failed: {str(e)}")
        
        return tables

    def _extract_tables_tabula(
        self, 
        pdf_path: str, 
        page_number: int, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract tables using Tabula library."""
        if not HAS_TABULA:
            return []
        
        tables = []
        try:
            # Extract tables
            tabula_tables = tabula.read_pdf(
                pdf_path,
                pages=page_number,
                multiple_tables=True,
                guess=True,  # Enable automatic table detection
                lattice=False,  # Better for borderless tables
                stream=True,   # Better for stream-based tables
                pandas_options={'header': None}  # Don't assume first row is header
            )
            
            # Process each table
            for table_df in tabula_tables:
                # Skip empty tables
                if table_df.empty:
                    continue
                
                # Clean the dataframe
                # Replace NaN with empty string and convert all cells to string
                table_df = table_df.fillna('')
                table_df = table_df.astype(str)
                
                # Extract header if configured
                header = None
                rows = table_df.values.tolist()
                
                if config.get('header_extraction', True) and len(rows) > 0:
                    header = rows[0]
                    rows = rows[1:]
                
                # Convert to standard format
                data = {
                    'header': header,
                    'rows': rows,
                    'shape': (len(rows), len(rows[0]) if rows and rows[0] else 0)
                }
                
                # Note: Tabula doesn't provide confidence scores or exact regions
                # We'll use a default confidence and None for region
                tables.append({
                    'data': data,
                    'region': None,
                    'confidence': 85,  # Default confidence for Tabula
                    'detection_method': 'tabula_stream',
                    'source': 'tabula'
                })
                
        except Exception as e:
            self.logger.warning(f"Tabula table extraction failed: {str(e)}")
        
        return tables

    def _extract_tables_pdfplumber(
        self, 
        pdf_path: str, 
        page_num: int,
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber library."""
        if not HAS_PDFPLUMBER:
            return []
            
        tables = []
        try:
            # Open the PDF with pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                # Get the specific page (adjust for 0-based indexing)
                page = pdf.pages[page_num - 1]
                
                # Extract tables
                plumber_tables = page.extract_tables()
                
                # Process extracted tables
                for i, table_data in enumerate(plumber_tables):
                    if not table_data or len(table_data) == 0:
                        continue
                    
                    # Clean the table data (remove None values)
                    cleaned_table = [[cell or "" for cell in row] for row in table_data]
                    
                    # Extract header if configured
                    header = None
                    rows = cleaned_table
                    if config.get('header_extraction', True) and len(cleaned_table) > 0:
                        header = cleaned_table[0]
                        rows = cleaned_table[1:]
                    
                    # Convert to standard format
                    data = {
                        'header': header,
                        'rows': rows,
                        'shape': (len(rows), len(rows[0]) if rows and rows[0] else 0)
                    }
                    
                    # Get table bounding box if available
                    # pdfplumber doesn't directly provide this, 
                    # but we can get it from the table's cells
                    region = None
                    try:
                        table_cells = page.find_tables()[i].cells
                        if table_cells:
                            # Get min/max x and y from all cells
                            x0 = min(cell[0] for cell in table_cells.values())
                            y0 = min(cell[1] for cell in table_cells.values())
                            x1 = max(cell[2] for cell in table_cells.values())
                            y1 = max(cell[3] for cell in table_cells.values())
                            region = (x0, y0, x1, y1)
                    except (IndexError, AttributeError):
                        # If we can't get the region, that's okay
                        pass
                    
                    tables.append({
                        'data': data,
                        'region': region,
                        'detection_method': 'pdfplumber',
                        'confidence': 85  # pdfplumber doesn't provide confidence scores
                    })
                
        except Exception as e:
            self.logger.warning(f"pdfplumber table extraction failed: {str(e)}")
            
        return tables

    def _select_best_tables_by_voting(self, table_candidates: List[Tuple[str, List[Dict]]]) -> List[Dict]:
        """Select best tables using a voting approach where multiple extractors processed the same tables."""
        if not table_candidates:
            return []
            
        # If only one extractor provided results, use those
        if len(table_candidates) == 1:
            return table_candidates[0][1]
            
        # Get all tables with their source
        all_tables = []
        for source, tables in table_candidates:
            for table in tables:
                table['source'] = source
                all_tables.append(table)
                
        # Group tables by similar regions
        region_groups = []
        for table in all_tables:
            region = table.get('region')
            if not region:
                # If no region, treat as unique table
                region_groups.append([table])
                continue
                
            # Check if this table overlaps with existing groups
            assigned = False
            for group in region_groups:
                if any(self._regions_overlap(region, t.get('region')) for t in group if t.get('region')):
                    group.append(table)
                    assigned = True
                    break
                    
            if not assigned:
                # Create new group
                region_groups.append([table])
                
        # Select best table from each group
        best_tables = []
        for group in region_groups:
            if len(group) == 1:
                best_tables.append(group[0])
            else:
                # Select based on confidence and source preference
                best_table = max(group, key=lambda t: (
                    t.get('confidence', 0),
                    2 if t.get('source') == 'camelot' and t.get('detection_method') == 'lattice' else
                    1.5 if t.get('source') == 'pdfplumber' else 
                    1 if t.get('source') == 'camelot' else
                    0.5
                ))
                best_tables.append(best_table)
                
        return best_tables

    def _detect_table_type(self, page: fitz.Page) -> str:
        """Detect table type on the page to determine best extractor."""
        # Get page content
        page_dict = page.get_text("dict")
        
        # Look for horizontal and vertical lines
        horizontal_lines = 0
        vertical_lines = 0
        
        for drawing in page_dict.get('drawings', []):
            for item in drawing.get('items', []):
                if item.get('type') == 'l':  # Line
                    x0, y0, x1, y1 = item['rect']
                    if abs(y1 - y0) < 1:  # Horizontal line
                        horizontal_lines += 1
                    if abs(x1 - x0) < 1:  # Vertical line
                        vertical_lines += 1
        
        # Check for scanned content
        is_scanned = len(page_dict.get('blocks', [])) < 5 and len(page.get_text().strip()) > 0
        
        # Determine table type
        if is_scanned:
            return 'scanned'
        elif horizontal_lines > 3 and vertical_lines > 3:
            return 'bordered'
        elif horizontal_lines > 3 or vertical_lines > 3:
            # Check for complex structure (merged cells)
            blocks = page_dict.get('blocks', [])
            has_merged_cells = False
            for block in blocks:
                if block.get('type') == 0:  # Text block
                    spans = block.get('spans', [])
                    if any(span.get('size', 0) > 0 for span in spans):
                        has_merged_cells = True
                        break
            
            return 'complex' if has_merged_cells else 'borderless'
        else:
            return 'borderless'

    def _heuristic_table_detection(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """
        Detect and extract tables using heuristic approaches when library-based methods fail.
        Uses layout analysis and pattern recognition to identify table structures.
        """
        tables = []
        try:
            # Get page content with layout information
            page_dict = page.get_text("dict")
            blocks = page_dict.get("blocks", [])
            
            # Find potential table regions
            table_regions = self._find_table_regions(page)
            
            for region in table_regions:
                # Extract text within the region
                region_text = page.get_text("text", clip=region).strip()
                if not region_text:
                    continue
                    
                # Split into lines and analyze structure
                lines = [line.strip() for line in region_text.split('\n') if line.strip()]
                if len(lines) < 2:  # Need at least header and one data row
                    continue
                    
                # Analyze column structure
                column_structure = self._analyze_column_structure(lines)
                if not column_structure['is_table']:
                    continue
                    
                # Extract table data
                table_data = self._extract_table_data(page, region)
                if not table_data:
                    continue
                    
                # Get table context (captions, references)
                context = self._get_table_context(page, region)
                    
                tables.append({
                    'data': table_data,
                    'region': region,
                    'confidence': 70,  # Lower confidence for heuristic detection
                    'detection_method': 'heuristic',
                    'source': 'built_in',
                    'context': context
                })
                
        except Exception as e:
            self.logger.warning(f"Heuristic table detection failed: {str(e)}")
        
        return tables

    def _analyze_column_structure(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze lines to determine if they form a table structure."""
        result = {
            'is_table': False,
            'num_columns': 0,
            'column_separators': [],
            'confidence': 0.0
        }
        
        if not lines:
            return result
        
        # Look for common separators
        separators = ['\t', '  ', ' | ', '|']
        separator_counts = {}
        column_counts = {}
        
        # Analyze each line for potential column structure
        for line in lines:
            for sep in separators:
                if sep in line:
                    parts = [p.strip() for p in line.split(sep)]
                    num_cols = len([p for p in parts if p])  # Count non-empty columns
                    
                    separator_counts[sep] = separator_counts.get(sep, 0) + 1
                    column_counts[num_cols] = column_counts.get(num_cols, 0) + 1
        
        if not separator_counts:
            return result
        
        # Find most common separator and column count
        best_separator = max(separator_counts.items(), key=lambda x: x[1])[0]
        most_common_cols = max(column_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate consistency
        total_lines = len(lines)
        separator_consistency = separator_counts[best_separator] / total_lines
        column_consistency = column_counts[most_common_cols] / total_lines
        
        # Determine if structure is table-like
        is_table = (
            most_common_cols >= 2 and  # At least 2 columns
            separator_consistency >= 0.8 and  # Consistent separator usage
            column_consistency >= 0.7  # Consistent column count
        )
        
        result.update({
            'is_table': is_table,
            'num_columns': most_common_cols,
            'column_separators': [best_separator],
            'confidence': min(separator_consistency, column_consistency) * 100
        })
        
        return result

    def _find_table_regions(self, page: fitz.Page) -> List[Tuple[float, float, float, float]]:
        """Find potential table regions using layout analysis."""
        regions = []
        
        # Get page text with layout info
        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", [])
        
        # Look for grid patterns and aligned text blocks
        current_region = None
        last_y = 0
        aligned_blocks = []
        
        for block in blocks:
            if block["type"] != 0:  # Skip non-text blocks
                continue
            
            x0, y0, x1, y1 = block["bbox"]
            
            # Check if block is aligned with previous blocks
            if aligned_blocks and abs(y0 - last_y) > self.config['row_tolerance']:
                # Process accumulated blocks as potential table
                if len(aligned_blocks) >= 3:  # Need at least 3 aligned blocks
                    region = self._get_region_from_blocks(aligned_blocks)
                    regions.append(region)
                aligned_blocks = []
            
            aligned_blocks.append(block)
            last_y = y0
        
        # Process final set of aligned blocks
        if len(aligned_blocks) >= 3:
            region = self._get_region_from_blocks(aligned_blocks)
            regions.append(region)
        
        # Also look for explicit table indicators (lines, borders)
        bordered_regions = self._find_bordered_regions(page)
        regions.extend(bordered_regions)
        
        # Merge overlapping regions
        return self._merge_overlapping_regions(regions)

    def _get_region_from_blocks(self, blocks: List[Dict]) -> Tuple[float, float, float, float]:
        """Calculate bounding box for a group of blocks."""
        x0 = min(block["bbox"][0] for block in blocks)
        y0 = min(block["bbox"][1] for block in blocks)
        x1 = max(block["bbox"][2] for block in blocks)
        y1 = max(block["bbox"][3] for block in blocks)
        return (x0, y0, x1, y1)

    def _find_bordered_regions(self, page: fitz.Page) -> List[Tuple[float, float, float, float]]:
        """Find regions enclosed by lines or borders."""
        regions = []
        
        # Get drawing elements
        page_dict = page.get_text("dict")
        drawings = page_dict.get('drawings', [])
        
        # Look for rectangles and line patterns
        for drawing in drawings:
            if drawing.get('type') == 'rect':
                regions.append(drawing['rect'])
            elif drawing.get('type') == 'line':
                # Analyze line patterns to detect table borders
                # This is a simplified version - could be enhanced
                pass
        
        return regions

    def _merge_overlapping_regions(
        self, 
        regions: List[Tuple[float, float, float, float]]
    ) -> List[Tuple[float, float, float, float]]:
        """Merge overlapping table regions."""
        if not regions:
            return []
        
        # Sort regions by x0 coordinate
        sorted_regions = sorted(regions, key=lambda r: r[0])
        merged = [sorted_regions[0]]
        
        for current in sorted_regions[1:]:
            previous = merged[-1]
            
            # Check if regions overlap
            if (current[0] <= previous[2] and  # x overlap
                current[1] <= previous[3] and  # y overlap
                current[2] >= previous[0] and
                current[3] >= previous[1]):
                # Merge regions
                merged[-1] = (
                    min(previous[0], current[0]),
                    min(previous[1], current[1]),
                    max(previous[2], current[2]),
                    max(previous[3], current[3])
                )
            else:
                merged.append(current)
        
        return merged

    def _extract_images_with_context(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Extract images from each page, possibly classify as diagrams."""
        images = []
        diagram_classifier = None

        # If we want to do advanced diagram detection:
        if HAS_VISION_MODELS and self.config.get('diagram_detection', {}).get('enabled', True):
            diagram_classifier = self._initialize_diagram_classifier()

        for page_num, page in enumerate(doc):
            page_text = page.get_text("dict")
            page_blocks = page_text.get("blocks", [])

            # Extract each image in the page.
            for img_index, img in enumerate(page.get_images()):
                xref = img[0]
                base_img = doc.extract_image(xref)
                if not base_img:
                    continue

                image_bytes = base_img["image"]
                try:
                    import PIL.Image
                    pil_image = PIL.Image.open(io.BytesIO(image_bytes))

                    # Skip tiny images.
                    minw = self.config['diagram_detection'].get('min_width', 100)
                    minh = self.config['diagram_detection'].get('min_height', 100)
                    if pil_image.width < minw or pil_image.height < minh:
                        continue

                    # Identify the bounding box on the page if available.
                    img_rect = None
                    for block in page_blocks:
                        if block.get("type") == 1 and block.get("xref") == xref:
                            img_rect = block.get("bbox")
                            break

                    # Get some surrounding text context for the image.
                    context = self._get_enhanced_image_context(page, img_rect, page_blocks)

                    # Diagram classification (OpenCV lines detection, or deep learning if classifier is available).
                    diagram_info = self._analyze_diagram(pil_image, diagram_classifier)

                    # If recognized as diagram and we have OCR instance, ask OCR to read embedded text if desired.
                    diagram_text = None
                    if diagram_info['is_diagram'] and self.ocr:
                        # We delegate to self.ocr for diagram text extraction.
                        diagram_text = self.ocr.extract_text_from_diagram(pil_image)

                    metadata = {
                        'page': page_num + 1,
                        'index': img_index,
                        'size': base_img.get("size", 0),
                        'format': base_img.get("ext", "unknown"),
                        'dimensions': (pil_image.width, pil_image.height),
                        'location': img_rect,
                        'context': context,
                        'diagram_type': diagram_info['diagram_type'],
                        'is_diagram': diagram_info['is_diagram'],
                        'confidence': diagram_info['confidence'],
                        'extracted_text': diagram_text
                    }
                    images.append({
                        'data': image_bytes,
                        'metadata': metadata
                    })

                except Exception as e:
                    self.logger.warning(f"Failed to analyze image on page {page_num+1}: {str(e)}")
                    images.append({
                        'data': image_bytes,
                        'metadata': {
                            'page': page_num + 1,
                            'index': img_index,
                            'size': base_img.get("size", 0),
                            'format': base_img.get("ext", "unknown"),
                            'error': str(e)
                        }
                    })

        return images

    def _initialize_diagram_classifier(self):
        """Initialize a diagram classifier if you have a trained model for specific diagram categories."""
        try:
            model = models.resnet50(pretrained=True)
            num_classes = len(DiagramType)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

            model_path = self.config['diagram_detection'].get('model_path')
            if model_path and os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))

            model.eval()
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return {'model': model, 'transform': transform}
        except Exception as e:
            self.logger.warning(f"Failed to initialize diagram classifier: {str(e)}")
            return None

    def _analyze_diagram(self, pil_image, classifier=None):
        """Heuristic + optional model-based approach to check if an image is a diagram."""
        result = {
            'is_diagram': False,
            'diagram_type': DiagramType.UNKNOWN.value,
            'confidence': 0.0
        }
        try:
            np_image = np.array(pil_image.convert('RGB'))
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

            line_count = 0 if lines is None else len(lines)
            # Heuristic: if many lines + certain features => probable diagram.
            result['is_diagram'] = (line_count > 5)
            result['confidence'] = 0.5 if result['is_diagram'] else 0.2

            # If we have a deep learning classifier, refine the type.
            if classifier and result['is_diagram']:
                model, transform = classifier['model'], classifier['transform']
                img_tensor = transform(pil_image).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    max_prob, class_idx = torch.max(probabilities, 0)

                    result['confidence'] = float(max_prob)
                    result['diagram_type'] = list(DiagramType)[class_idx].value

            # Simple fallback if we remain unknown but is_diagram is True.
            if result['diagram_type'] == DiagramType.UNKNOWN.value and result['is_diagram']:
                # We can do extra checks for charts, flowcharts, etc.
                pass

        except Exception as e:
            self.logger.warning(f"Diagram analysis error: {str(e)}")

        return result

    def _get_enhanced_image_context(self, page, img_rect, page_blocks):
        """Get enhanced context information around an image/diagram."""
        context = {
            'caption': None,
            'references': [],
            'surrounding_text': None,
            'subject_context': None,
            'educational_labels': []
        }
        
        if not img_rect:
            return context
        
        # Extract caption - usually text block right above or below the image
        x0, y0, x1, y1 = img_rect
        
        # Define regions to look for captions
        above_region = (x0 - 20, max(0, y0 - 100), x1 + 20, y0)
        below_region = (x0 - 20, y1, x1 + 20, min(page.rect.height, y1 + 100))
        
        # Common educational image/diagram labels
        edu_label_patterns = [
            r'(figure|fig\.)\s+\d',
            r'(diagram|diag\.)\s+\d',
            r'(illustration|illus\.)\s+\d',
            r'(chart)\s+\d',
            r'(graph)\s+\d',
            r'(plate)\s+\d',
            r'(map)\s+\d',
        ]
        
        # Check for caption above
        caption_above = page.get_text("text", clip=above_region).strip()
        if caption_above:
            for pattern in edu_label_patterns:
                if re.search(pattern, caption_above.lower()):
                    context['caption'] = caption_above
                    # Extract the specific label (e.g., "Figure 1.2")
                    match = re.search(pattern + r'[.:]?\s*[\d.]+', caption_above.lower())
                    if match:
                        context['educational_labels'].append(match.group(0))
                    break
                
        # If not found above, check below
        if not context['caption']:
            caption_below = page.get_text("text", clip=below_region).strip()
            if caption_below:
                for pattern in edu_label_patterns:
                    if re.search(pattern, caption_below.lower()):
                        context['caption'] = caption_below
                        # Extract the label
                        match = re.search(pattern + r'[.:]?\s*[\d.]+', caption_below.lower())
                        if match:
                            context['educational_labels'].append(match.group(0))
                        break
        
        # Get surrounding text for context
        surrounding_region = (
            max(0, x0 - 200),
            max(0, y0 - 200),
            min(page.rect.width, x1 + 200),
            min(page.rect.height, y1 + 200))
        
        context['surrounding_text'] = page.get_text("text", clip=surrounding_region).strip()
        
        # Look for references to the image in the text
        page_text = page.get_text("text")
        for label in context['educational_labels']:
            # Look for phrases like "as shown in Figure 1.2"
            ref_patterns = [
                rf'(as shown in|see|refer to|according to|in)\s+{re.escape(label)}',
                rf'{re.escape(label)}\s+shows',
                rf'illustrated in\s+{re.escape(label)}'
            ]
            
            for pattern in ref_patterns:
                for match in re.finditer(pattern, page_text, re.IGNORECASE):
                    # Get a snippet of text around the reference
                    start = max(0, match.start() - 50)
                    end = min(len(page_text), match.end() + 50)
                    ref_text = page_text[start:end].strip()
                    context['references'].append(ref_text)
        
        # Try to determine subject context from surrounding text
        context['subject_context'] = self._extract_subject_context(context['surrounding_text'])
        
        return context

    def _extract_pdf_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        metadata = doc.metadata
        return {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'keywords': metadata.get('keywords', ''),
            'creator': metadata.get('creator', ''),
            'producer': metadata.get('producer', ''),
            'page_count': doc.page_count,
            'file_size': doc.stream_length
        }

    def _get_table_context(self, page: fitz.Page, region: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """Get contextual information around a table region."""
        context = {
            'caption': None,
            'references': [],
            'surrounding_text': None
        }
        
        if not region:
            return context
        
        x0, y0, x1, y1 = region
        
        # Look for caption above/below table
        above_region = (x0, max(0, y0 - 100), x1, y0)
        below_region = (x0, y1, x1, min(page.rect.height, y1 + 100))
        
        caption_above = page.get_text("text", clip=above_region).strip()
        caption_below = page.get_text("text", clip=below_region).strip()
        
        # Check for table caption indicators
        for text in [caption_above, caption_below]:
            if re.search(r'(table|tbl\.)\s+\d', text.lower()):
                context['caption'] = text
                break
        
        # Get surrounding text for context
        surrounding = (
            max(0, x0 - 200),
            max(0, y0 - 200),
            min(page.rect.width, x1 + 200),
            min(page.rect.height, y1 + 200)
        )
        context['surrounding_text'] = page.get_text("text", clip=surrounding).strip()
        
        return context

    def _add_cross_references(self, document: 'Document'):
        """Add cross-references between document elements (tables, figures, sections)."""
        try:
            content = document.content
            doc_info = document.doc_info
            
            # Track references to tables
            if 'tables' in doc_info:
                for table in doc_info['tables']:
                    if caption := table.get('context', {}).get('caption'):
                        # Look for references to this table in the text
                        matches = re.finditer(
                            rf'(see|refer to|in|as shown in)\s+table\s+\d+',
                            content,
                            re.IGNORECASE
                        )
                        table['references'] = [
                            content[max(0, m.start() - 50):min(len(content), m.end() + 50)]
                            for m in matches
                        ]
            
            # Track references to figures/diagrams
            if 'diagrams' in doc_info:
                for diagram in doc_info['diagrams']:
                    if labels := diagram['metadata'].get('educational_labels', []):
                        for label in labels:
                            matches = re.finditer(
                                rf'(see|refer to|in|as shown in)\s+{re.escape(label)}',
                                content,
                                re.IGNORECASE
                            )
                            diagram['references'] = [
                                content[max(0, m.start() - 50):min(len(content), m.end() + 50)]
                                for m in matches
                            ]
            
            # Track references between sections if hierarchy exists
            if 'structure' in doc_info:
                for section in doc_info['structure']:
                    if 'text' in section:
                        matches = re.finditer(
                        rf'(see|refer to|in)\s+section\s+\d+(\.\d+)*',
                        content,
                        re.IGNORECASE
                    )
                        section['references'] = [
                            content[max(0, m.start() - 50):min(len(content), m.end() + 50)]
                            for m in matches
                        ]
        except Exception as e:
            self.logger.warning(f"Error adding cross-references: {str(e)}")

    def _check_if_scanned(self, doc: fitz.Document) -> bool:
        """
        Heuristic to determine if the PDF is scanned by checking the presence of extractable text.
        
        Args:
            doc: The PyMuPDF document.
        
        Returns:
            bool: True if the PDF appears to be scanned (i.e., minimal text detected), False otherwise.
        """
        scanned_pages = 0
        total_pages = len(doc)
        
        for page in doc:
            # Get the text from the page
            text = page.get_text("text").strip()
            # If very little text is found, consider this page "scanned"
            # Adjust the threshold (e.g., 10 characters) as needed.
            if len(text) < 10:
                scanned_pages += 1
        
        # If more than half of the pages lack sufficient text, consider the document scanned.
        if total_pages == 0:
            return False
        return (scanned_pages / total_pages) > 0.5

    def _determine_device(self) -> str:
        """Determine the best available device for acceleration."""
        config_device = self.config['acceleration']['device'].lower()
        
        if config_device == 'auto':
            try:
                import torch
                if torch.cuda.is_available():
                    return 'cuda'
                elif torch.backends.mps.is_available():
                    return 'mps'
            except ImportError:
                pass
            return 'cpu'
        
        return config_device

    def _init_ocr(self):
        """Initialize OCR if requested."""
        if self.config.get('use_ocr', False):
            try:
                from ..core.vision.ocr import OCR
                return OCR(
                    languages=['en'],
                    enhance_resolution=self.config.get('enhance_resolution', True),
                    preserve_layout=self.config.get('preserve_layout', True)
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize OCR: {e}")
                return None
