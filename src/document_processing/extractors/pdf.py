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
- OCR capabilities
- Error handling
- Performance optimization

Dependencies:
- PyMuPDF>=1.18.0
- pytesseract>=0.3.8
- numpy>=1.24.0
- Pillow>=8.0.0
- xgboost>=1.7.0
- torch>=2.0.0
- camelot-py>=0.10.1  # For table extraction
- tabula-py>=2.7.0    # For table extraction
- pdfplumber>=0.7.0   # Added for enhanced table extraction
- pandas>=1.5.0       # Required for table handling

Author: Keith Satuku
Version: 2.1.0
Created: 2025
License: MIT
"""

import fitz
from PIL import Image
import pytesseract
import numpy as np
import torch
import re
import pandas as pd
import io
import os
import tempfile  # Make sure tempfile is imported
from typing import Dict, Any, List, Optional, Tuple, Generator
from dataclasses import dataclass
from .models import Document
from .base import BaseExtractor, ExtractorResult
from enum import Enum
import cv2  # Add OpenCV for diagram analysis

# Import table extraction libraries with proper checks
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

# Add to class imports at top if needed
try:
    # Import vision libraries for diagram classification
    from torchvision import models, transforms
    HAS_VISION_MODELS = True
except ImportError:
    HAS_VISION_MODELS = False

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
        """Initialize PDF extractor.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        
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
                'method': 'auto',  # 'camelot', 'tabula', 'pdfplumber' or 'auto'
                'strategy': 'adaptive',  # 'adaptive', 'vote', or 'sequential'
                'flavor': 'lattice',  # For Camelot: 'lattice' or 'stream'
                'line_scale': 40,     # For Camelot lattice mode
                'min_confidence': 80, # Confidence threshold for detected tables (%)
                'header_extraction': True,  # Attempt to extract table headers
                'fallback_to_heuristic': True,  # Use built-in detection if libraries fail
                'table_types': {
                    'bordered': 'camelot',    # Preferred extractor for bordered tables
                    'borderless': 'tabula',   # Preferred extractor for borderless tables
                    'complex': 'camelot',     # Preferred for tables with merged cells
                    'scanned': 'tabula'       # Preferred for tables in scanned documents
                }
            }
        }
        
        # Merge provided config with defaults
        if config:
            default_config.update(config)
            
        super().__init__(config=default_config)
        self._init_components()
        self.base_font_size = 12.0  # Will be updated during processing

    def _init_components(self):
        """Initialize processing components."""
        try:
            self._init_ocr()
            self._init_layout_recognizer()
            self._init_device()
        except Exception as e:
            self.logger.error(f"Component initialization failed: {str(e)}")
            raise

    def _init_layout_recognizer(self):
        """Initialize layout recognition capabilities."""
        try:
            from ..core.vision.layout_recognizer import LayoutRecognizer
            
            # Get layout config
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
        """Initialize device settings."""
        try:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        except Exception:
            self.device = torch.device('cpu')

    def _init_ocr(self):
        """Initialize OCR capabilities."""
        try:
            from ..core.vision.ocr import OCR
            self.ocr = OCR()
            self.has_ocr = True
            self.logger.info("OCR initialized successfully")
        except Exception as e:
            self.logger.warning(f"OCR initialization failed: {str(e)}")
            self.has_ocr = False

    def _match_proj(self, text_block: Dict[str, Any]) -> bool:
        """Check if text block matches projection patterns for structure."""
        proj_patterns = [
            r"[0-9]+\.[0-9.]+(、|\.[ 　])",
            r"[A-Z]\.",
            r"[0-9]+\.",
            r"[\(（][0-9]+[）\)]",
            r"[•⚫➢①②③④⑤⑥⑦⑧⑨⑩]",
            r"[IVX]+\.",
            r"[a-z]\)",
            r"[A-Z]\)",
        ]
        return any(re.match(p, text_block["text"].strip()) for p in proj_patterns)

    def _updown_concat_features(self, up: Dict, down: Dict) -> List[float]:
        """Extract features for content concatenation decision."""
        features = [
            # Layout matching
            float(up.get("layout_type") == down.get("layout_type")),
            float(up.get("layout_type") == "text"),
            float(down.get("layout_type") == "text"),
            
            # Punctuation analysis
            float(bool(re.search(r"([.?!;+)）]|[a-z]\.)$", up["text"]))),
            float(bool(re.search(r"[,:'\"(+-]$", up["text"]))),
            float(bool(re.search(r"(^.?[/,?;:\],.;:'\"])", down["text"]))),
            
            # Special cases
            float(bool(re.match(r"[\(（][^\(\)（）]+[）\)]$", up["text"]))),
            float(bool(re.search(r"[,][^.]+$", up["text"]))),
            
            # Parentheses matching
            float(bool(re.search(r"[\(（][^\)）]+$", up["text"]) and 
                 re.search(r"[\)）]", down["text"]))),
                 
            # Character type analysis
            float(bool(re.match(r"[A-Z]", down["text"]))),
            float(bool(re.match(r"[A-Z]", up["text"][-1:]))),
            float(bool(re.match(r"[a-z0-9]", up["text"][-1:]))),
            
            # Distance metrics
            self._x_distance(up, down) / max(self._char_width(up), 0.000001),
            abs(self._block_height(up) - self._block_height(down)) / 
                max(min(self._block_height(up), self._block_height(down)), 0.000001)
        ]
        return features

    def _analyze_layout_with_rows(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Analyze page layout with row awareness."""
        layout_elements = []
        current_row = []
        last_y = 0
        
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 0:  # text block
                y_mid = (block["bbox"][1] + block["bbox"][3]) / 2
                
                # Check if new row
                if abs(y_mid - last_y) > self.config['row_tolerance'] and current_row:
                    # Process current row
                    layout_elements.extend(self._process_row(current_row))
                    current_row = []
                
                current_row.append(block)
                last_y = y_mid
        
        # Process final row
        if current_row:
            layout_elements.extend(self._process_row(current_row))
            
        return layout_elements

    def _process_row(self, row: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a row of text blocks."""
        # Sort by x coordinate
        row = sorted(row, key=lambda b: b["bbox"][0])
        
        # Analyze row characteristics
        row_height = max(b["bbox"][3] - b["bbox"][1] for b in row)
        row_font_sizes = [b["spans"][0]["size"] for b in row if b["spans"]]
        is_header = any(size > self.base_font_size * 1.2 for size in row_font_sizes)
        
        processed = []
        for block in row:
            block["in_row"] = len(row)
            block["is_row_header"] = is_header
            block["row_height"] = row_height
            processed.append(block)
            
        return processed

    def _concat_text_blocks(self, blocks: List[Dict[str, Any]]) -> str:
        """Concatenate text blocks intelligently."""
        result = []
        for i, block in enumerate(blocks):
            text = block["text"].strip()
            
            # Skip if empty
            if not text:
                continue
                
            # Check for concatenation with previous block
            if result and i > 0 and self._should_concat_with_previous(block, blocks[i-1]):
                # Add appropriate spacing/joining
                if re.match(r"[a-zA-Z0-9]", text[0]) and \
                   re.match(r"[a-zA-Z0-9]", result[-1][-1]):
                    result[-1] += " " + text
                else:
                    result[-1] += text
            else:
                result.append(text)
                
        return "\n".join(result)

    def _should_concat_with_previous(
        self, 
        current: Dict[str, Any], 
        previous: Dict[str, Any]
    ) -> bool:
        """Determine if blocks should be concatenated."""
        # Get coordinates
        prev_end = previous["bbox"][3]
        curr_start = current["bbox"][1]
        
        # Check vertical distance
        if curr_start - prev_end > self.config['line_spacing'] * 1.5:
            return False
            
        # Check if same paragraph
        if current.get("in_row", 1) > 1 or previous.get("in_row", 1) > 1:
            return False
            
        # Check for sentence completion
        if not re.search(r"[.!?]$", previous["text"]):
            return True
            
        return False

    def _char_width(self, block: Dict[str, Any]) -> float:
        """Calculate average character width in block."""
        width = block["bbox"][2] - block["bbox"][0]
        text_length = len(block["text"].strip())
        return width / max(text_length, 1)

    def _block_height(self, block: Dict[str, Any]) -> float:
        """Calculate block height."""
        return block["bbox"][3] - block["bbox"][1]

    def _x_distance(self, block1: Dict[str, Any], block2: Dict[str, Any]) -> float:
        """Calculate horizontal distance between blocks."""
        return abs(block1["bbox"][0] - block2["bbox"][0])

    async def extract(self, document: 'Document') -> 'Document':
        """Extract content from PDF document with layout awareness."""
        try:
            content = document.content
            if isinstance(content, str):
                # If it's a file path, read the file
                with open(content, 'rb') as f:
                    content = f.read()
                    
            # Open PDF from binary content
            doc = fitz.open(stream=content, filetype="pdf")
            
            # Process document structure
            structure = self._process_document_structure(doc)
            
            # Extract content with layout awareness
            extracted_text = self._extract_structured_text(doc, structure)
            
            # Update document with extracted content
            document.doc_info.update({
                'extracted_text': extracted_text,
                'structure': structure['hierarchy'],
                'metadata': self._extract_pdf_metadata(doc)
            })
            
            # Perform OCR if needed
            if self.has_ocr and self.config.get('use_ocr', False):
                ocr_text = await self._perform_ocr(doc)
                document.doc_info['ocr_text'] = ocr_text
            
            doc.close()
            return document
            
        except Exception as e:
            self.logger.error(f"PDF extraction error: {str(e)}")
            raise

    def _process_document_structure(self, doc: fitz.Document) -> Dict[str, Any]:
        """Process document structure and hierarchy."""
        structure = {
            'hierarchy': [],
            'page_layouts': [],
            'content_map': {}
        }
        
        # Process each page
        for page_num, page in enumerate(doc):
            layout = self._analyze_page_layout(page)
            structure['page_layouts'].append(layout)
            
            # Build content hierarchy
            for element in layout:
                if element.type in ['title', 'subtitle', 'heading']:
                    structure['hierarchy'].append({
                        'text': element.text,
                        'level': self._determine_heading_level(element),
                        'page': page_num + 1
                    })
                    
        return structure

    def _analyze_page_layout(self, page: fitz.Page) -> List[LayoutElement]:
        """Analyze page layout with enhanced recognition."""
        elements = []
        
        # Get raw layout information
        layout = page.get_text("dict")
        base_font_size = self._get_base_font_size(layout)
        
        for block in layout['blocks']:
            if block.get('type') == 0:  # Text block
                element = self._process_text_block(block, base_font_size)
                if element:
                    elements.append(element)
                    
        # Merge related elements
        elements = self._merge_related_elements(elements)
        
        return elements

    def _process_text_block(
        self,
        block: Dict[str, Any],
        base_font_size: float
    ) -> Optional[LayoutElement]:
        """Process text block with layout analysis."""
        try:
            # Extract text properties
            text = ' '.join(span['text'] for span in block['spans'])
            font_info = block['spans'][0]  # Use first span for font info
            
            # Calculate properties
            font_size = font_info.get('size', 0)
            is_bold = 'bold' in font_info.get('font', '').lower()
            
            # Determine element type
            element_type = self._determine_element_type(
                font_size,
                base_font_size,
                is_bold,
                block['bbox']
            )
            
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
        self,
        font_size: float,
        base_font_size: float,
        is_bold: bool,
        bbox: Tuple[float, float, float, float]
    ) -> str:
        """Determine element type based on properties."""
        # Title detection
        if font_size >= base_font_size * 1.5:
            return 'title'
            
        # Subtitle detection
        if font_size >= base_font_size * 1.2 or (
            font_size >= base_font_size * 1.1 and is_bold
        ):
            return 'subtitle'
            
        # Heading detection
        if is_bold or font_size > base_font_size:
            return 'heading'
            
        return 'text'

    def _merge_related_elements(
        self,
        elements: List[LayoutElement]
    ) -> List[LayoutElement]:
        """Merge related elements based on layout."""
        merged = []
        current = None
        
        for element in elements:
            if not current:
                current = element
                continue
                
            # Check if elements should be merged
            if self._should_merge_elements(current, element):
                current = self._merge_elements(current, element)
            else:
                merged.append(current)
                current = element
                
        if current:
            merged.append(current)
            
        return merged

    def _should_merge_elements(
        self,
        elem1: LayoutElement,
        elem2: LayoutElement
    ) -> bool:
        """Determine if elements should be merged."""
        # Check vertical distance
        vertical_gap = elem2.bbox[1] - elem1.bbox[3]
        
        # Check horizontal overlap
        horizontal_overlap = (
            min(elem1.bbox[2], elem2.bbox[2]) -
            max(elem1.bbox[0], elem2.bbox[0])
        )
        
        return (
            elem1.type == elem2.type and
            vertical_gap <= self.config['merge_tolerance'] and
            horizontal_overlap > 0
        )

    def _extract_tables_with_structure(
        self,
        doc: fitz.Document
    ) -> List[Dict[str, Any]]:
        """Extract tables with structural context using specialized libraries."""
        tables = []
        table_config = self.config.get('table_extraction', {})
        method = table_config.get('method', 'auto')
        strategy = table_config.get('strategy', 'adaptive')
        
        # Create a temporary file if needed for library-based extraction
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = temp_file.name
            doc.save(temp_path)
            
            try:
                for page_num, page in enumerate(doc):
                    page_tables = []
                    table_candidates = []
                    
                    # Determine table type for this page if using adaptive strategy
                    table_type = self._detect_table_type(page) if strategy == 'adaptive' else None
                    preferred_extractor = table_config.get('table_types', {}).get(table_type)
                    
                    # If using voting strategy, try all methods
                    if strategy == 'vote':
                        if HAS_CAMELOT:
                            camelot_tables = self._extract_tables_camelot(temp_path, page_num+1, table_config)
                            table_candidates.append(('camelot', camelot_tables))
                            
                        if HAS_TABULA:
                            tabula_tables = self._extract_tables_tabula(temp_path, page_num+1, table_config)
                            table_candidates.append(('tabula', tabula_tables))
                            
                        if HAS_PDFPLUMBER:
                            plumber_tables = self._extract_tables_pdfplumber(temp_path, page_num+1, table_config)
                            table_candidates.append(('pdfplumber', plumber_tables))
                            
                        # Select best tables via voting
                        page_tables = self._select_best_tables_by_voting(table_candidates)
                    
                    # Otherwise use sequential or adaptive approach
                    else:
                        # Try preferred extractor first if adaptive
                        if strategy == 'adaptive' and preferred_extractor:
                            if preferred_extractor == 'camelot' and HAS_CAMELOT:
                                page_tables = self._extract_tables_camelot(temp_path, page_num+1, table_config)
                            elif preferred_extractor == 'tabula' and HAS_TABULA:
                                page_tables = self._extract_tables_tabula(temp_path, page_num+1, table_config)
                            elif preferred_extractor == 'pdfplumber' and HAS_PDFPLUMBER:
                                page_tables = self._extract_tables_pdfplumber(temp_path, page_num+1, table_config)
                        
                        # If preferred extractor didn't work or not using adaptive, try methods sequentially
                        if not page_tables:
                            if method in ['auto', 'camelot'] and HAS_CAMELOT:
                                page_tables = self._extract_tables_camelot(temp_path, page_num+1, table_config)
                            
                            if not page_tables and method in ['auto', 'tabula'] and HAS_TABULA:
                                page_tables = self._extract_tables_tabula(temp_path, page_num+1, table_config)
                                
                            if not page_tables and method in ['auto', 'pdfplumber'] and HAS_PDFPLUMBER:
                                page_tables = self._extract_tables_pdfplumber(temp_path, page_num+1, table_config)
                    
                    # Fall back to heuristic detection if needed
                    if not page_tables and table_config.get('fallback_to_heuristic', True):
                        self.logger.info(f"Using heuristic table detection for page {page_num+1}")
                        table_regions = self._find_table_regions(page)
                        
                        for region in table_regions:
                            table_data = self._extract_table_data(page, region)
                            if table_data:
                                page_tables.append({
                                    'data': table_data,
                                    'region': region,
                                    'detection_method': 'heuristic',
                                    'confidence': 70  # Lower confidence for heuristic detection
                                })
                    
                    # Add context and metadata to tables
                    for table in page_tables:
                        table_region = table.get('region')
                        context = None
                        
                        # Get context if region is available
                        if table_region:
                            context = self._get_table_context(page, table_region)
                        
                        # Add to final tables list with metadata
                        tables.append({
                            'data': table.get('data'),
                            'page': page_num + 1,
                            'bbox': table_region,
                            'context': context,
                            'detection_method': table.get('detection_method', 'unknown'),
                            'confidence': table.get('confidence', 100),
                            'extractor': table.get('source', 'unknown')
                        })
                        
                        # Log successful extraction
                        self._record_extraction_metrics(
                            document=None,  # We don't have document object here
                            component="table_extraction",
                            success=True,
                            details={
                                'page': page_num + 1,
                                'method': table.get('detection_method', 'unknown'),
                                'confidence': table.get('confidence', 100)
                            }
                        )
            
            except Exception as e:
                self.logger.error(f"Table extraction error: {str(e)}")
                # Record extraction failure
                self._record_extraction_metrics(
                    document=None,
                    component="table_extraction",
                    success=False,
                    details={'error': str(e)}
                )
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                    self.logger.debug(f"Temporary file removed: {temp_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete temporary PDF: {str(e)}")
                
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
        
    def _regions_overlap(self, region1, region2):
        """Check if two regions overlap."""
        if not region1 or not region2:
            return False
            
        x0_1, y0_1, x1_1, y1_1 = region1
        x0_2, y0_2, x1_2, y1_2 = region2
        
        return not (x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1)

    def _extract_images_with_context(
        self,
        doc: fitz.Document
    ) -> List[Dict[str, Any]]:
        """Extract images and diagrams with enhanced context and classification."""
        images = []
        
        # Initialize diagram classifier if vision models are available
        diagram_classifier = None
        if HAS_VISION_MODELS and self.config.get('diagram_detection', {}).get('enabled', True):
            diagram_classifier = self._initialize_diagram_classifier()
        
        for page_num, page in enumerate(doc):
            # First extract page text for later context analysis
            page_text = page.get_text("dict")
            page_blocks = page_text.get("blocks", [])
            
            # Get all images from the page
            for img_index, img in enumerate(page.get_images()):
                xref = img[0]
                base_image = doc.extract_image(xref)
                
                if base_image:
                    # Convert image to numpy array for processing
                    image_bytes = base_image["image"]
                    try:
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        
                        # Skip tiny images (often icons or decorations)
                        if pil_image.width < self.config.get('diagram_detection', {}).get('min_width', 100) or \
                           pil_image.height < self.config.get('diagram_detection', {}).get('min_height', 100):
                            continue
                        
                        # Get image location if available
                        img_rect = None
                        for block in page_blocks:
                            if block.get("type") == 1:  # Image block
                                if block.get("xref") == xref:
                                    img_rect = block.get("bbox")
                                    break
                                
                        # Get enhanced context around the image
                        context = self._get_enhanced_image_context(page, img_rect, page_blocks)
                        
                        # Analyze image to determine if it's a diagram and its type
                        diagram_info = self._analyze_diagram(pil_image, diagram_classifier)
                        
                        # Extract text from diagram using OCR if appropriate
                        diagram_text = None
                        if diagram_info['is_diagram'] and self.has_ocr:
                            diagram_text = self._extract_text_from_diagram(pil_image)
                        
                        # Create comprehensive metadata for the diagram/image
                        metadata = {
                            'page': page_num + 1,
                            'index': img_index,
                            'size': base_image.get("size", 0),
                            'format': base_image.get("ext", "unknown"),
                            'dimensions': (pil_image.width, pil_image.height),
                            'location': img_rect,
                            'context': context,
                            'diagram_type': diagram_info['diagram_type'],
                            'is_diagram': diagram_info['is_diagram'],
                            'confidence': diagram_info['confidence'],
                            'extracted_text': diagram_text,
                            'educational_context': self._determine_educational_context(context, diagram_info)
                        }
                        
                        images.append({
                            'data': image_bytes,
                            'metadata': metadata
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to analyze image: {str(e)}")
                        # Add basic image data even if analysis fails
                        images.append({
                            'data': image_bytes,
                            'metadata': {
                                'page': page_num + 1,
                                'index': img_index,
                                'size': base_image.get("size", 0),
                                'format': base_image.get("ext", "unknown"),
                                'error': str(e)
                            }
                        })
                    
        return images

    def _initialize_diagram_classifier(self):
        """Initialize the diagram classifier model."""
        try:
            # Use a pre-trained model for diagram classification
            model = models.resnet50(pretrained=True)
            
            # Modify the output layer for our diagram types
            num_classes = len(DiagramType)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            
            # Load custom weights if available
            model_path = self.config.get('diagram_detection', {}).get('model_path')
            if model_path and os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
            
            model.eval()
            
            # Define image transformations
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            return {
                'model': model,
                'transform': transform
            }
        except Exception as e:
            self.logger.warning(f"Failed to initialize diagram classifier: {str(e)}")
            return None

    def _analyze_diagram(self, pil_image, classifier=None):
        """Analyze an image to determine if it's a diagram and its type."""
        # Default result
        result = {
            'is_diagram': False,
            'diagram_type': DiagramType.UNKNOWN.value,
            'confidence': 0.0
        }
        
        try:
            # Convert to numpy for OpenCV processing
            np_image = np.array(pil_image.convert('RGB'))
            
            # Basic diagram detection using OpenCV
            # Look for common diagram features like lines, shapes, text
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
            
            # Edge detection to find lines and shapes
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Line detection with Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            # Count lines as a basic diagram indicator
            line_count = 0 if lines is None else len(lines)
            
            # Check for text regions using contours
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours that might be text
            text_regions = [c for c in contours if 10 < cv2.contourArea(c) < 1000]
            
            # Basic heuristic: diagrams usually have lines and text
            result['is_diagram'] = line_count > 5 and len(text_regions) > 3
            result['confidence'] = 0.7 if result['is_diagram'] else 0.3
            
            # Use deep learning classifier if available for better accuracy
            if classifier and HAS_VISION_MODELS and result['is_diagram']:
                model = classifier['model']
                transform = classifier['transform']
                
                # Prepare image for model
                img_tensor = transform(pil_image).unsqueeze(0)
                
                # Get model prediction
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    
                    # Get the most likely class
                    max_prob, class_idx = torch.max(probabilities, 0)
                    
                    result['confidence'] = float(max_prob)
                    result['diagram_type'] = list(DiagramType)[class_idx].value
                    
            # Fallback: use simple heuristics if classifier wasn't used or failed
            if result['diagram_type'] == DiagramType.UNKNOWN.value and result['is_diagram']:
                # Check color distribution for chart/graph detection
                color_count = len(np.unique(np_image.reshape(-1, np_image.shape[2]), axis=0))
                
                # Check for circular shapes (pie charts, circular diagrams)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=300)
                
                if circles is not None and len(circles[0]) > 0:
                    result['diagram_type'] = DiagramType.CHART.value
                elif color_count > 100 and line_count > 10:
                    result['diagram_type'] = DiagramType.GRAPH.value
                elif line_count > 20:
                    result['diagram_type'] = DiagramType.FLOWCHART.value
                
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
            min(page.rect.height, y1 + 200)
        
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

    def _extract_text_from_diagram(self, pil_image):
        """Extract text from a diagram using OCR."""
        try:
            # Preprocess image for better OCR results
            np_image = np.array(pil_image)
            
            # Convert to grayscale
            if len(np_image.shape) == 3:
                gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = np_image
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
            
            # Convert back to PIL for OCR
            enhanced_img = Image.fromarray(denoised)
            
            # Perform OCR
            text = pytesseract.image_to_string(enhanced_img)
            
            return text.strip() if text else None
            
        except Exception as e:
            self.logger.warning(f"Diagram OCR error: {str(e)}")
            return None

    def _determine_educational_context(self, context, diagram_info):
        """Determine educational relevance and context of the diagram."""
        educational_context = {
            'subject': None,
            'topic': None,
            'educational_level': None,
            'curriculum_relevance': None,
            'recommended_use': None
        }
        
        # Extract subject from context
        if context['subject_context']:
            educational_context['subject'] = context['subject_context']
        
        # Analyze caption and surrounding text for educational context
        if context['caption']:
            caption_text = context['caption'].lower()
            
            # Look for subject indicators
            subjects = {
                'math': ['equation', 'formula', 'graph', 'function', 'coordinate', 'geometry', 'algebra'],
                'biology': ['cell', 'organism', 'anatomy', 'ecosystem', 'dna', 'species'],
                'chemistry': ['compound', 'element', 'reaction', 'molecule', 'bond', 'formula'],
                'physics': ['force', 'energy', 'motion', 'velocity', 'acceleration', 'circuit'],
                'geography': ['map', 'terrain', 'landform', 'climate', 'region'],
                'history': ['timeline', 'civilization', 'empire', 'dynasty', 'century', 'event']
            }
            
            # Check for subject keywords
            for subject, keywords in subjects.items():
                if any(keyword in caption_text for keyword in keywords):
                    educational_context['subject'] = subject
                    break
                
            # Look for educational level indicators
            levels = {
                'elementary': ['simple', 'basic', 'elementary', 'primary'],
                'middle_school': ['middle', 'intermediate', 'junior'],
                'high_school': ['high school', 'secondary', 'advanced'],
                'university': ['college', 'university', 'undergraduate', 'advanced']
            }
            
            # Check for level keywords
            for level, keywords in levels.items():
                if any(keyword in caption_text for keyword in keywords):
                    educational_context['educational_level'] = level
                    break
        
        # Further analyze based on diagram type
        if diagram_info['is_diagram']:
            diagram_type = diagram_info['diagram_type']
            
            # Make recommendations based on diagram type
            if diagram_type == DiagramType.CHART.value or diagram_type == DiagramType.GRAPH.value:
                educational_context['recommended_use'] = 'data_analysis'
            elif diagram_type == DiagramType.FLOWCHART.value:
                educational_context['recommended_use'] = 'process_understanding'
            elif diagram_type == DiagramType.SCIENTIFIC_DIAGRAM.value:
                educational_context['recommended_use'] = 'concept_explanation'
            elif diagram_type == DiagramType.CONCEPT_MAP.value:
                educational_context['recommended_use'] = 'relationship_mapping'
        
        return educational_context

    def _extract_subject_context(self, text):
        """Extract educational subject context from text."""
        if not text:
            return None
        
        # List of common educational subjects
        subjects = [
            'mathematics', 'math', 'algebra', 'geometry', 'calculus',
            'physics', 'chemistry', 'biology', 'earth science', 'astronomy',
            'history', 'geography', 'economics', 'civics', 'political science',
            'literature', 'language arts', 'grammar', 'composition',
            'computer science', 'programming', 'engineering',
            'art', 'music', 'physical education', 'health'
        ]
        
        # Look for subject mentions
        text_lower = text.lower()
        for subject in subjects:
            if subject in text_lower:
                return subject
            
        return None

    def _get_base_font_size(self, layout: Dict[str, Any]) -> float:
        """Determine base font size for the page."""
        font_sizes = []
        
        for block in layout['blocks']:
            if block.get('type') == 0:  # Text block
                for span in block.get('spans', []):
                    if size := span.get('size'):
                        font_sizes.append(size)
                        
        return np.median(font_sizes) if font_sizes else 12.0

    def _extract_pdf_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract comprehensive PDF metadata."""
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

    async def _perform_ocr(self, doc: fitz.Document) -> str:
        """Perform OCR on document pages."""
        if not self.has_ocr:
            return ""

        ocr_text = []
        for page in doc:
            try:
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Perform OCR
                page_text = await self.ocr.process_image(img)
                ocr_text.append(page_text)
                
            except Exception as e:
                self.logger.warning(f"OCR failed for page {page.number}: {str(e)}")
                continue

        return "\n\n".join(ocr_text) 

    def _find_table_regions(self, page: fitz.Page) -> List[Tuple[float, float, float, float]]:
        """Find potential table regions using heuristic approach."""
        regions = []
        
        # Get page text with bounding boxes
        blocks = page.get_text("dict")["blocks"]
        
        # Look for grid patterns (text blocks arranged in a grid)
        # This is a simplified heuristic - could be improved with ML approaches
        for i, block in enumerate(blocks):
            # Skip non-text blocks
            if block["type"] != 0:
                continue
            
            # Check if this block has multiple spans aligned in rows
            spans = block.get("lines", [])
            if len(spans) < 3:  # Need at least 3 rows for a table
                continue
            
            # Check for column-like alignment
            column_x_positions = []
            for line in spans:
                for span in line.get("spans", []):
                    if "origin" in span:
                        column_x_positions.append(span["origin"][0])
            
            # Count unique column positions
            unique_positions = set([round(x, 1) for x in column_x_positions])
            if len(unique_positions) >= 2:  # Need at least 2 columns
                # This might be a table - get its bounding box
                bbox = block["bbox"]
                regions.append(bbox)
        
        # Also check for bordered regions that might be tables
        page_dict = page.get_text("dict")
        rect_candidates = []
        
        for drawing in page_dict.get('drawings', []):
            for item in drawing.get('items', []):
                if item.get('type') == 'r':  # Rectangle
                    rect_candidates.append(item['rect'])
        
        # Add rectangles that might be tables
        for rect in rect_candidates:
            x0, y0, x1, y1 = rect
            # Check if the rectangle is large enough to be a table
            if (x1 - x0) > 100 and (y1 - y0) > 50:
                regions.append(rect)
        
        return regions

    def _extract_table_data(self, page: fitz.Page, region: Tuple[float, float, float, float]) -> Optional[Dict[str, Any]]:
        """Extract table data from a region using built-in methods."""
        try:
            # Extract text from the region
            text = page.get_text("text", clip=region)
            if not text.strip():
                return None
            
            # Split text into lines
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if not lines:
                return None
            
            # Try to determine columns by looking for consistent separators
            rows = []
            header = None
            
            # Check if first line might be a header (often has different formatting)
            if len(lines) > 1:
                header_candidates = [lines[0]]
                content_lines = lines[1:]
            else:
                header_candidates = []
                content_lines = lines
            
            # Process content lines into a table structure
            for line in content_lines:
                # Split by common separators
                for separator in ['\t', '  ', ' | ', '|']:
                    if separator in line:
                        cells = [cell.strip() for cell in line.split(separator)]
                        # Only use this split if it gives us multiple cells
                        if len(cells) > 1:
                            rows.append(cells)
                            break
                else:
                    # No separator found, treat as single cell
                    rows.append([line])
            
            # Process potential header the same way
            for header_line in header_candidates:
                for separator in ['\t', '  ', ' | ', '|']:
                    if separator in header_line:
                        header_cells = [cell.strip() for cell in header_line.split(separator)]
                        if len(header_cells) > 1:
                            header = header_cells
                            break
            
            # Ensure all rows have the same number of columns (pad with empty strings if needed)
            max_cols = max([len(row) for row in rows]) if rows else 0
            if header and len(header) < max_cols:
                header.extend([''] * (max_cols - len(header)))
            for row in rows:
                if len(row) < max_cols:
                    row.extend([''] * (max_cols - len(row)))
            
            # Return in standard format
            return {
                'header': header,
                'rows': rows,
                'shape': (len(rows), max_cols)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to extract table with built-in method: {str(e)}")
            return None

    def _get_table_context(self, page: fitz.Page, region: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """Get contextual information around a table."""
        context = {
            'caption': None,
            'reference': None,
            'notes': None
        }
        
        # Define regions for caption search (typically above or below the table)
        above_region = (region[0], max(0, region[1] - 100), region[2], region[1])
        below_region = (region[0], region[3], region[2], min(page.rect.height, region[3] + 100))
        
        # Search for caption above
        above_text = page.get_text("text", clip=above_region).strip()
        if above_text:
            # Look for typical caption indicators
            caption_patterns = [
                r'(table|tab\.)\s+\d',
                r'(figure|fig\.)\s+\d',
                r'(chart)\s+\d'
            ]
            
            for pattern in caption_patterns:
                if re.search(pattern, above_text.lower()):
                    context['caption'] = above_text
                    break
                
        # If not found above, check below
        if not context['caption']:
            below_text = page.get_text("text", clip=below_region).strip()
            if below_text:
                for pattern in caption_patterns:
                    if re.search(pattern, below_text.lower()):
                        context['caption'] = below_text
                        break
                    
        # Check for notes (usually marked with asterisks or "Note:")
        if below_text and re.search(r'(\*|\bNote:|\bNotes:)', below_text):
            # Extract just the notes part
            note_match = re.search(r'(\*|\bNote:|\bNotes:)(.*?)$', below_text, re.DOTALL)
            if note_match:
                context['notes'] = note_match.group(0).strip()
            else:
                context['notes'] = below_text
            
        # Look for references to the table in surrounding text
        # This is especially important for educational materials
        surrounding_region = (
            max(0, region[0] - 200), 
            max(0, region[1] - 200),
            min(page.rect.width, region[2] + 200),
            min(page.rect.height, region[3] + 200)
        )
        
        surrounding_text = page.get_text("text", clip=surrounding_region).strip()
        if surrounding_text:
            # Look for references like "as shown in Table 1"
            ref_patterns = [
                r'(as shown in|see|refer to|according to|in)\s+(table|tab\.)\s+\d',
                r'(table|tab\.)\s+\d\s+shows',
                r'illustrated in\s+(table|tab\.)\s+\d'
            ]
            
            for pattern in ref_patterns:
                ref_match = re.search(pattern, surrounding_text, re.IGNORECASE)
                if ref_match:
                    context['reference'] = ref_match.group(0)
                    break
                
        return context 