"""
Document Layout Recognition Module
-------------------------------

Advanced document layout analysis system using LayoutLMv3 for detecting
and classifying document elements and structure.

Key Features:
- Document element detection
- Layout structure analysis
- Element classification
- Spatial relationship analysis
- Reading order determination
- Layout hierarchy detection
- Style analysis

Technical Details:
- LayoutLMv3-based detection
- Element relationship graphs
- Layout structure trees
- Visual style extraction
- Semantic grouping

Dependencies:
- torch>=2.0.0
- transformers>=4.30.0
- numpy>=1.24.0
- networkx>=3.1.0
- Pillow>=9.0.0

Example Usage:
    # Basic layout analysis
    layout = LayoutRecognizer()
    results = layout.analyze('document.pdf')
    
    # With custom configuration
    layout = LayoutRecognizer(
        model_name='Kwan0/layoutlmv3-base-finetune-DocLayNet-100k',
        confidence_threshold=0.6,
        merge_boxes=True
    )
    
    # Advanced analysis
    results = layout.analyze(
        'document.pdf',
        extract_style=True,
        detect_reading_order=True,
        build_hierarchy=True
    )
    
    # Batch processing
    documents = ['doc1.pdf', 'doc2.pdf']
    results = layout.process_batch(
        documents,
        batch_size=8
    )

Element Types:
- Title
- Paragraph
- List
- Table
- Figure
- Header/Footer
- Sidebar
- Caption

Author: Keith Satuku
Version: 1.0.0
License: MIT
"""

from typing import List, Dict, Optional, Tuple, Union
import torch
import numpy as np
import logging
from pathlib import Path
from PIL import Image
import networkx as nx
from transformers import (
    LayoutLMv3Processor, 
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3FeatureExtractor,
    LayoutLMv3TokenizerFast
)
from .recognizer import Recognizer

class LayoutRecognizer(Recognizer):
    """Document layout analysis using LayoutLMv3."""
    
    def __init__(
        self,
        model_name: str = "Kwan0/layoutlmv3-base-finetune-DocLayNet-100k",
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        confidence_threshold: float = 0.5,
        merge_boxes: bool = True,
        label_list: Optional[List[str]] = None
    ):
        """Initialize layout recognizer.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to use (cuda/cpu)
            batch_size: Batch size for processing
            cache_dir: Directory to cache models
            confidence_threshold: Minimum confidence for predictions
            merge_boxes: Whether to merge adjacent boxes
            label_list: Custom label list (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.confidence_threshold = confidence_threshold
        self.merge_boxes = merge_boxes
        
        # Default DocLayNet labels if none provided
        self.label_list = label_list or [
            "text", "title", "list", "table", "figure",
            "header", "footer", "page_number", "caption",
            "section_header", "footnote", "formula", "annotation"
        ]
        
        self._init_model()

    def _init_model(self):
        """Initialize the LayoutLMv3 model and processor."""
        try:
            self.processor = LayoutLMv3Processor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                num_labels=len(self.label_list)
            ).to(self.device)
            
            self.feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.model.eval()
            self.logger.info(f"Model initialized on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise

    async def analyze(
        self,
        document: Union[str, Image.Image, np.ndarray],
        extract_style: bool = False,
        detect_reading_order: bool = True,
        build_hierarchy: bool = True
    ) -> Dict:
        """
        Analyze document layout.
        
        Args:
            document: Document image path or PIL Image
            extract_style: Extract style information
            detect_reading_order: Detect reading order
            build_hierarchy: Build element hierarchy
            
        Returns:
            Dictionary containing layout analysis results
        """
        try:
            # Load and preprocess image
            if isinstance(document, str):
                image = Image.open(document).convert("RGB")
            elif isinstance(document, np.ndarray):
                image = Image.fromarray(document).convert("RGB")
            else:
                image = document
                
            # Prepare inputs
            encoding = self.processor(
                image,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Move inputs to device
            for key, value in encoding.items():
                encoding[key] = value.to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**encoding)
                predictions = outputs.logits.softmax(dim=-1)
                
            # Process predictions
            elements = self._process_predictions(predictions[0], image)
            
            # Build result dictionary
            result = {
                'elements': elements,
                'page_size': image.size,
            }
            
            # Optional processing
            if detect_reading_order:
                result['reading_order'] = self._detect_reading_order(elements)
                
            if build_hierarchy:
                result['hierarchy'] = self._build_hierarchy(elements)
                
            if extract_style:
                result['styles'] = self._extract_styles(elements, image)
                
            return result
            
        except Exception as e:
            self.logger.error(f"Layout analysis failed: {str(e)}")
            raise

    def _process_predictions(
        self,
        predictions: torch.Tensor,
        image: Image.Image
    ) -> List[Dict]:
        """Process model predictions into structured elements."""
        elements = []
        scores, labels = predictions.max(dim=-1)
        
        for score, label in zip(scores, labels):
            if score >= self.confidence_threshold:
                element = {
                    'label': self.label_list[label],
                    'confidence': float(score),
                    'bbox': self._get_bbox(image),  # Implement bbox extraction
                    'id': len(elements)
                }
                elements.append(element)
                
        if self.merge_boxes:
            elements = self._merge_adjacent_elements(elements)
            
        return elements

    def _detect_reading_order(self, elements: List[Dict]) -> List[int]:
        """Detect natural reading order of elements."""
        if not elements:
            return []
            
        # Create a graph based on spatial relationships
        G = nx.DiGraph()
        
        # Add nodes
        for elem in elements:
            G.add_node(elem['id'])
            
        # Add edges based on layout rules (top-to-bottom, left-to-right)
        for elem1 in elements:
            for elem2 in elements:
                if elem1['id'] != elem2['id']:
                    if self._is_before(elem1, elem2):
                        G.add_edge(elem1['id'], elem2['id'])
        
        # Return topologically sorted nodes
        try:
            return list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            # If cycle detected, fall back to simple top-to-bottom ordering
            return sorted(
                [elem['id'] for elem in elements],
                key=lambda x: elements[x]['bbox'][1]  # Sort by y-coordinate
            )

    def _build_hierarchy(self, elements: List[Dict]) -> Dict:
        """Build hierarchical structure of document elements."""
        hierarchy = {
            'root': {
                'type': 'document',
                'children': []
            }
        }
        
        # Sort elements by vertical position
        sorted_elements = sorted(elements, key=lambda x: x['bbox'][1])
        
        current_section = None
        current_subsection = None
        
        for elem in sorted_elements:
            if elem['label'] == 'title':
                # New main section
                current_section = {
                    'type': 'section',
                    'title': elem,
                    'children': []
                }
                hierarchy['root']['children'].append(current_section)
                current_subsection = None
                
            elif elem['label'] == 'section_header':
                # New subsection
                current_subsection = {
                    'type': 'subsection',
                    'title': elem,
                    'children': []
                }
                if current_section:
                    current_section['children'].append(current_subsection)
                    
            else:
                # Add element to appropriate container
                if current_subsection:
                    current_subsection['children'].append(elem)
                elif current_section:
                    current_section['children'].append(elem)
                else:
                    hierarchy['root']['children'].append(elem)
        
        return hierarchy

    def _extract_styles(self, elements: List[Dict], image: Image.Image) -> Dict:
        """Extract style information for elements."""
        styles = {}
        
        for elem in elements:
            bbox = elem['bbox']
            region = image.crop(bbox)
            
            # Extract style features
            styles[elem['id']] = {
                'font_size': self._estimate_font_size(region),
                'is_bold': self._detect_bold(region),
                'alignment': self._detect_alignment(elem, image.size[0]),
                'indentation': bbox[0],  # Left margin
                'spacing': self._calculate_spacing(elem, elements)
            }
            
        return styles

    def _is_before(self, elem1: Dict, elem2: Dict) -> bool:
        """Determine if elem1 should come before elem2 in reading order."""
        y1 = elem1['bbox'][1]
        y2 = elem2['bbox'][1]
        x1 = elem1['bbox'][0]
        x2 = elem2['bbox'][0]
        
        # Consider elements on same line if y-coordinates are close
        same_line = abs(y1 - y2) < 20
        
        if same_line:
            return x1 < x2
        return y1 < y2

    def _merge_adjacent_elements(self, elements: List[Dict]) -> List[Dict]:
        """Merge adjacent elements of the same type."""
        if not elements:
            return elements
            
        merged = []
        current = elements[0]
        
        for next_elem in elements[1:]:
            if (current['label'] == next_elem['label'] and 
                self._are_adjacent(current, next_elem)):
                # Merge elements
                current = self._merge_elements(current, next_elem)
            else:
                merged.append(current)
                current = next_elem
                
        merged.append(current)
        return merged

    def _are_adjacent(self, elem1: Dict, elem2: Dict) -> bool:
        """Check if two elements are adjacent."""
        x1, y1, x2, y2 = elem1['bbox']
        x3, y3, x4, y4 = elem2['bbox']
        
        # Vertical overlap
        vertical_overlap = (
            (y3 <= y2 and y3 >= y1) or
            (y4 <= y2 and y4 >= y1) or
            (y1 <= y4 and y1 >= y3)
        )
        
        # Horizontal adjacency
        horizontal_adjacent = abs(x2 - x3) < 20 or abs(x4 - x1) < 20
        
        return vertical_overlap and horizontal_adjacent

    def _merge_elements(self, elem1: Dict, elem2: Dict) -> Dict:
        """Merge two elements into one."""
        return {
            'label': elem1['label'],
            'confidence': min(elem1['confidence'], elem2['confidence']),
            'bbox': (
                min(elem1['bbox'][0], elem2['bbox'][0]),  # x1
                min(elem1['bbox'][1], elem2['bbox'][1]),  # y1
                max(elem1['bbox'][2], elem2['bbox'][2]),  # x2
                max(elem1['bbox'][3], elem2['bbox'][3])   # y2
            ),
            'id': elem1['id']
        }



