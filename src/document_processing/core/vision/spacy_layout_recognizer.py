"""
SpaCy Layout Recognition Module
-------------------------------

Document layout analysis using spaCy and spaCy-layout for detecting
and classifying document elements and structure.

This module provides an alternative to the LayoutLMv3-based recognition
with a focus on accuracy and robustness through the spaCy ecosystem.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .recognizer import Recognizer, LayoutElement

logger = logging.getLogger(__name__)

class SpaCyLayoutRecognizer(Recognizer):
    """Document layout analysis using spaCy and spaCy-layout."""
    
    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        device: str = "cpu",
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        confidence_threshold: float = 0.5,
        merge_boxes: bool = True,
        label_list: Optional[List[str]] = None
    ):
        """Initialize spaCy layout recognizer.
        
        Args:
            model_name: spaCy model name
            device: Device to use (cuda/cpu)
            batch_size: Batch size for processing
            cache_dir: Directory to cache models
            confidence_threshold: Minimum confidence for predictions
            merge_boxes: Whether to merge adjacent boxes
            label_list: Custom label list (optional)
        """
        super().__init__(
            model_type="spacy",
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            cache_dir=cache_dir,
            confidence=confidence_threshold,
            merge_boxes=merge_boxes,
            label_list=label_list or self.DEFAULT_LABEL_LIST
        )
        
        self.confidence_threshold = confidence_threshold
        self._init_model()
        
    def _init_model(self):
        """Initialize spaCy model with layout processing."""
        try:
            import spacy
            from spacy_layout import spaCyLayout
            
            # Initialize a blank spaCy model and the spaCyLayout preprocessor
            self.nlp = spacy.blank("en") if self.model_name == "en" else spacy.load(self.model_name)
            self.layout_processor = spaCyLayout(self.nlp)
            
            logger.info(f"SpaCy layout model initialized successfully")
            self.model_loaded = True
            
        except ImportError:
            logger.error("Failed to import spacy or spacy_layout. Please install with: pip install spacy spacy-layout")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"Failed to initialize spaCy layout model: {str(e)}")
            self.model_loaded = False
    
    async def analyze(
        self,
        document: Union[str, Path],
        extract_style: bool = False,
        detect_reading_order: bool = True,
        build_hierarchy: bool = True
    ) -> Dict:
        """
        Analyze document layout using spaCy layout.
        
        Args:
            document: Path to PDF document
            extract_style: Whether to extract text styles
            detect_reading_order: Whether to detect reading order
            build_hierarchy: Whether to build document hierarchy
            
        Returns:
            Dict containing layout analysis results
        """
        import asyncio
        # Ensure this is a true coroutine by adding a small sleep
        await asyncio.sleep(0)
        
        if not self.model_loaded:
            logger.warning("SpaCy layout model not loaded, returning empty analysis")
            return {"elements": [], "reading_order": [], "hierarchy": {}}
        
        try:
            # Process the PDF and get a spaCy Doc object
            doc = self.layout_processor(str(document))
            
            # Extract layout elements
            elements = self._extract_layout_elements(doc)
            
            # Build result dictionary
            result = {
                "elements": elements,
                "reading_order": [],
                "hierarchy": {},
                "markdown": doc._.markdown if hasattr(doc._, "markdown") else ""
            }
            
            # Detect reading order if requested
            if detect_reading_order and elements:
                result["reading_order"] = self._detect_reading_order(elements)
                
            # Build hierarchy if requested
            if build_hierarchy and elements:
                result["hierarchy"] = self._build_hierarchy(elements)
                
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing document with spaCy layout: {str(e)}")
            return {"elements": [], "reading_order": [], "hierarchy": {}}
    
    def _extract_layout_elements(self, doc) -> List[Dict]:
        """
        Extract layout elements from spaCy Doc.
        
        Args:
            doc: spaCy Doc with layout information
            
        Returns:
            List of layout elements
        """
        elements = []
        
        # The doc._.pages attribute returns a list of tuples: (PageLayout, list of layout spans)
        for page_layout, spans in doc._.pages:
            for span in spans:
                # Skip elements with low confidence
                confidence = getattr(span._, "confidence", 1.0)
                if confidence < self.confidence_threshold:
                    continue
                    
                # Create element dictionary
                element = {
                    "type": span.label_.lower(),
                    "text": span.text,
                    "bbox": (
                        span._.layout.x, 
                        span._.layout.y, 
                        span._.layout.x + span._.layout.width, 
                        span._.layout.y + span._.layout.height
                    ),
                    "page": span._.layout.page_no,
                    "confidence": confidence,
                    "metadata": {
                        "start_char": span.start_char,
                        "end_char": span.end_char,
                        "heading": span._.heading.text if hasattr(span._, "heading") and span._.heading is not None else None
                    }
                }
                elements.append(element)
                
        # Merge adjacent elements if requested
        if self.merge_boxes and elements:
            elements = self._merge_adjacent_elements(elements)
            
        return elements
    
    def _detect_reading_order(self, elements: List[Dict]) -> List[int]:
        """
        Detect reading order of elements.
        
        Args:
            elements: List of layout elements
            
        Returns:
            List of indices representing reading order
        """
        # Sort elements by page, then y-coordinate, then x-coordinate
        sorted_indices = sorted(
            range(len(elements)),
            key=lambda i: (
                elements[i].get("page", 0),
                elements[i]["bbox"][1],  # y-coordinate
                elements[i]["bbox"][0]   # x-coordinate
            )
        )
        
        return sorted_indices
    
    def _build_hierarchy(self, elements: List[Dict]) -> Dict:
        """
        Build a hierarchical structure of document elements.
        
        Args:
            elements: List of document elements
            
        Returns:
            Dictionary with hierarchical structure
        """
        # Simple hierarchy based on element types and positions
        hierarchy = {
            "root": [],
            "sections": [],
            "blocks": []
        }
        
        # Group elements by type
        for i, elem in enumerate(elements):
            elem_type = elem["type"].lower()
            
            if elem_type in ["title", "heading", "header"]:
                hierarchy["root"].append(i)
            elif elem_type in ["section", "subsection"]:
                hierarchy["sections"].append(i)
            else:
                hierarchy["blocks"].append(i)
            
        return hierarchy
    
    def _merge_adjacent_elements(self, elements: List[Dict]) -> List[Dict]:
        """
        Merge adjacent elements of the same type.
        
        Args:
            elements: List of document elements
            
        Returns:
            List of merged elements
        """
        if not elements or len(elements) < 2:
            return elements
            
        # Sort elements by vertical position
        sorted_elements = sorted(elements, key=lambda e: (e["bbox"][1], e["bbox"][0]))
        
        merged_elements = []
        current_element = sorted_elements[0]
        
        for next_element in sorted_elements[1:]:
            # Check if elements are of the same type and adjacent
            if (current_element["type"] == next_element["type"] and
                self._are_elements_adjacent(current_element, next_element)):
                
                # Merge elements
                current_element = self._merge_element_pair(current_element, next_element)
            else:
                merged_elements.append(current_element)
                current_element = next_element
                
        # Add the last element
        merged_elements.append(current_element)
        
        return merged_elements
        
    def _are_elements_adjacent(self, elem1: Dict, elem2: Dict) -> bool:
        """Check if two elements are adjacent."""
        # Get bounding boxes
        bbox1 = elem1["bbox"]
        bbox2 = elem2["bbox"]
        
        # Check vertical adjacency (with small threshold)
        vertical_threshold = 10  # pixels
        if abs(bbox1[3] - bbox2[1]) <= vertical_threshold:
            # Check horizontal overlap
            horizontal_overlap = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
            return horizontal_overlap > 0
            
        return False
        
    def _merge_element_pair(self, elem1: Dict, elem2: Dict) -> Dict:
        """Merge two elements into one."""
        # Create a new merged element
        merged = elem1.copy()
        
        # Merge bounding boxes
        merged["bbox"] = [
            min(elem1["bbox"][0], elem2["bbox"][0]),  # x0
            min(elem1["bbox"][1], elem2["bbox"][1]),  # y0
            max(elem1["bbox"][2], elem2["bbox"][2]),  # x1
            max(elem1["bbox"][3], elem2["bbox"][3])   # y1
        ]
        
        # Merge text
        merged["text"] = elem1["text"] + " " + elem2["text"]
        
        # Average confidence
        merged["confidence"] = (elem1["confidence"] + elem2["confidence"]) / 2
        
        return merged
    
    async def process_page(self, page_image: Union[str, Image.Image, np.ndarray]) -> List[LayoutElement]:
        """
        Process a single page image.
        
        Args:
            page_image: Page image to process
            
        Returns:
            List of LayoutElement objects
        """
        logger.warning("SpaCyLayoutRecognizer.process_page is not fully implemented for single images")
        logger.warning("For best results, use the analyze method with a PDF file path")
        
        # Convert to PIL Image if needed
        if isinstance(page_image, str):
            image = Image.open(page_image)
        elif isinstance(page_image, np.ndarray):
            image = Image.fromarray(page_image)
        else:
            image = page_image
            
        # Save image to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            image.save(tmp_path)
            
        try:
            # Process the image
            doc = self.layout_processor(tmp_path)
            
            # Extract layout elements
            elements = self._extract_layout_elements(doc)
            
            # Convert to LayoutElement objects
            layout_elements = []
            for elem in elements:
                layout_elements.append(LayoutElement(
                    type=elem["type"],
                    text=elem["text"],
                    bbox=elem["bbox"],
                    confidence=elem["confidence"]
                ))
                
            return layout_elements
            
        except Exception as e:
            logger.error(f"Error processing page with spaCy layout: {str(e)}")
            return []
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path) 