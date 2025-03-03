"""
PDF Document Recognition Module
------------------------------

This module extends the base Recognizer class to provide specialized
functionality for PDF document analysis using spaCyLayout integration.

Key Features:
- PDF document loading and parsing
- Layout element extraction (headings, paragraphs, tables)
- Text extraction with structural information
- Table recognition and conversion to structured data
- Integration with spaCy NLP pipeline

Technical Details:
- Uses spaCyLayout for PDF processing
- Maintains document structure and hierarchy
- Preserves layout information (positions, styles)
- Supports multi-page documents
- Enables chunking for RAG pipelines

Dependencies:
- spacy>=3.5.0
- spacy-layout>=0.1.0
- pdf2image>=1.16.0
- pytesseract>=0.3.10

Example Usage:
    # Initialize the PDF recognizer
    pdf_recognizer = PDFRecognizer(
        spacy_model="en_core_web_sm",
        device="cuda"
    )
    
    # Process a PDF document
    result = pdf_recognizer.process_document("path/to/document.pdf")
    
    # Extract text with layout information
    text_blocks = pdf_recognizer.extract_text_blocks(result)
    
    # Extract tables as structured data
    tables = pdf_recognizer.extract_tables(result)

Author: Keith Satuku
Version: 1.0.0
License: MIT
"""

import logging
import os
import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple
import tempfile
from pathlib import Path

import spacy
from spacy_layout import spaCyLayout
import pandas as pd
from PIL import Image
import pdf2image

from .recognizer import Recognizer


class PDFRecognizer(Recognizer):
    """
    PDF document recognizer with spaCyLayout integration.
    
    This class extends the base Recognizer to provide specialized functionality
    for PDF document analysis, leveraging spaCyLayout for structured extraction.
    """
    
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        model_dir: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        label_list: Optional[List[str]] = None,
        task_name: str = "document_layout",
        chunk_size: int = 1000,
        extract_tables: bool = True,
        detect_headers: bool = True
    ):
        """
        Initialize the PDF recognizer.
        
        Args:
            spacy_model: Name of the spaCy model to use
            model_dir: Path to the directory containing model files
            device: Device to use (e.g., "cuda" or "cpu")
            batch_size: Batch size for processing
            cache_dir: Directory for caching models
            label_list: List of labels
            task_name: Specific task name for model selection
            chunk_size: Size of text chunks for RAG pipelines
            extract_tables: Whether to extract tables from the document
            detect_headers: Whether to detect and label headers/sections
        """
        super().__init__(
            model_dir=model_dir,
            device=device,
            batch_size=batch_size,
            cache_dir=cache_dir,
            label_list=label_list or ["Title", "Heading", "Paragraph", "List", "Table", "Figure", "Caption"],
            task_name=task_name
        )
        
        # Initialize spaCy with layout processing
        try:
            self.nlp = spacy.load(spacy_model)
            self.layout_processor = spaCyLayout(self.nlp)
            logging.info(f"Initialized spaCyLayout with model: {spacy_model}")
        except Exception as e:
            logging.error(f"Failed to load spaCy model: {e}")
            logging.info("Attempting to download the model...")
            try:
                os.system(f"python -m spacy download {spacy_model}")
                self.nlp = spacy.load(spacy_model)
                self.layout_processor = spaCyLayout(self.nlp)
                logging.info(f"Successfully downloaded and loaded model: {spacy_model}")
            except Exception as download_error:
                logging.error(f"Failed to download spaCy model: {download_error}")
                raise
        
        self.chunk_size = chunk_size
        self.extract_tables = extract_tables
        self.detect_headers = detect_headers
    
    def process_document(self, document_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a PDF document and extract structured information.
        
        Args:
            document_path: Path to the PDF document
            
        Returns:
            Dictionary containing structured document information
        """
        document_path = Path(document_path)
        if not document_path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        # Process the document with spaCyLayout
        try:
            doc = self.layout_processor(str(document_path))
            logging.info(f"Successfully processed document: {document_path.name}")
            
            # Extract structured information
            result = {
                "text": doc.text,
                "layout_spans": self._extract_layout_spans(doc),
                "pages": self._extract_pages_info(doc),
                "metadata": self._extract_metadata(doc),
            }
            
            # Extract tables if enabled
            if self.extract_tables and hasattr(doc._, "tables"):
                result["tables"] = self._extract_tables(doc)
            
            # Create chunks for RAG if needed
            result["chunks"] = self._create_text_chunks(doc)
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing document {document_path.name}: {e}")
            # Fallback to vision-based recognition if spaCyLayout fails
            return self._fallback_vision_recognition(document_path)
    
    def _extract_layout_spans(self, doc) -> List[Dict[str, Any]]:
        """Extract layout spans with their labels and positions."""
        layout_spans = []
        
        if "layout" in doc.spans:
            for span in doc.spans["layout"]:
                span_info = {
                    "text": span.text,
                    "label": span.label_,
                    "start": span.start,
                    "end": span.end,
                }
                
                # Add additional attributes if available
                if hasattr(span._, "page"):
                    span_info["page"] = span._.page
                if hasattr(span._, "bbox"):
                    span_info["bbox"] = span._.bbox
                if hasattr(span._, "font"):
                    span_info["font"] = span._.font
                
                layout_spans.append(span_info)
        
        return layout_spans
    
    def _extract_pages_info(self, doc) -> List[Dict[str, Any]]:
        """Extract information about document pages."""
        pages_info = []
        
        if hasattr(doc._, "pages"):
            for i, page in enumerate(doc._.pages):
                page_info = {
                    "page_number": i + 1,
                    "width": getattr(page, "width", None),
                    "height": getattr(page, "height", None),
                }
                pages_info.append(page_info)
        
        return pages_info
    
    def _extract_metadata(self, doc) -> Dict[str, Any]:
        """Extract document metadata."""
        metadata = {}
        
        if hasattr(doc._, "metadata"):
            metadata = doc._.metadata
        
        # Try to extract common metadata fields
        for field in ["title", "author", "subject", "creator", "producer", "creation_date"]:
            if hasattr(doc._, field):
                metadata[field] = getattr(doc._, field)
        
        return metadata
    
    def _extract_tables(self, doc) -> List[Dict[str, Any]]:
        """Extract tables from the document."""
        tables = []
        
        if hasattr(doc._, "tables"):
            for i, table in enumerate(doc._.tables):
                table_data = None
                if hasattr(table._, "data"):
                    # Convert to pandas DataFrame and then to dict for serialization
                    df = pd.DataFrame(table._.data)
                    table_data = df.to_dict(orient="records")
                
                table_info = {
                    "table_id": i,
                    "data": table_data,
                    "page": getattr(table._, "page", None),
                    "bbox": getattr(table._, "bbox", None),
                }
                
                tables.append(table_info)
        
        return tables
    
    def _create_text_chunks(self, doc) -> List[Dict[str, Any]]:
        """Create text chunks for RAG pipelines."""
        chunks = []
        
        # Simple chunking by layout spans
        current_chunk = ""
        current_chunk_metadata = {
            "spans": [],
            "page": None
        }
        
        if "layout" in doc.spans:
            for span in doc.spans["layout"]:
                # If adding this span would exceed chunk size, save current chunk and start new one
                if len(current_chunk) + len(span.text) > self.chunk_size and current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "metadata": current_chunk_metadata
                    })
                    current_chunk = ""
                    current_chunk_metadata = {"spans": [], "page": None}
                
                # Add span to current chunk
                current_chunk += span.text + "\n"
                current_chunk_metadata["spans"].append(span.label_)
                
                # Update page information
                if hasattr(span._, "page") and current_chunk_metadata["page"] is None:
                    current_chunk_metadata["page"] = span._.page
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append({
                    "text": current_chunk,
                    "metadata": current_chunk_metadata
                })
        
        return chunks
    
    def _fallback_vision_recognition(self, document_path: Path) -> Dict[str, Any]:
        """
        Fallback to vision-based recognition if spaCyLayout fails.
        
        This method converts the PDF to images and processes them using
        the base Recognizer's vision capabilities.
        """
        logging.info(f"Using vision-based fallback for document: {document_path.name}")
        
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(document_path)
            
            # Process images with the base recognizer
            results = []
            for i, img in enumerate(images):
                # Convert PIL Image to numpy array
                img_np = np.array(img)
                
                # Process with base recognizer
                page_result = self([img_np], thr=0.5)[0]
                
                # Add page number
                for item in page_result:
                    item["page"] = i + 1
                
                results.append(page_result)
            
            # Combine results
            combined_result = {
                "text": "Text extraction not available in fallback mode",
                "layout_spans": [],
                "pages": [{"page_number": i+1} for i in range(len(images))],
                "metadata": {},
                "vision_results": results,
                "fallback_mode": True
            }
            
            return combined_result
            
        except Exception as e:
            logging.error(f"Fallback vision recognition failed: {e}")
            return {
                "error": str(e),
                "fallback_mode": True,
                "success": False
            }
    
    def extract_text_blocks(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract text blocks with their types from the recognition result.
        
        Args:
            result: Result from process_document
            
        Returns:
            List of text blocks with type information
        """
        text_blocks = []
        
        # Extract from layout spans
        for span in result.get("layout_spans", []):
            text_blocks.append({
                "text": span["text"],
                "type": span["label"],
                "page": span.get("page"),
                "bbox": span.get("bbox")
            })
        
        # If no layout spans but fallback mode was used
        if not text_blocks and result.get("fallback_mode"):
            for page_idx, page_result in enumerate(result.get("vision_results", [])):
                for item in page_result:
                    # Convert bbox to x0, y0, x1, y1 format if needed
                    bbox = item.get("bbox", [0, 0, 0, 0])
                    
                    text_blocks.append({
                        "text": "",  # No text available in vision-only mode
                        "type": item.get("type", "unknown"),
                        "page": page_idx + 1,
                        "bbox": bbox
                    })
        
        return text_blocks
    
    def extract_tables(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract tables from the recognition result.
        
        Args:
            result: Result from process_document
            
        Returns:
            List of tables with their data
        """
        return result.get("tables", [])
    
    def get_document_text(self, result: Dict[str, Any]) -> str:
        """
        Get the full text of the document.
        
        Args:
            result: Result from process_document
            
        Returns:
            Full document text
        """
        return result.get("text", "")
    
    def get_document_chunks(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get document chunks for RAG pipelines.
        
        Args:
            result: Result from process_document
            
        Returns:
            List of text chunks with metadata
        """
        return result.get("chunks", [])


# Example usage
if __name__ == "__main__":
    # Initialize the PDF recognizer
    pdf_recognizer = PDFRecognizer(
        spacy_model="en_core_web_sm",
        device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    
    # Process a sample PDF document
    result = pdf_recognizer.process_document("sample.pdf")
    
    # Print document information
    print(f"Document has {len(result['pages'])} pages")
    print(f"Found {len(result['layout_spans'])} layout elements")
    
    if "tables" in result:
        print(f"Found {len(result['tables'])} tables")
    
    # Extract text blocks
    text_blocks = pdf_recognizer.extract_text_blocks(result)
    print(f"Extracted {len(text_blocks)} text blocks") 