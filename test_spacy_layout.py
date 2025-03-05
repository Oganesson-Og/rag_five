#!/usr/bin/env python
"""
Test script for spaCy layout integration in the RAG pipeline.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import the necessary modules
from src.document_processing.core.vision.spacy_layout_recognizer import SpaCyLayoutRecognizer
from src.document_processing.extractors.pdf import PDFExtractor
from src.rag.models import Document

async def test_spacy_layout_recognizer():
    """Test the SpaCyLayoutRecognizer class."""
    logger.info("Testing SpaCyLayoutRecognizer...")
    
    # Create a sample PDF file path
    # Replace with an actual PDF file path in your system
    pdf_path = "test_data/sample.pdf"
    
    if not os.path.exists(pdf_path):
        logger.warning(f"Sample PDF file not found at {pdf_path}")
        logger.info("Creating a test directory and downloading a sample PDF...")
        
        # Create test_data directory if it doesn't exist
        os.makedirs("test_data", exist_ok=True)
        
        # Download a sample PDF file
        import requests
        sample_pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        
        try:
            response = requests.get(sample_pdf_url)
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Sample PDF downloaded to {pdf_path}")
        except Exception as e:
            logger.error(f"Failed to download sample PDF: {str(e)}")
            return
    
    # Initialize the SpaCyLayoutRecognizer
    recognizer = SpaCyLayoutRecognizer(
        model_name="en_core_web_sm",
        device="cpu",
        confidence_threshold=0.5,
        merge_boxes=True
    )
    
    # Test the analyze method
    try:
        logger.info(f"Analyzing PDF: {pdf_path}")
        layout_info = await recognizer.analyze(
            document=pdf_path,
            extract_style=True,
            detect_reading_order=True,
            build_hierarchy=True
        )
        
        # Print the results
        logger.info(f"Layout analysis completed successfully")
        logger.info(f"Number of elements: {len(layout_info.get('elements', []))}")
        logger.info(f"Reading order: {layout_info.get('reading_order', [])}")
        logger.info(f"Hierarchy: {layout_info.get('hierarchy', {})}")
        
        # Print markdown if available
        if 'markdown' in layout_info and layout_info['markdown']:
            logger.info(f"Markdown preview: {layout_info['markdown'][:200]}...")
        
        return layout_info
    except Exception as e:
        logger.error(f"Error analyzing PDF with SpaCyLayoutRecognizer: {str(e)}")
        return None

async def test_pdf_extractor_with_spacy_layout():
    """Test the PDFExtractor with spaCy layout integration."""
    logger.info("Testing PDFExtractor with spaCy layout integration...")
    
    # Create a sample PDF file path
    pdf_path = "test_data/sample.pdf"
    
    if not os.path.exists(pdf_path):
        logger.warning(f"Sample PDF file not found at {pdf_path}")
        return
    
    # Create a minimal configuration for the PDFExtractor
    config = {
        'layout': {
            'engine': 'spacy',
            'spacy_model': 'en_core_web_sm',
            'confidence': 0.5,
            'merge_boxes': True
        }
    }
    
    # Initialize the PDFExtractor
    extractor = PDFExtractor(config)
    
    # Create a Document object
    document = Document(source=pdf_path, content="")
    
    # Extract the document
    try:
        logger.info(f"Extracting document: {pdf_path}")
        processed_document = await extractor.extract(document)
        
        # Print the results
        logger.info(f"Document extraction completed successfully")
        logger.info(f"Document content length: {len(processed_document.content)}")
        
        # Print layout information if available
        if 'layout' in processed_document.doc_info:
            layout_info = processed_document.doc_info['layout']
            logger.info(f"Number of pages: {len(layout_info.get('pages', []))}")
            logger.info(f"Number of tables: {len(layout_info.get('tables', []))}")
            
            # Print markdown if available
            if 'markdown' in layout_info and layout_info['markdown']:
                logger.info(f"Markdown preview: {layout_info['markdown'][:200]}...")
        
        return processed_document
    except Exception as e:
        logger.error(f"Error extracting document with PDFExtractor: {str(e)}")
        return None

async def main():
    """Main function to run the tests."""
    logger.info("Starting spaCy layout integration tests...")
    
    # Test the SpaCyLayoutRecognizer
    layout_info = await test_spacy_layout_recognizer()
    
    # Test the PDFExtractor with spaCy layout
    document = await test_pdf_extractor_with_spacy_layout()
    
    logger.info("Tests completed.")

if __name__ == "__main__":
    asyncio.run(main()) 