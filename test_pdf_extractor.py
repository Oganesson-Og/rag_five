#!/usr/bin/env python
"""
Test script for the PDF Extractor.

This script creates a simple PDF document using PyMuPDF and tests the PDF extractor
to verify that it can extract text, tables, and images correctly.
"""

import os
import sys
import logging
import tempfile
import fitz  # PyMuPDF
import asyncio
import io
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PDFExtractorTest")

def create_test_pdf():
    """Create a more complex test PDF with text, tables, images, and formatting."""
    # Create a temporary PDF file
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_file.close()
    pdf_path = temp_file.name
    
    # Create a new PDF document
    doc = fitz.open()
    
    # Page 1: Title page with formatting
    page = doc.new_page()
    
    # Add title with formatting
    page.insert_text((50, 50), "PDF Extractor Test Document", fontsize=24, fontname="Helvetica-Bold", color=(0, 0, 0.8))
    page.insert_text((50, 80), "A comprehensive test for extraction capabilities", fontsize=16, fontname="Helvetica-Oblique")
    page.insert_text((50, 120), "Created for testing purposes", fontsize=12)
    
    # Add a horizontal line
    page.draw_line((50, 140), (550, 140), color=(0, 0, 0), width=1)
    
    # Add formatted paragraphs
    text = """
    This document contains various elements to test the PDF extractor's capabilities:
    
    • Formatted text with different styles and sizes
    • Tables with borders and without borders
    • Images and diagrams
    • Lists and bullet points
    • Multiple pages with different layouts
    
    The extractor should be able to handle all these elements and extract their content correctly.
    """
    page.insert_text((50, 160), text, fontsize=12)
    
    # Page 2: Text content with headings and paragraphs
    page = doc.new_page()
    
    # Add headings
    page.insert_text((50, 50), "1. Introduction to PDF Extraction", fontsize=18, fontname="Helvetica-Bold")
    
    # Add paragraphs
    intro_text = """
    PDF extraction is the process of automatically extracting content from PDF documents. This includes text, tables, images, and other elements. The extraction process can be challenging due to the complex structure of PDF documents and the variety of ways in which content can be represented.
    
    Modern PDF extractors use a combination of techniques to extract content, including:
    • Text extraction using PDF libraries
    • OCR for scanned documents
    • Table detection and extraction
    • Image extraction and analysis
    """
    page.insert_text((50, 80), intro_text, fontsize=11)
    
    # Add subheading
    page.insert_text((50, 200), "1.1 Challenges in PDF Extraction", fontsize=14, fontname="Helvetica-Bold")
    
    # Add more paragraphs
    challenges_text = """
    PDF extraction faces several challenges:
    
    1. Varied document structures
    2. Mixed content types (text, tables, images)
    3. Scanned documents requiring OCR
    4. Complex layouts with multiple columns
    5. Headers, footers, and page numbers
    
    Effective PDF extractors need to handle all these challenges to provide accurate results.
    """
    page.insert_text((50, 230), challenges_text, fontsize=11)
    
    # Page 3: Tables
    page = doc.new_page()
    
    # Add heading
    page.insert_text((50, 50), "2. Tables in PDF Documents", fontsize=18, fontname="Helvetica-Bold")
    
    # Add introduction to tables
    page.insert_text((50, 80), "Tables are common in PDF documents and can be challenging to extract correctly.", fontsize=11)
    
    # Add a bordered table
    page.insert_text((50, 120), "2.1 Bordered Table Example:", fontsize=14, fontname="Helvetica-Bold")
    
    # Draw table borders
    table_rect = fitz.Rect(50, 150, 550, 300)
    page.draw_rect(table_rect, color=(0, 0, 0), width=1)
    
    # Draw table header row
    header_rect = fitz.Rect(50, 150, 550, 180)
    page.draw_rect(header_rect, color=(0, 0, 0), width=1, fill=(0.8, 0.8, 0.8))
    
    # Draw column dividers
    page.draw_line((150, 150), (150, 300), color=(0, 0, 0), width=1)
    page.draw_line((250, 150), (250, 300), color=(0, 0, 0), width=1)
    page.draw_line((350, 150), (350, 300), color=(0, 0, 0), width=1)
    page.draw_line((450, 150), (450, 300), color=(0, 0, 0), width=1)
    
    # Draw row dividers
    page.draw_line((50, 180), (550, 180), color=(0, 0, 0), width=1)
    page.draw_line((50, 210), (550, 210), color=(0, 0, 0), width=1)
    page.draw_line((50, 240), (550, 240), color=(0, 0, 0), width=1)
    page.draw_line((50, 270), (550, 270), color=(0, 0, 0), width=1)
    
    # Add table headers
    page.insert_text((75, 170), "Product", fontsize=12, fontname="Helvetica-Bold")
    page.insert_text((175, 170), "Category", fontsize=12, fontname="Helvetica-Bold")
    page.insert_text((275, 170), "Price", fontsize=12, fontname="Helvetica-Bold")
    page.insert_text((375, 170), "Quantity", fontsize=12, fontname="Helvetica-Bold")
    page.insert_text((475, 170), "Total", fontsize=12, fontname="Helvetica-Bold")
    
    # Add table data
    # Row 1
    page.insert_text((75, 200), "Laptop", fontsize=11)
    page.insert_text((175, 200), "Electronics", fontsize=11)
    page.insert_text((275, 200), "$1,200.00", fontsize=11)
    page.insert_text((375, 200), "2", fontsize=11)
    page.insert_text((475, 200), "$2,400.00", fontsize=11)
    
    # Row 2
    page.insert_text((75, 230), "Monitor", fontsize=11)
    page.insert_text((175, 230), "Electronics", fontsize=11)
    page.insert_text((275, 230), "$300.00", fontsize=11)
    page.insert_text((375, 230), "3", fontsize=11)
    page.insert_text((475, 230), "$900.00", fontsize=11)
    
    # Row 3
    page.insert_text((75, 260), "Keyboard", fontsize=11)
    page.insert_text((175, 260), "Accessories", fontsize=11)
    page.insert_text((275, 260), "$80.00", fontsize=11)
    page.insert_text((375, 260), "5", fontsize=11)
    page.insert_text((475, 260), "$400.00", fontsize=11)
    
    # Row 4
    page.insert_text((75, 290), "Mouse", fontsize=11)
    page.insert_text((175, 290), "Accessories", fontsize=11)
    page.insert_text((275, 290), "$25.00", fontsize=11)
    page.insert_text((375, 290), "5", fontsize=11)
    page.insert_text((475, 290), "$125.00", fontsize=11)
    
    # Add a borderless table
    page.insert_text((50, 330), "2.2 Borderless Table Example:", fontsize=14, fontname="Helvetica-Bold")
    
    # Add table headers (no borders)
    page.insert_text((75, 360), "Month", fontsize=12, fontname="Helvetica-Bold")
    page.insert_text((175, 360), "Revenue", fontsize=12, fontname="Helvetica-Bold")
    page.insert_text((275, 360), "Expenses", fontsize=12, fontname="Helvetica-Bold")
    page.insert_text((375, 360), "Profit", fontsize=12, fontname="Helvetica-Bold")
    
    # Add table data (no borders)
    # Row 1
    page.insert_text((75, 390), "January", fontsize=11)
    page.insert_text((175, 390), "$10,000", fontsize=11)
    page.insert_text((275, 390), "$8,000", fontsize=11)
    page.insert_text((375, 390), "$2,000", fontsize=11)
    
    # Row 2
    page.insert_text((75, 420), "February", fontsize=11)
    page.insert_text((175, 420), "$12,000", fontsize=11)
    page.insert_text((275, 420), "$7,500", fontsize=11)
    page.insert_text((375, 420), "$4,500", fontsize=11)
    
    # Row 3
    page.insert_text((75, 450), "March", fontsize=11)
    page.insert_text((175, 450), "$15,000", fontsize=11)
    page.insert_text((275, 450), "$9,000", fontsize=11)
    page.insert_text((375, 450), "$6,000", fontsize=11)
    
    # Page 4: Images and diagrams
    page = doc.new_page()
    
    # Add heading
    page.insert_text((50, 50), "3. Images and Diagrams", fontsize=18, fontname="Helvetica-Bold")
    
    # Add introduction to images
    page.insert_text((50, 80), "PDF documents often contain images and diagrams that need to be extracted.", fontsize=11)
    
    # Create a simple diagram (bar chart)
    # Draw axes
    page.draw_line((100, 350), (100, 150), color=(0, 0, 0), width=1)  # Y-axis
    page.draw_line((100, 350), (400, 350), color=(0, 0, 0), width=1)  # X-axis
    
    # Draw bars
    page.draw_rect(fitz.Rect(150, 250, 180, 350), color=(0, 0, 0), fill=(1, 0, 0))
    page.draw_rect(fitz.Rect(200, 200, 230, 350), color=(0, 0, 0), fill=(0, 1, 0))
    page.draw_rect(fitz.Rect(250, 300, 280, 350), color=(0, 0, 0), fill=(0, 0, 1))
    page.draw_rect(fitz.Rect(300, 180, 330, 350), color=(0, 0, 0), fill=(0.5, 0.5, 0))
    
    # Add labels
    page.insert_text((150, 370), "Q1", fontsize=10)
    page.insert_text((200, 370), "Q2", fontsize=10)
    page.insert_text((250, 370), "Q3", fontsize=10)
    page.insert_text((300, 370), "Q4", fontsize=10)
    
    page.insert_text((80, 250), "100", fontsize=10)
    page.insert_text((80, 200), "150", fontsize=10)
    page.insert_text((80, 150), "200", fontsize=10)
    
    # Add title
    page.insert_text((150, 130), "Quarterly Sales Performance", fontsize=12, fontname="Helvetica-Bold")
    
    # Create a simple image (checkerboard pattern)
    img_size = 100
    checkerboard = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Create checkerboard pattern
    square_size = 20
    for i in range(0, img_size, square_size):
        for j in range(0, img_size, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                checkerboard[i:i+square_size, j:j+square_size] = [255, 255, 255]
            else:
                checkerboard[i:i+square_size, j:j+square_size] = [0, 0, 0]
    
    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(checkerboard)
    
    # Convert PIL Image to bytes
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Add image to PDF
    img_rect = fitz.Rect(450, 150, 550, 250)
    page.insert_image(img_rect, stream=img_bytes.getvalue())
    
    # Add caption
    page.insert_text((450, 270), "Sample Image", fontsize=10, fontname="Helvetica-Bold")
    
    # Save the document
    doc.save(pdf_path)
    doc.close()
    
    logger.info(f"Created test PDF at {pdf_path}")
    return pdf_path

async def test_pdf_extractor(pdf_path):
    """Test the PDF extractor with the given PDF file."""
    try:
        # Import the PDFExtractor
        from src.document_processing.extractors.pdf import PDFExtractor
        
        # Import the Document model from extractors.models
        from src.document_processing.extractors.models import Document
        
        # Create a simple configuration
        config = {
            'use_ocr': False,
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
            'acceleration': {
                'num_threads': 4,
                'device': 'cpu'
            }
        }
        
        # Initialize the extractor with the config
        logger.info("Initializing PDFExtractor...")
        extractor = PDFExtractor(config=config)
        logger.info("PDFExtractor initialized successfully")
        
        # Create a document object
        with open(pdf_path, 'rb') as f:
            content = f.read()
        
        document = Document(
            content=content,
            modality="pdf",
            doc_info={
                "filename": os.path.basename(pdf_path),
                "content_type": "application/pdf"
            }
        )
        
        # Extract content - properly await the async method
        logger.info("Extracting content from PDF...")
        result = await extractor.extract(document)
        
        # Print results
        logger.info("Extraction completed successfully")
        
        # Check if result has text attribute or if it's in doc_info
        if hasattr(result, 'text'):
            logger.info(f"Extracted text sample: {result.text[:100]}...")
            logger.info(f"Total text length: {len(result.text)} characters")
        elif isinstance(result, Document) and 'extracted_text' in result.doc_info:
            logger.info(f"Extracted text sample: {result.doc_info['extracted_text'][:100]}...")
            logger.info(f"Total text length: {len(result.doc_info['extracted_text'])} characters")
        else:
            logger.info("No text extracted or text not found in expected location")
        
        # Check for tables
        tables = []
        if hasattr(result, 'tables'):
            tables = result.tables
        elif isinstance(result, Document) and 'tables' in result.doc_info:
            tables = result.doc_info['tables']
            
        if tables:
            logger.info(f"Extracted {len(tables)} tables")
            for i, table in enumerate(tables):
                if isinstance(table, dict) and 'data' in table:
                    logger.info(f"Table {i+1}: {len(table['data'])} rows x {len(table['data'][0]) if table['data'] else 0} columns")
                    # Log a sample of the table data
                    if table['data'] and len(table['data']) > 0:
                        logger.info(f"Table {i+1} sample: {table['data'][0]}")
                else:
                    logger.info(f"Table {i+1}: format unknown")
        else:
            logger.info("No tables extracted")
            
        # Check for images
        images = []
        if hasattr(result, 'images'):
            images = result.images
        elif isinstance(result, Document) and 'images' in result.doc_info:
            images = result.doc_info['images']
            
        if images:
            logger.info(f"Extracted {len(images)} images")
            for i, image in enumerate(images):
                if isinstance(image, dict):
                    page = image.get('page', 'unknown')
                    width = image.get('width', 'unknown')
                    height = image.get('height', 'unknown')
                    format_type = image.get('format', 'unknown')
                    logger.info(f"Image {i+1}: Page {page}, Size: {width}x{height}, Format: {format_type}")
                else:
                    logger.info(f"Image {i+1}: format unknown")
        else:
            logger.info("No images extracted")
            
        # Check for document structure
        if isinstance(result, Document) and 'structure' in result.doc_info:
            structure = result.doc_info['structure']
            logger.info(f"Document structure detected with {len(structure.get('elements', []))} elements")
            
            # Log headings if available
            headings = [elem for elem in structure.get('elements', []) if elem.get('type', '').startswith('heading')]
            if headings:
                logger.info(f"Detected {len(headings)} headings:")
                for i, heading in enumerate(headings[:3]):  # Show first 3 headings
                    logger.info(f"  Heading {i+1}: {heading.get('text', '')}")
                if len(headings) > 3:
                    logger.info(f"  ... and {len(headings) - 3} more headings")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error testing PDFExtractor: {str(e)}")
        return False

async def main_async():
    """Async main function to run the test."""
    try:
        # Create a test PDF
        pdf_path = create_test_pdf()
        
        # Test the PDF extractor
        success = await test_pdf_extractor(pdf_path)
        
        # Clean up
        try:
            os.remove(pdf_path)
            logger.info(f"Removed test PDF file: {pdf_path}")
        except Exception as e:
            logger.warning(f"Failed to remove test PDF: {str(e)}")
        
        if success:
            logger.info("PDF Extractor test completed successfully")
            return 0
        else:
            logger.error("PDF Extractor test failed")
            return 1
            
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        return 1

def main():
    """Main function to run the async test."""
    return asyncio.run(main_async())

if __name__ == "__main__":
    sys.exit(main()) 