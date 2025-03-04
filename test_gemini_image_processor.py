#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for Gemini image processing in the PDF extractor.
"""

import os
import sys
import logging
import tempfile
import yaml
from pathlib import Path
import base64
from PIL import Image
import io
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger('GeminiImageTest')

# Add the project root to the Python path
def get_project_base_directory():
    """Get the base directory of the project."""
    return os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, get_project_base_directory())

# Import the PDFExtractor
from src.document_processing.extractors.pdf import PDFExtractor

def create_test_pdf_with_image():
    """Create a test PDF with an embedded image."""
    # Create a temporary PDF file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        pdf_path = tmp.name
    
    # Create a new PDF document
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    
    # Add some text
    page.insert_text((50, 50), "Test PDF with Image", fontsize=16)
    page.insert_text((50, 80), "This is a test PDF with an embedded image for testing Gemini image processing.", fontsize=12)
    
    # Create a simple image
    img = Image.new('RGB', (300, 300), color=(73, 109, 137))
    
    # Add a simple diagram (a circle)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse((50, 50, 250, 250), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.line((50, 150, 250, 150), fill=(0, 0, 0), width=2)
    draw.line((150, 50, 150, 250), fill=(0, 0, 0), width=2)
    
    # Add some text to the image
    draw.text((120, 30), "Simple Diagram", fill=(0, 0, 0))
    draw.text((60, 160), "X-axis", fill=(0, 0, 0))
    draw.text((160, 60), "Y-axis", fill=(0, 0, 0))
    
    # Save the image to a bytes buffer
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Insert the image into the PDF
    page.insert_image(
        fitz.Rect(50, 100, 350, 400),
        stream=img_bytes.getvalue()
    )
    
    # Save the PDF
    doc.save(pdf_path)
    doc.close()
    
    logger.info(f"Created test PDF at {pdf_path}")
    return pdf_path

def update_config_for_gemini():
    """Update the configuration to use Gemini for image processing."""
    config_path = os.path.join(get_project_base_directory(), "config/rag_config.yaml")
    
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found at {config_path}")
        return None
    
    # Load the existing configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the configuration to use Gemini for image processing
    if 'pdf' not in config:
        config['pdf'] = {}
    
    if 'picture_annotation' not in config['pdf']:
        config['pdf']['picture_annotation'] = {}
    
    # Set the model type to Gemini
    config['pdf']['picture_annotation']['enabled'] = True
    config['pdf']['picture_annotation']['model_type'] = 'gemini'
    config['pdf']['picture_annotation']['model_name'] = 'gemini-2.0-pro-exp-02-05'
    
    # Make sure we have an API key
    if 'model' in config and 'gemini_api_key' in config['model']:
        api_key = config['model']['gemini_api_key']
        # Ensure the API key is set in the picture_annotation section
        config['pdf']['picture_annotation']['api_key'] = api_key
        
        # Also ensure the API key is set in the model section
        if 'api_key' not in config['model']:
            config['model']['api_key'] = api_key
            
        # Add API configuration for DoclingExtractor
        config['pdf']['picture_annotation']['api_config'] = {
            'url': 'https://generativelanguage.googleapis.com/v1/models/gemini-2.0-pro-exp-02-05:generateContent',
            'api_key': api_key,
            'headers': {
                'Content-Type': 'application/json'
            },
            'params': {
                'model': 'gemini-2.0-pro-exp-02-05',
                'temperature': 0.7,
                'max_output_tokens': 1024,
                'top_p': 0.9
            },
            'timeout': 90
        }
    else:
        logger.warning("No Gemini API key found in configuration")
        # For testing purposes, we'll use a placeholder API key
        # In a real scenario, you would need a valid API key
        if 'model' not in config:
            config['model'] = {}
        config['model']['gemini_api_key'] = "YOUR_API_KEY"
        config['pdf']['picture_annotation']['api_key'] = config['model']['gemini_api_key']
        
        # Add API configuration for DoclingExtractor
        config['pdf']['picture_annotation']['api_config'] = {
            'url': 'https://generativelanguage.googleapis.com/v1/models/gemini-2.0-pro-exp-02-05:generateContent',
            'api_key': config['model']['gemini_api_key'],
            'headers': {
                'Content-Type': 'application/json'
            },
            'params': {
                'model': 'gemini-2.0-pro-exp-02-05',
                'temperature': 0.7,
                'max_output_tokens': 1024,
                'top_p': 0.9
            },
            'timeout': 90
        }
    
    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as tmp:
        yaml.dump(config, tmp)
        tmp_config_path = tmp.name
    
    logger.info(f"Created temporary configuration at {tmp_config_path}")
    logger.info(f"Updated configuration: {config['pdf']['picture_annotation']}")
    return tmp_config_path

def test_gemini_image_processor():
    """Test the Gemini image processor functionality."""
    # Create a test PDF with an image
    pdf_path = create_test_pdf_with_image()
    
    try:
        # Update the configuration to use Gemini
        config_path = update_config_for_gemini()
        
        if not config_path:
            logger.error("Failed to update configuration")
            return
        
        # Load the configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Print the configuration
        logger.info("Configuration:")
        logger.info(f"PDF picture_annotation: {config.get('pdf', {}).get('picture_annotation', {})}")
        logger.info(f"Model: {config.get('model', {})}")
        
        # Initialize the PDF extractor with the PDF-specific configuration
        logger.info("Initializing PDFExtractor...")
        # Extract just the PDF configuration section
        pdf_config = config.get('pdf', {})
        # Add the model configuration to ensure API keys are available
        pdf_config['model'] = config.get('model', {})
        
        logger.info(f"Passing configuration to PDFExtractor: {pdf_config}")
        extractor = PDFExtractor(pdf_config)
        
        # Check if Gemini image processor is initialized
        if hasattr(extractor, 'gemini_image_processor') and extractor.gemini_image_processor:
            logger.info("Gemini image processor initialized successfully")
        else:
            logger.warning("Gemini image processor not initialized")
            if hasattr(extractor, 'gemini_image_processor'):
                logger.info("gemini_image_processor attribute exists but is None")
            else:
                logger.info("gemini_image_processor attribute does not exist")
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Extract images
        logger.info("Extracting images from PDF...")
        images = extractor._extract_images_with_context(doc)
        
        # Check the results
        if not images:
            logger.warning("No images extracted from the PDF")
        else:
            logger.info(f"Extracted {len(images)} images")
            
            for i, image in enumerate(images):
                logger.info(f"Image {i+1}:")
                logger.info(f"  Page: {image['page']}")
                logger.info(f"  Size: {image['width']}x{image['height']}")
                logger.info(f"  Format: {image['format']}")
                
                if 'description' in image:
                    logger.info(f"  Description: {image['description']}")
                else:
                    logger.info("  No description available")
                
                if 'metadata' in image and 'diagram_type' in image['metadata']:
                    logger.info(f"  Diagram Type: {image['metadata']['diagram_type']}")
                else:
                    logger.info("  No diagram type available")
        
        # Close the PDF
        doc.close()
        
        logger.info("Test completed successfully")
        
    finally:
        # Clean up
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
            logger.info(f"Removed test PDF file: {pdf_path}")
        
        if 'config_path' in locals() and os.path.exists(config_path):
            os.unlink(config_path)
            logger.info(f"Removed temporary configuration file: {config_path}")

if __name__ == "__main__":
    test_gemini_image_processor() 