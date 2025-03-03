"""
Test script for OCR implementation
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Import the OCR class
    from src.document_processing.core.vision.ocr import OCR, HAS_PALIGEMMA, HAS_OPENCV
    
    # Print configuration information
    print("\n=== OCR Configuration ===")
    print(f"OpenCV: {'Available' if HAS_OPENCV else 'Not available'}")
    print(f"PaliGemma: {'Available' if HAS_PALIGEMMA else 'Not available'}")
    
    # Initialize OCR
    ocr = OCR()
    print(f"OCR initialized. Advanced features: {'Available' if ocr.has_advanced_features() else 'Limited'}")
    
    # Print OCR configuration
    print("\n=== OCR Settings ===")
    print(f"Languages: {ocr.languages}")
    print(f"Preserve Layout: {ocr.preserve_layout}")
    print(f"Enhance Resolution: {ocr.enhance_resolution}")
    print(f"Confidence Threshold: {ocr.confidence_threshold}")
    print(f"Use PaliGemma: {ocr.use_paligemma}")
    
    print("\nOCR test completed successfully!")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc() 