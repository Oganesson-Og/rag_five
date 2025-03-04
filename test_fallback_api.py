#!/usr/bin/env python3
"""
Test script to verify the API fallback mechanism for rate limiting.
"""

import yaml
import sys
from pathlib import Path

def main():
    """Test the fallback mechanism for rate limiting."""
    try:
        print("Starting test...")
        
        # Load configuration
        config_path = Path('config/rag_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Loaded configuration from {config_path}")
        
        # Import PDFExtractor
        print("Importing PDFExtractor...")
        from src.document_processing.extractors.pdf import PDFExtractor
        
        # Initialize PDFExtractor with configuration
        print("Initializing PDFExtractor...")
        pdf_config = config['pdf']
        extractor = PDFExtractor(config=pdf_config)
        
        print("PDFExtractor initialized successfully")
        
        # Check if primary and secondary API keys are available
        if hasattr(extractor, 'primary_api_key') and extractor.primary_api_key:
            print(f"Primary API key: {extractor.primary_api_key[:10]}... (truncated)")
        else:
            print("Primary API key not found")
            
        if hasattr(extractor, 'secondary_api_key') and extractor.secondary_api_key:
            print(f"Secondary API key: {extractor.secondary_api_key[:10]}... (truncated)")
        else:
            print("Secondary API key not found")
            
        # Check model names
        if hasattr(extractor, 'primary_model_name'):
            print(f"Primary model: {extractor.primary_model_name}")
        
        if hasattr(extractor, 'secondary_model_name'):
            print(f"Secondary model: {extractor.secondary_model_name}")
            
        # Test fallback mechanism
        print("Testing fallback mechanism...")
        
        # Simulate a rate limit error and switch to fallback
        if hasattr(extractor, '_switch_to_fallback_api'):
            result = extractor._switch_to_fallback_api()
            print(f"Switch to fallback API result: {result}")
            
            if result:
                print(f"Current API key: {extractor.current_api_key[:10]}... (truncated)")
                print(f"Current model: {extractor.current_model_name}")
                print(f"Rate limited: {extractor.rate_limited}")
                
                # Test switching back to primary
                print("Testing switch back to primary API...")
                back_result = extractor._switch_to_primary_api()
                print(f"Switch back to primary API result: {back_result}")
                
                if back_result:
                    print(f"Current API key: {extractor.current_api_key[:10]}... (truncated)")
                    print(f"Current model: {extractor.current_model_name}")
                    print(f"Rate limited: {extractor.rate_limited}")
        else:
            print("Fallback mechanism not implemented")
            
        # Test DoclingExtractor initialization
        if hasattr(extractor, 'docling_extractor') and extractor.docling_extractor:
            print("DoclingExtractor initialized successfully")
            
            # Check if DoclingExtractor has fallback support
            if hasattr(extractor.docling_extractor, 'secondary_api_config'):
                print("DoclingExtractor has fallback support")
                
                if extractor.docling_extractor.secondary_api_config:
                    print("Secondary API config is available for DoclingExtractor")
                else:
                    print("Secondary API config is not available for DoclingExtractor")
        else:
            print("DoclingExtractor not initialized")
        
        print("Test completed successfully")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main() 