#!/usr/bin/env python
"""
Debug script for TableStructureRecognizer initialization.
"""

import os
import logging
import yaml
import sys

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TableStructureRecognizerDebug")

def get_project_base_directory():
    """Get the base directory of the project."""
    return os.path.dirname(os.path.abspath(__file__))

def debug_table_structure_recognizer():
    """Debug the TableStructureRecognizer initialization."""
    try:
        # Get the path to the config file
        config_path = os.path.join(get_project_base_directory(), "config/rag_config.yaml")
        
        # Check if the file exists
        if not os.path.exists(config_path):
            logger.error(f"Config file not found at {config_path}")
            return False
        
        # Read the config file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract the cross_modal config
        pdf_config = config.get('pdf', {})
        educational_config = pdf_config.get('educational', {})
        cross_modal_config = educational_config.get('cross_modal', {})
        
        # Print the cross_modal config
        logger.info(f"Cross-modal config: {cross_modal_config}")
        
        # Check the model_name
        model_name = cross_modal_config.get('model_name', '')
        logger.info(f"Model name: {model_name} (type: {type(model_name)})")
        
        # Try to use the model_name
        if not isinstance(model_name, str):
            logger.warning(f"model_name is not a string but a {type(model_name).__name__}. Converting to string representation.")
            try:
                model_name = str(model_name)
                logger.info(f"Converted model_name: {model_name}")
            except Exception as e:
                logger.error(f"Failed to convert model_name to string: {str(e)}")
                model_name = ''
        
        if model_name and model_name.lower().startswith('gemini'):
            logger.info(f"Model name starts with 'gemini': {model_name}")
        else:
            logger.info(f"Model name does not start with 'gemini': {model_name}")
        
        # Try to import the TableStructureRecognizer
        try:
            sys.path.append(get_project_base_directory())
            from src.document_processing.core.vision.table_structure_recognizer import TableStructureRecognizer
            
            # Initialize the TableStructureRecognizer
            logger.info("Initializing TableStructureRecognizer...")
            table_structure_recognizer = TableStructureRecognizer(config_path)
            logger.info("TableStructureRecognizer initialized successfully")
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize TableStructureRecognizer: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        logger.error(f"Error debugging TableStructureRecognizer: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_table_structure_recognizer()
    if success:
        print("TableStructureRecognizer debug completed successfully!")
    else:
        print("Failed to debug TableStructureRecognizer.") 