#!/usr/bin/env python
"""
Fix script for TableStructureRecognizer to handle non-string model_name values.
"""

import os
import logging
import re
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TableStructureRecognizerFix")

def get_project_base_directory():
    """Get the base directory of the project."""
    return os.path.dirname(os.path.abspath(__file__))

def fix_table_structure_recognizer():
    """Fix the TableStructureRecognizer class to handle non-string model names."""
    try:
        # Get the path to the TableStructureRecognizer file
        tsr_path = os.path.join(
            get_project_base_directory(),
            "src/document_processing/core/vision/table_structure_recognizer.py"
        )
        
        # Check if the file exists
        if not os.path.exists(tsr_path):
            logger.error(f"TableStructureRecognizer file not found at {tsr_path}")
            return False
        
        # Read the file
        with open(tsr_path, 'r') as f:
            content = f.read()
        
        # Let's completely rewrite the file to ensure it's correct
        # First, find the class definition
        class_start = content.find("class TableStructureRecognizer(Recognizer):")
        if class_start == -1:
            logger.error("Could not find TableStructureRecognizer class definition")
            return False
        
        # Find the imports and everything before the class
        header = content[:class_start]
        
        # Find the end of the file
        file_end = len(content)
        
        # Create a completely new implementation of the class
        new_class = '''class TableStructureRecognizer(Recognizer):
    labels = [
        "table",
        "table column",
        "table row",
        "table column header",
        "table projected row header",
        "table spanning cell",
    ]

    def __init__(self, config_path=None):
        """
        Initialize the TableStructureRecognizer with optional configuration.
        
        Args:
            config_path: Path to configuration file (defaults to config/rag_config.yaml)
        """
        # Load configuration
        if config_path is None:
            config_path = os.path.join(get_project_base_directory(), "config/rag_config.yaml")
        
        config = {}
        picture_config = {}
        educational_config = {}
        cross_modal_config = {}
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # Get model configuration from config file
                pdf_config = config.get('pdf', {})
                picture_config = pdf_config.get('picture_annotation', {})
                educational_config = pdf_config.get('educational', {})
                cross_modal_config = educational_config.get('cross_modal', {})
                
                # Check if we should use a local vision model
                if picture_config.get('enabled', True) and picture_config.get('model_type') == 'local':
                    model_name = picture_config.get('model_name')
                    if model_name:
                        logging.info(f"Using local vision model: {model_name}")
                        # Will use this model_name later
                
                # Check if we should use Gemini model
                model_name = cross_modal_config.get('model_name', '')
                
                # Handle case where model_name might be a list or other non-string type
                if not isinstance(model_name, str):
                    logging.warning(f"model_name is not a string but a {type(model_name).__name__}. Converting to string representation.")
                    try:
                        model_name = str(model_name)
                    except Exception as e:
                        logging.warning(f"Failed to convert model_name to string: {str(e)}")
                        model_name = ''
                
                if model_name and model_name.lower().startswith('gemini'):
                    logging.info(f"Using Gemini model: {model_name}")
                    # Will use Gemini configuration later
                else:
                    logging.info(f"Using non-Gemini model: {model_name}")
                    
            except Exception as e:
                logging.warning(f"Failed to load config from {config_path}: {str(e)}")
                logging.info("Falling back to default model")
        
        # Check if we should use Gemini model
        use_gemini = False
        model_name = cross_modal_config.get('model_name', '')
        
        # Handle case where model_name might be a list or other non-string type
        if not isinstance(model_name, str):
            logging.warning(f"model_name is not a string but a {type(model_name).__name__}. Converting to string representation.")
            try:
                model_name = str(model_name)
            except Exception as e:
                logging.warning(f"Failed to convert model_name to string: {str(e)}")
                model_name = ''
        
        if model_name and model_name.lower().startswith('gemini'):
            try:
                # Try to import and initialize Gemini
                import google.generativeai as genai
                from PIL import Image
                
                # Get API key from config or environment
                api_key = config.get('model', {}).get('google_api_key', os.environ.get('GOOGLE_API_KEY'))
                if not api_key:
                    logging.warning("No Google API key found, falling back to default model")
                else:
                    # Configure the Gemini API
                    genai.configure(api_key=api_key)
                    
                    # Initialize the Gemini model
                    self.gemini_model = genai.GenerativeModel(model_name)
                    
                    # Set generation config
                    self.gemini_generation_config = {
                        'temperature': cross_modal_config.get('temperature', 0.7),
                        'top_p': cross_modal_config.get('top_p', 0.9),
                        'max_output_tokens': cross_modal_config.get('max_length', 2048),
                    }
                    
                    # Set the prompt for table structure recognition
                    self.gemini_prompt = """
                    Analyze this table image and extract its structure. 
                    Identify the following elements:
                    1. Table boundaries
                    2. Column headers
                    3. Row headers
                    4. Data cells
                    5. Spanning cells
                    
                    Return the result in JSON format with the following structure:
                    {
                        "headers": ["header1", "header2", ...],
                        "rows": [
                            {"header": "row_header", "cells": ["cell1", "cell2", ...]},
                            ...
                        ],
                        "spanning_cells": [
                            {"text": "cell_text", "row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1}
                        ]
                    }
                    """
                    
                    use_gemini = True
                    logging.info(f"Successfully initialized Gemini model: {model_name}")
            except Exception as e:
                logging.warning(f"Failed to initialize Gemini: {str(e)}")
                logging.info("Falling back to default model")
        
        if use_gemini:
            # If using Gemini, we still need to initialize the base class
            # but we'll override the __call__ method
            super().__init__(self.labels, "tsr", os.path.join(
                    get_project_base_directory(),
                    "rag/res/deepdoc"))
            # Set a flag to indicate we're using Gemini
            self.use_gemini = True
            return
        
        try:
            # Try to use the model specified in config
            if picture_config.get('enabled', True) and picture_config.get('model_type') == 'local':
                model_name = picture_config.get('model_name')
                if model_name:
                    # Use the specified local model
                    device = picture_config.get('device', 'auto')
                    if device == 'auto':
                        device = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch, 'mps') and torch.backends.mps.is_available() else 'cpu'
                    
                    super().__init__(
                        self.labels, 
                        "tsr", 
                        model_dir=os.path.join(get_project_base_directory(), "models", model_name),
                        device=device
                    )
                    self.use_gemini = False
                    return
            
            # If no local model specified or configuration failed, try the default path
            super().__init__(self.labels, "tsr", os.path.join(
                    get_project_base_directory(),
                    "rag/res/deepdoc"))
            self.use_gemini = False
        except Exception:
            # Fall back to downloading from HuggingFace
            super().__init__(self.labels, "tsr", snapshot_download(repo_id="InfiniFlow/deepdoc",
                                              local_dir=os.path.join(get_project_base_directory(), "rag/res/deepdoc"),
                                              local_dir_use_symlinks=False))
            self.use_gemini = False'''
        
        # Find the __call__ method
        call_method_start = content.find("def __call__(self, images, thr=0.2):")
        if call_method_start == -1:
            logger.error("Could not find __call__ method")
            return False
        
        # Extract everything after the class definition but before the __call__ method
        middle = content[class_start:call_method_start]
        
        # Extract everything after the __call__ method
        rest = content[call_method_start:]
        
        # Combine everything
        new_content = header + new_class + rest
        
        # Write the updated content back to the file
        with open(tsr_path, 'w') as f:
            f.write(new_content)
        
        logger.info("Successfully updated TableStructureRecognizer")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing TableStructureRecognizer: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_table_structure_recognizer()
    if success:
        print("TableStructureRecognizer fixed successfully!")
    else:
        print("Failed to fix TableStructureRecognizer.") 