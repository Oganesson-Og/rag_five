import os
import logging
import yaml

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TableStructureRecognizerTest")

def get_project_base_directory():
    """Get the base directory of the project."""
    return os.path.dirname(os.path.abspath(__file__))

def fix_table_structure_recognizer():
    """Fix the TableStructureRecognizer class to handle non-Gemini model names better."""
    try:
        # Get the path to the TableStructureRecognizer file
        tsr_path = os.path.join(
            get_project_base_directory(),
            "src/document_processing/core/vision/table_structure_recognizer.py"
        )
        
        # Check if the file exists
        if not os.path.exists(tsr_path):
            logger.error(f"TableStructureRecognizer file not found at {tsr_path}")
            return
        
        # Read the file
        with open(tsr_path, 'r') as f:
            content = f.read()
        
        # Check if the file already contains our fix
        if "isinstance(model_name, str) and model_name.lower().startswith('gemini')" in content:
            logger.info("TableStructureRecognizer already contains the fix")
            return
        
        # Replace the problematic code
        content = content.replace(
            "if cross_modal_config.get('model_name', '').startswith('gemini'):",
            "model_name = cross_modal_config.get('model_name', '')\n                if isinstance(model_name, str) and model_name.lower().startswith('gemini'):"
        )
        
        # Replace the second occurrence
        content = content.replace(
            "if cross_modal_config.get('model_name', '').startswith('gemini'):",
            "model_name = cross_modal_config.get('model_name', '')\n        if isinstance(model_name, str) and model_name.lower().startswith('gemini'):"
        )
        
        # Update the __call__ method
        content = content.replace(
            "if hasattr(self, 'use_gemini') and self.use_gemini:",
            "if hasattr(self, 'use_gemini') and self.use_gemini and hasattr(self, 'gemini_model'):"
        )
        
        # Add try-except block to the __call__ method
        call_method_start = content.find("def __call__(self, images, thr=0.2):")
        if call_method_start != -1:
            # Find the start of the method body
            method_body_start = content.find(":", call_method_start) + 1
            # Find the next line after the if statement
            next_line_after_if = content.find("\n", content.find("return self._process_with_gemini(images)", method_body_start)) + 1
            
            # Insert try-except block
            content = (
                content[:next_line_after_if] + 
                "\n        # Otherwise use the default processing\n        try:" + 
                content[next_line_after_if:content.find("return res", next_line_after_if)] + 
                "return res\n        except Exception as e:\n            logging.error(f\"Error processing images with default model: {str(e)}\")\n            return [[] for _ in range(len(images) if isinstance(images, list) else 1)]" + 
                content[content.find("return res", next_line_after_if) + len("return res"):]
            )
        
        # Write the updated content back to the file
        with open(tsr_path, 'w') as f:
            f.write(content)
        
        logger.info("Successfully updated TableStructureRecognizer")
        
    except Exception as e:
        logger.error(f"Error fixing TableStructureRecognizer: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_table_structure_recognizer() 