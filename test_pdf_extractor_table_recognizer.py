import os
import logging
import yaml

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PDFExtractorTest")

def get_project_base_directory():
    """Get the base directory of the project."""
    return os.path.dirname(os.path.abspath(__file__))

def fix_pdf_extractor():
    """Fix the PDFExtractor class to handle TableStructureRecognizer initialization errors better."""
    try:
        # Get the path to the PDFExtractor file
        pdf_path = os.path.join(
            get_project_base_directory(),
            "src/document_processing/extractors/pdf.py"
        )
        
        # Check if the file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDFExtractor file not found at {pdf_path}")
            return
        
        # Read the file
        with open(pdf_path, 'r') as f:
            content = f.read()
        
        # Find the TableStructureRecognizer initialization code
        tsr_init_start = content.find("# Initialize TableStructureRecognizer if available")
        if tsr_init_start == -1:
            logger.error("TableStructureRecognizer initialization code not found")
            return
        
        # Find the end of the try-except block
        try_except_end = content.find("# Process each page", tsr_init_start)
        if try_except_end == -1:
            logger.error("End of try-except block not found")
            return
        
        # Extract the current initialization code
        current_init_code = content[tsr_init_start:try_except_end]
        
        # Create the updated initialization code
        updated_init_code = """# Initialize TableStructureRecognizer if available
        table_structure_recognizer = None
        try:
            # Pass the config path to TableStructureRecognizer
            config_path = os.path.join(get_project_base_directory(), "config/rag_config.yaml")
            
            # Check if config file exists before initializing
            if os.path.exists(config_path):
                try:
                    from ..core.vision.table_structure_recognizer import TableStructureRecognizer
                    table_structure_recognizer = TableStructureRecognizer(config_path)
                    self.logger.info("TableStructureRecognizer initialized successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize TableStructureRecognizer: {str(e)}")
            else:
                self.logger.warning(f"Config file not found at {config_path}, skipping TableStructureRecognizer initialization")
        except Exception as e:
            self.logger.warning(f"Failed to initialize TableStructureRecognizer: {str(e)}")
            
        """
        
        # Replace the current initialization code with the updated code
        content = content.replace(current_init_code, updated_init_code)
        
        # Write the updated content back to the file
        with open(pdf_path, 'w') as f:
            f.write(content)
        
        logger.info("Successfully updated PDFExtractor")
        
    except Exception as e:
        logger.error(f"Error fixing PDFExtractor: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_pdf_extractor() 