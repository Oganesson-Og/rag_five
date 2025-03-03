import os
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TableStructureRecognizerTest")

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Import the TableStructureRecognizer
    from src.document_processing.core.vision.table_structure_recognizer import TableStructureRecognizer
    
    # Get the config path
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/rag_config.yaml")
    
    # Initialize the TableStructureRecognizer
    logger.info("Initializing TableStructureRecognizer...")
    table_structure_recognizer = TableStructureRecognizer(config_path)
    
    # Check if initialization was successful
    logger.info("TableStructureRecognizer initialized successfully")
    
    # Check if using Gemini
    if hasattr(table_structure_recognizer, 'use_gemini'):
        logger.info(f"Using Gemini: {table_structure_recognizer.use_gemini}")
    else:
        logger.warning("use_gemini attribute not found")
    
    # Check if the model is available
    if hasattr(table_structure_recognizer, 'gemini_model'):
        logger.info("Gemini model is available")
    else:
        logger.info("Gemini model is not available")
    
except Exception as e:
    logger.error(f"Error initializing TableStructureRecognizer: {str(e)}")
    import traceback
    traceback.print_exc() 