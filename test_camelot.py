import camelot
import logging
import os
import sys

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CamelotTest")

# Check if Ghostscript is available through Camelot
try:
    # Check if Camelot can find Ghostscript
    logger.info("Checking if Camelot can find Ghostscript...")
    
    # Print Camelot version
    logger.info(f"Camelot version: {camelot.__version__}")
    
    # Check if the ghostscript dependency is available
    if hasattr(camelot, 'utils') and hasattr(camelot.utils, 'get_ghostscript_path'):
        gs_path = camelot.utils.get_ghostscript_path()
        logger.info(f"Ghostscript path found by Camelot: {gs_path}")
    else:
        logger.warning("Camelot does not have the get_ghostscript_path method")
        
    # Try to import the specific module that checks for Ghostscript
    try:
        from camelot.utils import get_ghostscript_path
        gs_path = get_ghostscript_path()
        logger.info(f"Ghostscript path from direct import: {gs_path}")
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import get_ghostscript_path: {str(e)}")
    
    # Print environment variables that might affect Ghostscript detection
    logger.info(f"PATH: {os.environ.get('PATH', '')}")
    logger.info(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}")
    logger.info(f"DYLD_LIBRARY_PATH: {os.environ.get('DYLD_LIBRARY_PATH', '')}")
    
    # Print Python path
    logger.info(f"Python path: {sys.path}")
    
except Exception as e:
    logger.error(f"Error checking Ghostscript with Camelot: {str(e)}")
    
print("Test completed. Check the logs above for results.") 