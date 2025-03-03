import camelot
import logging
import os
import sys
import importlib.util

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CamelotGhostscriptTest")

# Check if Ghostscript is available through Camelot's backend
try:
    # Import the Ghostscript backend directly
    spec = importlib.util.spec_from_file_location(
        "ghostscript_backend", 
        "rag_env/lib/python3.11/site-packages/camelot/backends/ghostscript_backend.py"
    )
    ghostscript_backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ghostscript_backend)
    
    # Create a Ghostscript backend instance
    gs_backend = ghostscript_backend.GhostscriptBackend()
    
    # Check if Ghostscript is installed
    is_installed = gs_backend.installed()
    logger.info(f"Ghostscript is installed according to Camelot: {is_installed}")
    
    # Also check using the functions directly
    if sys.platform in ["linux", "darwin"]:
        posix_result = ghostscript_backend.installed_posix()
        logger.info(f"Ghostscript POSIX check result: {posix_result}")
    elif sys.platform == "win32":
        windows_result = ghostscript_backend.installed_windows()
        logger.info(f"Ghostscript Windows check result: {windows_result}")
    
    # Print environment variables that might affect Ghostscript detection
    logger.info(f"PATH: {os.environ.get('PATH', '')}")
    logger.info(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}")
    logger.info(f"DYLD_LIBRARY_PATH: {os.environ.get('DYLD_LIBRARY_PATH', '')}")
    
    # Print Python path
    logger.info(f"Python path: {sys.path}")
    
except Exception as e:
    logger.error(f"Error checking Ghostscript with Camelot: {str(e)}")
    import traceback
    traceback.print_exc()
    
print("Test completed. Check the logs above for results.") 