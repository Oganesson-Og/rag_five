import logging
import sys
from ctypes.util import find_library
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GhostscriptTest")

def check_ghostscript_installed() -> bool:
    """
    Check if Ghostscript is installed on the system.
    
    This method checks for Ghostscript in the same way that Camelot does.
    
    Returns:
        True if Ghostscript is installed, False otherwise
    """
    try:
        # First try the Camelot way of checking for Ghostscript
        logger.debug("Checking for Ghostscript using ctypes.util.find_library")
        
        # Check based on platform (same as Camelot does)
        if sys.platform in ["linux", "darwin"]:
            # For Linux and macOS
            library = find_library("gs")
            result = library is not None
            logger.debug(f"Ghostscript library check result: {result}, library path: {library}")
            if result:
                return True
        elif sys.platform == "win32":
            # For Windows
            import ctypes
            library = find_library(
                "".join(("gsdll", str(ctypes.sizeof(ctypes.c_voidp) * 8), ".dll"))
            )
            result = library is not None
            logger.debug(f"Ghostscript library check result: {result}, library path: {library}")
            if result:
                return True
        
        # If the library check failed, fall back to checking for the executable
        logger.debug("Library check failed, falling back to executable check")
        
        # Try to find the Ghostscript executable
        gs_command = "gs"
        
        # Check if we're on Windows
        if sys.platform == "win32":
            gs_command = "gswin64c"  # 64-bit Ghostscript on Windows
        
        logger.debug(f"Checking for Ghostscript using command: {gs_command}")
            
        # Try to run Ghostscript with version flag
        result = subprocess.run(
            [gs_command, "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            timeout=5
        )
        
        # Log the result for debugging
        logger.debug(f"Ghostscript check result: returncode={result.returncode}, stdout={result.stdout.decode().strip()}, stderr={result.stderr.decode().strip()}")
        
        # If the command succeeded, Ghostscript is installed
        return result.returncode == 0
        
    except (subprocess.SubprocessError, FileNotFoundError, ImportError) as e:
        # Log the specific error
        logger.debug(f"Ghostscript check failed with error: {str(e)}")
        # If the command failed or the executable wasn't found
        return False

# Run the test
test_result = check_ghostscript_installed()
print(f"Updated Ghostscript detection test result: {test_result}")

# Also try to import the ghostscript Python package
try:
    import ghostscript
    print(f"Ghostscript Python package is installed: {ghostscript.__file__}")
except ImportError as e:
    print(f"Ghostscript Python package is not installed: {str(e)}")

# Try to run a simple Ghostscript command using the Python package
try:
    import ghostscript
    args = ["gs", "-v"]
    ghostscript.Ghostscript(*args)
    print("Successfully ran Ghostscript command using Python package")
except Exception as e:
    print(f"Failed to run Ghostscript command using Python package: {str(e)}") 