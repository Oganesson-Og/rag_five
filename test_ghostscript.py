import subprocess
import os
import logging

def test_ghostscript_detection():
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("GhostscriptTest")
    
    # Print environment PATH
    logger.debug(f"PATH environment: {os.environ.get('PATH', '')}")
    
    try:
        # Try to find the Ghostscript executable
        gs_command = "gs"
        
        # Check if we're on Windows
        if os.name == 'nt':
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
        
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        # Log the specific error
        logger.debug(f"Ghostscript check failed with error: {str(e)}")
        # If the command failed or the executable wasn't found
        return False

# Run the test
test_result = test_ghostscript_detection()
print(f"Ghostscript detection test result: {test_result}")

# Also try with full path to gs
try:
    result = subprocess.run(
        ["/opt/homebrew/bin/gs", "--version"], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        timeout=5
    )
    print(f"Direct path test: returncode={result.returncode}, stdout={result.stdout.decode().strip()}")
except Exception as e:
    print(f"Direct path test failed: {str(e)}") 