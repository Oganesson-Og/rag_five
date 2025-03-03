import os
import yaml
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TableRecognizerTest")

def get_project_base_directory():
    """Get the base directory of the project."""
    return os.path.dirname(os.path.abspath(__file__))

def check_config():
    """Check the configuration file for issues."""
    config_path = os.path.join(get_project_base_directory(), "config/rag_config.yaml")
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}")
        return
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check PDF configuration
        pdf_config = config.get('pdf', {})
        if not isinstance(pdf_config, dict):
            logger.error(f"PDF config is not a dictionary: {type(pdf_config)}")
            return
        
        # Check educational configuration
        educational_config = pdf_config.get('educational', {})
        if not isinstance(educational_config, dict):
            logger.error(f"Educational config is not a dictionary: {type(educational_config)}")
            return
        
        # Check cross_modal configuration
        cross_modal_config = educational_config.get('cross_modal', {})
        if not isinstance(cross_modal_config, dict):
            logger.error(f"Cross-modal config is not a dictionary: {type(cross_modal_config)}")
            return
        
        # Check model_name
        model_name = cross_modal_config.get('model_name')
        logger.info(f"Model name: {model_name} (type: {type(model_name)})")
        
        if isinstance(model_name, list):
            logger.error(f"Model name is a list, which will cause errors: {model_name}")
            
            # Fix the config
            logger.info("Fixing the config file...")
            if model_name and len(model_name) > 0:
                # Use the first item in the list
                cross_modal_config['model_name'] = model_name[0]
            else:
                # Set a default value
                cross_modal_config['model_name'] = "gemini-pro-vision"
            
            # Save the fixed config
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Fixed model name: {cross_modal_config['model_name']}")
        
        # Check other configurations
        picture_config = pdf_config.get('picture_annotation', {})
        logger.info(f"Picture annotation config: {picture_config}")
        
        # Check API keys
        api_key = config.get('model', {}).get('google_api_key')
        if api_key:
            logger.info("Google API key is set")
        else:
            logger.warning("Google API key is not set")
        
        gemini_api_key = config.get('model', {}).get('gemini_api_key')
        if gemini_api_key:
            logger.info("Gemini API key is set")
        else:
            logger.warning("Gemini API key is not set")
        
    except Exception as e:
        logger.error(f"Error checking config: {str(e)}")

if __name__ == "__main__":
    check_config() 