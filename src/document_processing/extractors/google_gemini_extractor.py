import logging
from pathlib import Path
from typing import Any, Dict, Optional
from google import genai
import yaml

class GeminiProExtractor:
    """
    Extractor for scanned images using Google Gemini Pro API.
    This extractor uses the google-genai SDK to extract structured text/content
    from scanned images.
    """
    def __init__(self, api_key: str, config_path: Optional[str] = None, model_id: str = "gemini-2.0-pro-exp-02-05"):
        self.api_key = api_key
        self.model_id = model_id
        
        # Load configuration if provided
        self.config = {}
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    # Get relevant config sections
                    self.config = config.get('model', {})
                    if 'google' in config:
                        self.config.update(config['google'])
            except Exception as e:
                logging.warning(f"Failed to load config from {config_path}: {str(e)}")
        
        # Initialize the Gemini client
        self.client = genai.Client(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text and structured information from a scanned image file.

        Args:
            file_path: Path to the scanned image file.

        Returns:
            A dictionary containing extracted text and any structured data.
        """
        try:
            display_name = file_path.stem
            # Upload the file via the Gemini client's file API
            uploaded_file = self.client.files.upload(
                file=str(file_path),
                config={'display_name': display_name}
            )
            
            # Get configuration values or use defaults
            temperature = self.config.get('temperature', 0.7)
            max_tokens = self.config.get('max_tokens', 1000)
            
            # Define a prompt for text extraction that guides Gemini to capture
            # both printed and handwritten text in a structured JSON format.
            prompt = (
                "Extract all textual content from the provided scanned pdf. "
                "Pay attention to graphs, tables, and formatting. When you see a graph or image, "
                "describe it in the text field. When you see a table, describe it in the text field. "
                "Return a valid JSON object that contains a 'text' field with the complete extracted string."
            )
            
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt, uploaded_file],
                config={
                    'response_mime_type': 'application/json',
                    'temperature': temperature,
                    'max_output_tokens': max_tokens
                }
            )
            result = response.parsed
            self.logger.debug(f"Gemini extraction result for {file_path}: {result}")
            return result if isinstance(result, dict) else {"text": str(result)}
        except Exception as e:
            self.logger.error(f"Gemini extraction failed for file {file_path}: {str(e)}", exc_info=True)
            raise e 