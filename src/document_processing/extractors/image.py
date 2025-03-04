from typing import Optional, Dict
from PIL import Image
import io
import logging
from ...database.models import DocumentDB
from ...rag.models import Document as PydanticDocument
from .base import BaseExtractor, ExtractorResult

logger = logging.getLogger(__name__)

class ImageExtractor(BaseExtractor):
    """Handles image document extraction with OCR capabilities."""
    
    async def extract(self, document: 'DocumentDB') -> 'PydanticDocument':
        """Extract content from image document.
        
        Args:
            document: Document instance containing image data
            
        Returns:
            Updated document with extracted text and metadata
        """
        try:
            # Get content bytes from document
            content = document.content
            if not content:
                raise ValueError("Document has no content")

            # Convert bytes to image
            image = Image.open(io.BytesIO(content))
            
            # Use OCR to extract text - dynamically import to avoid circular imports
            from ..core.vision.ocr import OCR
            
            # Try to load config, but handle case where it might not exist
            try:
                import yaml
                import os
                from src.utils.file_utils import get_project_base_directory
                
                config_path = os.path.join(get_project_base_directory(), "config", "rag_config.yaml")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    ocr = OCR(config=config)
                else:
                    logger.warning(f"Config file not found at {config_path}, using default OCR settings")
                    ocr = OCR(config={})
            except Exception as e:
                logger.warning(f"Failed to initialize OCR with config: {str(e)}, using default settings")
                ocr = OCR(config={})
                
            extracted_text = ocr.process_image(image)
            
            # Update document metadata
            document.doc_info.update({
                'extracted_text': extracted_text,
                'image_metadata': {
                    'width': image.width,
                    'height': image.height,
                    'format': image.format,
                    'mode': image.mode
                },
                'extraction_method': 'ocr'
            })
            
            return document
            
        except Exception as e:
            logger.error(f"Image extraction failed: {str(e)}")
            raise