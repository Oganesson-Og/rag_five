"""
DoclingExtractor: A comprehensive document processing toolkit

This module provides a robust interface to Docling's document processing capabilities,
specializing in extracting structured content from various document formats with a focus on PDFs.

Key Features:
    - Advanced OCR capabilities for both scanned and native PDFs
    - Structured text extraction preserving document layout
    - Table detection and structure preservation
    - Support for mathematical formulas and code blocks
    - Multiple export formats (markdown, HTML, JSON, plain text)
    - Batch processing capabilities
    - Offline processing support
    - Progress tracking and detailed error handling

Example usage:
    ```python
    from docling_test import DoclingExtractor
    
    # Initialize extractor
    extractor = DoclingExtractor()
    
    # Process single file
    result = extractor.process_file("path/to/document.pdf")
    print(result.markdown)
    
    # Batch process directory
    results = extractor.process_directory("path/to/docs", output_dir="processed")
    ```
"""

import os
import logging
from pathlib import Path
from typing import Union, Dict, List, Optional, Any
from dataclasses import dataclass
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    granite_picture_description
)
from docling.document_converter import PdfFormatOption
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    """
    Container for document extraction results in various formats.
    
    Attributes:
        filename (str): Original filename
        markdown (str): Markdown formatted content
        text (str): Plain text content
        html (str): HTML formatted content
        json (str): JSON structured content
        success (bool): Extraction success status
        error (Optional[str]): Error message if extraction failed
    """
    filename: str
    markdown: str = ""
    text: str = ""
    html: str = ""
    json: str = ""
    success: bool = True
    error: Optional[str] = None

class DoclingExtractor:
    """
    A comprehensive document processing toolkit using Docling.
    
    This class provides methods for extracting structured content from documents,
    with special emphasis on PDF processing capabilities including OCR.
    
    Attributes:
        converter (DocumentConverter): Configured Docling document converter
        offline_mode (bool): Whether to use offline processing mode
        model_type (str): Type of model to use ('local' or 'remote')
        model_name (str): Name of the model to use
        image_scale (float): Scale factor for image processing
        picture_prompt (str): Prompt for image description
        api_config (Optional[Dict[str, Any]]): Configuration for remote API (required if model_type is 'remote')
        num_threads (int): Number of threads for acceleration
        device (AcceleratorDevice): Selected accelerator device
    """
    
    def __init__(
        self,
        offline_mode: bool = False,
        artifacts_path: Optional[str] = None,
        model_type: str = "local",
        model_name: str = "ibm-granite/granite-vision-3.1-2b-preview",
        image_scale: float = 2.0,
        picture_prompt: str = "Describe the image in three sentences. Be concise and accurate.",
        api_config: Optional[Dict[str, Any]] = None,
        num_threads: int = 8,
        enable_remote_services: bool = True
    ):
        """
        Initialize the DoclingExtractor.
        
        Args:
            offline_mode (bool): If True, uses offline processing mode
            artifacts_path (Optional[str]): Path to model artifacts for offline usage
            model_type (str): Type of model to use ('local' or 'remote')
            model_name (str): Name of the model to use
            image_scale (float): Scale factor for image processing
            picture_prompt (str): Prompt for image description
            api_config (Optional[Dict[str, Any]]): Configuration for remote API (required if model_type is 'remote')
            num_threads (int): Number of threads for acceleration
            enable_remote_services (bool): Whether to enable remote services for Docling
        """
        self.offline_mode = offline_mode
        self.model_type = model_type
        self.model_name = model_name
        self.image_scale = image_scale
        self.picture_prompt = picture_prompt
        self.api_config = api_config
        self.num_threads = num_threads
        self.enable_remote_services = enable_remote_services
        self.device = self._determine_device()
        self.converter = self._initialize_converter(artifacts_path)
        
    def _determine_device(self) -> AcceleratorDevice:
        """Determine the best available accelerator device."""
        try:
            import torch
            if torch.cuda.is_available():
                return AcceleratorDevice.CUDA
            elif torch.backends.mps.is_available():
                return AcceleratorDevice.MPS
        except ImportError:
            pass
        return AcceleratorDevice.CPU

    def _initialize_converter(self, artifacts_path: Optional[str]) -> DocumentConverter:
        """
        Initialize and configure the Docling document converter.
        
        Args:
            artifacts_path (Optional[str]): Path to model artifacts for offline usage
            
        Returns:
            DocumentConverter: Configured document converter
        """
        # Configure accelerator
        accelerator_options = AcceleratorOptions(
            num_threads=self.num_threads,
            device=self.device
        )

        pipeline_options = PdfPipelineOptions(
            enable_remote_services=self.enable_remote_services,
            accelerator_options=accelerator_options
        )

        # Configure picture description
        pipeline_options.do_picture_description = True
        pipeline_options.images_scale = self.image_scale
        pipeline_options.generate_picture_images = True

        if self.model_type == 'local':
            pipeline_options.picture_description_options = self._configure_local_model()
        else:
            pipeline_options.picture_description_options = self._configure_remote_model()

        if self.offline_mode and artifacts_path:
            pipeline_options.artifacts_path = str(artifacts_path)
            pipeline_options.do_table_structure = True
            pipeline_options.do_code_formula = True
            pipeline_options.do_image_classification = True

        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
        
        return DocumentConverter(format_options=format_options)
    
    def _configure_local_model(self) -> PictureDescriptionApiOptions:
        """Configure local model options."""
        return PictureDescriptionApiOptions(
            url="http://localhost:8000/v1/chat/completions",
            params=dict(
                model=self.model_name,
                seed=42,
                max_completion_tokens=200,
            ),
            prompt=self.picture_prompt,
            timeout=90,
        )

    def _configure_remote_model(self) -> PictureDescriptionApiOptions:
        """Configure remote model options."""
        if not self.api_config:
            raise ValueError("API configuration required for remote model")

        if 'watsonx' in self.api_config:
            return self._configure_watsonx_model()
        else:
            return PictureDescriptionApiOptions(
                url=self.api_config['url'],
                params=self.api_config['params'],
                headers=self.api_config.get('headers', {}),
                prompt=self.picture_prompt,
                timeout=self.api_config.get('timeout', 60),
            )

    def _configure_watsonx_model(self) -> PictureDescriptionApiOptions:
        """Configure WatsonX.ai model options."""
        load_dotenv()
        api_key = os.environ.get("WX_API_KEY")
        project_id = os.environ.get("WX_PROJECT_ID")

        if not api_key or not project_id:
            raise ValueError("WX_API_KEY and WX_PROJECT_ID environment variables required")

        access_token = self._get_watsonx_token(api_key)

        return PictureDescriptionApiOptions(
            url="https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29",
            params=dict(
                model_id=self.model_name,
                project_id=project_id,
                parameters=dict(
                    max_new_tokens=400,
                ),
            ),
            headers={"Authorization": f"Bearer {access_token}"},
            prompt=self.picture_prompt,
            timeout=60,
        )

    @staticmethod
    def _get_watsonx_token(api_key: str) -> str:
        """Get WatsonX.ai access token."""
        response = requests.post(
            url="https://iam.cloud.ibm.com/identity/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}",
        )
        response.raise_for_status()
        return response.json()["access_token"]
    
    def process_file(self, file_path_or_content: Union[str, Path, bytes]) -> ExtractionResult:
        """
        Process a single document file or binary content.
        
        Args:
            file_path_or_content (Union[str, Path, bytes]): Path to the document file or binary content
            
        Returns:
            ExtractionResult: Extraction results in various formats
            
        Example:
            ```python
            extractor = DoclingExtractor()
            
            # Process from file path
            result = extractor.process_file("document.pdf")
            
            # Or process from binary content
            with open("document.pdf", "rb") as f:
                content = f.read()
            result = extractor.process_file(content)
            
            if result.success:
                print(result.markdown)
            else:
                print(f"Error: {result.error}")
            ```
        """
        # Handle binary content
        if isinstance(file_path_or_content, bytes):
            # Create a temporary filename for logging purposes
            filename = "document_from_bytes.pdf"
            result = ExtractionResult(filename=filename)
            
            try:
                logger.info(f"Processing binary content ({len(file_path_or_content)} bytes)")
                # Use convert_from_bytes method if available, otherwise use a temporary file
                try:
                    # Try to use convert_from_bytes if available
                    conv_result = self.converter.convert_from_bytes(file_path_or_content, "pdf")
                except AttributeError:
                    # If not available, use a temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                        temp_file.write(file_path_or_content)
                        temp_path = temp_file.name
                    
                    try:
                        conv_result = self.converter.convert(temp_path)
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                
                doc = conv_result.document
                
                # Extract content in different formats
                result.markdown = doc.export_to_markdown()
                result.text = doc.export_to_text()
                result.html = doc.export_to_html()
                result.json = doc.export_to_json()
                
            except Exception as e:
                result.success = False
                result.error = str(e)
                logger.error(f"Error processing binary content: {e}")
                
            return result
        
        # Handle file path
        file_path = Path(file_path_or_content)
        result = ExtractionResult(filename=file_path.name)
        
        try:
            logger.info(f"Processing file: {file_path}")
            conv_result = self.converter.convert(str(file_path))
            doc = conv_result.document
            
            # Extract content in different formats
            result.markdown = doc.export_to_markdown()
            result.text = doc.export_to_text()
            result.html = doc.export_to_html()
            result.json = doc.export_to_json()
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Error processing {file_path}: {e}")
            
        return result
    
    def process_directory(self, 
                         input_dir: Union[str, Path],
                         output_dir: Optional[Union[str, Path]] = None,
                         file_types: List[str] = ['.pdf', '.docx', '.png', '.jpg'],
                         export_format: str = 'markdown') -> List[ExtractionResult]:
        """
        Process all supported documents in a directory.
        
        Args:
            input_dir (Union[str, Path]): Input directory path
            output_dir (Optional[Union[str, Path]]): Output directory path
            file_types (List[str]): List of file extensions to process
            export_format (str): Format to save results ('markdown', 'text', 'html', 'json')
            
        Returns:
            List[ExtractionResult]: List of extraction results
            
        Example:
            ```python
            extractor = DoclingExtractor()
            results = extractor.process_directory(
                "input_docs",
                output_dir="processed",
                file_types=['.pdf', '.docx'],
                export_format='markdown'
            )
            ```
        """
        input_dir = Path(input_dir)
        results = []
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        for file_type in file_types:
            for file_path in input_dir.glob(f"*{file_type}"):
                result = self.process_file(file_path)
                results.append(result)
                
                if output_dir and result.success:
                    output_path = output_dir / f"{file_path.stem}.{export_format}"
                    content = getattr(result, export_format)
                    output_path.write_text(content, encoding='utf-8')
                    logger.info(f"Saved output to: {output_path}")
        
        return results
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """
        Get list of supported document formats.
        
        Returns:
            List[str]: List of supported file extensions
        """
        return ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.png', '.jpg', '.jpeg']
    
    def extract_tables(self, file_path_or_content: Union[str, Path, bytes]) -> List[Dict[str, Any]]:
        """
        Extract tables from a document.
        
        Args:
            file_path_or_content (Union[str, Path, bytes]): Path to the document file or binary content
            
        Returns:
            List[Dict[str, Any]]: List of extracted tables with their structure
        """
        result = self.process_file(file_path_or_content)
        if not result.success:
            return []
        
        # Parse JSON content to extract tables
        # Note: Implementation depends on Docling's JSON structure
        # This is a placeholder for the actual implementation
        return []

def main():
    """CLI interface for the DoclingExtractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract structured content from documents using Docling")
    parser.add_argument("input_path", help="Path to input file or directory")
    parser.add_argument("--output-dir", help="Output directory for processed files")
    parser.add_argument("--format", choices=['markdown', 'text', 'html', 'json'],
                       default='markdown', help="Output format")
    parser.add_argument("--offline", action="store_true", help="Use offline mode")
    parser.add_argument("--artifacts-path", help="Path to model artifacts for offline usage")
    
    args = parser.parse_args()
    
    extractor = DoclingExtractor(offline_mode=args.offline, artifacts_path=args.artifacts_path)
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        result = extractor.process_file(input_path)
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{input_path.stem}.{args.format}"
            content = getattr(result, args.format)
            output_path.write_text(content, encoding='utf-8')
            print(f"Output saved to: {output_path}")
        else:
            print(getattr(result, args.format))
    
    elif input_path.is_dir():
        results = extractor.process_directory(
            input_path,
            output_dir=args.output_dir,
            export_format=args.format
        )
        print(f"Processed {len(results)} files")

if __name__ == "__main__":
    main()