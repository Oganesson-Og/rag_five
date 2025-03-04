"""
Optical Character Recognition Module
---------------------------------

Advanced OCR system for extracting text from document images with
support for multiple languages and document formats.

Key Features:
- Multi-language text recognition (Tesseract or custom models)
- Layout-aware text extraction (if needed)
- Handwriting recognition (extendable)
- Document format support (PDF, images)
- Text confidence scoring (Tesseract's confidences or custom)
- Language detection (optional)

Technical Details:
- Tesseract-based or ONNX-based pipelines.
- Quality enhancement preprocessing.
- Additional methods for diagram text extraction.

Dependencies:
- tesseract>=5.3.0
- pytesseract>=0.3.10
- opencv-python>=4.8.0
- numpy>=1.24.0
- pdf2image>=1.16.3 (if you want to convert PDF pages to images)
- onnxruntime, etc. (if using custom ONNX OCR models)

Author: Keith Satuku
Version: 1.1.0
License: MIT
"""

import os
import cv2
import math
import time
import logging
import numpy as np
import pytesseract
import fitz
from typing import List, Dict, Union, Optional, Any
from pdf2image import convert_from_path
from PIL import Image
import yaml
from pathlib import Path
import google.generativeai as genai
from base64 import b64encode
from io import BytesIO

# Default tesseract configuration
DEFAULT_TESSERACT_CONFIG = "--oem 1 --psm 6"

# Default languages
DEFAULT_LANGUAGES = ['eng']

# Default confidence threshold
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# Default PaliGemma model
DEFAULT_PALIGEMMA_MODEL = "google/paligemma-3b-mix-224"

# Default Gemini model
DEFAULT_GEMINI_MODEL = "gemini-pro-vision"

# Check if PaliGemma is available
try:
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    HAS_PALIGEMMA = True
except ImportError:
    HAS_PALIGEMMA = False

# Check if OpenCV is available
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# Check if Tesseract is available
try:
    pytesseract.get_tesseract_version()
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

# Configure logging
logger = logging.getLogger(__name__)

# If you have your custom text detection/recognition operators:
# from .operators import ...
# from .postprocess import build_post_process

# Check for OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    print("Warning: OpenCV not available. Image preprocessing will be limited.")
    HAS_OPENCV = False

# Check if PaliGemma is available
try:
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    HAS_PALIGEMMA = True
except ImportError:
    HAS_PALIGEMMA = False

# Check if Tesseract is available
try:
    pytesseract.get_tesseract_version()
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

# # Load configuration
# def load_config():
#     try:
#         # Try multiple possible locations for the config file
#         possible_paths = [
#             Path(__file__).parents[4] / "config" / "rag_config.yaml",  # From module location
#             Path("config/rag_config.yaml"),                            # From current directory
#             Path("../config/rag_config.yaml"),                         # One level up
#             Path("../../config/rag_config.yaml"),                      # Two levels up
#             Path("../../../config/rag_config.yaml"),                   # Three levels up
#             Path("../../../../config/rag_config.yaml"),                # Four levels up
#         ]
        
#         config_path = None
#         for path in possible_paths:
#             if path.exists():
#                 config_path = path
#                 break
                
#         if config_path:
#             print(f"Loading configuration from {config_path}")
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#             return config
#         else:
#             print("Warning: Configuration file not found. Using default settings.")
#             # Return a minimal default configuration
#             return {
#                 "pdf": {
#                     "use_ocr": True,
#                     "preserve_layout": True,
#                     "enhance_resolution": True,
#                     "use_paligemma": True
#                 },
#                 "model": {
#                     "temperature": 0.7,
#                     "max_tokens": 200,
#                     "hf_api_key": "",
#                     "gemini_api_key": ""
#                 }
#             }
#     except Exception as e:
#         print(f"Error loading configuration: {str(e)}")
#         return {}

# # Get configuration
# config = load_config()

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format=image.format if image.format else "PNG")
    return b64encode(buffered.getvalue()).decode("utf-8")

class OCR:
    """
    Advanced OCR processor that handles:
      - OCR extraction from PDF pages (e.g. scanned PDFs)
      - OCR extraction from in-memory images (e.g. diagrams)
      - Multi-language detection and recognition
      - Layout preservation and structural analysis
      - Quality enhancement and preprocessing
      - Confidence scoring and validation
    """
    def __init__(
        self,
        config: Dict[str, Any],
        languages: List[str] = None,
        preserve_layout: bool = None,
        enhance_resolution: bool = None,
        confidence_threshold: float = None,
        use_paligemma: bool = False
    ):
        # Load configuration
        self.config = config
        
        # Set defaults from configuration or fallback to hardcoded defaults
        self.languages = languages or self.config.get("pdf", {}).get("languages", ['en'])
        self.preserve_layout = preserve_layout if preserve_layout is not None else self.config.get("pdf", {}).get("preserve_layout", True)
        self.enhance_resolution = enhance_resolution if enhance_resolution is not None else self.config.get("pdf", {}).get("enhance_resolution", True)
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else self.config.get("pdf", {}).get("confidence_threshold", 0.7)
        self.use_paligemma = use_paligemma if use_paligemma is not None else self.config.get("pdf", {}).get("use_paligemma", False)
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize Gemini if available
        self.use_gemini = False
        self.gemini_model = None
        
        # Try to initialize Gemini if PaliGemma is not available
        if not HAS_PALIGEMMA:
            try:                # Get API key from different possible locations in config
                api_key = (
                    self.config.get("layout", {}).get("api_key")
                )
                
                # Get model name from different possible locations in config
                model_name = (
                    self.config.get("layout", {}).get("model_name") 
                )
                
                if api_key:
                    self.logger.info(f"Initializing Gemini model {model_name} for image understanding")
                    genai.configure(api_key=api_key)
                    self.gemini_model = genai.GenerativeModel(model_name)
                    self.use_gemini = True
                else:
                    self.logger.warning("No Gemini API key found in configuration. Advanced image understanding will be disabled.")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Gemini model: {str(e)}")
        
        # Default preprocessing options
        self.preprocessing_options = {
            'deskew': True,
            'denoise': True,
            'contrast_enhancement': True,
            'remove_borders': True
        }
        
        # Check if advanced features are available
        self._check_advanced_features()

    def _check_advanced_features(self):
        """Check if advanced OCR features are available and log status."""
        if not HAS_OPENCV:
            self.logger.warning("OpenCV not available. Image preprocessing will be limited.")
            # Disable preprocessing options that require OpenCV
            self.preprocessing_options['deskew'] = False
            self.preprocessing_options['denoise'] = False
            self.preprocessing_options['contrast_enhancement'] = False
            self.preprocessing_options['remove_borders'] = False
        
        # Check for advanced image understanding capabilities
        has_advanced_image_understanding = False
        
        # Check if PaliGemma is available
        if self.use_paligemma and HAS_PALIGEMMA:
            has_advanced_image_understanding = True
            self.logger.info("PaliGemma model is available for advanced image understanding.")
        elif hasattr(self, 'use_gemini') and self.use_gemini and self.gemini_model:
            has_advanced_image_understanding = True
            self.logger.info("Gemini model is available for advanced image understanding.")
        else:
            self.logger.warning("Advanced image understanding is not available.")
            if self.use_paligemma and not HAS_PALIGEMMA:
                self.logger.warning("PaliGemma could not be loaded. See earlier logs for details.")
            if hasattr(self, 'use_gemini') and not self.use_gemini:
                self.logger.warning("Gemini model could not be initialized. Check API key configuration.")
        
        self.has_advanced_image_understanding = has_advanced_image_understanding

    def has_advanced_features(self) -> bool:
        """Check if advanced OCR features are available."""
        # Consider either PaliGemma or Gemini as valid for advanced features
        has_advanced_models = (self.use_paligemma and HAS_PALIGEMMA) or (hasattr(self, 'use_gemini') and self.use_gemini and self.gemini_model is not None)
        return HAS_OPENCV and has_advanced_models

    async def process_document(self, doc) -> str:
        """
        Process an entire document with advanced OCR capabilities.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            Structured OCR text with layout information
        """
        all_text = []
        document_language = None
        
        try:
            # Detect primary document language if not specified
            if len(self.languages) == 1 and self.languages[0] == 'en':
                document_language = await self._detect_document_language(doc)
                if document_language:
                    self.languages = [document_language]
            
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                # Enhanced resolution with configurable zoom
                zoom = 2.0 if self.enhance_resolution else 1.0
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Preprocess image for better OCR quality
                processed_image = self._preprocess_image(pil_image)
                
                # Extract text with layout preservation and confidence scores
                page_result = await self._process_page(
                    processed_image,
                    languages=self.languages,
                    preserve_layout=self.preserve_layout
                )
                
                # Format page text with structural information
                page_text = self._format_page_text(page_result, page_index)
                all_text.append(page_text)
                
        except Exception as e:
            self.logger.error(f"Document OCR failed: {str(e)}")
            raise
            
        return "\n\n".join(all_text)

    async def _process_page(
        self,
        image: Image.Image,
        languages: List[str],
        preserve_layout: bool
    ) -> Dict[str, Any]:
        """
        Process a single page with advanced OCR features.
        
        Returns:
            Dict containing text blocks, confidence scores, and layout information
        """
        try:
            config = []
            if preserve_layout:
                config.append("--psm 6")  # Assume uniform block of text
            else:
                config.append("--psm 3")  # Fully automatic page segmentation
                
            # Join languages with '+' for pytesseract
            lang_str = "+".join(languages)
            
            # Use pytesseract for OCR
            import pytesseract
            ocr_data = pytesseract.image_to_data(
                image, 
                lang=lang_str,
                config=" ".join(config),
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text blocks with confidence scores
            blocks = self._extract_text_blocks(ocr_data)
            
            # Filter blocks by confidence threshold
            validated_blocks = self._validate_blocks(blocks)
            
            # Calculate overall page confidence
            page_confidence = self._calculate_page_confidence(validated_blocks)
            
            # Extract layout information
            layout_info = self._extract_layout_info(ocr_data)
            
            # Use advanced image understanding if available
            advanced_insights = None
            
            # Try PaliGemma first if enabled and available
            if self.use_paligemma and HAS_PALIGEMMA:
                try:
                    advanced_insights = await self._get_paligemma_original_insights(image)
                    if advanced_insights:
                        self.logger.info("Successfully obtained PaliGemma insights")
                except Exception as e:
                    self.logger.warning(f"PaliGemma processing failed: {str(e)}")
            
            # Try Gemini as fallback if PaliGemma failed or is not available
            if not advanced_insights and hasattr(self, 'use_gemini') and self.use_gemini and self.gemini_model:
                try:
                    advanced_insights = await self._get_gemini_insights(image)
                    if advanced_insights:
                        self.logger.info("Successfully obtained Gemini insights")
                except Exception as e:
                    self.logger.warning(f"Gemini processing failed: {str(e)}")
            
            return {
                'blocks': validated_blocks,
                'page_confidence': page_confidence,
                'layout_info': layout_info,
                'advanced_insights': advanced_insights
            }
            
        except Exception as e:
            self.logger.error(f"Page OCR failed: {str(e)}")
            return {
                'blocks': [],
                'page_confidence': 0.0,
                'layout_info': {},
                'advanced_insights': None
            }

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess an image to improve OCR results.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed PIL Image
        """
        # If OpenCV is not available, return the original image
        if not HAS_OPENCV:
            return image
            
        try:
            # Convert PIL image to OpenCV format
            img_array = np.array(image.convert('RGB'))
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Apply preprocessing steps based on options
            if self.preprocessing_options.get('deskew', True):
                img_cv = self._deskew_image(img_cv)
                
            if self.preprocessing_options.get('denoise', True):
                img_cv = self._denoise_image(img_cv)
                
            if self.preprocessing_options.get('contrast_enhancement', True):
                img_cv = self._enhance_contrast(img_cv)
                
            if self.preprocessing_options.get('remove_borders', True):
                img_cv = self._remove_borders(img_cv)
                
            # Convert back to PIL image
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {str(e)}")
            return image
            
    def _deskew_image(self, img: np.ndarray) -> np.ndarray:
        """Deskew an image to straighten text."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Threshold the image
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # Find all contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find largest contour
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Find minimum area rectangle
                rect = cv2.minAreaRect(largest_contour)
                angle = rect[2]
                
                # Adjust angle
                if angle < -45:
                    angle = 90 + angle
                
                # Rotate image
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return rotated
            
            return img
        except Exception as e:
            self.logger.warning(f"Deskew failed: {str(e)}")
            return img
            
    def _denoise_image(self, img: np.ndarray) -> np.ndarray:
        """Remove noise from an image."""
        try:
            # Apply non-local means denoising
            return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        except Exception as e:
            self.logger.warning(f"Denoising failed: {str(e)}")
            return img
            
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Enhance contrast in an image."""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Split channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge channels
            merged = cv2.merge((cl, a, b))
            
            # Convert back to BGR
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        except Exception as e:
            self.logger.warning(f"Contrast enhancement failed: {str(e)}")
            return img
            
    def _remove_borders(self, img: np.ndarray) -> np.ndarray:
        """Remove borders from an image."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find bounding rectangle
            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Crop image
                return img[y:y+h, x:x+w]
            
            return img
        except Exception as e:
            self.logger.warning(f"Border removal failed: {str(e)}")
            return img

    def extract_text_from_diagram(self, pil_image: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Enhanced diagram text extraction with preprocessing and structural analysis.
        """
        try:
            # Preprocess the diagram image
            processed = self._preprocess_image(pil_image)
            
            # Extract text with layout preservation
            ocr_result = pytesseract.image_to_data(
                processed,
                lang="+".join(self.languages),
                config="--psm 6",
                output_type=pytesseract.Output.DICT
            )
            
            # Process and validate results
            blocks = self._process_ocr_data(ocr_result)
            validated_blocks = self._validate_ocr_results(blocks)
            
            if not validated_blocks:
                return None
                
            return {
                'text_blocks': validated_blocks,
                'confidence': self._calculate_page_confidence(validated_blocks),
                'layout': self._extract_layout_info(ocr_result)
            }
            
        except Exception as e:
            self.logger.warning(f"Diagram OCR error: {str(e)}")
            return None

    def _process_ocr_data(self, ocr_data: Dict) -> List[Dict[str, Any]]:
        """Process raw OCR data into structured blocks."""
        blocks = []
        current_block = None
        
        for i in range(len(ocr_data['text'])):
            if ocr_data['text'][i].strip():
                conf = float(ocr_data['conf'][i])
                if conf == -1:  # Skip invalid confidence
                    continue
                    
                block_data = {
                    'text': ocr_data['text'][i],
                    'confidence': conf,
                    'bbox': (
                        ocr_data['left'][i],
                        ocr_data['top'][i],
                        ocr_data['left'][i] + ocr_data['width'][i],
                        ocr_data['top'][i] + ocr_data['height'][i]
                    ),
                    'line_num': ocr_data['line_num'][i],
                    'block_num': ocr_data['block_num'][i]
                }
                
                if current_block and current_block['block_num'] == block_data['block_num']:
                    current_block['text_elements'].append(block_data)
                else:
                    if current_block:
                        blocks.append(current_block)
                    current_block = {
                        'block_num': block_data['block_num'],
                        'text_elements': [block_data],
                        'confidence': conf
                    }
                    
        if current_block:
            blocks.append(current_block)
            
        return blocks

    def _validate_ocr_results(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate OCR results and filter low-confidence sections."""
        validated = []
        
        for block in blocks:
            # Calculate average confidence for the block
            confidences = [elem['confidence'] for elem in block['text_elements']]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            if avg_confidence >= self.confidence_threshold * 100:
                validated.append(block)
                
        return validated

    def _calculate_page_confidence(self, blocks: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for the page."""
        if not blocks:
            return 0.0
            
        confidences = []
        for block in blocks:
            block_confidences = [elem['confidence'] for elem in block['text_elements']]
            if block_confidences:
                confidences.append(sum(block_confidences) / len(block_confidences))
                
        return sum(confidences) / len(confidences) if confidences else 0.0

    def _extract_layout_info(self, ocr_data: Dict) -> Dict[str, Any]:
        """Extract layout information from OCR data."""
        return {
            'blocks': list(set(ocr_data['block_num'])),
            'lines': list(set(ocr_data['line_num'])),
            'word_count': len([t for t in ocr_data['text'] if t.strip()]),
            'average_line_height': np.mean(ocr_data['height']) if ocr_data['height'] else 0
        }

    def _format_page_text(self, page_result: Dict[str, Any], page_index: int) -> str:
        """Format page text with structural information."""
        if not page_result['blocks']:
            return f"--- Page {page_index + 1} ---\n[No valid text detected]"
            
        text_parts = [f"--- Page {page_index + 1} ---"]
        text_parts.append(f"Confidence: {page_result['page_confidence']:.1f}%")
        
        # Add PaliGemma insights if available
        if page_result.get('advanced_insights'):
            text_parts.append("\n=== Document Analysis ===")
            for insight in page_result['advanced_insights'].get('insights', []):
                text_parts.append(f"Q: {insight['prompt'].replace('<image> ', '')}")
                text_parts.append(f"A: {insight['response']}\n")
        
        text_parts.append("=== Extracted Text ===")
        for block in page_result['blocks']:
            block_text = " ".join(
                elem['text'] for elem in block['text_elements']
            )
            text_parts.append(block_text)
            
        return "\n".join(text_parts)

    async def _detect_document_language(self, doc) -> Optional[str]:
        """Detect primary document language."""
        try:
            # Sample first few pages for language detection
            sample_text = ""
            for i in range(min(3, doc.page_count)):
                page = doc.load_page(i)
                sample_text += page.get_text()
                
            if not sample_text.strip():
                return None
                
            # Use langdetect or similar library
            from langdetect import detect
            return detect(sample_text)
            
        except Exception as e:
            self.logger.warning(f"Language detection failed: {str(e)}")
            return None

    async def _get_paligemma_original_insights(self, image: Image.Image) -> Dict[str, Any]:
        """Original PaliGemma implementation."""
        # Original PaliGemma code here...
        if not HAS_PALIGEMMA or not self.use_paligemma:
            return None
            
        try:
            # Get temperature from config
            temperature = config.get("model", {}).get("temperature", 0.7)
            
            # Try to get custom prompt from config
            custom_prompt = config.get("pdf", {}).get("picture_annotation", {}).get("prompt", "")
            
            # Generic document understanding prompts
            default_prompts = [
                "<image> Describe the main content and structure of this document.",
                "<image> What type of document is this and what information does it contain?",
                "<image> Extract the key information from this document."
            ]
            
            # Use custom prompt if available
            prompts = ["<image> " + custom_prompt] if custom_prompt else default_prompts
            
            results = []
            for prompt in prompts:
                inputs = processor(
                    text=prompt, 
                    images=image, 
                    return_tensors="pt"
                )
                inputs.pop("last_cache_position", None)
                
                for key, tensor in inputs.items():
                    if key in ["input_ids", "attention_mask"]:
                        inputs[key] = tensor.to(device)
                    else:
                        inputs[key] = tensor.to(device, dtype=torch.float16)
                
                input_len = inputs["input_ids"].shape[-1]
                
                with torch.inference_mode():
                    generation = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id
                    )
                    generation = generation[0][input_len:]
                    decoded = processor.decode(generation, skip_special_tokens=True)
                    results.append({
                        'prompt': prompt,
                        'response': decoded
                    })
            
            return {
                'insights': results
            }
            
        except Exception as e:
            self.logger.warning(f"PaliGemma insights failed: {str(e)}")
            return None

    async def _get_gemini_insights(self, image: Image.Image) -> Dict[str, Any]:
        """Get insights using Gemini model."""
        try:
            # Get custom prompt from config
            custom_prompt = self.config.get("pdf", {}).get("picture_annotation", {}).get("prompt", "")
            
            # Default prompts if no custom prompt
            default_prompts = [
                "Describe the main content and structure of this document.",
                "What type of document is this and what information does it contain?",
                "Extract the key information from this document."
            ]
            
            prompts = [custom_prompt] if custom_prompt else default_prompts
            results = []
            
            # Convert PIL Image to base64
            image_data = encode_image_to_base64(image)
            
            for prompt in prompts:
                try:
                    response = self.gemini_model.generate_content([
                        prompt,
                        {"mime_type": "image/png", "data": image_data}
                    ])
                    
                    results.append({
                        'prompt': prompt,
                        'response': response.text
                    })
                except Exception as e:
                    self.logger.warning(f"Gemini prompt failed: {str(e)}")
                    results.append({
                        'prompt': prompt,
                        'response': f"Error processing image: {str(e)}"
                    })
            
            self.logger.info(f"Successfully processed image with Gemini model")
            return {
                'insights': results,
                'model': 'gemini'
            }
            
        except Exception as e:
            self.logger.warning(f"Gemini image analysis failed: {str(e)}")
            return None

    def _extract_text_blocks(self, ocr_data: Dict) -> List[Dict[str, Any]]:
        """Extract text blocks from OCR data."""
        blocks = []
        current_block = None
        current_block_num = -1
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            if not text:
                continue
                
            block_num = ocr_data['block_num'][i]
            line_num = ocr_data['line_num'][i]
            confidence = ocr_data['conf'][i]
            
            # New block
            if block_num != current_block_num:
                if current_block:
                    blocks.append(current_block)
                    
                current_block = {
                    'block_num': block_num,
                    'text_elements': [],
                    'bounding_box': {
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i]
                    }
                }
                current_block_num = block_num
                
            # Add text element to current block
            current_block['text_elements'].append({
                'text': text,
                'line_num': line_num,
                'confidence': float(confidence) if confidence != '-1' else 0.0,
                'position': {
                    'x': ocr_data['left'][i],
                    'y': ocr_data['top'][i],
                    'width': ocr_data['width'][i],
                    'height': ocr_data['height'][i]
                }
            })
            
        # Add the last block
        if current_block:
            blocks.append(current_block)
            
        return blocks
    
    def _validate_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate text blocks based on confidence threshold."""
        validated = []
        
        for block in blocks:
            # Calculate average confidence for the block
            confidences = [elem['confidence'] for elem in block['text_elements']]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            if avg_confidence >= self.confidence_threshold * 100:
                validated.append(block)
                
        return validated

    def process_image(self, image: Image.Image) -> str:
        """
        Process a single image and extract text using OCR.
        
        Args:
            image: PIL Image object to process
            
        Returns:
            Extracted text from the image
        """
        try:
            # Preprocess the image
            processed_image = self._preprocess_image(image)
            
            # Use Tesseract for OCR
            if HAS_TESSERACT:
                # Convert languages to Tesseract format
                lang_str = '+'.join(self.languages)
                
                # Extract text using Tesseract
                text = pytesseract.image_to_string(
                    processed_image, 
                    lang=lang_str,
                    config=DEFAULT_TESSERACT_CONFIG
                )
                
                # If text is empty or very short, try advanced methods if available
                if len(text.strip()) < 10:
                    if self.use_paligemma and HAS_PALIGEMMA:
                        # Try PaliGemma for image understanding
                        text = self._extract_text_with_paligemma(processed_image)
                    elif self.use_gemini and self.gemini_model:
                        # Try Gemini for image understanding
                        text = self._extract_text_with_gemini(processed_image)
                
                return text.strip()
            else:
                self.logger.warning("Tesseract not available. Using alternative methods.")
                
                # Try alternative methods
                if self.use_paligemma and HAS_PALIGEMMA:
                    return self._extract_text_with_paligemma(processed_image)
                elif self.use_gemini and self.gemini_model:
                    return self._extract_text_with_gemini(processed_image)
                else:
                    self.logger.error("No OCR methods available.")
                    return ""
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return ""
            
    def _extract_text_with_paligemma(self, image: Image.Image) -> str:
        """Extract text using PaliGemma model."""
        try:
            if not HAS_PALIGEMMA:
                return ""
                
            # Load model and processor
            processor = AutoProcessor.from_pretrained(DEFAULT_PALIGEMMA_MODEL)
            model = AutoModelForVision2Seq.from_pretrained(DEFAULT_PALIGEMMA_MODEL)
            
            # Process image
            inputs = processor(images=image, return_tensors="pt")
            
            # Generate text
            generated_ids = model.generate(
                pixel_values=inputs["pixel_values"],
                max_length=512,
                num_beams=3
            )
            
            # Decode text
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text
        except Exception as e:
            self.logger.error(f"PaliGemma extraction failed: {str(e)}")
            return ""
            
    def _extract_text_with_gemini(self, image: Image.Image) -> str:
        """Extract text using Gemini model."""
        try:
            if not self.use_gemini or not self.gemini_model:
                return ""
                
            # Encode image to base64
            image_data = encode_image_to_base64(image)
            
            # Generate prompt
            prompt = "Extract all text visible in this image. Include any text in tables, diagrams, or other elements."
            
            # Generate response
            response = self.gemini_model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": image_data}
            ])
            
            return response.text
        except Exception as e:
            self.logger.error(f"Gemini extraction failed: {str(e)}")
            return ""

from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
import torch
from PIL import Image
import os
from pathlib import Path
from pdf2image import convert_from_path

# Model setup
try:
    # Get configuration values for HuggingFace
    hf_api_key = config.get("model", {}).get("hf_api_key")
    model_id = config.get("model", {}).get("hf_captioning_model", "google/paligemma2-10b-mix-448")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model ID: {model_id}")

    torch.set_float32_matmul_precision('high')

    if not hf_api_key:
        print("Warning: No Hugging Face API token found in configuration.")
        print("To use PaliGemma, add your Hugging Face API token to the config/rag_config.yaml file under model.hf_api_key")
        raise ValueError("Missing Hugging Face API token")

    print(f"Attempting to load PaliGemma model with API token: {hf_api_key[:4]}{'*' * (len(hf_api_key) - 8)}{hf_api_key[-4:] if len(hf_api_key) > 8 else ''}")

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_api_key
    ).eval()

    processor = PaliGemmaProcessor.from_pretrained(
        model_id,
        token=hf_api_key
    )
    HAS_PALIGEMMA = True
    print("Successfully loaded PaliGemma model")
except Exception as e:
    print(f"Warning: Could not load PaliGemma model: {str(e)}")
    print("Falling back to basic OCR functionality")
    print("To use PaliGemma, you need to:")
    print("1. Create a Hugging Face account")
    print("2. Accept the model terms at https://huggingface.co/google/paligemma2-10b-mix-448")
    print("3. Create an access token at https://huggingface.co/settings/tokens")
    print("4. Add your token to config/rag_config.yaml under model.hf_api_key")
    HAS_PALIGEMMA = False
    model = None
    processor = None

# Process passport directory
passports_dir = Path("port")

def process_passport_image(image, filename, page_num=None):
    try:
        # Check if PaliGemma is available
        if not HAS_PALIGEMMA:
            print(f"Using basic OCR for {filename} (Page {page_num if page_num is not None else 1})")
            # Save the image to a temporary file
            temp_img_path = f"temp_{filename.replace('.pdf', '')}_{page_num if page_num is not None else 1}.png"
            image.save(temp_img_path)
            print(f"Saved image to {temp_img_path}")
            
            # Use basic OCR to extract text
            try:
                import pytesseract
                text = pytesseract.image_to_string(image)
                
                # Save the extracted text
                text_file = f"extracted_{filename.replace('.pdf', '')}_{page_num if page_num is not None else 1}.txt"
                with open(text_file, "w") as f:
                    f.write(text)
                print(f"Extracted text saved to {text_file}")
                
            except ImportError:
                print("pytesseract not available. Install it with: pip install pytesseract")
                print("You'll also need to install Tesseract OCR on your system.")
            
            return
            
        # Get prompts from configuration if available
        default_prompts = [
            "<image> What are the towage charges for a vessel with a net registered tonnage (NRT) of 4500 in the harbour areas of HaminaKotka?",
            "<image> What is the minimum charge for towage assistance, and how is the time for the service calculated?",
            "<image> What are the additional charges for towage assistance provided on weekends or public holidays in HaminaKotka?",
            "<image> What is the cancellation fee for tug services if the tug has not left the station on a working day?",
            "<image> How much is the surcharge for ordering towage assistance outside normal working hours without prior notification during public holidays?"
        ]
        
        # Try to get custom prompt from config
        custom_prompt = config.get("pdf", {}).get("picture_annotation", {}).get("prompt", "")
        if custom_prompt:
            queries = ["<image> " + custom_prompt]
            print(f"Using custom prompt from configuration: {custom_prompt}")
        else:
            queries = default_prompts
            
        # Get temperature from config
        temperature = config.get("model", {}).get("temperature", 0.7)

        for prompt in queries:
            inputs = processor(
                text=prompt, 
                images=image, 
                return_tensors="pt"
            )
            inputs.pop("last_cache_position", None)
            
            for key, tensor in inputs.items():
                if key in ["input_ids", "attention_mask"]:
                    inputs[key] = tensor.to(device)
                else:
                    inputs[key] = tensor.to(device, dtype=torch.float16)
            
            input_len = inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                generation = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
                generation = generation[0][input_len:]
                decoded = processor.decode(generation, skip_special_tokens=True)
                
                page_info = f" (Page {page_num})" if page_num is not None else ""
                print(f"\nProcessing: {filename}{page_info}")
                print("-" * 50)
                print("Prompt:", prompt, "\n")
                print("Caption:", decoded)
                print("-" * 50)
                
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    if not passports_dir.exists():
        print(f"Directory {passports_dir} not found!")
        return
    
    print(f"Looking for PDFs in: {passports_dir.absolute()}")
    
    # Print configuration information
    print("\n=== Configuration ===")
    print(f"Hugging Face API Key: {'Configured' if hf_api_key else 'Not configured'}")
    print(f"OpenCV: {'Available' if HAS_OPENCV else 'Not available'}")
    print(f"PaliGemma: {'Available' if HAS_PALIGEMMA else 'Not available'}")
    print(f"Device: {device}")
    
    # Print PDF configuration
    pdf_config = config.get("pdf", {})
    print("\n=== PDF Processing Configuration ===")
    print(f"Use OCR: {pdf_config.get('use_ocr', True)}")
    print(f"Preserve Layout: {pdf_config.get('preserve_layout', True)}")
    print(f"Enhance Resolution: {pdf_config.get('enhance_resolution', True)}")
    print(f"Use PaliGemma: {pdf_config.get('use_paligemma', True)}")
    print("=" * 20)
    
    # Initialize OCR with configuration
    ocr = OCR()
    
    print(f"\nOCR initialized. Advanced features: {'Available' if ocr.has_advanced_features() else 'Limited'}")
    if not ocr.has_advanced_features():
        print("Note: Some advanced features are not available. Basic OCR will be used.")
        if not HAS_OPENCV:
            print("OpenCV is not available. Image preprocessing will be limited.")
        if not HAS_PALIGEMMA:
            print("PaliGemma model is not available. Advanced image understanding will be disabled.")
            print("To use PaliGemma, you need to:")
            print("1. Create a Hugging Face account")
            print("2. Accept the model terms at https://huggingface.co/google/paligemma2-10b-mix-448")
            print("3. Create an access token at https://huggingface.co/settings/tokens")
            print("4. Ensure your token is in the config/rag_config.yaml file under model.hf_api_key")
    
    # Process all PDF files in the directory
    found_pdfs = False
    
    for file_path in passports_dir.iterdir():
        if file_path.suffix.lower() == '.pdf':
            found_pdfs = True
            print(f"\nProcessing PDF: {file_path.name}")
            
            try:
                # Convert PDF to images
                images = convert_from_path(file_path)
                
                # Process each page
                for i, image in enumerate(images, 1):
                    print(f"Processing page {i} of {file_path.name}")
                    
                    # Save the image to a temporary file
                    temp_img_path = f"temp_{file_path.stem}_page{i}.png"
                    image.save(temp_img_path)
                    print(f"Saved image to {temp_img_path}")
                    
                    # Create a PyMuPDF document from the image
                    import fitz
                    doc = fitz.open()
                    doc.new_page(width=image.width, height=image.height)
                    page = doc[0]
                    page.insert_image(fitz.Rect(0, 0, image.width, image.height), filename=temp_img_path)
                    
                    # Process the document with OCR
                    import asyncio
                    result = asyncio.run(ocr.process_document(doc))
                    
                    # Save the result
                    output_file = f"extracted_{file_path.stem}_page{i}.txt"
                    with open(output_file, "w") as f:
                        f.write(result)
                    print(f"Extracted text saved to {output_file}")
                    
                    # Clean up
                    doc.close()
                    
                    # If PaliGemma is available, also process with the original function
                    if HAS_PALIGEMMA and ocr.use_paligemma:
                        process_passport_image(image, file_path.name, i)
                    
            except Exception as e:
                print(f"Error processing PDF {file_path}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    if not found_pdfs:
        print("No PDF files found in the passports directory!")

if __name__ == "__main__":
    main()


