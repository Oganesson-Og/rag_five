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
from typing import List, Dict, Union, Optional, Any
from pdf2image import convert_from_path
from PIL import Image

# If you have your custom text detection/recognition operators:
# from .operators import ...
# from .postprocess import build_post_process


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
        languages: List[str] = ['en'],
        enhance_resolution: bool = False,
        preserve_layout: bool = True,
        confidence_threshold: float = 0.6,
        preprocessing_options: Optional[Dict] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.languages = languages
        self.enhance_resolution = enhance_resolution
        self.preserve_layout = preserve_layout
        self.confidence_threshold = confidence_threshold
        
        # Default preprocessing options
        self._preprocessing_options = {
            'denoise': True,
            'deskew': True,
            'contrast_enhancement': True,
            'remove_borders': True,
            'adaptive_thresholding': True
        }
        if preprocessing_options:
            self._preprocessing_options.update(preprocessing_options)

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
                
            # Add any additional Tesseract configurations
            config.append("--oem 3")  # Default, LSTM only
            config.append("-c preserve_interword_spaces=1")
            
            lang_str = "+".join(languages)
            
            # Get detailed OCR data including confidence scores
            ocr_data = pytesseract.image_to_data(
                image,
                lang=lang_str,
                config=" ".join(config),
                output_type=pytesseract.Output.DICT
            )
            
            # Process OCR results into structured format
            blocks = self._process_ocr_data(ocr_data)
            
            # Validate results and filter low-confidence sections
            validated_blocks = self._validate_ocr_results(blocks)
            
            return {
                'blocks': validated_blocks,
                'page_confidence': self._calculate_page_confidence(validated_blocks),
                'layout_info': self._extract_layout_info(ocr_data)
            }
            
        except Exception as e:
            self.logger.error(f"Page OCR processing failed: {str(e)}")
            return {'blocks': [], 'page_confidence': 0.0, 'layout_info': None}

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Apply preprocessing steps to improve OCR quality.
        """
        try:
            # Convert to numpy array for OpenCV operations
            np_image = np.array(image)
            
            if self._preprocessing_options['denoise']:
                np_image = cv2.fastNlMeansDenoisingColored(np_image)
                
            if self._preprocessing_options['deskew']:
                np_image = self._deskew_image(np_image)
                
            if self._preprocessing_options['contrast_enhancement']:
                # Convert to LAB color space for better contrast enhancement
                lab = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                lab = cv2.merge((l,a,b))
                np_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                
            if self._preprocessing_options['remove_borders']:
                np_image = self._remove_borders(np_image)
                
            if self._preprocessing_options['adaptive_thresholding']:
                gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
                np_image = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                
            return Image.fromarray(np_image)
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {str(e)}")
            return image

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

    def _deskew_image(self, np_image: np.ndarray) -> np.ndarray:
        """Deskew image using contour detection."""
        try:
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            if lines is None:
                return np_image
                
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                angles.append(angle)
                
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:  # Only correct if skew is significant
                (h, w) = np_image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                return cv2.warpAffine(np_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                
        except Exception as e:
            self.logger.warning(f"Deskewing failed: {str(e)}")
            
        return np_image

    def _remove_borders(self, np_image: np.ndarray) -> np.ndarray:
        """Remove potential borders from the image."""
        try:
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return np_image
                
            # Find the main content contour
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Add small margin
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(np_image.shape[1] - x, w + 2 * margin)
            h = min(np_image.shape[0] - y, h + 2 * margin)
            
            return np_image[y:y+h, x:x+w]
            
        except Exception as e:
            self.logger.warning(f"Border removal failed: {str(e)}")
            return np_image



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
model_id = "google/paligemma2-10b-mix-448"
api_token = ""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.set_float32_matmul_precision('high')

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    token=api_token
).eval()

processor = PaliGemmaProcessor.from_pretrained(
    model_id,
    token=api_token
)

# Process passport directory
passports_dir = Path("port")

def process_passport_image(image, filename, page_num=None):
    try:
        # Revised prompt
        queries = [
        "<image> What are the towage charges for a vessel with a net registered tonnage (NRT) of 4500 in the harbour areas of HaminaKotka?",
        "<image> What is the minimum charge for towage assistance, and how is the time for the service calculated?",
        "<image>  What are the additional charges for towage assistance provided on weekends or public holidays in HaminaKotka?",
        "<image> What is the cancellation fee for tug services if the tug has not left the station on a working day?",
        "<image> How much is the surcharge for ordering towage assistance outside normal working hours without prior notification during public holidays?"
        ]

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
                    temperature=0.7,
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
    
    # Process all PDF files in the directory
    found_pdfs = False
    
    for file_path in passports_dir.iterdir():
        if file_path.suffix.lower() == '.pdf':
            found_pdfs = True
            print(f"Processing PDF: {file_path.name}")
            
            try:
                # Convert PDF to images
                images = convert_from_path(file_path)
                
                # Process each page
                for i, image in enumerate(images, 1):
                    process_passport_image(image, file_path.name, i)
                    
            except Exception as e:
                print(f"Error converting PDF {file_path}: {str(e)}")
    
    if not found_pdfs:
        print("No PDF files found in the passports directory!")

if __name__ == "__main__":
    main()


