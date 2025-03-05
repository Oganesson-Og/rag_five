"""
Base Vision Recognition Module
----------------------------

Abstract base class providing core functionality for all vision recognition
models in the document processing pipeline.

Key Features:
- Support for multiple vision models (Gemini, Ollama)
- Spatial reasoning and layout organization
- Batch processing support
- Model caching
- PDF layout analysis integration

Dependencies:
- google-generativeai>=0.3.0
- ollama>=0.1.0
- numpy>=1.24.0
- PIL>=9.5.0

Example Usage:
    # Initialize for PDF processing
    recognizer = Recognizer(
        model_type="gemini",
        model_name="gemini-2.0-pro-exp-02-05",
        label_list=["title", "text", "list", "table", "figure",
                   "header", "footer", "sidebar", "caption"]
    )
"""

import logging
import os
import math
import numpy as np
import base64
from copy import deepcopy
from PIL import Image
from typing import Union, List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import google.generativeai as genai
import requests
from io import BytesIO

from src.utils.file_utils import get_project_base_directory
from .operators import *  
from .operators import preprocess
from . import operators

@dataclass
class LayoutElement:
    """Structure for layout elements."""
    type: str
    text: str = ""
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    font_size: float = 0
    font_name: str = ""
    is_bold: bool = False
    in_row: int = 1
    row_height: float = 0
    is_row_header: bool = False
    confidence: float = 1.0

class Recognizer:
    """Base class for vision recognition models."""
    
    DEFAULT_LABEL_LIST = [
        "title", "text", "list", "table", "figure",
        "header", "footer", "sidebar", "caption"
    ]
    
    def __init__(
        self,
        model_type: str = "gemini",  # "gemini" or "ollama"
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        label_list: Optional[List[str]] = None,
        task_name: str = "document_layout",
        ollama_host: str = "http://localhost:11434",
        model_dir: str = "",  # Added for compatibility
        confidence: float = 0.5,  # Added for compatibility
        merge_boxes: bool = True  # Added for compatibility
    ):
        """
        Initialize the Recognizer.

        Args:
            model_type: Type of vision model to use ("gemini" or "ollama")
            model_name: Name of the specific model to use
            api_key: API key for Gemini (if using Gemini)
            device: Device to use (for Ollama)
            batch_size: Batch size for processing
            cache_dir: Directory for caching models
            label_list: List of labels (defaults to PDF layout labels)
            task_name: Specific task name
            ollama_host: Host URL for Ollama API
            model_dir: Directory for model files (compatibility)
            confidence: Confidence threshold (compatibility)
            merge_boxes: Whether to merge overlapping boxes (compatibility)
        """
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.label_list = label_list or self.DEFAULT_LABEL_LIST
        self.task_name = task_name
        self.ollama_host = ollama_host
        self.confidence_threshold = confidence
        self.merge_boxes = merge_boxes

        if self.model_type == "gemini":
            if not api_key:
                raise ValueError("API key required for Gemini model")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name or "gemini-2.0-pro-exp-02-05")
        elif self.model_type == "ollama":
            self.model_name = self.model_name or "Qwen/Qwen2.5-VL-7B"
            try:
                response = requests.get(f"{self.ollama_host}/api/tags")
                if response.status_code != 200:
                    raise ConnectionError(f"Failed to connect to Ollama at {self.ollama_host}")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Ollama: {str(e)}")
        elif self.model_type == "spacy":
            # For spaCy model type, initialization is handled in the SpaCyLayoutRecognizer class
            pass
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _encode_image(self, image: Union[str, Image.Image, np.ndarray]) -> str:
        """Convert image to base64 string."""
        if isinstance(image, str):
            with open(image, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if isinstance(image, Image.Image):
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        raise ValueError("Unsupported image type")

    def _prepare_prompt(self, image: Union[str, Image.Image, np.ndarray]) -> dict:
        """Prepare prompt for vision models with PDF-specific instructions."""
        base64_image = self._encode_image(image)
        
        prompt_text = f"""
        Analyze this document image and identify layout elements. For each element, provide:
        1. type (one of: {', '.join(self.label_list)})
        2. bbox coordinates [x0,y0,x1,y1]
        3. confidence score

        Return results in JSON format with fields:
        - type: element type
        - bbox: [x0,y0,x1,y1] coordinates
        - score: confidence (0-1)
        """
        
        if self.model_type == "gemini":
            return {
                "contents": [{
                    "parts": [
                        {"text": prompt_text},
                        {"inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image
                        }}
                    ]
                }]
            }
        else:  # ollama
            return {
                "model": self.model_name,
                "prompt": prompt_text,
                "images": [base64_image],
                "stream": False
            }

    def _box_to_layout_element(self, box: Dict[str, Any]) -> LayoutElement:
        """Convert a box detection to a LayoutElement."""
        return LayoutElement(
            type=box["type"],
            bbox=tuple(box["bbox"]),
            confidence=box.get("score", 1.0)
        )

    async def process_page(self, page_image: Union[str, Image.Image, np.ndarray]) -> List[LayoutElement]:
        """
        Process a single page image and return layout elements.
        This is the main method used by the PDF extractor.
        """
        boxes = await self._get_vision_model_response(page_image)
        
        # Filter by confidence
        boxes = [box for box in boxes if box.get('score', 0) >= self.confidence_threshold]
        
        # Apply spatial reasoning if merge_boxes is True
        if self.merge_boxes and boxes:
            boxes = self.sort_Y_firstly(boxes, 10)
            boxes = self.sort_X_firstly(boxes, 10)
            boxes = self.layouts_cleanup(boxes, boxes)
        
        # Convert to LayoutElements
        return [self._box_to_layout_element(box) for box in boxes]

    async def _get_vision_model_response(self, image: Union[str, Image.Image, np.ndarray]) -> List[Dict]:
        """Get response from vision model."""
        prompt = self._prepare_prompt(image)
        
        try:
            if self.model_type == "gemini":
                response = await self.model.generate_content(**prompt)
                # Parse Gemini response to extract bounding boxes
                return self._parse_gemini_response(response.text)
            else:  # ollama
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json=prompt
                )
                if response.status_code != 200:
                    raise Exception(f"Ollama API error: {response.text}")
                # Parse Ollama response to extract bounding boxes
                return self._parse_ollama_response(response.json()["response"])
        except Exception as e:
            logging.error(f"Vision model error: {str(e)}")
            return []

    def _parse_gemini_response(self, response_text: str) -> List[Dict]:
        """Parse Gemini response into standardized format."""
        try:
            import json
            boxes = json.loads(response_text)
            return boxes
        except Exception as e:
            logging.error(f"Failed to parse Gemini response: {str(e)}")
            return []

    def _parse_ollama_response(self, response_text: str) -> List[Dict]:
        """Parse Ollama response into standardized format."""
        try:
            import json
            boxes = json.loads(response_text)
            return boxes
        except Exception as e:
            logging.error(f"Failed to parse Ollama response: {str(e)}")
            return []

    async def __call__(self, image_list: List[Union[str, Image.Image, np.ndarray]], 
                      thr: float = 0.7, 
                      batch_size: int = 16) -> List[List[Dict]]:
        """Process images and return detected elements."""
        results = []
        
        # Process images in batches
        for i in range(0, len(image_list), batch_size):
            batch = image_list[i:i + batch_size]
            batch_results = []
            
            # Process each image in the batch
            for image in batch:
                # Get predictions from vision model
                boxes = await self._get_vision_model_response(image)
                
                # Filter by confidence threshold
                boxes = [box for box in boxes if box.get('score', 0) >= thr]
                
                # Apply spatial reasoning
                if boxes:
                    # First sort vertically to establish reading order
                    boxes = self.sort_Y_firstly(boxes, 10)
                    
                    # Sort horizontally within same vertical regions
                    boxes = self.sort_X_firstly(boxes, 10)
                    
                    # Find and handle overlapping elements
                    boxes_sorted = sorted(boxes, key=lambda x: x["top"])
                    for idx, box in enumerate(boxes):
                        # Find overlapping boxes
                        overlapped_idx = self.find_overlapped(box, boxes_sorted)
                        if overlapped_idx is not None:
                            # If significant overlap found, keep the one with higher score
                            if boxes[overlapped_idx].get('score', 0) > box.get('score', 0):
                                boxes[idx] = None
                    
                    # Remove None values from boxes list
                    boxes = [b for b in boxes if b is not None]
                    
                    # Clean up remaining overlaps
                    boxes = self.layouts_cleanup(boxes, boxes)
                    
                    # Find horizontally aligned elements
                    for idx, box in enumerate(boxes):
                        closest_idx = self.find_horizontally_tightest_fit(box, boxes[:idx] + boxes[idx+1:])
                        if closest_idx is not None:
                            # Mark horizontally aligned elements with same layout number
                            box['layoutno'] = boxes[closest_idx].get('layoutno', str(idx))
                    
                    # Final sort based on layout numbers and position
                    boxes = self.sort_C_firstly(boxes) if any('C' in box for box in boxes) else \
                           self.sort_R_firstly(boxes) if any('R' in box for box in boxes) else \
                           boxes
                
                batch_results.append(boxes)
            
            results.extend(batch_results)
        
        return results

    @staticmethod
    def sort_Y_firstly(arr, threashold):
        # sort using y1 first and then x1
        arr = sorted(arr, key=lambda r: (r["top"], r["x0"]))
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                # restore the order using th
                if abs(arr[j + 1]["top"] - arr[j]["top"]) < threashold \
                        and arr[j + 1]["x0"] < arr[j]["x0"]:
                    tmp = deepcopy(arr[j])
                    arr[j] = deepcopy(arr[j + 1])
                    arr[j + 1] = deepcopy(tmp)
        return arr

    @staticmethod
    def sort_X_firstly(arr, threashold, copy=True):
        # sort using y1 first and then x1
        arr = sorted(arr, key=lambda r: (r["x0"], r["top"]))
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                # restore the order using th
                if abs(arr[j + 1]["x0"] - arr[j]["x0"]) < threashold \
                        and arr[j + 1]["top"] < arr[j]["top"]:
                    tmp = deepcopy(arr[j]) if copy else arr[j]
                    arr[j] = deepcopy(arr[j + 1]) if copy else arr[j + 1]
                    arr[j + 1] = deepcopy(tmp) if copy else tmp
        return arr

    @staticmethod
    def sort_C_firstly(arr, thr=0):
        # sort using y1 first and then x1
        # sorted(arr, key=lambda r: (r["x0"], r["top"]))
        arr = Recognizer.sort_X_firstly(arr, thr)
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                # restore the order using th
                if "C" not in arr[j] or "C" not in arr[j + 1]:
                    continue
                if arr[j + 1]["C"] < arr[j]["C"] \
                        or (
                        arr[j + 1]["C"] == arr[j]["C"]
                        and arr[j + 1]["top"] < arr[j]["top"]
                ):
                    tmp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = tmp
        return arr

        return sorted(arr, key=lambda r: (r.get("C", r["x0"]), r["top"]))

    @staticmethod
    def sort_R_firstly(arr, thr=0):
        # sort using y1 first and then x1
        # sorted(arr, key=lambda r: (r["top"], r["x0"]))
        arr = Recognizer.sort_Y_firstly(arr, thr)
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                if "R" not in arr[j] or "R" not in arr[j + 1]:
                    continue
                if arr[j + 1]["R"] < arr[j]["R"] \
                        or (
                        arr[j + 1]["R"] == arr[j]["R"]
                        and arr[j + 1]["x0"] < arr[j]["x0"]
                ):
                    tmp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = tmp
        return arr

    @staticmethod
    def overlapped_area(a, b, ratio=True):
        tp, btm, x0, x1 = a["top"], a["bottom"], a["x0"], a["x1"]
        if b["x0"] > x1 or b["x1"] < x0:
            return 0
        if b["bottom"] < tp or b["top"] > btm:
            return 0
        x0_ = max(b["x0"], x0)
        x1_ = min(b["x1"], x1)
        assert x0_ <= x1_, "Bbox mismatch! T:{},B:{},X0:{},X1:{} ==> {}".format(
            tp, btm, x0, x1, b)
        tp_ = max(b["top"], tp)
        btm_ = min(b["bottom"], btm)
        assert tp_ <= btm_, "Bbox mismatch! T:{},B:{},X0:{},X1:{} => {}".format(
            tp, btm, x0, x1, b)
        ov = (btm_ - tp_) * (x1_ - x0_) if x1 - \
                                           x0 != 0 and btm - tp != 0 else 0
        if ov > 0 and ratio:
            ov /= (x1 - x0) * (btm - tp)
        return ov

    @staticmethod
    def layouts_cleanup(boxes, layouts, far=2, thr=0.7):
        def notOverlapped(a, b):
            return any([a["x1"] < b["x0"],
                        a["x0"] > b["x1"],
                        a["bottom"] < b["top"],
                        a["top"] > b["bottom"]])

        i = 0
        while i + 1 < len(layouts):
            j = i + 1
            while j < min(i + far, len(layouts)) \
                    and (layouts[i].get("type", "") != layouts[j].get("type", "")
                         or notOverlapped(layouts[i], layouts[j])):
                j += 1
            if j >= min(i + far, len(layouts)):
                i += 1
                continue
            if Recognizer.overlapped_area(layouts[i], layouts[j]) < thr \
                    and Recognizer.overlapped_area(layouts[j], layouts[i]) < thr:
                i += 1
                continue

            if layouts[i].get("score") and layouts[j].get("score"):
                if layouts[i]["score"] > layouts[j]["score"]:
                    layouts.pop(j)
                else:
                    layouts.pop(i)
                continue

            area_i, area_i_1 = 0, 0
            for b in boxes:
                if not notOverlapped(b, layouts[i]):
                    area_i += Recognizer.overlapped_area(b, layouts[i], False)
                if not notOverlapped(b, layouts[j]):
                    area_i_1 += Recognizer.overlapped_area(b, layouts[j], False)

            if area_i > area_i_1:
                layouts.pop(j)
            else:
                layouts.pop(i)

        return layouts

    def create_inputs(self, imgs, im_info):
        """generate input for different model type
        Args:
            imgs (list(numpy)): list of images (np.ndarray)
            im_info (list(dict)): list of image info
        Returns:
            inputs (dict): input of model
        """
        inputs = {}

        im_shape = []
        scale_factor = []
        if len(imgs) == 1:
            inputs['image'] = np.array((imgs[0],)).astype('float32')
            inputs['im_shape'] = np.array(
                (im_info[0]['im_shape'],)).astype('float32')
            inputs['scale_factor'] = np.array(
                (im_info[0]['scale_factor'],)).astype('float32')
            return inputs

        for e in im_info:
            im_shape.append(np.array((e['im_shape'],)).astype('float32'))
            scale_factor.append(np.array((e['scale_factor'],)).astype('float32'))

        inputs['im_shape'] = np.concatenate(im_shape, axis=0)
        inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

        imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
        max_shape_h = max([e[0] for e in imgs_shape])
        max_shape_w = max([e[1] for e in imgs_shape])
        padding_imgs = []
        for img in imgs:
            im_c, im_h, im_w = img.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape_h, max_shape_w), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = img
            padding_imgs.append(padding_im)
        inputs['image'] = np.stack(padding_imgs, axis=0)
        return inputs

    @staticmethod
    def find_overlapped(box, boxes_sorted_by_y, naive=False):
        if not boxes_sorted_by_y:
            return
        bxs = boxes_sorted_by_y
        s, e, ii = 0, len(bxs), 0
        while s < e and not naive:
            ii = (e + s) // 2
            pv = bxs[ii]
            if box["bottom"] < pv["top"]:
                e = ii
                continue
            if box["top"] > pv["bottom"]:
                s = ii + 1
                continue
            break
        while s < ii:
            if box["top"] > bxs[s]["bottom"]:
                s += 1
            break
        while e - 1 > ii:
            if box["bottom"] < bxs[e - 1]["top"]:
                e -= 1
            break

        max_overlaped_i, max_overlaped = None, 0
        for i in range(s, e):
            ov = Recognizer.overlapped_area(bxs[i], box)
            if ov <= max_overlaped:
                continue
            max_overlaped_i = i
            max_overlaped = ov

        return max_overlaped_i

    @staticmethod
    def find_horizontally_tightest_fit(box, boxes):
        if not boxes:
            return
        min_dis, min_i = 1000000, None
        for i,b in enumerate(boxes):
            if box.get("layoutno", "0") != b.get("layoutno", "0"):
                continue
            dis = min(abs(box["x0"] - b["x0"]), abs(box["x1"] - b["x1"]), abs(box["x0"]+box["x1"] - b["x1"] - b["x0"])/2)
            if dis < min_dis:
                min_i = i
                min_dis = dis
        return min_i

    @staticmethod
    def find_overlapped_with_threashold(box, boxes, thr=0.3):
        if not boxes:
            return
        max_overlapped_i, max_overlapped, _max_overlapped = None, thr, 0
        s, e = 0, len(boxes)
        for i in range(s, e):
            ov = Recognizer.overlapped_area(box, boxes[i])
            _ov = Recognizer.overlapped_area(boxes[i], box)
            if (ov, _ov) < (max_overlapped, _max_overlapped):
                continue
            max_overlapped_i = i
            max_overlapped = ov
            _max_overlapped = _ov

        return max_overlapped_i

    def preprocess(self, image_list):
        inputs = []
        if "scale_factor" in self.input_names:
            preprocess_ops = []
            for op_info in [
                {'interp': 2, 'keep_ratio': False, 'target_size': [800, 608], 'type': 'LinearResize'},
                {'is_scale': True, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'type': 'StandardizeImage'},
                {'type': 'Permute'},
                {'stride': 32, 'type': 'PadStride'}
            ]:
                new_op_info = op_info.copy()
                op_type = new_op_info.pop('type')
                preprocess_ops.append(getattr(operators, op_type)(**new_op_info))

            for im_path in image_list:
                im, im_info = preprocess(im_path, preprocess_ops)
                inputs.append({"image": np.array((im,)).astype('float32'),
                               "scale_factor": np.array((im_info["scale_factor"],)).astype('float32')})
        else:
            hh, ww = self.input_shape
            for img in image_list:
                h, w = img.shape[:2]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(np.array(img).astype('float32'), (ww, hh))
                # Scale input pixel values to 0 to 1
                img /= 255.0
                img = img.transpose(2, 0, 1)
                img = img[np.newaxis, :, :, :].astype(np.float32)
                inputs.append({self.input_names[0]: img, "scale_factor": [w/ww, h/hh]})
        return inputs

    def postprocess(self, boxes, inputs, thr):
        if "scale_factor" in self.input_names:
            bb = []
            for b in boxes:
                clsid, bbox, score = int(b[0]), b[2:], b[1]
                if score < thr:
                    continue
                if clsid >= len(self.label_list):
                    continue
                bb.append({
                    "type": self.label_list[clsid].lower(),
                    "bbox": [float(t) for t in bbox.tolist()],
                    "score": float(score)
                })
            return bb

        def xywh2xyxy(x):
            # [x, y, w, h] to [x1, y1, x2, y2]
            y = np.copy(x)
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            return y

        def compute_iou(box, boxes):
            # Compute xmin, ymin, xmax, ymax for both boxes
            xmin = np.maximum(box[0], boxes[:, 0])
            ymin = np.maximum(box[1], boxes[:, 1])
            xmax = np.minimum(box[2], boxes[:, 2])
            ymax = np.minimum(box[3], boxes[:, 3])

            # Compute intersection area
            intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

            # Compute union area
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            union_area = box_area + boxes_area - intersection_area

            # Compute IoU
            iou = intersection_area / union_area

            return iou

        def iou_filter(boxes, scores, iou_threshold):
            sorted_indices = np.argsort(scores)[::-1]

            keep_boxes = []
            while sorted_indices.size > 0:
                # Pick the last box
                box_id = sorted_indices[0]
                keep_boxes.append(box_id)

                # Compute IoU of the picked box with the rest
                ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

                # Remove boxes with IoU over the threshold
                keep_indices = np.where(ious < iou_threshold)[0]

                # print(keep_indices.shape, sorted_indices.shape)
                sorted_indices = sorted_indices[keep_indices + 1]

            return keep_boxes

        boxes = np.squeeze(boxes).T
        # Filter out object confidence scores below threshold
        scores = np.max(boxes[:, 4:], axis=1)
        boxes = boxes[scores > thr, :]
        scores = scores[scores > thr]
        if len(boxes) == 0:
            return []

        # Get the class with the highest confidence
        class_ids = np.argmax(boxes[:, 4:], axis=1)
        boxes = boxes[:, :4]
        input_shape = np.array([inputs["scale_factor"][0], inputs["scale_factor"][1], inputs["scale_factor"][0], inputs["scale_factor"][1]])
        boxes = np.multiply(boxes, input_shape, dtype=np.float32)
        boxes = xywh2xyxy(boxes)

        unique_class_ids = np.unique(class_ids)
        indices = []
        for class_id in unique_class_ids:
            class_indices = np.where(class_ids == class_id)[0]
            class_boxes = boxes[class_indices, :]
            class_scores = scores[class_indices]
            class_keep_boxes = iou_filter(class_boxes, class_scores, 0.2)
            indices.extend(class_indices[class_keep_boxes])

        return [{
            "type": self.label_list[class_ids[i]].lower(),
            "bbox": [float(t) for t in boxes[i].tolist()],
            "score": float(scores[i])
        } for i in indices]

    def __call__(self, image_list, thr=0.7, batch_size=16):
        res = []
        imgs = []
        for i in range(len(image_list)):
            if not isinstance(image_list[i], np.ndarray):
                imgs.append(np.array(image_list[i]))
            else:
                imgs.append(image_list[i])

        batch_loop_cnt = math.ceil(float(len(imgs)) / batch_size)
        for i in range(batch_loop_cnt):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(imgs))
            batch_image_list = imgs[start_index:end_index]
            inputs = self.preprocess(batch_image_list)
            logging.debug("preprocess")
            for ins in inputs:
                bb = self.postprocess(self.ort_sess.run(None, {k:v for k,v in ins.items() if k in self.input_names}, self.run_options)[0], ins, thr)
                res.append(bb)

        #seeit.save_results(image_list, res, self.label_list, threshold=thr)

        return res



