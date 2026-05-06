import argparse
import json
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm
import logging
import cv2

# --- Configuration ---
GROUNDING_MODEL_ID = 'IDEA-Research/grounding-dino-tiny'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CAPTION_TYPE_ALIASES = {
    "minimal": "handmade",
    "handmade": "handmade",
    "contextual": "llm",
    "llm": "llm",
}


def normalize_caption_type(caption_type: str) -> str:
    key = caption_type.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in CAPTION_TYPE_ALIASES:
        raise ValueError("caption_type must be minimal/handmade or contextual/llm")
    return CAPTION_TYPE_ALIASES[key]


class ObjectPresenceEvaluator:
    """
    Evaluates object presence and optionally background whiteness in images
    based on captions from a JSON file.
    """
    def __init__(self, grounding_model_name: str = GROUNDING_MODEL_ID):
        self.processor, self.model = self._load_grounding_model(grounding_model_name)
        
        # Define special-case object handling
        self.pair_objects = {
            'sneakers', 'slippers', 'boots', 'gloves',
            'sandals', 'high heels', 'leather shoes', 'chopsticks'
        }
        self.unbounded_objects = {'cleaning products'}

    def _load_grounding_model(self, model_name: str):
        """Loads the Grounding DINO model and processor."""
        logger.info(f"Loading grounding model '{model_name}' on {DEVICE}...")
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(DEVICE)
            logger.info("Grounding model loaded successfully.")
            return processor, model
        except Exception as e:
            logger.error(f"Failed to load grounding model: {e}", exc_info=True)
            raise

    def _check_object_count(self, count: int, obj_name: str) -> bool:
        """Applies rules to determine if the object count is valid."""
        if obj_name in self.unbounded_objects:
            return count >= 1
        elif obj_name in self.pair_objects:
            return 1 <= count <= 2
        else:
            return count == 1

    def _check_background_whiteness(self, image: np.ndarray, object_mask: np.ndarray) -> bool:
        """
        Checks if the background of an image is acceptably white using a generated mask.
        Uses hyperparameters from the user's test script.
        """
        brightness_threshold = 190
        white_tolerance = 0.7

        # Invert the mask to get the background
        background_mask = cv2.bitwise_not(object_mask)

        # Convert the original image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Select only the background pixels
        background_pixels = gray_image[background_mask != 0]

        if background_pixels.size == 0:
            return True # No background to fail the test

        white_pixel_count = np.sum(background_pixels > brightness_threshold)
        white_ratio = white_pixel_count / background_pixels.size

        return white_ratio >= white_tolerance

    def run_evaluation(self, args):
        """Runs the full evaluation pipeline."""
        try:
            with open(args.input_json, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load JSON file at {args.input_json}: {e}")
            return

        verified_samples = []
        for item in tqdm(all_data, desc="Evaluating Images"):
            image_id, objects = item.get("id"), item.get("objects")
            if image_id is None or objects is None:
                continue

            caption_type = normalize_caption_type(args.caption_type)
            image_path = Path(args.image_folder) / f"concept_{image_id:05d}_{caption_type}.webp"
            if not image_path.exists():
                continue

            try:
                image_pil = Image.open(image_path).convert("RGB")
                image_np = np.array(image_pil)
            except Exception as e:
                logger.error(f"Error opening image {image_path}: {e}")
                continue

            # --- Step 1: Object Presence and Count Check ---
            prompt = ". ".join(objects) + "."
            inputs = self.processor(images=image_pil, text=prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            post_processed = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=0.4, text_threshold=0.3,
                target_sizes=[image_pil.size[::-1]]
            )

            detected_labels = [label.lower() for label in post_processed[0]["labels"]]
            
            objects_ok = True
            for obj in objects:
                count = detected_labels.count(obj.lower())
                if not self._check_object_count(count, obj.lower()):
                    objects_ok = False
                    break
            
            if not objects_ok:
                continue

            # --- Step 2: Optional White Background Check ---
            background_ok = True
            if args.check_background:
                # Create a mask from all detected bounding boxes
                object_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
                boxes = post_processed[0]["boxes"]
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(object_mask, (x1, y1), (x2, y2), 255, -1)
                
                if not self._check_background_whiteness(image_np, object_mask):
                    background_ok = False
            
            if background_ok:
                verified_samples.append(item)

        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(verified_samples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation complete. Found {len(verified_samples)} valid samples. Saved to '{args.output_json}'")


def main():
    parser = argparse.ArgumentParser(description="Evaluate object presence and optional background whiteness in generated images.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--caption_type", type=normalize_caption_type, required=True, help="Image set to evaluate: minimal/handmade or contextual/llm.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the verified JSON file.")
    parser.add_argument("--check_background", action='store_true', help="Enable the white background check (for Minimal images).")
    args = parser.parse_args()

    evaluator = ObjectPresenceEvaluator()
    evaluator.run_evaluation(args)

if __name__ == "__main__":
    main()
