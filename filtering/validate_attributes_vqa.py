import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

sys.path.append(str(Path(__file__).resolve().parents[1]))
from autocomp_tasks import (
    COLOR_VOCAB,
    SIZE_RELATION_VOCAB,
    SPATIAL_RELATION_VOCAB,
    TASK_CHOICES,
    caption_file_suffix,
    get_item_attributes,
    get_task_spec,
    normalize_caption_source,
    normalize_task_type,
)


VQA_MODEL_ID = "google/gemma-3-12b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)
MAX_VQA_ATTEMPTS = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AttributeValidator:
    """
    Validates attribute or relation correctness in generated images using a VQA model.
    """

    def __init__(self, task_type: str, model_name: str = VQA_MODEL_ID):
        self.task_type = normalize_task_type(task_type)
        self.spec = get_task_spec(self.task_type)
        self.color_vocab = list(COLOR_VOCAB)
        self.spatial_relation_vocab = list(SPATIAL_RELATION_VOCAB)
        self.size_relation_vocab = list(SIZE_RELATION_VOCAB)
        self.processor, self.model = self._load_vqa_model(model_name)

    def _load_vqa_model(self, model_name: str):
        logger.info(f"Loading VQA model '{model_name}' on {DEVICE}...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=TORCH_DTYPE,
                device_map="auto",
            ).eval()
            processor = AutoProcessor.from_pretrained(model_name)
            logger.info("VQA model loaded successfully.")
            return processor, model
        except Exception as exc:
            logger.error(f"Failed to load VQA model: {exc}", exc_info=True)
            raise

    def _query_vqa(self, image: Image.Image, prompt: str) -> str:
        messages = [{"role": "user", "content": [prompt, image]}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=50)

        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("user\n")[-1].split("assistant\n")[-1].strip()
        logger.debug(f"Prompt: '{prompt}' | Raw VQA Answer: '{answer}'")
        return answer

    def _validate_color(self, image: Image.Image, obj: str, expected_color: str) -> bool:
        prompt = (
            f"In the image, what color is the '{obj}'? "
            f"Choose only from: {', '.join(self.color_vocab)}."
        )
        for _ in range(MAX_VQA_ATTEMPTS):
            answer = self._query_vqa(image, prompt)
            if expected_color.lower() in answer.lower():
                logger.debug(f"'{obj}': expected '{expected_color}', got '{answer}'")
                return True
            time.sleep(1)
        logger.warning(
            f"'{obj}': expected '{expected_color}', but failed after "
            f"{MAX_VQA_ATTEMPTS} attempts."
        )
        return False

    def _validate_position(self, image: Image.Image, obj1: str, obj2: str, expected_relation: str) -> bool:
        prompt = (
            f"In the image, what is the spatial relationship between the '{obj1}' "
            f"and the '{obj2}'? Choose only from: {', '.join(self.spatial_relation_vocab)}."
        )
        return self._validate_relation_answer(image, prompt, obj1, obj2, expected_relation)

    def _validate_relative_size(
        self,
        image: Image.Image,
        obj1: str,
        obj2: str,
        expected_relation: str,
    ) -> bool:
        prompt = (
            f"In the image, what is the comparative size relationship from the '{obj1}' "
            f"to the '{obj2}'? Choose only from: {', '.join(self.size_relation_vocab)}."
        )
        return self._validate_relation_answer(image, prompt, obj1, obj2, expected_relation)

    def _validate_relation_answer(
        self,
        image: Image.Image,
        prompt: str,
        obj1: str,
        obj2: str,
        expected_relation: str,
    ) -> bool:
        for _ in range(MAX_VQA_ATTEMPTS):
            answer = self._query_vqa(image, prompt)
            if expected_relation.lower() in answer.lower():
                logger.debug(f"'{obj1}'->'{obj2}': expected '{expected_relation}', got '{answer}'")
                return True
            time.sleep(1)
        logger.warning(
            f"'{obj1}'->'{obj2}': expected '{expected_relation}', but failed after "
            f"{MAX_VQA_ATTEMPTS} attempts."
        )
        return False

    def _validate_item(self, image: Image.Image, item: dict) -> bool:
        objects = item.get("objects")
        if not objects:
            return False
        attributes = get_item_attributes(item)

        if self.spec.is_attribute_binding:
            for obj, color in zip(objects, attributes):
                if not self._validate_color(image, obj, color):
                    return False
            return True

        for i, relation in enumerate(attributes):
            if self.task_type == "relative_size":
                valid = self._validate_relative_size(image, objects[i], objects[i + 1], relation)
            else:
                valid = self._validate_position(image, objects[i], objects[i + 1], relation)
            if not valid:
                return False
        return True

    def run(self, args):
        try:
            with open(args.input_json, "r", encoding="utf-8") as handle:
                data_to_check = json.load(handle)
            logger.info(
                f"Loaded {len(data_to_check)} samples from '{args.input_json}' "
                "that passed object presence."
            )
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.error(f"Failed to load input JSON: {exc}")
            return

        suffix = caption_file_suffix(args.caption_type)
        fully_verified_samples = []
        for item in tqdm(data_to_check, desc=f"Validating {self.spec.display_name} labels"):
            image_id = item.get("id")
            if image_id is None:
                continue
            image_path = Path(args.image_folder) / f"concept_{image_id:05d}_{suffix}.webp"

            if not image_path.exists():
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue

            if self._validate_item(image, item):
                fully_verified_samples.append(item)
                logger.info(f"Image {image_id} PASSED all validation stages.")

        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(fully_verified_samples, handle, indent=2, ensure_ascii=False)
        logger.info(
            f"Finished. Found {len(fully_verified_samples)} fully verified samples. "
            f"Saved to '{args.output_json}'."
        )


def main():
    parser = argparse.ArgumentParser(description="Validate image attributes or relations using a VQA model.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to JSON file that passed object presence.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument(
        "--caption_type",
        type=normalize_caption_source,
        required=True,
        help="Image set to validate: minimal/handmade or contextual/llm.",
    )
    parser.add_argument(
        "--task_type",
        type=normalize_task_type,
        required=True,
        choices=TASK_CHOICES,
        help="Task family: color, shape_color, position, or relative_size.",
    )
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the verified JSON file.")
    args = parser.parse_args()

    validator = AttributeValidator(task_type=args.task_type)
    validator.run(args)


if __name__ == "__main__":
    main()
