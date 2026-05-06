import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

sys.path.append(str(Path(__file__).resolve().parents[1]))
from autocomp_tasks import TASK_CHOICES, get_article, get_task_spec, normalize_task_type


DEFAULT_GEMMA_MODEL_ID = "google/gemma-3-12b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CaptionGenerator:
    """
    Generates paired Minimal and Contextual captions for all Auto-Comp task families.
    """

    def __init__(
        self,
        task_type: str,
        model_name: str = DEFAULT_GEMMA_MODEL_ID,
        max_regeneration_attempts: int = 3,
    ):
        self.task_type = normalize_task_type(task_type)
        self.spec = get_task_spec(self.task_type)
        self.objects = list(self.spec.object_vocab)
        self.attributes = list(self.spec.attribute_vocab)
        self.max_regeneration_attempts = max_regeneration_attempts
        self._load_model_and_processor(model_name)

    def _load_model_and_processor(self, model_name: str):
        logger.info(f"Loading model '{model_name}' on {DEVICE} with dtype {TORCH_DTYPE}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=TORCH_DTYPE,
                device_map="auto",
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_name)
            if self.processor.tokenizer.pad_token_id is None:
                self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
            logger.info("Model and processor loaded successfully.")
        except Exception as exc:
            logger.error(f"Error loading model {model_name}: {exc}", exc_info=True)
            raise

    def _generate_attribute_minimal(self, objects_with_attrs: List[Tuple[str, str]]) -> str:
        caption_parts = []
        for obj, attr in objects_with_attrs:
            article = get_article(obj, attr)
            caption_parts.append(f"{article} {attr} {obj.lower()}")

        if len(caption_parts) == 1:
            return f"{caption_parts[0]} on a white background."

        last_part = caption_parts.pop()
        return ", ".join(caption_parts) + f", and {last_part} on a white background."

    def _generate_relational_minimal(self, objects: List[str], relations: List[str]) -> str:
        article1 = get_article(objects[0], objects[0])
        article2 = get_article(objects[1], objects[1])
        caption = (
            f"{article1.capitalize()} {objects[0].lower()} is "
            f"{relations[0]} {article2} {objects[1].lower()}"
        )

        for i in range(1, len(relations)):
            next_obj = objects[i + 1]
            article_next = get_article(next_obj, next_obj)
            caption += f", which is {relations[i]} {article_next} {next_obj.lower()}"

        caption += " on a white background."
        return caption

    def _is_llm_output_valid(self, caption: str, objects: List[str], attributes: List[str]) -> bool:
        if not caption:
            return False

        caption_lower = caption.lower()
        if self.spec.is_attribute_binding:
            for obj, attr in zip(objects, attributes):
                obj_l = obj.lower()
                attr_l = attr.lower()
                if not re.search(
                    rf"\b{re.escape(attr_l)}\s+(?:[a-zA-Z]+\s+)?{re.escape(obj_l)}\b",
                    caption_lower,
                ):
                    return False
            return True

        for i, relation in enumerate(attributes):
            obj1_l = objects[i].lower()
            obj2_l = objects[i + 1].lower()
            rel_l = relation.lower()
            if not re.search(
                rf"\b{re.escape(obj1_l)}\b.*?\b{re.escape(rel_l)}\b.*?\b{re.escape(obj2_l)}\b",
                caption_lower,
            ):
                return False
        return True

    def _prepare_attribute_prompt(self, objects: List[str], attributes: List[str]) -> List[Dict]:
        subject = "shapes" if self.spec.object_type == "shape" else "objects"
        obj_attr_pairs = [f"'{attr} {obj.lower()}'" for obj, attr in zip(objects, attributes)]
        obj_attr_text = ", ".join(obj_attr_pairs)

        sys_prompt = (
            "You are an expert image caption writer. Your task is to generate a single, "
            f"natural-sounding sentence accurately describing a scene with specific {subject} "
            "and their colors. You MUST include ALL specified items and their assigned colors. "
            "Generate ONLY the caption text without any preamble, explanations, or markdown."
        )

        example = (
            "for 'a purple sphere' and 'a gold pyramid', a good caption is: "
            "'A shimmering gold pyramid stands beside a smooth, purple sphere on a desert landscape.'"
            if self.spec.object_type == "shape"
            else "for 'a green chair' and 'a yellow lamp', a good caption is: "
            "'A green chair stands beside a yellow lamp in a brightly lit room.'"
        )

        user_prompt = f"""Rewrite the following concept into a natural, single-sentence image caption: {obj_attr_text}.
Keep these {subject} as the main focus.
Explicitly mention each item with its assigned color: {obj_attr_text}.
Place them in a realistic context or background.
Example: {example}
Caption:"""

        return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]

    def _prepare_relational_prompt(self, objects: List[str], relations: List[str]) -> List[Dict]:
        chain_desc = f"'{objects[0].lower()} {relations[0]} {objects[1].lower()}'"
        for i in range(1, len(relations)):
            chain_desc += f", which is then '{relations[i]} {objects[i + 1].lower()}'"

        relation_kind = (
            "comparative size relationship"
            if self.task_type == "relative_size"
            else "spatial relationship"
        )
        sys_prompt = (
            "You are an expert image caption writer. Your task is to generate a single, "
            f"natural-sounding sentence that accurately describes a scene with multiple objects "
            f"in a specific chained {relation_kind}. Generate ONLY the caption text without any "
            "preamble, explanations, or markdown."
        )

        example = (
            "for 'ladder larger than car', a good caption is: "
            "'A towering ladder leans against a house, larger than a parked car in the driveway.'"
            if self.task_type == "relative_size"
            else "for 'chair to the left of table', a good caption is: "
            "'A wooden chair is positioned to the left of a small table in a bright room.'"
        )

        user_prompt = f"""Rewrite the following chained relationship into a natural, single-sentence image caption: {chain_desc}.
Keep all objects and relation phrases exactly represented.
Place the objects in a realistic context.
Example: {example}
Caption:"""

        return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]

    def _prepare_llm_prompt(self, objects: List[str], attributes: List[str]) -> List[Dict]:
        if self.spec.is_attribute_binding:
            return self._prepare_attribute_prompt(objects, attributes)
        return self._prepare_relational_prompt(objects, attributes)

    def _generate_with_llm(self, chat_prompt: List[Dict]) -> str:
        try:
            inputs = self.processor.apply_chat_template(
                chat_prompt,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            prompt_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)
            generated_text = self.processor.decode(outputs[0][prompt_len:], skip_special_tokens=True)
            return generated_text.strip()
        except Exception as exc:
            logger.error(f"Error during LLM generation: {exc}", exc_info=True)
            return ""

    def _build_record(
        self,
        record_id: int,
        num_objects: int,
        objects: List[str],
        attributes: List[str],
        minimal_caption: str,
        contextual_caption: str,
    ) -> Dict:
        record = {
            "id": record_id,
            "task_type": self.task_type,
            "binding_type": self.spec.binding_type,
            "object_type": self.spec.object_type,
            "num_objects": num_objects,
            "objects": objects,
            "attributes": attributes,
            "minimal_caption": minimal_caption,
            "contextual_caption": contextual_caption,
            "handmade_caption": minimal_caption,
            "llm_caption": contextual_caption,
        }
        if self.spec.is_attribute_binding:
            record["colors"] = attributes
        else:
            record["relations"] = attributes
        return record

    def generate_captions(self, num_objects: int, num_captions: int, output_path: str):
        num_attributes_needed = self.spec.num_attributes(num_objects)
        if num_attributes_needed < 1:
            raise ValueError(f"{self.spec.display_name} requires at least 2 objects.")
        if num_objects > len(self.objects):
            raise ValueError(
                f"Requested {num_objects} objects, but {self.spec.display_name} only has "
                f"{len(self.objects)} available {self.spec.object_type}s."
            )
        if num_attributes_needed > len(self.attributes):
            raise ValueError(
                f"Requested {num_attributes_needed} attributes/relations, but "
                f"{self.spec.display_name} only has {len(self.attributes)} available."
            )

        results = []
        desc = f"Generating {self.spec.display_name} Captions (N={num_objects})"
        for concept_index in tqdm(range(num_captions), desc=desc):
            sampled_objects = random.sample(self.objects, num_objects)
            sampled_attributes = random.sample(self.attributes, num_attributes_needed)

            if self.spec.is_attribute_binding:
                minimal_caption = self._generate_attribute_minimal(
                    list(zip(sampled_objects, sampled_attributes))
                )
            else:
                minimal_caption = self._generate_relational_minimal(
                    sampled_objects,
                    sampled_attributes,
                )

            contextual_caption = ""
            for _ in range(self.max_regeneration_attempts):
                prompt = self._prepare_llm_prompt(sampled_objects, sampled_attributes)
                candidate_caption = self._generate_with_llm(prompt)
                if self._is_llm_output_valid(candidate_caption, sampled_objects, sampled_attributes):
                    contextual_caption = candidate_caption
                    break

            if not contextual_caption:
                logger.warning(
                    f"Failed to generate valid LLM caption for concept {concept_index + 1}. Skipping."
                )
                continue

            results.append(
                self._build_record(
                    record_id=len(results) + 1,
                    num_objects=num_objects,
                    objects=sampled_objects,
                    attributes=sampled_attributes,
                    minimal_caption=minimal_caption,
                    contextual_caption=contextual_caption,
                )
            )

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        logger.info(f"Successfully generated and saved {len(results)} captions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate compositional captions for VLM benchmarks.")
    parser.add_argument(
        "--task_type",
        type=normalize_task_type,
        required=True,
        choices=TASK_CHOICES,
        help="Task family: color, shape_color, position, or relative_size.",
    )
    parser.add_argument("--num_objects", type=int, required=True, help="The number of objects per caption.")
    parser.add_argument("--num_captions", type=int, default=1000, help="The total number of captions to generate.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSON file.")
    args = parser.parse_args()

    if not args.output_file:
        args.output_file = f"generated_captions_{args.task_type}_n{args.num_objects}.json"

    generator = CaptionGenerator(task_type=args.task_type)
    generator.generate_captions(
        num_objects=args.num_objects,
        num_captions=args.num_captions,
        output_path=args.output_file,
    )


if __name__ == "__main__":
    main()
