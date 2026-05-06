import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from autocomp_tasks import (
    TASK_CHOICES,
    generate_confusion_negatives,
    generate_swap_negatives,
    normalize_caption_source,
    normalize_task_type,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HardNegativeGenerator:
    """
    Generates Swap and Confusion hard negatives for any configured Auto-Comp task.
    """

    def __init__(self, task_type: str, caption_source: str = "minimal"):
        self.task_type = normalize_task_type(task_type)
        self.caption_source = normalize_caption_source(caption_source)
        logger.info(
            f"Initialized HardNegativeGenerator for task '{self.task_type}' "
            f"and caption source '{self.caption_source}'."
        )

    def _generate_swap_negatives(self, item: Dict) -> List[str]:
        return generate_swap_negatives(item, self.task_type, self.caption_source)

    def _generate_confusion_negatives(self, item: Dict) -> List[str]:
        return generate_confusion_negatives(item, self.task_type, self.caption_source)

    def process_file(self, input_path: str, output_path: str):
        try:
            with open(input_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.error(f"Failed to load input file '{input_path}': {exc}")
            return

        augmented_data = []
        for item in tqdm(data, desc="Generating Hard Negatives"):
            item["swap_negatives"] = self._generate_swap_negatives(item)
            item["confusion_negatives"] = self._generate_confusion_negatives(item)
            item["negative_caption_source"] = self.caption_source
            augmented_data.append(item)

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(augmented_data, handle, indent=2, ensure_ascii=False)

        logger.info(
            f"Processing complete. Saved {len(augmented_data)} items with hard negatives "
            f"to '{output_path}'."
        )


def main():
    parser = argparse.ArgumentParser(description="Generate and save hard negatives for compositional benchmarks.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file of positive samples.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the output JSON file with negatives.")
    parser.add_argument(
        "--task_type",
        type=normalize_task_type,
        required=True,
        choices=TASK_CHOICES,
        help="Task family: color, shape_color, position, or relative_size.",
    )
    parser.add_argument(
        "--caption_source",
        type=normalize_caption_source,
        default="minimal",
        help="Caption source to perturb: minimal/handmade or contextual/llm.",
    )
    args = parser.parse_args()

    generator = HardNegativeGenerator(task_type=args.task_type, caption_source=args.caption_source)
    generator.process_file(args.input_json, args.output_json)


if __name__ == "__main__":
    main()
