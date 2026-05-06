"""
Utilities for evaluating contrastive VLMs on Auto-Comp benchmarks.
"""

import logging
import sys
from pathlib import Path
from typing import List

from datasets import load_dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from autocomp_tasks import (
    generate_confusion_candidates,
    generate_swap_negatives,
    get_caption,
    get_image,
    get_task_spec,
    normalize_caption_source,
    normalize_task_type,
)


logger = logging.getLogger(__name__)


def load_hf_dataset(hf_repo_id: str, num_objects: int):
    logger.info(f"Loading dataset for N={num_objects} from HF Hub: {hf_repo_id}...")
    try:
        hf_dataset = load_dataset(hf_repo_id, split="train")
        filtered_dataset = hf_dataset.filter(lambda example: example.get("num_objects") == num_objects)
        logger.info(f"Loaded and filtered {len(filtered_dataset)} entries.")
        return list(filtered_dataset)
    except Exception as exc:
        logger.error(f"Failed to load dataset '{hf_repo_id}': {exc}")
        return []


def _collect_batch_images(batch: list, caption_source: str):
    images = []
    entries = []
    for entry in batch:
        image = get_image(entry, caption_source)
        if image is None:
            continue
        images.append(image)
        entries.append(entry)
    return images, entries


def run_single_benchmark(
    benchmark_name: str,
    data: list,
    model,
    batch_size: int,
    caption_source: str,
    task_type: str,
):
    task_type = normalize_task_type(task_type)
    caption_source = normalize_caption_source(caption_source)
    spec = get_task_spec(task_type)
    logger.info(
        f"--- Starting {benchmark_name} Benchmark "
        f"({spec.display_name}, {caption_source}) ---"
    )

    results: List[bool] = []
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        images, valid_entries = _collect_batch_images(batch, caption_source)
        if not valid_entries:
            continue

        img_embs = model.compute_image_embeddings(images)
        all_candidate_captions = []
        entry_infos = []

        for image_index, entry in enumerate(valid_entries):
            positive_caption = get_caption(entry, caption_source)
            if benchmark_name == "Swap":
                negatives = generate_swap_negatives(entry, task_type, caption_source)
                candidates = [positive_caption] + negatives
                label = 0
            else:
                candidates, label = generate_confusion_candidates(entry, task_type, caption_source)
                if label < 0:
                    continue

            all_candidate_captions.extend(candidates)
            entry_infos.append(
                {"num_candidates": len(candidates), "label": label, "image_index": image_index}
            )

        if not all_candidate_captions:
            continue

        txt_embs = model.compute_text_embeddings(all_candidate_captions)

        text_offset = 0
        for info in entry_infos:
            num_cands = info["num_candidates"]
            img_emb = img_embs[info["image_index"]]
            candidate_embs = txt_embs[text_offset : text_offset + num_cands]
            sims = (img_emb @ candidate_embs.T).squeeze(0)
            pred_idx = sims.argmax().item()

            results.append(pred_idx == info["label"])
            text_offset += num_cands

    accuracy = sum(results) / len(results) if results else 0
    logger.info(f"{benchmark_name} Done: {sum(results)}/{len(results)} correct ({accuracy:.2%})")
    return accuracy


def run_benchmarks(
    data: list,
    model,
    batch_size: int,
    caption_source: str,
    task_type: str,
    log_wandb: bool,
):
    task_type = normalize_task_type(task_type)
    caption_source = normalize_caption_source(caption_source)
    swap_acc = run_single_benchmark("Swap", data, model, batch_size, caption_source, task_type)
    conf_acc = run_single_benchmark("Confusion", data, model, batch_size, caption_source, task_type)

    if log_wandb:
        try:
            import wandb

            wandb.log(
                {
                    f"{caption_source}/{task_type}/swap_accuracy": swap_acc,
                    f"{caption_source}/{task_type}/confusion_accuracy": conf_acc,
                }
            )
        except ImportError:
            logger.warning("wandb not installed. Skipping logging.")

    return {"swap_accuracy": swap_acc, "confusion_accuracy": conf_acc}
