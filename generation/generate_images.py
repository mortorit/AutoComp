import torch
import os
from diffusers import StableDiffusion3Pipeline
from pathlib import Path
import json
import argparse
import math
import logging
from tqdm import tqdm

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion 3 from a caption file.")
    
    # File I/O Arguments
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to the input JSON file containing the captions.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base directory to save output images. 'handmade' (Minimal) and 'llm' (Contextual) subfolders will be created here.")

    # Distribution/Batching Arguments
    parser.add_argument("--segment_index", type=int, required=True,
                        help="Index of the data segment to process (e.g., 0).")
    parser.add_argument("--total_segments", type=int, required=True,
                        help="Total number of data segments the input is divided into.")
    parser.add_argument("--batch_size_multiplier", type=int, default=16,
                        help="Number of images per GPU per batch. Total batch size = num_gpus * this_value.")
    
    # Hardware/Model Arguments
    parser.add_argument("--model_repo", type=str, default="stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--gpu_memory", type=str, default="78GiB",
                        help="Max memory per GPU for 'balanced' device_map (e.g., '38GiB' for 40GB VRAM).")

    return parser.parse_args()

def get_device_map(max_mem_str: str) -> dict | None:
    """Creates a memory map for multi-GPU balanced loading."""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return None
    logger.info(f"Found {num_gpus} GPUs. Max memory per GPU for balanced mode: {max_mem_str}.")
    return {i: max_mem_str for i in range(num_gpus)}

def generate_and_save_images(prompts: list, output_paths: list, seeds: list, pipe: StableDiffusion3Pipeline):
    """Generates and saves a batch of images."""
    if not prompts:
        return

    logger.info(f"Generating batch of {len(prompts)} images...")
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        generators = [torch.Generator(device=gen_device).manual_seed(s) for s in seeds]
        images = pipe(
            prompt=prompts,
            num_inference_steps=28,
            guidance_scale=4.5,
            height=1024,
            width=1024,
            generator=generators,
        ).images

        for img, path, seed in zip(images, output_paths, seeds):
            img.save(path)
            logger.debug(f"Saved: {path} (Seed: {seed})")
    except Exception as e:
        logger.error(f"Error during batch generation: {e}", exc_info=True)

def main():
    args = parse_arguments()

    # --- Setup Directories ---
    base_output_dir = Path(args.output_dir)
    handmade_dir = base_output_dir / "handmade"
    llm_dir = base_output_dir / "llm"
    handmade_dir.mkdir(parents=True, exist_ok=True)
    llm_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Outputting images to: {base_output_dir}")

    # --- Load Model ---
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map_config = get_device_map(args.gpu_memory)
    
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_repo,
        torch_dtype=dtype,
        device_map="balanced" if device_map_config else None,
        max_memory=device_map_config
    )
    if not device_map_config:
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Model '{args.model_repo}' loaded.")

    # --- Load and Slice Data ---
    try:
        with open(args.input_json, 'r') as f:
            all_caption_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Could not load or parse input JSON: {args.input_json}. Error: {e}")
        return

    total_items = len(all_caption_data)
    items_per_segment = math.ceil(total_items / args.total_segments)
    start_index = args.segment_index * items_per_segment
    end_index = min((start_index + items_per_segment), total_items)
    segment_data = all_caption_data[start_index:end_index]
    
    if not segment_data:
        logger.warning(f"No data for segment {args.segment_index}. Exiting.")
        return
    logger.info(f"Processing segment {args.segment_index + 1}/{args.total_segments}: items {start_index} to {end_index-1}.")

    # --- Process in Batches ---
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    batch_size = num_gpus * args.batch_size_multiplier
    logger.info(f"Effective batch size: {batch_size}")

    prompts_batch, paths_batch, seeds_batch = [], [], []
    for item in tqdm(segment_data, desc=f"Segment {args.segment_index}"):
        concept_id = f"concept_{item.get('id', 'unknown'):05d}"
        handmade_caption = item.get('handmade_caption') or item.get('minimal_caption')
        llm_caption = item.get('llm_caption') or item.get('contextual_caption')

        if not all([handmade_caption, llm_caption]):
            logger.warning(f"Skipping item with missing caption: {item}")
            continue

        seed = torch.randint(0, 2**32, (1,)).item()
        prompts_batch.extend([handmade_caption, llm_caption])
        paths_batch.extend([handmade_dir / f"{concept_id}_handmade.webp", llm_dir / f"{concept_id}_llm.webp"])
        seeds_batch.extend([seed, seed])

        if len(prompts_batch) >= batch_size:
            generate_and_save_images(prompts_batch, paths_batch, seeds_batch, pipe)
            prompts_batch, paths_batch, seeds_batch = [], [], []
    
    # Process any remaining items in the last batch
    if prompts_batch:
        generate_and_save_images(prompts_batch, paths_batch, seeds_batch, pipe)

    logger.info(f"Segment {args.segment_index + 1}/{args.total_segments} processing complete!")

if __name__ == "__main__":
    main()
