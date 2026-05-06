import argparse
import torch
from PIL import Image
import wandb
import open_clip
from transformers import SiglipProcessor, SiglipModel
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from autocomp_tasks import TASK_CHOICES, normalize_caption_source, normalize_task_type
from hf_benchmarks import load_hf_dataset, run_benchmarks

class ModelWrapper:
    """Base class for model wrappers."""
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_image_embeddings(self, images):
        raise NotImplementedError
        
    def compute_text_embeddings(self, texts):
        raise NotImplementedError

class OpenCLIPWrapper(ModelWrapper):
    """Wraps an OpenCLIP model."""
    def __init__(self, model_name, pretrained, device=None):
        super().__init__(model_name, device)
        self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def compute_image_embeddings(self, images):
        inputs = torch.stack([self.preprocess_val(img) for img in images]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(inputs)
        return torch.nn.functional.normalize(feats, dim=-1).cpu()

    def compute_text_embeddings(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
        return torch.nn.functional.normalize(feats, dim=-1).cpu()

class SigLIPWrapper(ModelWrapper):
    """Wraps a Hugging Face SigLIP model."""
    def __init__(self, model_name, device=None):
        super().__init__(model_name, device)
        self.text_max_length = 128
        self.processor = SiglipProcessor.from_pretrained(model_name, max_position_embeddings=128)
        self.model = SiglipModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def compute_image_embeddings(self, images):
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            embeddings = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(embeddings, dim=-1).cpu()

    def compute_text_embeddings(self, texts):
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding="max_length", max_length=self.text_max_length, 
            truncation=True).to(self.device)
            embeddings = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(embeddings, dim=-1).cpu()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a contrastive VLM on a compositional benchmark.")
    
    # Model arguments
    parser.add_argument("--model_family", type=str, required=True, choices=['clip', 'siglip'], help="Family of the model to evaluate.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model (OpenCLIP format or HF Hub ID).")
    parser.add_argument("--clip_pretrained", type=str, default='openai', help="Pretrained tag for OpenCLIP models.")

    # Data arguments
    parser.add_argument("--hf_repo_id", type=str, required=True, help="Hugging Face repo ID of the dataset.")
    parser.add_argument(
        "--task_type",
        type=normalize_task_type,
        required=True,
        choices=TASK_CHOICES,
        help="Benchmark task type: color, shape_color, position, or relative_size.",
    )
    parser.add_argument("--num_objects", type=int, required=True, help="Number of objects (N) to filter the dataset for.")
    parser.add_argument(
        "--caption_source",
        type=normalize_caption_source,
        default='minimal',
        help="Caption source to evaluate: minimal/handmade or contextual/llm.",
    )
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=32)
    
    # Logging arguments
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Auto-Comp-Eval")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    args = parser.parse_args()

    # Load data using the unified benchmark module
    data = load_hf_dataset(args.hf_repo_id, args.num_objects)
    if not data:
        print("No data loaded. Exiting.")
        return

    # Initialize the correct model wrapper
    if args.model_family == 'clip':
        model_wrapper = OpenCLIPWrapper(args.model_name, args.clip_pretrained)
    else: # siglip
        model_wrapper = SigLIPWrapper(args.model_name)

    # Initialize W&B
    if args.log_wandb:
        run_name = args.wandb_run_name or f"{args.model_family}_{args.model_name.replace('/', '--')}_{args.task_type}_N{args.num_objects}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Run benchmarks using the unified benchmark module
    run_benchmarks(
        data=data,
        model=model_wrapper,
        batch_size=args.batch_size,
        caption_source=args.caption_source,
        task_type=args.task_type,
        log_wandb=args.log_wandb
    )

    if args.log_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
