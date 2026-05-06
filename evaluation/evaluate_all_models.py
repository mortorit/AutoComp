import shlex
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from autocomp_tasks import TASK_SPECS


# Define all datasets to be evaluated. Update the repo ID pattern if your hosted
# benchmark names differ from the default Auto-Comp naming convention.
BENCHMARK_CONFIGS = []
for condition, sources in (
    ("Minimal", ["minimal"]),
    ("Contextual", ["contextual"]),
    ("Paired", ["minimal", "contextual"]),
):
    for spec in TASK_SPECS.values():
        for num_objects in (2, 3):
            BENCHMARK_CONFIGS.append(
                {
                    "name": f"{condition} {spec.display_name} N={num_objects}",
                    "hf_repo_id": (
                        f"AutoComp/AutoComp-{condition}-N{num_objects}-{spec.repo_suffix}"
                    ),
                    "task_type": spec.name,
                    "num_objects": num_objects,
                    "sources": sources,
                }
            )


CLIP_MODELS = [
    {"family": "clip", "name": "ViT-B-32-quickgelu", "pretrained": "openai"},
    {"family": "clip", "name": "ViT-B-16-quickgelu", "pretrained": "openai"},
    {"family": "clip", "name": "ViT-L-14-quickgelu", "pretrained": "openai"},
    {"family": "clip", "name": "ViT-H-14-quickgelu", "pretrained": "openai"},
    {"family": "clip", "name": "ViT-H-14-378-quickgelu", "pretrained": "dfn5b"},
    {"family": "clip", "name": "ViT-SO400M-14-SigLIP-384", "pretrained": "siglip-384"},
    {"family": "clip", "name": "EVA02-B-16", "pretrained": "laion2b_s34b_b88k"},
    {"family": "clip", "name": "EVA02-E-14", "pretrained": "laion2b_s32b_b79k"},
]

SIGLIP_MODELS = [
    {"family": "siglip", "name": "google/siglip-base-patch16-224"},
    {"family": "siglip", "name": "google/siglip-base-patch16-384"},
    {"family": "siglip", "name": "google/siglip-large-patch16-256"},
    {"family": "siglip", "name": "google/siglip-large-patch16-384"},
    {"family": "siglip", "name": "google/siglip-so400m-patch14-384"},
    {"family": "siglip", "name": "google/siglip2-base-patch16-224"},
    {"family": "siglip", "name": "google/siglip2-base-patch16-256"},
    {"family": "siglip", "name": "google/siglip2-base-patch16-384"},
    {"family": "siglip", "name": "google/siglip2-large-patch16-256"},
    {"family": "siglip", "name": "google/siglip2-large-patch16-384"},
    {"family": "siglip", "name": "google/siglip2-so400m-patch14-384"},
    {"family": "siglip", "name": "google/siglip2-giant-opt-patch16-256"},
]

MODELS_TO_EVALUATE = CLIP_MODELS + SIGLIP_MODELS

BATCH_SIZE = 4
LOG_WANDB = True
WANDB_PROJECT = "Auto-Comp-Eval"


def main():
    for bench_config in BENCHMARK_CONFIGS:
        for model_config in MODELS_TO_EVALUATE:
            for source in bench_config["sources"]:
                model_family = model_config["family"]
                model_name = model_config["name"]

                print("\n" + "=" * 80)
                print(f"Queueing Evaluation for: {bench_config['name']}")
                print(f"  - Model: {model_name} ({model_family})")
                print(f"  - Caption Source: {source}")
                print("=" * 80)

                cmd_parts = [
                    "python",
                    "evaluate_model.py",
                    "--model_family",
                    model_family,
                    "--model_name",
                    model_name,
                    "--hf_repo_id",
                    bench_config["hf_repo_id"],
                    "--task_type",
                    bench_config["task_type"],
                    "--num_objects",
                    str(bench_config["num_objects"]),
                    "--caption_source",
                    source,
                    "--batch_size",
                    str(BATCH_SIZE),
                ]

                if model_family == "clip" and "pretrained" in model_config:
                    cmd_parts.extend(["--clip_pretrained", model_config["pretrained"]])

                if LOG_WANDB:
                    sanitized_name = model_name.replace("/", "--")
                    run_name = (
                        f"{sanitized_name}_{bench_config['task_type']}_"
                        f"N{bench_config['num_objects']}_{source}"
                    )
                    cmd_parts.extend(
                        [
                            "--log_wandb",
                            "--wandb_project",
                            WANDB_PROJECT,
                            "--wandb_run_name",
                            run_name,
                        ]
                    )

                print(f"Executing: {' '.join(shlex.quote(part) for part in cmd_parts)}\n")

                try:
                    subprocess.run(cmd_parts, check=True)
                except subprocess.CalledProcessError as exc:
                    print(
                        f"\n--- Evaluation FAILED for {model_name} "
                        f"with error code {exc.returncode} ---\n"
                    )


if __name__ == "__main__":
    main()
