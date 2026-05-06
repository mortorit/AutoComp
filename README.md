# Auto-Comp Pipeline: Code Supplement

This repository contains the anonymized code supplement for the Auto-Comp pipeline, a concept-driven benchmark generator for compositional probing of contrastive vision-language models. The implementation supports the four benchmark families described in the paper:

| Task type | Binding family | Vocabulary used | Attribute/relation slots |
| --- | --- | --- | --- |
| `color` | Attribute binding | real objects + colors | `N` colors |
| `shape_color` | Attribute binding | geometric shapes + colors | `N` colors |
| `position` | Relational binding | real objects + spatial relations | `N - 1` relations |
| `relative_size` | Relational binding | real objects + size relations | `N - 1` relations |

The code generalizes through the shared task registry in `autocomp_tasks.py`. To add another benchmark family, define a new task spec with an object/shape vocabulary, an attribute or relation vocabulary, and a binding type (`attribute` or `relational`). The generation, filtering, Swap, and Confusion logic will then use the correct slot count and hard-negative protocol.

## Benchmarks on Hugging Face

Generated benchmark datasets can be hosted on the Hugging Face Hub. The default evaluation configuration assumes this naming pattern:

```text
AutoComp/AutoComp-{Minimal|Contextual|Paired}-N{2|3}-{C|SC|P|RS}
```

If your anonymized dataset repository names differ, edit `BENCHMARK_CONFIGS` in `evaluation/evaluate_all_models.py`.

## Setup

```bash
git clone [URL_TO_ANONYMOUS_REPO]
cd [REPO_NAME]
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows, use the appropriate virtual-environment activation command for your shell.

## Benchmark Generation Pipeline

The pipeline has three required stages and one optional hard-negative materialization stage. Scripts accept `minimal`/`contextual` as paper terminology and `handmade`/`llm` as legacy file-name aliases.

### Step 1: Generate Captions

```bash
cd generation
python generate_captions.py \
    --task_type [color|shape_color|position|relative_size] \
    --num_objects [N] \
    --num_captions [TOTAL] \
    --output_file [path/to/captions.json]
```

For attribute-binding tasks (`color`, `shape_color`), each concept has `N` attributes. For relational tasks (`position`, `relative_size`), each concept has `N - 1` chained relations.

### Step 2: Generate Images

```bash
cd generation
python generate_images.py \
    --input_json [path/to/captions.json] \
    --output_dir [path/to/save/images] \
    --segment_index 0 \
    --total_segments 4
```

The script writes Minimal images to `handmade/` and Contextual images to `llm/`, using the historical file suffixes expected by the filtering scripts.

### Step 3: Validate Generated Data

Object presence and optional Minimal background validation:

```bash
cd filtering
python evaluate_object_presence.py \
    --input_json [path/to/captions.json] \
    --image_folder [path/to/generated/images/handmade] \
    --caption_type minimal \
    --output_json [path/to/object_presence_verified.json] \
    --check_background
```

Attribute or relation correctness validation:

```bash
cd filtering
python validate_attributes_vqa.py \
    --input_json [path/to/object_presence_verified.json] \
    --image_folder [path/to/generated/images/handmade] \
    --caption_type minimal \
    --task_type [color|shape_color|position|relative_size] \
    --output_json [path/to/fully_validated_data.json]
```

Use `--caption_type contextual` and the `llm/` image folder for Contextual images. Keep `--check_background` only for Minimal images.

### Step 4: Optional Offline Hard Negatives

The evaluation scripts generate negatives in real time. To store them in JSON instead:

```bash
cd generation
python generate_swap_confusion.py \
    --input_json [path/to/fully_validated_data.json] \
    --output_json [path/to/data_with_negatives.json] \
    --task_type [color|shape_color|position|relative_size] \
    --caption_source minimal
```

Swap negatives use `N! - 1` permutations. Attribute-binding tasks permute attributes over fixed objects/shapes; relational tasks permute objects over fixed relations. Confusion candidates are generated with replacement from the same concept vocabulary, using `N^(2N)` candidates for attribute binding and `N^N * (N - 1)^(N - 1)` candidates for relational binding, including the positive caption.

## Evaluation

Configure model and dataset lists in `evaluation/evaluate_all_models.py`, then run:

```bash
cd evaluation
python evaluate_all_models.py
```

For a single benchmark:

```bash
cd evaluation
python evaluate_model.py \
    --model_family [clip|siglip] \
    --model_name [MODEL_NAME_OR_HF_ID] \
    --hf_repo_id [DATASET_REPO_ID] \
    --task_type [color|shape_color|position|relative_size] \
    --num_objects [2|3] \
    --caption_source [minimal|contextual]
```
