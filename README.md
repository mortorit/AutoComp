# Auto-Comp

**Beyond Bag-of-Words: Diagnosing Compositional Binding Failures in Vision-Language Models**
*NeurIPS 2026*

[[Paper]](https://arxiv.org/abs/2602.02043) [[Benchmarks on Hugging Face]](https://huggingface.co/AutoComp)

Auto-Comp is a fully automated, concept-driven pipeline for generating compositional benchmarks for contrastive vision-language models such as CLIP and SigLIP. Each concept is rendered as a paired image set from a **Minimal** caption (objects on a plain white background) and an LLM-generated **Contextual** caption (the same objects in a realistic scene). This gives a controlled A/B test that separates core binding ability from visio-linguistic complexity.

For every sample, Auto-Comp produces two kinds of hard negatives:

- **Swap negatives**, which permute attributes or objects (for example, "a red cube and a blue sphere" vs. "a blue cube and a red sphere").
- **Confusion negatives**, which sample attributes and objects with replacement, producing low-entropy distractors such as repeated objects or repeated colors. Our experiments show these expose failures that go beyond the known bag-of-words limitations.

## Benchmark families

The code supports the four benchmark families described in the paper:

| Task type | Binding family | Vocabulary used | Attribute/relation slots |
| --- | --- | --- | --- |
| `color` | Attribute binding | real objects + colors | `N` colors |
| `shape_color` | Attribute binding | geometric shapes + colors | `N` colors |
| `position` | Relational binding | real objects + spatial relations | `N - 1` relations |
| `relative_size` | Relational binding | real objects + size relations | `N - 1` relations |

New families can be added through the shared task registry in `autocomp_tasks.py`. Define a task spec with an object/shape vocabulary, an attribute or relation vocabulary, and a binding type (`attribute` or `relational`). The generation, filtering, Swap, and Confusion logic will then use the correct slot count and hard-negative protocol automatically.

## Pre-generated benchmarks

All benchmarks used in the paper are available on the Hugging Face Hub at [huggingface.co/AutoComp](https://huggingface.co/AutoComp), following this naming pattern:

```text
AutoComp/AutoComp-{Minimal|Contextual|Paired}-N{2|3}-{C|SC|P|RS}
```

where `C` = color, `SC` = shape-color, `P` = position, and `RS` = relative size. For example:

```python
from datasets import load_dataset

ds = load_dataset("AutoComp/AutoComp-Minimal-N2-C")
```

If you host your own generated benchmarks under different names, edit `BENCHMARK_CONFIGS` in `evaluation/evaluate_all_models.py`.

## Setup

```bash
git clone [REPO_URL]
cd [REPO_NAME]
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the virtual environment with the command for your shell (e.g. `venv\Scripts\activate`).

## Generating a benchmark

The pipeline has three required stages and one optional stage for materializing hard negatives. Scripts accept `minimal`/`contextual` (the paper's terminology) as well as `handmade`/`llm` as legacy file-name aliases.

### Step 1: Generate captions

```bash
cd generation
python generate_captions.py \
    --task_type [color|shape_color|position|relative_size] \
    --num_objects [N] \
    --num_captions [TOTAL] \
    --output_file [path/to/captions.json]
```

For attribute-binding tasks (`color`, `shape_color`), each concept has `N` attributes. For relational tasks (`position`, `relative_size`), each concept has `N - 1` chained relations.

### Step 2: Generate images

```bash
cd generation
python generate_images.py \
    --input_json [path/to/captions.json] \
    --output_dir [path/to/save/images] \
    --segment_index 0 \
    --total_segments 4
```

`--segment_index` and `--total_segments` let you split generation across multiple jobs or GPUs. Minimal images are written to `handmade/` and Contextual images to `llm/`, matching the file names expected by the filtering scripts.

### Step 3: Validate generated data

Check object presence (and, for Minimal images, the plain background):

```bash
cd filtering
python evaluate_object_presence.py \
    --input_json [path/to/captions.json] \
    --image_folder [path/to/generated/images/handmade] \
    --caption_type minimal \
    --output_json [path/to/object_presence_verified.json] \
    --check_background
```

Check attribute or relation correctness:

```bash
cd filtering
python validate_attributes_vqa.py \
    --input_json [path/to/object_presence_verified.json] \
    --image_folder [path/to/generated/images/handmade] \
    --caption_type minimal \
    --task_type [color|shape_color|position|relative_size] \
    --output_json [path/to/fully_validated_data.json]
```

For Contextual images, use `--caption_type contextual` with the `llm/` image folder, and drop `--check_background`.

### Step 4 (optional): Store hard negatives offline

The evaluation scripts generate negatives on the fly. To save them to JSON instead:

```bash
cd generation
python generate_swap_confusion.py \
    --input_json [path/to/fully_validated_data.json] \
    --output_json [path/to/data_with_negatives.json] \
    --task_type [color|shape_color|position|relative_size] \
    --caption_source minimal
```

Swap negatives use all `N! - 1` permutations. Attribute-binding tasks permute attributes over fixed objects/shapes; relational tasks permute objects over fixed relations.

Confusion candidates are sampled with replacement from the same concept vocabulary, giving `N^(2N)` candidates for attribute binding and `N^N * (N - 1)^(N - 1)` candidates for relational binding (both counts include the positive caption).

## Evaluation

To evaluate many models across many benchmarks, configure the model and dataset lists in `evaluation/evaluate_all_models.py` and run:

```bash
cd evaluation
python evaluate_all_models.py
```

To evaluate a single model on a single benchmark:

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

## Citation

If you use Auto-Comp or our benchmarks, please cite:

```bibtex
@inproceedings{[citation_key],
  title     = {Beyond Bag-of-Words: Diagnosing Compositional Binding Failures in Vision-Language Models},
  author    = {[Author list]},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.02043}
}
```

## License

[License name]. See `LICENSE` for details.
