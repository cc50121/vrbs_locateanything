# VRBS LocateAnything Tag Selection

This project reuses the VRBS video-understanding setup from:

`/mnt/si00068187c7/default/myc/code/VRBS/VRBS_vllm-qwen3.6_eval_boxes-original_final-version.py`

Pipeline:

1. Run the first-round video-understanding prompt with the distilled VLM.
2. Extract a target category / grounding phrase from each user question with a small LM when configured, or a deterministic fallback.
3. Run LocateAnything on each test scene to get 2D proposals.
4. Render proposal boxes with numeric tags on the scene image.
5. Ask the VLM, using the first-round video context and the inquirer image, which tag is correct.
6. Convert the selected tag back to a 2D bbox and evaluate with IoU/precision.

## Quick Start On The GPU Server

```bash
cd /mnt/si00068187c7/default/myc/projects/vrbs_locateanything
source /opt/conda/etc/profile.d/conda.sh
conda activate Qwen3-vl

export PYTHONPATH=/mnt/si00068187c7/default/myc/code/Eagle/Embodied:$PYTHONPATH
python run_pipeline.py --max-queries 1 --hf-endpoint https://hf-mirror.com --disable-hf-xet
```

Default important paths:

```text
base data:     /mnt/si00068187c7/default/myc/data/VRBS/input/dengnan
VLM model:     /mnt/si00068187c7/default/myc/models/Qwen3.6-35B-A3B/models
small LM:      /mnt/si00068187c7/default/myc/models/Qwen3.5-0.8B
small LM deps: /mnt/si00068187c7/default/myc/projects/vrbs_locateanything/.deps/qwen35_transformers
Eagle repo:    /mnt/si00068187c7/default/myc/code/Eagle/Embodied
outputs:       /mnt/si00068187c7/default/myc/data/VRBS/output/dengnan_locateanything_tag_select
```

## VS Code Debugging

Open this project folder on the GPU server, then use `.vscode/launch.json`.

Available launch configs:

- `VRBS smoke: debug wiring only`: fast path that uses GT boxes and the first proposal selector, without loading LocateAnything or the VLM.
- `VRBS q1: LocateAnything + Qwen VLM`: one real q1 run with Qwen/Qwen3.5-0.8B for the LocateAnything grounding phrase, LocateAnything proposals, and Qwen3.6 VLM tag selection.

## Smoke Test Without Loading Models

This only verifies project wiring and evaluation logic. It does not measure the real method.

```bash
python run_pipeline.py \
  --max-queries 1 \
  --proposal-source gt_for_smoke \
  --selector-source first_proposal \
  --output-dir /mnt/si00068187c7/default/myc/data/VRBS/output/dengnan_locateanything_smoke
```

## Using A Small LM For Category Extraction

By default, the pipeline uses:

```text
/mnt/si00068187c7/default/myc/models/Qwen3.5-0.8B
```

You can override it with another local small language model path or Hugging Face model id:

```bash
python run_pipeline.py --small-lm-model /path/to/small-lm --max-queries 1
```

The extracted `prompt` is written to `text/category_predictions.jsonl` and passed directly to LocateAnything as the grounding phrase. If the small model cannot load, the extractor falls back to a simple rule-based parser and records `source=rules` in `text/category_predictions.jsonl`.

Qwen/Qwen3.5-0.8B needs a newer transformers version than LocateAnything currently uses, so this project loads it in an isolated subprocess with dependencies under `.deps/qwen35_transformers`. The main process still uses the conda environment's LocateAnything-compatible transformers.

If the server cannot reach Hugging Face directly, pass:

```bash
--hf-endpoint https://hf-mirror.com
```

For unstable Xet/CAS downloads, also pass `--disable-hf-xet`.
