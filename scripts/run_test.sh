#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/si00068187c7/default/myc/projects/vrbs_locateanything"
source /opt/conda/etc/profile.d/conda.sh
conda activate Qwen3-vl

export PYTHONPATH="/mnt/si00068187c7/default/myc/code/Eagle/Embodied:${PYTHONPATH:-}"
cd "${PROJECT_DIR}"

python run_pipeline.py --max-queries 1 "$@"

