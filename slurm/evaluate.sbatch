#!/bin/bash
#SBATCH --job-name=avagen-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/hpc/tmp/%u/video-persona-gen/%x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/video-persona-gen}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/evaluate_motion.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
PREDICTED_ROOT="${PREDICTED_ROOT:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"

cd "$PROJECT_ROOT"

COMMAND=("$PYTHON_BIN" scripts/evaluate_motion.py --config "$CONFIG_PATH")
if [[ -n "$MANIFEST_PATH" ]]; then
  COMMAND+=(--manifest-path "$MANIFEST_PATH")
fi
if [[ -n "$PREDICTED_ROOT" ]]; then
  COMMAND+=(--predicted-root "$PREDICTED_ROOT")
fi
if [[ -n "$OUTPUT_PATH" ]]; then
  COMMAND+=(--output-path "$OUTPUT_PATH")
fi

"${COMMAND[@]}"
