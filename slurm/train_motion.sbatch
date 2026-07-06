#!/bin/bash
#SBATCH --job-name=avagen-train-motion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/hpc/tmp/%u/video-persona-gen/%x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/video-persona-gen}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/train_motion_gru.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-}"

cd "$PROJECT_ROOT"

COMMAND=("$PYTHON_BIN" scripts/train_motion.py --config "$CONFIG_PATH")
if [[ -n "$MANIFEST_PATH" ]]; then
  COMMAND+=(--manifest-path "$MANIFEST_PATH")
fi
if [[ -n "$EXPERIMENT_DIR" ]]; then
  COMMAND+=(--experiment-dir "$EXPERIMENT_DIR")
fi

"${COMMAND[@]}"
