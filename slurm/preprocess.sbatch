#!/bin/bash
#SBATCH --job-name=avagen-preprocess
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/tmp/%x-%j.out

set -euo pipefail

REMOTE_STORAGE_ROOT="${REMOTE_STORAGE_ROOT:-/mnt/hpc/tmp/${USER}/video-persona-gen}"
PROCESSED_ROOT="${PROCESSED_ROOT:-$REMOTE_STORAGE_ROOT/data/processed}"

mkdir -p "$PROCESSED_ROOT"

python scripts/preprocess_dataset.py --output-root "$PROCESSED_ROOT" "$@"
