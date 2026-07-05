#!/bin/bash
#SBATCH --job-name=avagen-preprocess
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/tmp/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
LIVEPORTRAIT_ENV="${LIVEPORTRAIT_ENV:-$REPO_ROOT/.conda/liveportrait}"
PYTHON_BIN="${PYTHON_BIN:-$LIVEPORTRAIT_ENV/bin/python}"
REMOTE_STORAGE_ROOT="${REMOTE_STORAGE_ROOT:-/mnt/hpc/tmp/${USER}/video-persona-gen}"
PROCESSED_ROOT="${PROCESSED_ROOT:-$REMOTE_STORAGE_ROOT/data/processed}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing python executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$PROCESSED_ROOT"
cd "$REPO_ROOT"

"$PYTHON_BIN" scripts/preprocess_dataset.py --output-root "$PROCESSED_ROOT" "$@"
