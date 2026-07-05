#!/bin/bash
#SBATCH --job-name=avagen-motion-extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/tmp/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
LIVEPORTRAIT_ENV="${LIVEPORTRAIT_ENV:-$REPO_ROOT/.conda/liveportrait}"
REMOTE_STORAGE_ROOT="${REMOTE_STORAGE_ROOT:-/mnt/hpc/tmp/${USER}/video-persona-gen}"
PROCESSED_ROOT="${PROCESSED_ROOT:-$REMOTE_STORAGE_ROOT/data/processed}"
WORK_ROOT="${WORK_ROOT:-$REMOTE_STORAGE_ROOT/motion_template_work}"
MANIFEST_PATH="${MANIFEST_PATH:-$PROCESSED_ROOT/self_001/manifest.jsonl}"
TMP_ROOT="${TMP_ROOT:-/tmp/${USER}/extract-motion-${SLURM_JOB_ID}}"
PERSIST_WEIGHTS_DIR="${PERSIST_WEIGHTS_DIR:-$REMOTE_STORAGE_ROOT/liveportrait_weights}"
LIVEPORTRAIT_ROOT="${LIVEPORTRAIT_ROOT:-$REPO_ROOT/external/LivePortrait}"
STAGED_LIVEPORTRAIT_ROOT="$TMP_ROOT/LivePortrait"
PYTHON_BIN="$LIVEPORTRAIT_ENV/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing python executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$LIVEPORTRAIT_ROOT" ]]; then
  echo "Missing LivePortrait checkout: $LIVEPORTRAIT_ROOT" >&2
  exit 1
fi

if [[ ! -d "$PERSIST_WEIGHTS_DIR" ]]; then
  echo "Missing persistent LivePortrait weights: $PERSIST_WEIGHTS_DIR" >&2
  exit 1
fi

mkdir -p "$WORK_ROOT"
rm -rf "$TMP_ROOT"
mkdir -p "$TMP_ROOT"
cp -a "$LIVEPORTRAIT_ROOT" "$STAGED_LIVEPORTRAIT_ROOT"
rm -rf "$STAGED_LIVEPORTRAIT_ROOT/pretrained_weights"
ln -s "$PERSIST_WEIGHTS_DIR" "$STAGED_LIVEPORTRAIT_ROOT/pretrained_weights"

"$PYTHON_BIN" scripts/extract_motion.py \
  --config "$REPO_ROOT/configs/extract_motion.yaml" \
  --manifest-path "$MANIFEST_PATH" \
  --liveportrait-root "$STAGED_LIVEPORTRAIT_ROOT" \
  --python-executable "$PYTHON_BIN" \
  --work-root "$WORK_ROOT" \
  "$@"
