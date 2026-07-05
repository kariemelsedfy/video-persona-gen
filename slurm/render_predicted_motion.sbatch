#!/bin/bash
#SBATCH --job-name=avagen-render-predicted-motion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/tmp/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
LIVEPORTRAIT_ENV="${LIVEPORTRAIT_ENV:-$REPO_ROOT/.conda/liveportrait}"
REMOTE_STORAGE_ROOT="${REMOTE_STORAGE_ROOT:-/mnt/hpc/tmp/${USER}/video-persona-gen}"
MANIFEST_PATH="${MANIFEST_PATH:-$REMOTE_STORAGE_ROOT/data/processed/self_001/manifest.jsonl}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-$REPO_ROOT/configs/train_motion_gru.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REMOTE_STORAGE_ROOT/render_predicted_motion}"
PREDICTED_OUTPUT_ROOT="${PREDICTED_OUTPUT_ROOT:-$OUTPUT_ROOT/predicted_motion}"
TMP_ROOT="${TMP_ROOT:-/tmp/${USER}/render-predicted-motion-${SLURM_JOB_ID}}"
PERSIST_WEIGHTS_DIR="${PERSIST_WEIGHTS_DIR:-$REMOTE_STORAGE_ROOT/liveportrait_weights}"
LIVEPORTRAIT_ROOT="${LIVEPORTRAIT_ROOT:-$REPO_ROOT/external/LivePortrait}"
STAGED_LIVEPORTRAIT_ROOT="$TMP_ROOT/LivePortrait"
PYTHON_BIN="$LIVEPORTRAIT_ENV/bin/python"

if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "Missing CHECKPOINT_PATH." >&2
  exit 1
fi

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

mkdir -p "$OUTPUT_ROOT" "$PREDICTED_OUTPUT_ROOT"
rm -rf "$TMP_ROOT"
mkdir -p "$TMP_ROOT"
cp -a "$LIVEPORTRAIT_ROOT" "$STAGED_LIVEPORTRAIT_ROOT"
rm -rf "$STAGED_LIVEPORTRAIT_ROOT/pretrained_weights"
ln -s "$PERSIST_WEIGHTS_DIR" "$STAGED_LIVEPORTRAIT_ROOT/pretrained_weights"

"$PYTHON_BIN" scripts/render_predicted_motion.py \
  --config "$REPO_ROOT/configs/render_predicted_motion.yaml" \
  --checkpoint "$CHECKPOINT_PATH" \
  --model-config "$MODEL_CONFIG_PATH" \
  --manifest-path "$MANIFEST_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --predicted-output-root "$PREDICTED_OUTPUT_ROOT" \
  --liveportrait-root "$STAGED_LIVEPORTRAIT_ROOT" \
  --python-executable "$PYTHON_BIN" \
  "$@"
