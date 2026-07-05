#!/bin/bash
#SBATCH --job-name=avagen-prepare-manifest
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
IDENTITY_DIR="$(dirname "$MANIFEST_PATH")"
IDENTITY_ID="${IDENTITY_ID:-$(basename "$IDENTITY_DIR")}"
PROCESSED_ROOT="${PROCESSED_ROOT:-$(dirname "$IDENTITY_DIR")}"
WORK_ROOT="${WORK_ROOT:-$REMOTE_STORAGE_ROOT/motion_template_work/$IDENTITY_ID}"
TMP_ROOT="${TMP_ROOT:-/tmp/${USER}/prepare-manifest-${SLURM_JOB_ID}}"
PERSIST_WEIGHTS_DIR="${PERSIST_WEIGHTS_DIR:-$REMOTE_STORAGE_ROOT/liveportrait_weights}"
LIVEPORTRAIT_ROOT="${LIVEPORTRAIT_ROOT:-$REPO_ROOT/external/LivePortrait}"
STAGED_LIVEPORTRAIT_ROOT="$TMP_ROOT/LivePortrait"
PYTHON_BIN="${PYTHON_BIN:-$LIVEPORTRAIT_ENV/bin/python}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-0}"
SKIP_MOTION_TEMPLATE_EXTRACTION="${SKIP_MOTION_TEMPLATE_EXTRACTION:-0}"
SKIP_SPLIT_ASSIGNMENT="${SKIP_SPLIT_ASSIGNMENT:-0}"
SKIP_AUDIO_FEATURE_EXTRACTION="${SKIP_AUDIO_FEATURE_EXTRACTION:-0}"
SKIP_MOTION_FEATURE_EXTRACTION="${SKIP_MOTION_FEATURE_EXTRACTION:-0}"
OVERWRITE_MOTION_TEMPLATES="${OVERWRITE_MOTION_TEMPLATES:-0}"
OVERWRITE_AUDIO_FEATURES="${OVERWRITE_AUDIO_FEATURES:-0}"
OVERWRITE_MOTION_FEATURES="${OVERWRITE_MOTION_FEATURES:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing python executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$WORK_ROOT"

if [[ "$SKIP_MOTION_TEMPLATE_EXTRACTION" != "1" ]]; then
  if [[ ! -d "$LIVEPORTRAIT_ROOT" ]]; then
    echo "Missing LivePortrait checkout: $LIVEPORTRAIT_ROOT" >&2
    exit 1
  fi

  if [[ ! -d "$PERSIST_WEIGHTS_DIR" ]]; then
    echo "Missing persistent LivePortrait weights: $PERSIST_WEIGHTS_DIR" >&2
    exit 1
  fi

  rm -rf "$TMP_ROOT"
  mkdir -p "$TMP_ROOT"
  cp -a "$LIVEPORTRAIT_ROOT" "$STAGED_LIVEPORTRAIT_ROOT"
  rm -rf "$STAGED_LIVEPORTRAIT_ROOT/pretrained_weights"
  ln -s "$PERSIST_WEIGHTS_DIR" "$STAGED_LIVEPORTRAIT_ROOT/pretrained_weights"
fi

echo "manifest_path=$MANIFEST_PATH"
echo "identity_id=$IDENTITY_ID"
echo "processed_root=$PROCESSED_ROOT"
echo "work_root=$WORK_ROOT"

if [[ "$SKIP_MOTION_TEMPLATE_EXTRACTION" != "1" ]]; then
  echo "=== step: extract_motion ==="
  motion_args=(
    --config "$REPO_ROOT/configs/extract_motion.yaml"
    --manifest-path "$MANIFEST_PATH"
    --liveportrait-root "$STAGED_LIVEPORTRAIT_ROOT"
    --python-executable "$PYTHON_BIN"
    --work-root "$WORK_ROOT"
  )
  if [[ "$OVERWRITE_MOTION_TEMPLATES" == "1" ]]; then
    motion_args+=(--overwrite)
  fi
  "$PYTHON_BIN" scripts/extract_motion.py "${motion_args[@]}"
fi

if [[ "$SKIP_SPLIT_ASSIGNMENT" != "1" ]]; then
  echo "=== step: create_splits ==="
  "$PYTHON_BIN" scripts/create_splits.py \
    --identity-id "$IDENTITY_ID" \
    --processed-root "$PROCESSED_ROOT" \
    --manifest-path "$MANIFEST_PATH" \
    --train-fraction "$TRAIN_FRACTION" \
    --val-fraction "$VAL_FRACTION" \
    --seed "$SPLIT_SEED"
fi

if [[ "$SKIP_AUDIO_FEATURE_EXTRACTION" != "1" ]]; then
  echo "=== step: extract_audio_features ==="
  audio_args=(
    --config "$REPO_ROOT/configs/extract_audio_features.yaml"
    --manifest-path "$MANIFEST_PATH"
  )
  if [[ "$OVERWRITE_AUDIO_FEATURES" == "1" ]]; then
    audio_args+=(--overwrite)
  fi
  "$PYTHON_BIN" scripts/extract_audio_features.py "${audio_args[@]}"
fi

if [[ "$SKIP_MOTION_FEATURE_EXTRACTION" != "1" ]]; then
  echo "=== step: extract_motion_features ==="
  motion_feature_args=(
    --config "$REPO_ROOT/configs/extract_motion_features.yaml"
    --manifest-path "$MANIFEST_PATH"
  )
  if [[ "$OVERWRITE_MOTION_FEATURES" == "1" ]]; then
    motion_feature_args+=(--overwrite)
  fi
  "$PYTHON_BIN" scripts/extract_motion_features.py "${motion_feature_args[@]}"
fi
