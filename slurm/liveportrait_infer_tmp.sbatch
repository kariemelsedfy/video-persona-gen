#!/bin/bash
#SBATCH --job-name=liveportrait-infer-tmp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/tmp/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
LIVEPORTRAIT_ENV="${LIVEPORTRAIT_ENV:-$REPO_ROOT/.conda/liveportrait}"
LIVEPORTRAIT_ROOT="${LIVEPORTRAIT_ROOT:-$REPO_ROOT/external/LivePortrait}"
TMP_ROOT="${TMP_ROOT:-/tmp/${USER}/liveportrait-${SLURM_JOB_ID}}"
SOURCE_REL="${SOURCE_REL:-assets/examples/source/s0.jpg}"
DRIVING_REL="${DRIVING_REL:-assets/examples/driving/d0.mp4}"
PERSIST_OUTPUT_DIR="${PERSIST_OUTPUT_DIR:-}"

PYTHON_BIN="$LIVEPORTRAIT_ENV/bin/python"
HF_BIN="$LIVEPORTRAIT_ENV/bin/hf"
HF_LOG="$TMP_ROOT/hf.log"
INFER_LOG="$TMP_ROOT/inference.log"
OUTPUT_DIR="$TMP_ROOT/output"
STAGED_ROOT="$TMP_ROOT/LivePortrait"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing python executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -x "$HF_BIN" ]]; then
  echo "Missing hf executable: $HF_BIN" >&2
  exit 1
fi

if [[ ! -d "$LIVEPORTRAIT_ROOT" ]]; then
  echo "Missing LivePortrait checkout: $LIVEPORTRAIT_ROOT" >&2
  exit 1
fi

cleanup_and_report() {
  local status=$?

  if [[ $status -ne 0 ]]; then
    echo "liveportrait_tmp_run_failed=$status"
    echo "hf_log_tail:"
    tail -n 40 "$HF_LOG" 2>/dev/null || true
    echo "inference_log_tail:"
    tail -n 80 "$INFER_LOG" 2>/dev/null || true
  fi

  if [[ -n "$PERSIST_OUTPUT_DIR" ]]; then
    mkdir -p "$PERSIST_OUTPUT_DIR" 2>/dev/null || true
    if [[ -d "$OUTPUT_DIR" ]]; then
      cp -a "$OUTPUT_DIR"/. "$PERSIST_OUTPUT_DIR"/ 2>/dev/null || true
    fi
    cp "$HF_LOG" "$INFER_LOG" "$PERSIST_OUTPUT_DIR"/ 2>/dev/null || true
  fi
}

trap cleanup_and_report EXIT

rm -rf "$TMP_ROOT"
mkdir -p "$TMP_ROOT"
cp -a "$LIVEPORTRAIT_ROOT" "$STAGED_ROOT"
rm -rf "$STAGED_ROOT/pretrained_weights"

export HF_HOME="$TMP_ROOT/hf_home"
export XDG_CACHE_HOME="$TMP_ROOT/xdg_cache"

"$HF_BIN" download KlingTeam/LivePortrait --local-dir "$STAGED_ROOT/pretrained_weights" >"$HF_LOG" 2>&1

SOURCE_PATH="$STAGED_ROOT/$SOURCE_REL"
DRIVING_PATH="$STAGED_ROOT/$DRIVING_REL"

cd "$STAGED_ROOT"
"$PYTHON_BIN" inference.py -s "$SOURCE_PATH" -d "$DRIVING_PATH" -o "$OUTPUT_DIR" >"$INFER_LOG" 2>&1

echo "tmp_root=$TMP_ROOT"
echo "output_dir=$OUTPUT_DIR"
if [[ -n "$PERSIST_OUTPUT_DIR" ]]; then
  echo "persist_output_dir=$PERSIST_OUTPUT_DIR"
fi
find "$OUTPUT_DIR" -maxdepth 2 -type f | sort
