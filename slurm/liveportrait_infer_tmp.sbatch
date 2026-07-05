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
FETCH_WAIT_SECONDS="${FETCH_WAIT_SECONDS:-900}"

DEFAULT_REMOTE_STORAGE_ROOT=""
if [[ -d "/mnt/hpc/tmp/${USER}" && -w "/mnt/hpc/tmp/${USER}" ]]; then
  DEFAULT_REMOTE_STORAGE_ROOT="/mnt/hpc/tmp/${USER}/video-persona-gen"
fi

REMOTE_STORAGE_ROOT="${REMOTE_STORAGE_ROOT:-$DEFAULT_REMOTE_STORAGE_ROOT}"
PERSIST_OUTPUT_DIR="${PERSIST_OUTPUT_DIR:-}"
PERSIST_WEIGHTS_DIR="${PERSIST_WEIGHTS_DIR:-}"

if [[ -n "$REMOTE_STORAGE_ROOT" ]]; then
  mkdir -p "$REMOTE_STORAGE_ROOT"
  if [[ -z "$PERSIST_OUTPUT_DIR" ]]; then
    PERSIST_OUTPUT_DIR="$REMOTE_STORAGE_ROOT/liveportrait_runs/${SLURM_JOB_ID}"
  fi
  if [[ -z "$PERSIST_WEIGHTS_DIR" ]]; then
    PERSIST_WEIGHTS_DIR="$REMOTE_STORAGE_ROOT/liveportrait_weights"
  fi
fi

PYTHON_BIN="$LIVEPORTRAIT_ENV/bin/python"
HF_BIN="$LIVEPORTRAIT_ENV/bin/hf"
HF_LOG="$TMP_ROOT/hf.log"
INFER_LOG="$TMP_ROOT/inference.log"
OUTPUT_DIR="$TMP_ROOT/output"
STAGED_ROOT="$TMP_ROOT/LivePortrait"
STATUS_FILE="$TMP_ROOT/status.env"
OUTPUT_FILES_LIST="$TMP_ROOT/output_files.txt"
PICKUP_FILE="$TMP_ROOT/pickup.done"
PERSIST_LOG_DIR=""
PERSIST_OUTPUT_FILES_DIR=""
PERSIST_STATUS_FILE=""
PERSIST_HF_LOG=""
PERSIST_INFER_LOG=""
PERSIST_OUTPUT_FILES_LIST=""

if [[ -n "$PERSIST_OUTPUT_DIR" ]]; then
  PERSIST_LOG_DIR="$PERSIST_OUTPUT_DIR/logs"
  PERSIST_OUTPUT_FILES_DIR="$PERSIST_OUTPUT_DIR/output"
  PERSIST_STATUS_FILE="$PERSIST_LOG_DIR/status.env"
  PERSIST_HF_LOG="$PERSIST_LOG_DIR/hf.log"
  PERSIST_INFER_LOG="$PERSIST_LOG_DIR/inference.log"
  PERSIST_OUTPUT_FILES_LIST="$PERSIST_LOG_DIR/output_files.txt"
fi

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

copy_persisted_outputs() {
  if [[ -n "$PERSIST_OUTPUT_DIR" ]]; then
    mkdir -p "$PERSIST_LOG_DIR" "$PERSIST_OUTPUT_FILES_DIR" 2>/dev/null || true
    if [[ -d "$OUTPUT_DIR" ]]; then
      cp -a "$OUTPUT_DIR"/. "$PERSIST_OUTPUT_FILES_DIR"/ 2>/dev/null || true
    fi
    cp "$HF_LOG" "$PERSIST_HF_LOG" 2>/dev/null || true
    cp "$INFER_LOG" "$PERSIST_INFER_LOG" 2>/dev/null || true
    cp "$STATUS_FILE" "$PERSIST_STATUS_FILE" 2>/dev/null || true
    cp "$OUTPUT_FILES_LIST" "$PERSIST_OUTPUT_FILES_LIST" 2>/dev/null || true
  fi
}

wait_for_pickup() {
  local deadline

  if [[ "$FETCH_WAIT_SECONDS" -le 0 ]]; then
    return 0
  fi

  deadline=$((SECONDS + FETCH_WAIT_SECONDS))
  while [[ ! -f "$PICKUP_FILE" && $SECONDS -lt $deadline ]]; do
    sleep 5
  done
}

rm -rf "$TMP_ROOT"
mkdir -p "$TMP_ROOT"
cp -a "$LIVEPORTRAIT_ROOT" "$STAGED_ROOT"
rm -rf "$STAGED_ROOT/pretrained_weights"

export HF_HOME="$TMP_ROOT/hf_home"
export XDG_CACHE_HOME="$TMP_ROOT/xdg_cache"

SOURCE_PATH="$STAGED_ROOT/$SOURCE_REL"
DRIVING_PATH="$STAGED_ROOT/$DRIVING_REL"
FAILED_STAGE="none"
RUN_EXIT_CODE=0
RUN_STATE="SUCCESS"
WEIGHTS_DIR="$STAGED_ROOT/pretrained_weights"
WEIGHTS_READY_MARKER=""

if [[ -n "$PERSIST_WEIGHTS_DIR" ]]; then
  mkdir -p "$(dirname "$PERSIST_WEIGHTS_DIR")"
  WEIGHTS_DIR="$PERSIST_WEIGHTS_DIR"
  WEIGHTS_READY_MARKER="$PERSIST_WEIGHTS_DIR/.hf-download-complete"
fi

if [[ -n "$PERSIST_WEIGHTS_DIR" ]]; then
  if [[ -f "$WEIGHTS_READY_MARKER" ]]; then
    printf 'Reusing persisted weights at %s\n' "$PERSIST_WEIGHTS_DIR" >"$HF_LOG"
  else
    rm -rf "$PERSIST_WEIGHTS_DIR"
    if ! "$HF_BIN" download KlingTeam/LivePortrait --local-dir "$PERSIST_WEIGHTS_DIR" >"$HF_LOG" 2>&1; then
      FAILED_STAGE="hf_download"
      RUN_EXIT_CODE=$?
      RUN_STATE="FAILURE"
    else
      touch "$WEIGHTS_READY_MARKER"
    fi
  fi
  ln -s "$PERSIST_WEIGHTS_DIR" "$STAGED_ROOT/pretrained_weights"
elif ! "$HF_BIN" download KlingTeam/LivePortrait --local-dir "$STAGED_ROOT/pretrained_weights" >"$HF_LOG" 2>&1; then
  FAILED_STAGE="hf_download"
  RUN_EXIT_CODE=$?
  RUN_STATE="FAILURE"
fi

if [[ "$RUN_STATE" == "SUCCESS" ]]; then
  cd "$STAGED_ROOT"
  if ! "$PYTHON_BIN" inference.py -s "$SOURCE_PATH" -d "$DRIVING_PATH" -o "$OUTPUT_DIR" >"$INFER_LOG" 2>&1; then
    FAILED_STAGE="inference"
    RUN_EXIT_CODE=$?
    RUN_STATE="FAILURE"
  fi
fi

if [[ -d "$OUTPUT_DIR" ]]; then
  find "$OUTPUT_DIR" -maxdepth 2 -type f | sort >"$OUTPUT_FILES_LIST"
else
  : >"$OUTPUT_FILES_LIST"
fi

cat >"$STATUS_FILE" <<EOF
state=$RUN_STATE
failed_stage=$FAILED_STAGE
exit_code=$RUN_EXIT_CODE
job_id=$SLURM_JOB_ID
node=${SLURMD_NODENAME:-unknown}
tmp_root=$TMP_ROOT
output_dir=$OUTPUT_DIR
hf_log=$HF_LOG
infer_log=$INFER_LOG
output_files=$OUTPUT_FILES_LIST
fetch_wait_seconds=$FETCH_WAIT_SECONDS
persist_output_dir=$PERSIST_OUTPUT_DIR
persist_log_dir=$PERSIST_LOG_DIR
persist_output_files_dir=$PERSIST_OUTPUT_FILES_DIR
persist_status_file=$PERSIST_STATUS_FILE
remote_storage_root=$REMOTE_STORAGE_ROOT
weights_dir=$WEIGHTS_DIR
source_rel=$SOURCE_REL
driving_rel=$DRIVING_REL
EOF

copy_persisted_outputs

echo "tmp_root=$TMP_ROOT"
echo "output_dir=$OUTPUT_DIR"
echo "status_file=$STATUS_FILE"
echo "state=$RUN_STATE"
if [[ -n "$PERSIST_OUTPUT_DIR" ]]; then
  echo "persist_output_dir=$PERSIST_OUTPUT_DIR"
fi
if [[ -n "$REMOTE_STORAGE_ROOT" ]]; then
  echo "remote_storage_root=$REMOTE_STORAGE_ROOT"
fi
echo "weights_dir=$WEIGHTS_DIR"
cat "$OUTPUT_FILES_LIST"

wait_for_pickup

if [[ "$RUN_STATE" != "SUCCESS" ]]; then
  echo "liveportrait_tmp_run_failed=$RUN_EXIT_CODE"
  echo "hf_log_tail:"
  tail -n 40 "$HF_LOG" 2>/dev/null || true
  echo "inference_log_tail:"
  tail -n 80 "$INFER_LOG" 2>/dev/null || true
  exit "$RUN_EXIT_CODE"
fi
