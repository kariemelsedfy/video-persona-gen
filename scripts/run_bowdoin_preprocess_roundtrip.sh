#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_bowdoin_preprocess_roundtrip.sh \
    --identity-id hdtf_cmr \
    --local-raw-dir data/raw/hdtf_cmr

Upload a local raw identity directory to Bowdoin scratch, submit the tracked
preprocess Slurm job from a fresh scratch clone, wait for completion, and
download a small local inspection bundle.

Required:
  --identity-id    Identity id to pass to scripts/preprocess_dataset.py.
  --local-raw-dir  Local directory containing raw source videos.

Optional:
  --local-dir               Local output directory. Default: outputs/bowdoin_preprocess/job-<jobid>
  --remote-repo-url         Git URL to clone on Bowdoin. Default: https://github.com/kariemelsedfy/video-persona-gen.git
  --git-ref                 Git ref to clone on Bowdoin. Default: main
  --liveportrait-env        Remote conda env root. Default: /home/kelsedfy/video-persona-gen/.conda/liveportrait
  --job-script              Remote sbatch script relative to repo root. Default: slurm/preprocess.sbatch
  --job-name                Slurm job name override. Default: preprocess-fetch
  --partition               Slurm partition. Default: main
  --cpus                    CPUs per task override. Default: 8
  --mem                     Memory override. Default: 32G
  --time                    Time limit override. Default: 08:00:00
  --remote-storage-root     Persistent Bowdoin storage root. Default: /mnt/hpc/tmp/<user>/video-persona-gen
  --remote-raw-dir          Remote raw dataset directory. Default: <remote-storage-root>/data/raw/<identity-id>
  --remote-processed-root   Remote processed root. Default: <remote-storage-root>/data/processed
  --remote-user             Remote HPC username. Defaults to BOWDOIN_HPC_USER from .env.hpc.local.
  --poll-seconds            Poll interval while waiting for completion. Default: 10.
  --timeout                 Max seconds to wait for job completion. Default: 3600.
  --fps                     Target output FPS. Default: 25.0
  --audio-sample-rate       Target audio sample rate. Default: 16000
  --split                   Split label for newly preprocessed clips. Default: train
  --face-margin             Face crop margin. Default: 0.2
  --disable-center-crop-fallback  Disable fallback center crop.
  --overwrite               Replace existing remote raw and processed identity data.
  --keep-remote-clone       Leave the remote scratch clone in place after download.
  --extra-arg               Extra argument to forward to scripts/preprocess_dataset.py. Repeatable.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_helper="$repo_root/scripts/run_bowdoin_hpc_command.sh"
upload_script="$repo_root/scripts/upload_to_bowdoin.sh"
fetch_script="$repo_root/scripts/fetch_bowdoin_preprocess_output.sh"

identity_id=""
local_raw_dir=""
local_dir=""
remote_repo_url="https://github.com/kariemelsedfy/video-persona-gen.git"
git_ref="main"
liveportrait_env="/home/kelsedfy/video-persona-gen/.conda/liveportrait"
job_script="slurm/preprocess.sbatch"
job_name="preprocess-fetch"
partition="main"
cpus="8"
mem="32G"
time_limit="08:00:00"
remote_storage_root=""
remote_raw_dir=""
remote_processed_root=""
remote_user=""
poll_seconds=10
timeout_seconds=3600
fps="25.0"
audio_sample_rate="16000"
split_label="train"
face_margin="0.2"
disable_center_crop_fallback=0
overwrite=0
keep_remote_clone=0
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --identity-id)
      identity_id="${2:?missing value for --identity-id}"
      shift 2
      ;;
    --local-raw-dir)
      local_raw_dir="${2:?missing value for --local-raw-dir}"
      shift 2
      ;;
    --local-dir)
      local_dir="${2:?missing value for --local-dir}"
      shift 2
      ;;
    --remote-repo-url)
      remote_repo_url="${2:?missing value for --remote-repo-url}"
      shift 2
      ;;
    --git-ref)
      git_ref="${2:?missing value for --git-ref}"
      shift 2
      ;;
    --liveportrait-env)
      liveportrait_env="${2:?missing value for --liveportrait-env}"
      shift 2
      ;;
    --job-script)
      job_script="${2:?missing value for --job-script}"
      shift 2
      ;;
    --job-name)
      job_name="${2:?missing value for --job-name}"
      shift 2
      ;;
    --partition)
      partition="${2:?missing value for --partition}"
      shift 2
      ;;
    --cpus)
      cpus="${2:?missing value for --cpus}"
      shift 2
      ;;
    --mem)
      mem="${2:?missing value for --mem}"
      shift 2
      ;;
    --time)
      time_limit="${2:?missing value for --time}"
      shift 2
      ;;
    --remote-storage-root)
      remote_storage_root="${2:?missing value for --remote-storage-root}"
      shift 2
      ;;
    --remote-raw-dir)
      remote_raw_dir="${2:?missing value for --remote-raw-dir}"
      shift 2
      ;;
    --remote-processed-root)
      remote_processed_root="${2:?missing value for --remote-processed-root}"
      shift 2
      ;;
    --remote-user)
      remote_user="${2:?missing value for --remote-user}"
      shift 2
      ;;
    --poll-seconds)
      poll_seconds="${2:?missing value for --poll-seconds}"
      shift 2
      ;;
    --timeout)
      timeout_seconds="${2:?missing value for --timeout}"
      shift 2
      ;;
    --fps)
      fps="${2:?missing value for --fps}"
      shift 2
      ;;
    --audio-sample-rate)
      audio_sample_rate="${2:?missing value for --audio-sample-rate}"
      shift 2
      ;;
    --split)
      split_label="${2:?missing value for --split}"
      shift 2
      ;;
    --face-margin)
      face_margin="${2:?missing value for --face-margin}"
      shift 2
      ;;
    --disable-center-crop-fallback)
      disable_center_crop_fallback=1
      shift
      ;;
    --overwrite)
      overwrite=1
      shift
      ;;
    --keep-remote-clone)
      keep_remote_clone=1
      shift
      ;;
    --extra-arg)
      extra_args+=("${2:?missing value for --extra-arg}")
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$identity_id" || -z "$local_raw_dir" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -d "$local_raw_dir" ]]; then
  echo "Local raw directory not found: $local_raw_dir" >&2
  exit 1
fi

if [[ -z "$remote_user" && -f "$repo_root/.env.hpc.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$repo_root/.env.hpc.local"
  set +a
  remote_user="${BOWDOIN_HPC_USER:-}"
fi

if [[ -z "$remote_user" ]]; then
  echo "Unable to determine remote user. Pass --remote-user or define BOWDOIN_HPC_USER in .env.hpc.local." >&2
  exit 1
fi

if [[ -z "$remote_storage_root" ]]; then
  remote_storage_root="/mnt/hpc/tmp/${remote_user}/video-persona-gen"
fi

if [[ -z "$remote_raw_dir" ]]; then
  remote_raw_dir="$remote_storage_root/data/raw/$identity_id"
fi

if [[ -z "$remote_processed_root" ]]; then
  remote_processed_root="$remote_storage_root/data/processed"
fi

remote_identity_dir="$remote_processed_root/$identity_id"

local_inputs=()
while IFS= read -r local_input; do
  local_inputs+=("$local_input")
done < <(find "$local_raw_dir" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.avi' -o -iname '*.mkv' \) | sort)
if (( ${#local_inputs[@]} == 0 )); then
  echo "No supported video files were found in $local_raw_dir" >&2
  exit 1
fi

shell_quote() {
  printf "%q" "$1"
}

remote_input_args=""
for local_input in "${local_inputs[@]}"; do
  remote_input_path="$remote_raw_dir/$(basename "$local_input")"
  remote_input_args+=" $(shell_quote "$remote_input_path")"
done

preprocess_args=(
  --identity-id "$identity_id"
  --fps "$fps"
  --audio-sample-rate "$audio_sample_rate"
  --split "$split_label"
  --face-margin "$face_margin"
)

if (( disable_center_crop_fallback == 1 )); then
  preprocess_args+=(--disable-center-crop-fallback)
fi

if (( overwrite == 1 )); then
  preprocess_args+=(--overwrite)
fi

if (( ${#extra_args[@]} > 0 )); then
  for extra_arg in "${extra_args[@]}"; do
    preprocess_args+=("$extra_arg")
  done
fi

preprocess_args_string=""
for arg in "${preprocess_args[@]}"; do
  preprocess_args_string+=" $(shell_quote "$arg")"
done

upload_args=(
  --local-path "$local_raw_dir"
  --remote-path "$remote_raw_dir"
)
if (( overwrite == 1 )); then
  upload_args+=(--overwrite)
fi

"$upload_script" "${upload_args[@]}"

run_slug="preprocess-$(date +%Y%m%d-%H%M%S)-$RANDOM"
remote_clone_root="$remote_storage_root/verifications/$run_slug/repo"

if (( overwrite == 1 )); then
  "$remote_helper" --command "rm -rf $(shell_quote "$remote_identity_dir") && mkdir -p $(shell_quote "$remote_identity_dir")" >/dev/null
else
  "$remote_helper" --command "if [ -e $(shell_quote "$remote_identity_dir") ]; then echo 'Remote processed identity already exists: $(shell_quote "$remote_identity_dir")' >&2; exit 3; fi; mkdir -p $(shell_quote "$remote_identity_dir")" >/dev/null
fi

remote_command=$(
  cat <<EOF
set -euo pipefail
mkdir -p $(shell_quote "$(dirname "$remote_clone_root")")
git clone --branch $(shell_quote "$git_ref") --single-branch $(shell_quote "$remote_repo_url") $(shell_quote "$remote_clone_root")
cd $(shell_quote "$remote_clone_root")
job_id=\$(REPO_ROOT=$(shell_quote "$remote_clone_root") LIVEPORTRAIT_ENV=$(shell_quote "$liveportrait_env") PYTHON_BIN=$(shell_quote "$liveportrait_env/bin/python") REMOTE_STORAGE_ROOT=$(shell_quote "$remote_storage_root") PROCESSED_ROOT=$(shell_quote "$remote_processed_root") sbatch --parsable -J $(shell_quote "$job_name") -p $(shell_quote "$partition") -c $(shell_quote "$cpus") --mem=$(shell_quote "$mem") -t $(shell_quote "$time_limit") --output=$(shell_quote "$remote_identity_dir/slurm-%j.out") $(shell_quote "$job_script") --input$remote_input_args$preprocess_args_string)
job_id=\${job_id%%;*}
printf 'job_id=%s\n' "\$job_id"
printf 'remote_identity_dir=%s\n' $(shell_quote "$remote_identity_dir")
printf 'repo_clone_root=%s\n' $(shell_quote "$remote_clone_root")
EOF
)

submit_output="$("$remote_helper" --command "$remote_command")"
job_id="$(printf '%s\n' "$submit_output" | tr -d '\r' | awk -F= '$1 == "job_id" {print $2}' | tail -n 1)"
reported_identity_dir="$(printf '%s\n' "$submit_output" | tr -d '\r' | awk -F= '$1 == "remote_identity_dir" {print $2}' | tail -n 1)"
reported_clone_root="$(printf '%s\n' "$submit_output" | tr -d '\r' | awk -F= '$1 == "repo_clone_root" {print $2}' | tail -n 1)"

if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Unexpected submission output: $submit_output" >&2
  exit 1
fi

if [[ -z "$reported_identity_dir" || -z "$reported_clone_root" ]]; then
  echo "Submission output was missing remote paths: $submit_output" >&2
  exit 1
fi

if [[ -z "$local_dir" ]]; then
  local_dir="$repo_root/outputs/bowdoin_preprocess/job-$job_id"
fi

echo "Submitted Bowdoin job $job_id"
echo "Local download target: $local_dir"
echo "Remote identity dir: $reported_identity_dir"
echo "Remote scratch clone: $reported_clone_root"

"$fetch_script" \
  --job-id "$job_id" \
  --remote-identity-dir "$reported_identity_dir" \
  --local-dir "$local_dir" \
  --remote-user "$remote_user" \
  --poll-seconds "$poll_seconds" \
  --timeout "$timeout_seconds"

if (( keep_remote_clone == 0 )); then
  "$remote_helper" --command "rm -rf $(shell_quote "$reported_clone_root")" >/dev/null 2>&1 || true
fi

echo "Preprocess round-trip complete for Bowdoin job $job_id"
