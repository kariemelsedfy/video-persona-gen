#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_bowdoin_predicted_render_roundtrip.sh \
    --checkpoint-path /mnt/hpc/tmp/<user>/video-persona-gen/.../best.pt \
    --manifest-path /mnt/hpc/tmp/<user>/video-persona-gen/data/processed/<identity>/manifest.jsonl \
    [options]

Submit the tracked Bowdoin predicted-render Slurm job from a fresh scratch clone,
wait for completion from the local side, and download the persisted output root.

Required:
  --checkpoint-path  Remote Bowdoin checkpoint path for scripts/render_predicted_motion.py.
  --manifest-path    Remote Bowdoin manifest path for scripts/render_predicted_motion.py.

Optional:
  --local-dir              Local output directory. Default: outputs/bowdoin_predicted_render/job-<jobid>
  --model-config-path      Model config path. Relative paths resolve inside the remote scratch clone. Default: configs/train_motion_gru.yaml
  --source-path            Optional remote source portrait override.
  --clip-id                Optional clip filter. Repeat to select multiple clips.
  --device                 Predictor device override, for example auto or cuda.
  --extra-arg              Extra argument to forward to scripts/render_predicted_motion.py. Repeatable.
  --remote-repo-url        Git URL to clone on Bowdoin. Default: https://github.com/kariemelsedfy/video-persona-gen.git
  --git-ref                Git ref to clone on Bowdoin. Default: main
  --remote-liveportrait-root  Remote upstream LivePortrait checkout. Default: /home/kelsedfy/video-persona-gen/external/LivePortrait
  --liveportrait-env       Remote conda env root. Default: /home/kelsedfy/video-persona-gen/.conda/liveportrait
  --job-script             Remote sbatch script relative to repo root. Default: slurm/render_predicted_motion.sbatch
  --job-name               Slurm job name override. Default: predicted-render-fetch
  --gpu-gres               GRES request. Default: gpu:pro6000:1
  --partition              Slurm partition. Default: gpu
  --cpus                   CPUs per task override. Default: 4
  --mem                    Memory override. Default: 32G
  --time                   Time limit override. Default: 08:00:00
  --remote-storage-root    Persistent Bowdoin storage root. Default: /mnt/hpc/tmp/<user>/video-persona-gen
  --remote-user            Remote HPC username. Defaults to BOWDOIN_HPC_USER from .env.hpc.local.
  --poll-seconds           Poll interval while waiting for completion. Default: 10.
  --timeout                Max seconds to wait for job completion. Default: 3600.
  --keep-remote-clone      Leave the remote scratch clone in place after download.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_helper="$repo_root/scripts/run_bowdoin_hpc_command.sh"
fetch_script="$repo_root/scripts/fetch_bowdoin_predicted_render_output.sh"

checkpoint_path=""
manifest_path=""
local_dir=""
model_config_path="configs/train_motion_gru.yaml"
source_path=""
device=""
remote_repo_url="https://github.com/kariemelsedfy/video-persona-gen.git"
git_ref="main"
remote_liveportrait_root="/home/kelsedfy/video-persona-gen/external/LivePortrait"
liveportrait_env="/home/kelsedfy/video-persona-gen/.conda/liveportrait"
job_script="slurm/render_predicted_motion.sbatch"
job_name="predicted-render-fetch"
gpu_gres="gpu:pro6000:1"
partition="gpu"
cpus="4"
mem="32G"
time_limit="08:00:00"
remote_storage_root=""
remote_user=""
poll_seconds=10
timeout_seconds=3600
keep_remote_clone=0
clip_ids=()
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-path)
      checkpoint_path="${2:?missing value for --checkpoint-path}"
      shift 2
      ;;
    --manifest-path)
      manifest_path="${2:?missing value for --manifest-path}"
      shift 2
      ;;
    --local-dir)
      local_dir="${2:?missing value for --local-dir}"
      shift 2
      ;;
    --model-config-path)
      model_config_path="${2:?missing value for --model-config-path}"
      shift 2
      ;;
    --source-path)
      source_path="${2:?missing value for --source-path}"
      shift 2
      ;;
    --clip-id)
      clip_ids+=("${2:?missing value for --clip-id}")
      shift 2
      ;;
    --device)
      device="${2:?missing value for --device}"
      shift 2
      ;;
    --extra-arg)
      extra_args+=("${2:?missing value for --extra-arg}")
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
    --remote-liveportrait-root)
      remote_liveportrait_root="${2:?missing value for --remote-liveportrait-root}"
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
    --gpu-gres)
      gpu_gres="${2:?missing value for --gpu-gres}"
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
    --keep-remote-clone)
      keep_remote_clone=1
      shift
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

if [[ -z "$checkpoint_path" || -z "$manifest_path" ]]; then
  usage >&2
  exit 2
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

shell_quote() {
  printf "%q" "$1"
}

run_slug="predicted-render-$(date +%Y%m%d-%H%M%S)-$RANDOM"
remote_clone_root="$remote_storage_root/verifications/$run_slug/repo"
remote_output_root="$remote_storage_root/render_predicted_motion/$run_slug"
remote_model_config_path="$model_config_path"
if [[ "$remote_model_config_path" != /* ]]; then
  remote_model_config_path="$remote_clone_root/$remote_model_config_path"
fi

render_args=()
if [[ -n "$source_path" ]]; then
  render_args+=(--source "$source_path")
fi
if [[ -n "$device" ]]; then
  render_args+=(--device "$device")
fi
if (( ${#clip_ids[@]} > 0 )); then
  for clip_id in "${clip_ids[@]}"; do
    render_args+=(--clip-id "$clip_id")
  done
fi
if (( ${#extra_args[@]} > 0 )); then
  for extra_arg in "${extra_args[@]}"; do
    # Use = form so values starting with '-' (e.g. LivePortrait --driving-multiplier)
    # are not misread by argparse as the next option.
    render_args+=("--extra-arg=$extra_arg")
  done
fi

render_args_string=""
if (( ${#render_args[@]} > 0 )); then
  for arg in "${render_args[@]}"; do
    render_args_string+=" $(shell_quote "$arg")"
  done
fi

remote_command=$(
  cat <<EOF
set -euo pipefail
mkdir -p $(shell_quote "$(dirname "$remote_clone_root")") $(shell_quote "$remote_output_root")
git clone --branch $(shell_quote "$git_ref") --single-branch $(shell_quote "$remote_repo_url") $(shell_quote "$remote_clone_root")
cd $(shell_quote "$remote_clone_root")
job_id=\$(REPO_ROOT=$(shell_quote "$remote_clone_root") LIVEPORTRAIT_ENV=$(shell_quote "$liveportrait_env") LIVEPORTRAIT_ROOT=$(shell_quote "$remote_liveportrait_root") CHECKPOINT_PATH=$(shell_quote "$checkpoint_path") MODEL_CONFIG_PATH=$(shell_quote "$remote_model_config_path") MANIFEST_PATH=$(shell_quote "$manifest_path") OUTPUT_ROOT=$(shell_quote "$remote_output_root") PREDICTED_OUTPUT_ROOT=$(shell_quote "$remote_output_root/predicted_motion") sbatch --parsable -J $(shell_quote "$job_name") -p $(shell_quote "$partition") --gres=$(shell_quote "$gpu_gres") -c $(shell_quote "$cpus") --mem=$(shell_quote "$mem") -t $(shell_quote "$time_limit") --output=$(shell_quote "$remote_output_root/slurm-%j.out") $(shell_quote "$job_script")$render_args_string)
job_id=\${job_id%%;*}
printf 'job_id=%s\n' "\$job_id"
printf 'output_root=%s\n' $(shell_quote "$remote_output_root")
printf 'repo_clone_root=%s\n' $(shell_quote "$remote_clone_root")
EOF
)

submit_output="$("$remote_helper" --command "$remote_command")"
job_id="$(printf '%s\n' "$submit_output" | tr -d '\r' | awk -F= '$1 == "job_id" {print $2}' | tail -n 1)"
reported_output_root="$(printf '%s\n' "$submit_output" | tr -d '\r' | awk -F= '$1 == "output_root" {print $2}' | tail -n 1)"
reported_clone_root="$(printf '%s\n' "$submit_output" | tr -d '\r' | awk -F= '$1 == "repo_clone_root" {print $2}' | tail -n 1)"

if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Unexpected submission output: $submit_output" >&2
  exit 1
fi

if [[ -z "$reported_output_root" || -z "$reported_clone_root" ]]; then
  echo "Submission output was missing remote paths: $submit_output" >&2
  exit 1
fi

if [[ -z "$local_dir" ]]; then
  local_dir="$repo_root/outputs/bowdoin_predicted_render/job-$job_id"
fi

echo "Submitted Bowdoin job $job_id"
echo "Local download target: $local_dir"
echo "Remote output root: $reported_output_root"
echo "Remote scratch clone: $reported_clone_root"

"$fetch_script" \
  --job-id "$job_id" \
  --remote-output-root "$reported_output_root" \
  --local-dir "$local_dir" \
  --remote-user "$remote_user" \
  --poll-seconds "$poll_seconds" \
  --timeout "$timeout_seconds"

if (( keep_remote_clone == 0 )); then
  "$remote_helper" --command "rm -rf $(shell_quote "$reported_clone_root")" >/dev/null 2>&1 || true
fi

echo "Predicted-render round-trip complete for Bowdoin job $job_id"
