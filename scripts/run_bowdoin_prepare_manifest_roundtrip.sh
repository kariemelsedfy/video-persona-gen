#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_bowdoin_prepare_manifest_roundtrip.sh \
    --manifest-path /mnt/hpc/tmp/<user>/video-persona-gen/data/processed/<identity>/manifest.jsonl

Submit the tracked Bowdoin manifest-preparation Slurm job from a fresh scratch clone,
wait for completion from the local side, and download a small local inspection bundle.

Required:
  --manifest-path  Remote Bowdoin manifest path for slurm/prepare_dataset_manifest.sbatch.

Optional:
  --local-dir                Local output directory. Default: outputs/bowdoin_prepare_manifest/job-<jobid>
  --remote-repo-url          Git URL to clone on Bowdoin. Default: https://github.com/kariemelsedfy/video-persona-gen.git
  --git-ref                  Git ref to clone on Bowdoin. Default: main
  --remote-liveportrait-root Remote upstream LivePortrait checkout. Default: /home/kelsedfy/video-persona-gen/external/LivePortrait
  --liveportrait-env         Remote conda env root. Default: /home/kelsedfy/video-persona-gen/.conda/liveportrait
  --job-script               Remote sbatch script relative to repo root. Default: slurm/prepare_dataset_manifest.sbatch
  --job-name                 Slurm job name override. Default: prepare-manifest-fetch
  --gpu-gres                 GRES request. Default: gpu:pro6000:1
  --partition                Slurm partition. Default: gpu
  --cpus                     CPUs per task override. Default: 4
  --mem                      Memory override. Default: 64G
  --time                     Time limit override. Default: 08:00:00
  --remote-storage-root      Persistent Bowdoin storage root. Default: /mnt/hpc/tmp/<user>/video-persona-gen
  --remote-user              Remote HPC username. Defaults to BOWDOIN_HPC_USER from .env.hpc.local.
  --poll-seconds             Poll interval while waiting for completion. Default: 10.
  --timeout                  Max seconds to wait for job completion. Default: 3600.
  --keep-remote-clone        Leave the remote scratch clone in place after download.
  --skip-motion-template-extraction  Skip extract_motion.py inside the job.
  --skip-split-assignment    Skip create_splits.py inside the job.
  --skip-audio-feature-extraction    Skip extract_audio_features.py inside the job.
  --skip-motion-feature-extraction   Skip extract_motion_features.py inside the job.
  --overwrite-motion-templates       Overwrite existing motion_template.pkl outputs.
  --overwrite-audio-features         Overwrite existing audio_features.npz outputs.
  --overwrite-motion-features        Overwrite existing motion_features.npz outputs.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_helper="$repo_root/scripts/run_bowdoin_hpc_command.sh"
fetch_script="$repo_root/scripts/fetch_bowdoin_preprocess_output.sh"

manifest_path=""
local_dir=""
remote_repo_url="https://github.com/kariemelsedfy/video-persona-gen.git"
git_ref="main"
remote_liveportrait_root="/home/kelsedfy/video-persona-gen/external/LivePortrait"
liveportrait_env="/home/kelsedfy/video-persona-gen/.conda/liveportrait"
job_script="slurm/prepare_dataset_manifest.sbatch"
job_name="prepare-manifest-fetch"
gpu_gres="gpu:pro6000:1"
partition="gpu"
cpus="4"
mem="64G"
time_limit="08:00:00"
remote_storage_root=""
remote_user=""
poll_seconds=10
timeout_seconds=3600
keep_remote_clone=0
skip_motion_template_extraction=0
skip_split_assignment=0
skip_audio_feature_extraction=0
skip_motion_feature_extraction=0
overwrite_motion_templates=0
overwrite_audio_features=0
overwrite_motion_features=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest-path)
      manifest_path="${2:?missing value for --manifest-path}"
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
    --skip-motion-template-extraction)
      skip_motion_template_extraction=1
      shift
      ;;
    --skip-split-assignment)
      skip_split_assignment=1
      shift
      ;;
    --skip-audio-feature-extraction)
      skip_audio_feature_extraction=1
      shift
      ;;
    --skip-motion-feature-extraction)
      skip_motion_feature_extraction=1
      shift
      ;;
    --overwrite-motion-templates)
      overwrite_motion_templates=1
      shift
      ;;
    --overwrite-audio-features)
      overwrite_audio_features=1
      shift
      ;;
    --overwrite-motion-features)
      overwrite_motion_features=1
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

if [[ -z "$manifest_path" ]]; then
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

identity_dir="$(dirname "$manifest_path")"
run_slug="prepare-manifest-$(date +%Y%m%d-%H%M%S)-$RANDOM"
remote_clone_root="$remote_storage_root/verifications/$run_slug/repo"

job_env=(
  "REPO_ROOT=$remote_clone_root"
  "LIVEPORTRAIT_ENV=$liveportrait_env"
  "LIVEPORTRAIT_ROOT=$remote_liveportrait_root"
  "MANIFEST_PATH=$manifest_path"
  "REMOTE_STORAGE_ROOT=$remote_storage_root"
)

if (( skip_motion_template_extraction == 1 )); then
  job_env+=("SKIP_MOTION_TEMPLATE_EXTRACTION=1")
fi
if (( skip_split_assignment == 1 )); then
  job_env+=("SKIP_SPLIT_ASSIGNMENT=1")
fi
if (( skip_audio_feature_extraction == 1 )); then
  job_env+=("SKIP_AUDIO_FEATURE_EXTRACTION=1")
fi
if (( skip_motion_feature_extraction == 1 )); then
  job_env+=("SKIP_MOTION_FEATURE_EXTRACTION=1")
fi
if (( overwrite_motion_templates == 1 )); then
  job_env+=("OVERWRITE_MOTION_TEMPLATES=1")
fi
if (( overwrite_audio_features == 1 )); then
  job_env+=("OVERWRITE_AUDIO_FEATURES=1")
fi
if (( overwrite_motion_features == 1 )); then
  job_env+=("OVERWRITE_MOTION_FEATURES=1")
fi

job_env_string=""
for env_var in "${job_env[@]}"; do
  job_env_string+=" $(shell_quote "$env_var")"
done

remote_command=$(
  cat <<EOF
set -euo pipefail
mkdir -p $(shell_quote "$(dirname "$remote_clone_root")")
git clone --branch $(shell_quote "$git_ref") --single-branch $(shell_quote "$remote_repo_url") $(shell_quote "$remote_clone_root")
cd $(shell_quote "$remote_clone_root")
job_id=\$(env$job_env_string sbatch --parsable -J $(shell_quote "$job_name") -p $(shell_quote "$partition") --gres=$(shell_quote "$gpu_gres") -c $(shell_quote "$cpus") --mem=$(shell_quote "$mem") -t $(shell_quote "$time_limit") --output=$(shell_quote "$identity_dir/slurm-%j.out") $(shell_quote "$job_script"))
job_id=\${job_id%%;*}
printf 'job_id=%s\n' "\$job_id"
printf 'identity_dir=%s\n' $(shell_quote "$identity_dir")
printf 'repo_clone_root=%s\n' $(shell_quote "$remote_clone_root")
EOF
)

submit_output="$("$remote_helper" --command "$remote_command")"
job_id="$(printf '%s\n' "$submit_output" | tr -d '\r' | awk -F= '$1 == "job_id" {print $2}' | tail -n 1)"
reported_identity_dir="$(printf '%s\n' "$submit_output" | tr -d '\r' | awk -F= '$1 == "identity_dir" {print $2}' | tail -n 1)"
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
  local_dir="$repo_root/outputs/bowdoin_prepare_manifest/job-$job_id"
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

echo "Prepare-manifest round-trip complete for Bowdoin job $job_id"
