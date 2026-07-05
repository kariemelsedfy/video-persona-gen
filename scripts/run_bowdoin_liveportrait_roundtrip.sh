#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_bowdoin_liveportrait_roundtrip.sh [options]

Submit the tracked Bowdoin LivePortrait Slurm job, wait for it to finish inference,
download logs and outputs into the local repository, and then release the remote job.

Optional:
  --source-rel        Source asset path relative to external/LivePortrait. Default: assets/examples/source/s0.jpg
  --driving-rel       Driving asset path relative to external/LivePortrait. Default: assets/examples/driving/d0.mp4
  --local-dir         Local output directory. Default: outputs/bowdoin_liveportrait/job-<jobid>
  --remote-repo-root  Remote repo root. Default: /home/kelsedfy/video-persona-gen
  --job-script        Remote sbatch script relative to repo root. Default: slurm/liveportrait_infer_tmp.sbatch
  --job-name          Slurm job name override. Default: liveportrait-infer-fetch
  --fetch-wait        Seconds the remote job should stay alive for pickup. Default: 1800
  --gpu-gres          GRES request. Default: gpu:rtx3080:1
  --partition         Slurm partition. Default: gpu
  --cpus              CPUs per task override. Default: 2
  --mem               Memory override. Default: 32G
  --time              Time limit override. Default: 01:00:00
  --remote-storage-root  Persistent Bowdoin storage root. Default: /mnt/hpc/tmp/<user>/video-persona-gen
  --persist-weights-dir  Persistent Bowdoin LivePortrait weights directory. Default: <remote-storage-root>/liveportrait_weights
  --remote-user       Remote HPC username. Defaults to BOWDOIN_HPC_USER from .env.hpc.local.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_helper="$repo_root/scripts/run_bowdoin_hpc_command.sh"
fetch_script="$repo_root/scripts/fetch_bowdoin_liveportrait_output.sh"

source_rel="assets/examples/source/s0.jpg"
driving_rel="assets/examples/driving/d0.mp4"
local_dir=""
remote_repo_root="/home/kelsedfy/video-persona-gen"
job_script="slurm/liveportrait_infer_tmp.sbatch"
job_name="liveportrait-infer-fetch"
fetch_wait_seconds=1800
gpu_gres="gpu:rtx3080:1"
partition="gpu"
cpus="2"
mem="32G"
time_limit="01:00:00"
remote_storage_root=""
persist_weights_dir=""
remote_user=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-rel)
      source_rel="${2:?missing value for --source-rel}"
      shift 2
      ;;
    --driving-rel)
      driving_rel="${2:?missing value for --driving-rel}"
      shift 2
      ;;
    --local-dir)
      local_dir="${2:?missing value for --local-dir}"
      shift 2
      ;;
    --remote-repo-root)
      remote_repo_root="${2:?missing value for --remote-repo-root}"
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
    --fetch-wait)
      fetch_wait_seconds="${2:?missing value for --fetch-wait}"
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
    --persist-weights-dir)
      persist_weights_dir="${2:?missing value for --persist-weights-dir}"
      shift 2
      ;;
    --remote-user)
      remote_user="${2:?missing value for --remote-user}"
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

if [[ -z "$persist_weights_dir" ]]; then
  persist_weights_dir="$remote_storage_root/liveportrait_weights"
fi

job_script_path="$repo_root/$job_script"
if [[ ! -f "$job_script_path" ]]; then
  echo "Local job script not found: $job_script_path" >&2
  exit 1
fi

shell_quote() {
  printf "%q" "$1"
}

job_script_content="$(<"$job_script_path")"

remote_command=$(
  cat <<EOF
mkdir -p $(shell_quote "$remote_repo_root/$(dirname "$job_script")") && \
cat > $(shell_quote "$remote_repo_root/$job_script") <<'EOF_BOWDOIN_JOB'
$job_script_content
EOF_BOWDOIN_JOB
chmod +x $(shell_quote "$remote_repo_root/$job_script") && \
cd $(shell_quote "$remote_repo_root") && \
SOURCE_REL=$(shell_quote "$source_rel") \
DRIVING_REL=$(shell_quote "$driving_rel") \
FETCH_WAIT_SECONDS=$(shell_quote "$fetch_wait_seconds") \
REPO_ROOT=$(shell_quote "$remote_repo_root") \
REMOTE_STORAGE_ROOT=$(shell_quote "$remote_storage_root") \
PERSIST_WEIGHTS_DIR=$(shell_quote "$persist_weights_dir") \
sbatch --parsable \
  -J $(shell_quote "$job_name") \
  -p $(shell_quote "$partition") \
  --gres=$(shell_quote "$gpu_gres") \
  -c $(shell_quote "$cpus") \
  --mem=$(shell_quote "$mem") \
  -t $(shell_quote "$time_limit") \
  $(shell_quote "$job_script")
EOF
)

submit_output="$("$remote_helper" --command "$remote_command")"
job_id="$(printf '%s\n' "$submit_output" | tr -d '\r' | awk 'NF {print $1}' | tail -n 1)"
job_id="${job_id%%;*}"

if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Unexpected sbatch output: $submit_output" >&2
  exit 1
fi

if [[ -z "$local_dir" ]]; then
  local_dir="$repo_root/outputs/bowdoin_liveportrait/job-$job_id"
fi

echo "Submitted Bowdoin job $job_id"
echo "Local download target: $local_dir"
echo "Remote storage root: $remote_storage_root"

"$fetch_script" \
  --job-id "$job_id" \
  --local-dir "$local_dir" \
  --remote-user "$remote_user" \
  --remote-storage-root "$remote_storage_root"
