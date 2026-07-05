#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  fetch_bowdoin_preprocess_output.sh \
    --job-id 12345 \
    --remote-identity-dir /mnt/hpc/tmp/<user>/video-persona-gen/data/processed/hdtf_cmr \
    --local-dir outputs/bowdoin_preprocess/job-12345

Required:
  --job-id               Slurm job ID for a preprocess run.
  --remote-identity-dir  Persisted Bowdoin processed identity directory.
  --local-dir            Local directory for the downloaded inspection bundle.

Optional:
  --poll-seconds  Poll interval while waiting for Slurm completion. Default: 10.
  --timeout       Max seconds to wait for job completion. Default: 3600.
  --remote-user   Remote HPC username. Defaults to BOWDOIN_HPC_USER from .env.hpc.local.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_helper="$repo_root/scripts/run_bowdoin_hpc_command.sh"
job_id=""
remote_identity_dir=""
local_dir=""
poll_seconds=10
timeout_seconds=3600
remote_user=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-id)
      job_id="${2:?missing value for --job-id}"
      shift 2
      ;;
    --remote-identity-dir)
      remote_identity_dir="${2:?missing value for --remote-identity-dir}"
      shift 2
      ;;
    --local-dir)
      local_dir="${2:?missing value for --local-dir}"
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

if [[ -z "$job_id" || -z "$remote_identity_dir" || -z "$local_dir" ]]; then
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

mkdir -p "$local_dir"
payload_file="$(mktemp)"

cleanup() {
  rm -f "$payload_file"
}

trap cleanup EXIT

shell_quote() {
  printf "%q" "$1"
}

run_remote() {
  local command=$1
  "$remote_helper" --command "$command"
}

decode_to_tar() {
  local payload_path=$1
  local destination=$2

  if base64 --help >/dev/null 2>&1; then
    base64 --decode <"$payload_path" | tar -xf - -C "$destination"
  else
    base64 -D <"$payload_path" | tar -xf - -C "$destination"
  fi
}

job_state=""
job_exit_code=""
deadline=$((SECONDS + timeout_seconds))
while (( SECONDS < deadline )); do
  raw_queue_state="$(run_remote "squeue -h -j $job_id -o '%T' 2>/dev/null || true")"
  queue_state="$(printf '%s\n' "$raw_queue_state" | tr -d '\r' | awk 'NF {print $1; exit}')"
  if [[ -n "$queue_state" ]]; then
    sleep "$poll_seconds"
    continue
  fi

  raw_sacct_state="$(run_remote "sacct -j $job_id --format=JobIDRaw,State,ExitCode -n -P 2>/dev/null || true")"
  sacct_line="$(printf '%s\n' "$raw_sacct_state" | tr -d '\r' | awk -F'|' -v job_id="$job_id" '$1 == job_id {print; exit}')"
  if [[ -n "$sacct_line" ]]; then
    job_state="$(printf '%s\n' "$sacct_line" | awk -F'|' '{print $2}')"
    job_exit_code="$(printf '%s\n' "$sacct_line" | awk -F'|' '{print $3}')"
    break
  fi

  sleep "$poll_seconds"
done

if [[ -z "$job_state" ]]; then
  echo "Timed out waiting for Bowdoin job $job_id to finish." >&2
  exit 1
fi

bundle_command=$(
  cat <<EOF
set -euo pipefail
identity_dir=$(shell_quote "$remote_identity_dir")
if [ ! -d "\$identity_dir" ]; then
  echo "Missing remote identity dir: \$identity_dir" >&2
  exit 1
fi
bundle_root=\$(mktemp -d)
trap 'rm -rf "\$bundle_root"' EXIT
mkdir -p "\$bundle_root/clip_summaries"
if [ -f "\$identity_dir/manifest.jsonl" ]; then
  cp "\$identity_dir/manifest.jsonl" "\$bundle_root/manifest.jsonl"
fi
if [ -f "\$identity_dir/dataset_report.json" ]; then
  cp "\$identity_dir/dataset_report.json" "\$bundle_root/dataset_report.json"
fi
find "\$identity_dir" -mindepth 2 -maxdepth 2 -name metadata.json | sort | while IFS= read -r metadata_path; do
  clip_dir=\$(dirname "\$metadata_path")
  clip_id=\$(basename "\$clip_dir")
  clip_bundle_dir="\$bundle_root/clip_summaries/\$clip_id"
  mkdir -p "\$clip_bundle_dir"
  cp "\$metadata_path" "\$clip_bundle_dir/metadata.json"
  if [ -d "\$clip_dir/face_crops" ]; then
    first_crop=\$(find "\$clip_dir/face_crops" -maxdepth 1 -type f | sort | head -n 1 || true)
    if [ -n "\$first_crop" ]; then
      cp "\$first_crop" "\$clip_bundle_dir/\$(basename "\$first_crop")"
    fi
  fi
done
tar -C "\$bundle_root" -cf - . | base64 | tr -d '\n'
EOF
)

run_remote "$bundle_command" >"$payload_file"
if [[ ! -s "$payload_file" ]]; then
  echo "Remote preprocess bundle was empty: $remote_identity_dir" >&2
  exit 1
fi

decode_to_tar "$payload_file" "$local_dir"

cat >"$local_dir/status.env" <<EOF
job_id=$job_id
state=$job_state
exit_code=$job_exit_code
remote_identity_dir=$remote_identity_dir
remote_user=$remote_user
EOF

echo "Fetched Bowdoin preprocess bundle for job $job_id to $local_dir"
echo "state=$job_state"
echo "remote_identity_dir=$remote_identity_dir"

if [[ "$job_state" != "COMPLETED" ]]; then
  exit 1
fi
