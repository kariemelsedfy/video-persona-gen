#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  fetch_bowdoin_liveportrait_output.sh --job-id 12345 --local-dir outputs/bowdoin/job-12345

Required:
  --job-id      Slurm job ID for a running liveportrait_infer_tmp job.
  --local-dir   Local directory where logs and output files will be extracted.

Optional:
  --poll-seconds  Poll interval while waiting for the remote status file. Default: 10.
  --timeout       Max seconds to wait for remote status before failing. Default: 3600.
  --remote-user   Remote HPC username. Defaults to BOWDOIN_HPC_USER from .env.hpc.local.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_helper="$repo_root/scripts/run_bowdoin_hpc_command.sh"
job_id=""
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

if [[ -z "$job_id" || -z "$local_dir" ]]; then
  usage >&2
  exit 2
fi

if [[ -z "$remote_user" ]]; then
  if [[ -f "$repo_root/.env.hpc.local" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$repo_root/.env.hpc.local"
    set +a
    remote_user="${BOWDOIN_HPC_USER:-}"
  fi
fi

if [[ -z "$remote_user" ]]; then
  echo "Unable to determine remote user. Pass --remote-user or define BOWDOIN_HPC_USER in .env.hpc.local." >&2
  exit 1
fi

tmp_root="/tmp/${remote_user}/liveportrait-${job_id}"
status_file="$tmp_root/status.env"
pickup_file="$tmp_root/pickup.done"

mkdir -p "$local_dir"
logs_dir="$local_dir/logs"
output_dir="$local_dir/output"
mkdir -p "$logs_dir" "$output_dir"
logs_payload_file="$(mktemp)"
output_payload_file="$(mktemp)"

cleanup() {
  rm -f "$logs_payload_file" "$output_payload_file"
}

trap cleanup EXIT

run_remote() {
  local command=$1
  "$remote_helper" --command "$command"
}

decode_to_tar() {
  local payload_file=$1
  local destination=$2

  if base64 --help >/dev/null 2>&1; then
    base64 --decode <"$payload_file" | tar -xf - -C "$destination"
  else
    base64 -D <"$payload_file" | tar -xf - -C "$destination"
  fi
}

status_payload=""
deadline=$((SECONDS + timeout_seconds))
while (( SECONDS < deadline )); do
  raw_status_payload="$(run_remote "srun --jobid $job_id --overlap bash -lc \"cat '$status_file' 2>/dev/null || true\"")"
  status_payload="$(printf '%s\n' "$raw_status_payload" | tr -d '\r' | sed '/^[[:space:]]*$/d')"
  if printf '%s\n' "$status_payload" | grep -q '^state='; then
    break
  fi
  sleep "$poll_seconds"
done

if ! printf '%s\n' "$status_payload" | grep -q '^state='; then
  echo "Timed out waiting for remote status file: $status_file" >&2
  exit 1
fi

printf '%s\n' "$status_payload" >"$logs_dir/status.env"
state="$(awk -F= '$1 == "state" {print $2}' "$logs_dir/status.env")"

run_remote "srun --jobid $job_id --overlap bash -lc \"tar -C '$tmp_root' -cf - status.env hf.log inference.log output_files.txt 2>/dev/null | base64 | tr -d '\n'\"" >"$logs_payload_file"
decode_to_tar "$logs_payload_file" "$logs_dir"

if grep -q '^/' "$logs_dir/output_files.txt" 2>/dev/null; then
  run_remote "srun --jobid $job_id --overlap bash -lc \"if [ -d '$tmp_root/output' ]; then tar -C '$tmp_root/output' -cf - . | base64 | tr -d '\n'; fi\"" >"$output_payload_file"
  decode_to_tar "$output_payload_file" "$output_dir"
fi

run_remote "srun --jobid $job_id --overlap bash -lc \"touch '$pickup_file'\" >/dev/null 2>&1 || true"

run_remote "while squeue -h -j $job_id | grep -q $job_id; do sleep 2; done; sacct -j $job_id --format=JobID,State,ExitCode -n -P"

echo "Fetched Bowdoin job $job_id to $local_dir"
echo "state=$state"

if [[ "$state" != "SUCCESS" ]]; then
  exit 1
fi
