#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  upload_to_bowdoin.sh --local-path data/raw/hdtf_cmr --remote-path /mnt/hpc/tmp/<user>/video-persona-gen/data/raw/hdtf_cmr

Required:
  --local-path   Local file or directory to upload.
  --remote-path  Remote Bowdoin destination path.

Optional:
  --env-file    Local env file. Default: .env.hpc.local
  --overwrite   Replace an existing remote file or directory at --remote-path.
  --help        Show this message.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_helper="$repo_root/scripts/run_bowdoin_hpc_command.sh"

env_file="$repo_root/.env.hpc.local"
local_path=""
remote_path=""
overwrite=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local-path)
      local_path="${2:?missing value for --local-path}"
      shift 2
      ;;
    --remote-path)
      remote_path="${2:?missing value for --remote-path}"
      shift 2
      ;;
    --env-file)
      env_file="${2:?missing value for --env-file}"
      shift 2
      ;;
    --overwrite)
      overwrite=1
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

if [[ -z "$local_path" || -z "$remote_path" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -e "$local_path" ]]; then
  echo "Local path not found: $local_path" >&2
  exit 1
fi

if [[ ! -f "$env_file" ]]; then
  echo "Env file not found: $env_file" >&2
  exit 1
fi

if ! command -v expect >/dev/null 2>&1; then
  echo "The 'expect' command is required on the local machine." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

for required_var in BOWDOIN_HPC_HOST BOWDOIN_HPC_USER BOWDOIN_HPC_PASSWORD; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "Missing $required_var in $env_file" >&2
    exit 1
  fi
done

shell_quote() {
  printf "%q" "$1"
}

remote_parent="$(dirname "$remote_path")"

if [[ -d "$local_path" ]]; then
  local_basename="$(basename "$local_path")"
  upload_target="$remote_parent"
  staged_remote_path="$remote_parent/$local_basename"
else
  upload_target="$remote_path"
  staged_remote_path="$remote_path"
fi

if (( overwrite == 1 )); then
  "$remote_helper" --env-file "$env_file" --command "rm -rf $(shell_quote "$remote_path") && mkdir -p $(shell_quote "$remote_parent")" >/dev/null
else
  "$remote_helper" --env-file "$env_file" --command "if [ -e $(shell_quote "$remote_path") ]; then echo 'Remote path already exists: $(shell_quote "$remote_path")' >&2; exit 3; fi; mkdir -p $(shell_quote "$remote_parent")" >/dev/null
fi

export BOWDOIN_UPLOAD_LOCAL_PATH="$local_path"
export BOWDOIN_UPLOAD_TARGET="$upload_target"

if [[ -d "$local_path" ]]; then
  export BOWDOIN_UPLOAD_RECURSIVE="1"
else
  export BOWDOIN_UPLOAD_RECURSIVE="0"
fi

expect <<'EOF'
set timeout -1
log_user 0

set user $env(BOWDOIN_HPC_USER)
set pass $env(BOWDOIN_HPC_PASSWORD)
set host $env(BOWDOIN_HPC_HOST)
set local_path $env(BOWDOIN_UPLOAD_LOCAL_PATH)
set upload_target $env(BOWDOIN_UPLOAD_TARGET)
set recursive $env(BOWDOIN_UPLOAD_RECURSIVE)
set target "${user}@${host}:${upload_target}"
set sent_password 0

set scp_cmd [list \
  scp \
  -o StrictHostKeyChecking=accept-new \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  -o NumberOfPasswordPrompts=1 \
  -o ConnectTimeout=20]

if {$recursive == "1"} {
  lappend scp_cmd -r
}

lappend scp_cmd -- $local_path $target

eval spawn -noecho $scp_cmd

expect {
  -re "(?i)yes/no" {
    send "yes\r"
    exp_continue
  }
  -re "(?i)password:" {
    send "$pass\r"
    set sent_password 1
    log_user 1
    exp_continue
  }
  -re {Permission denied \(.+\)|[Aa]uthentication failed|Permission denied, please try again\.} {
    puts stderr "Authentication failed for ${user}@${host}"
    exit 2
  }
  timeout {
    puts stderr "Timed out uploading to ${user}@${host}"
    exit 124
  }
  eof {
    if {!$sent_password} {
      log_user 1
    }
    catch wait result
    set exit_status [lindex $result 3]
    exit $exit_status
  }
}
EOF

if [[ -d "$local_path" && "$staged_remote_path" != "$remote_path" ]]; then
  "$remote_helper" --env-file "$env_file" --command "mv $(shell_quote "$staged_remote_path") $(shell_quote "$remote_path")"
fi

echo "Uploaded $local_path to $remote_path"
