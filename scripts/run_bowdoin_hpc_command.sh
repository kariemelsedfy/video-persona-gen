#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_bowdoin_hpc_command.sh --command "hostname"
  run_bowdoin_hpc_command.sh --env-file /abs/path/.env.hpc.local --command "squeue -u $USER"

Required:
  --command   Remote shell command to execute on Bowdoin HPC.

Optional:
  --env-file  Path to a local env file. Defaults to .env.hpc.local in the current directory.
  --help      Show this message.
EOF
}

env_file=".env.hpc.local"
remote_command=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      env_file="${2:?missing value for --env-file}"
      shift 2
      ;;
    --command)
      remote_command="${2:?missing value for --command}"
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

if [[ -z "$remote_command" ]]; then
  echo "Missing required --command argument." >&2
  usage >&2
  exit 2
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

export BOWDOIN_HPC_REMOTE_COMMAND="$remote_command"

expect <<'EOF'
set timeout -1
log_user 0

set user $env(BOWDOIN_HPC_USER)
set pass $env(BOWDOIN_HPC_PASSWORD)
set host $env(BOWDOIN_HPC_HOST)
set remote_command $env(BOWDOIN_HPC_REMOTE_COMMAND)
set target "${user}@${host}"
set sent_password 0

set ssh_cmd [list \
  ssh \
  -o StrictHostKeyChecking=accept-new \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  -o NumberOfPasswordPrompts=1 \
  -o ConnectTimeout=20 \
  $target \
  $remote_command]

eval spawn -noecho $ssh_cmd

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
    puts stderr "Authentication failed for $target"
    exit 2
  }
  timeout {
    puts stderr "Timed out connecting to $target"
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
