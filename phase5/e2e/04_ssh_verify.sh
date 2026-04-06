#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <host> <port> [tmux_session]"
  exit 1
fi

HOST="$1"
PORT="$2"
SESSION="${3:-train}"

SSH_OPTS=("-p" "${PORT}" "-o" "StrictHostKeyChecking=no")

echo "Running remote smoke checks on ${HOST}:${PORT}"
ssh "${SSH_OPTS[@]}" "root@${HOST}" 'bash -lc "nvidia-smi || true"'
ssh "${SSH_OPTS[@]}" "root@${HOST}" 'bash -lc "ls -lah /workspace/datasets/phase5/e2e/nanochat/base_data_climbmix || true"'
ssh "${SSH_OPTS[@]}" "root@${HOST}" 'bash -lc "tmux ls || true"'
ssh "${SSH_OPTS[@]}" "root@${HOST}" 'bash -lc "tail -n 120 /workspace/training.log || true"'

echo
echo "Attach tmux session:"
echo "ssh -p ${PORT} root@${HOST} 'tmux attach -t ${SESSION}'"
