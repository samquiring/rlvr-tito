#!/usr/bin/env bash
# Push the local secrets file to a pod as /workspace/secrets.env (chmod 600).
#
# The key never appears in any terminal, log, or shell history: this script
# only copies a file. Put your key in ~/.rlvr-tito-secrets.env first:
#
#   ANTHROPIC_API_KEY=sk-ant-...
#
# Usage:
#   scripts/push_secrets.sh <pod-ip> <ssh-port>
#   scripts/push_secrets.sh 38.147.83.32 19849
#
# Env overrides:
#   SECRETS_SRC   local secrets file      (default ~/.rlvr-tito-secrets.env)
#   SSH_KEY       ssh private key         (default ~/.ssh/id_ed25519)

set -euo pipefail

POD_IP="${1:?usage: push_secrets.sh <pod-ip> <ssh-port>}"
POD_PORT="${2:?usage: push_secrets.sh <pod-ip> <ssh-port>}"
SECRETS_SRC="${SECRETS_SRC:-$HOME/.rlvr-tito-secrets.env}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

if [ ! -f "$SECRETS_SRC" ]; then
    echo "No secrets file at $SECRETS_SRC"
    echo "Create it with a single line:  ANTHROPIC_API_KEY=sk-ant-..."
    echo "Then: chmod 600 $SECRETS_SRC"
    exit 1
fi

# Sanity-check the file has the key WITHOUT printing any part of the value.
if ! grep -q '^ANTHROPIC_API_KEY=..*' "$SECRETS_SRC"; then
    echo "$SECRETS_SRC does not contain an ANTHROPIC_API_KEY=... line"
    exit 1
fi

chmod 600 "$SECRETS_SRC"
scp -P "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
    "$SECRETS_SRC" "root@${POD_IP}:/workspace/secrets.env"
ssh -p "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
    "root@${POD_IP}" "chmod 600 /workspace/secrets.env && wc -c < /workspace/secrets.env"
echo "secrets.env deployed (byte count above is the only thing read back)."
