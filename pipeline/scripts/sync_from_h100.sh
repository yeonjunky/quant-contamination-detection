#!/usr/bin/env bash
# rsync wrapper: pulls data/raw/ down from the H100 box to this machine.
# pipeline_build_plan.md's "Raw data schema + sync" section: excludes model-
# weight caches (only the raw/manifests tree is synced, not HF_HOME), and
# should always be run with --dry-run first before validation/main artifact sync.
#
# Usage:
#   scripts/sync_from_h100.sh <ssh-alias> [<remote-repo-path>] [-- <extra rsync args>]
#
# Example:
#   scripts/sync_from_h100.sh h100-box                      # uses default remote path
#   scripts/sync_from_h100.sh h100-box ~/repo -- --dry-run   # dry run first
#
# The remote repo path defaults to ~/quant-contamination-detection — override
# with the second positional argument if the H100 checkout lives elsewhere.

set -euo pipefail

if [ "${1:-}" = "" ]; then
    echo "usage: $0 <ssh-alias> [<remote-repo-path>] [-- <extra rsync args>]" >&2
    exit 1
fi

SSH_ALIAS="$1"
shift

REMOTE_REPO_PATH="${1:-~/quant-contamination-detection}"
if [ "${1:-}" != "" ] && [ "${1:-}" != "--" ]; then
    shift
fi

if [ "${1:-}" = "--" ]; then
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
LOCAL_DEST="${LOCAL_PIPELINE_DIR}/../data/raw/"

mkdir -p "$LOCAL_DEST"

echo "Syncing ${SSH_ALIAS}:${REMOTE_REPO_PATH}/data/raw/ -> ${LOCAL_DEST}"
echo "(run with a trailing '-- --dry-run' first to preview)"

rsync -avz --progress \
    --exclude 'hf_cache/' \
    --exclude '*.safetensors' \
    --exclude '*.bin' \
    "${SSH_ALIAS}:${REMOTE_REPO_PATH}/data/raw/" \
    "$LOCAL_DEST" \
    "$@"
