#!/usr/bin/env bash
# rsync wrapper: pulls main-study raw data down from the H100 box to this machine.
# pipeline_build_plan.md's "Raw data schema + sync" section: excludes model-
# weight caches (only the raw/manifests tree is synced, not HF_HOME), and
# should always be run with --dry-run first before validation/main artifact sync.
#
# Paper §4.6 keeps engineering-validation output and study data in two
# namespaces (`data/raw/validation/` and `data/raw/main/`). This script used to
# pull `data/raw/` whole, which merged them into one local tree. It now syncs
# one namespace per rsync invocation, into its own destination: main study by
# default, and the validation namespace only when asked for with
# --with-validation.
#
# Usage:
#   scripts/sync_from_h100.sh [--with-validation|--validation-only] \
#       <ssh-alias> [<remote-repo-path>] [-- <extra rsync args>]
#
# Example:
#   scripts/sync_from_h100.sh h100-box                       # main study only
#   scripts/sync_from_h100.sh h100-box ~/repo -- --dry-run   # dry run first
#   scripts/sync_from_h100.sh --with-validation h100-box     # both namespaces
#
# The remote repo path defaults to ~/quant-contamination-detection — override
# with the second positional argument if the H100 checkout lives elsewhere.

set -euo pipefail

SYNC_MAIN=1
SYNC_VALIDATION=0

while [ "${1:-}" != "" ]; do
    case "$1" in
        --with-validation) SYNC_VALIDATION=1; shift ;;
        --validation-only) SYNC_VALIDATION=1; SYNC_MAIN=0; shift ;;
        --main-only) SYNC_VALIDATION=0; SYNC_MAIN=1; shift ;;
        *) break ;;
    esac
done

if [ "${1:-}" = "" ]; then
    echo "usage: $0 [--with-validation|--validation-only] <ssh-alias> [<remote-repo-path>] [-- <extra rsync args>]" >&2
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
# Resolved rather than left as `pipeline/../data/raw`, so the paths this script
# prints are the ones a reader can compare against the repository layout.
LOCAL_RAW_DIR="$(cd "${LOCAL_PIPELINE_DIR}/.." && pwd)/data/raw"

# Reads one run directory's `study_phase` out of its manifest, which is the
# field paper §4.6's boundary is actually enforced on (io/manifest.py). Plain
# sed so the script keeps working without the project's python environment; a
# tree with no manifest is reported as such rather than assumed to be study data.
report_study_phase() {
    local namespace_dir="$1"
    local found=0
    while IFS= read -r manifest; do
        found=1
        local phase
        phase="$(sed -n 's/.*"study_phase"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$manifest" | head -1)"
        echo "  $(dirname "$manifest"): study_phase=${phase:-<absent>}"
    done < <(find "$namespace_dir" -maxdepth 3 -name manifest.json 2>/dev/null | sort)
    if [ "$found" -eq 0 ]; then
        echo "  (no manifest.json under $namespace_dir — nothing synced yet, or a --dry-run)"
    fi
}

sync_namespace() {
    local namespace="$1"
    shift  # the rest is the caller's extra rsync args
    local destination="${LOCAL_RAW_DIR}/${namespace}/"

    mkdir -p "$destination"
    echo "Syncing ${SSH_ALIAS}:${REMOTE_REPO_PATH}/data/raw/${namespace}/ -> ${destination}"
    echo "(run with a trailing '-- --dry-run' first to preview)"

    rsync -avz --progress \
        --exclude 'hf_cache/' \
        --exclude '*.safetensors' \
        --exclude '*.bin' \
        "${SSH_ALIAS}:${REMOTE_REPO_PATH}/data/raw/${namespace}/" \
        "$destination" \
        "$@"

    echo "study_phase recorded in ${destination}:"
    report_study_phase "$destination"
}

if [ "$SYNC_MAIN" -eq 1 ]; then
    sync_namespace main "$@"
fi
if [ "$SYNC_VALIDATION" -eq 1 ]; then
    sync_namespace validation "$@"
fi
