#!/usr/bin/env bash
# Linux-only M0 acceptance check. Mount changes exist only in a private namespace.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "usage: $0 /absolute/path/to/clean-venv/bin/python" >&2
    exit 2
fi
python_bin=$(realpath -s "$1")
repo_dir=$(cd "$(dirname "$0")/.." && pwd)
unshare --user --map-root-user --mount --propagation private /bin/bash -s -- "$python_bin" "$repo_dir" <<'INNER'
set -euo pipefail
python_bin=$1
repo_dir=$2
# This host keeps all installed toolkits below /usr/local; hide all of them.
mount -t tmpfs -o size=1m tmpfs /usr/local
cd /tmp
exec env -u LD_LIBRARY_PATH -u CUDA_HOME -u CUDA_PATH -u PYTHONPATH \
    PATH="$(dirname "$python_bin")" PYTHONNOUSERSITE=1 \
    "$python_bin" -m pytest "$repo_dir/tests" -v -p no:cacheprovider
INNER
