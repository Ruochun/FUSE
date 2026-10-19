#!/usr/bin/env bash
# Driver-only container: no CUDA toolkit is present in the Python base image.
set -euo pipefail
python_version=${1:?Python version required}
expected_gpu=${2:?SM label required}
repo_dir=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$repo_dir/build/gpu-evidence"
# Pin the resolved image identity in the evidence for each validation run.
image="python:${python_version}-slim-bookworm"
docker pull "$image"
docker image inspect "$image" > "$repo_dir/build/gpu-evidence/container.json"
docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e EXPECTED_GPU="$expected_gpu" \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$repo_dir:/project:ro" \
    -v "$repo_dir/build/gpu-evidence:/evidence" \
    -w /tmp "$image" bash -euc '
        python -m venv /tmp/runtime
        /tmp/runtime/bin/python -m pip install --only-binary=:all: --find-links=/project/wheelhouse --no-index --no-deps fusemesh
        /tmp/runtime/bin/python -m pip install --only-binary=:all: --find-links=/project/wheelhouse "fusemesh[quality]" "pytest>=8,<9"
        /tmp/runtime/bin/python -m pip check
        /tmp/runtime/bin/python -m pip freeze > /evidence/packages.txt
        /tmp/runtime/bin/python /project/scripts/runtime_evidence.py --expected-gpu "$EXPECTED_GPU" > /evidence/runtime.json
        /tmp/runtime/bin/python -m pytest /project/tests -q -p no:cacheprovider --junitxml=/evidence/tests.xml
        for native_test in /project/build/native-tests/fuse_*; do
            "$native_test" >> /evidence/native-tests.txt 2>&1
        done
        if [[ "$EXPECTED_GPU" == sm90 ]]; then
        CUDA_FORCE_PTX_JIT=1 /tmp/runtime/bin/python -m pytest /project/tests/test_smoke.py /project/tests/test_m3.py -q -p no:cacheprovider --junitxml=/evidence/ptx.xml
        fi
    '
