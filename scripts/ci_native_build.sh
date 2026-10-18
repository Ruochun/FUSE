#!/usr/bin/env bash
# Run inside the pinned manylinux CUDA build image with source archive mounted.
set -euo pipefail
archive=${1:?source archive required}
out=${2:?output directory required}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$out"
tar -xzf "$archive" -C "$work" --strip-components=1
cmake -S "$work" -B "$work/build" \
    -DFUSE_BUILD_PYTHON=OFF -DFUSE_BUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build "$work/build" --parallel 2
ctest --test-dir "$work/build" -R 'fuse_(version|cells)$' --output-on-failure
for test in version smoke cells reconstruct extraction; do
    cp "$work/build/fuse_$test" "$out/"
done
cmake --version > "$out/build-tools.txt"
/usr/local/cuda/bin/nvcc --version >> "$out/build-tools.txt"
g++ --version >> "$out/build-tools.txt"
