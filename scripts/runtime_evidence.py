"""Record driver-only GPU execution and reject unintended CUDA runtime loading."""

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess

import fusemesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gpu", required=True, choices=["sm80", "sm86", "sm89", "sm90"])
    args = parser.parse_args()
    if shutil.which("nvcc") or list(Path("/usr/local").glob("cuda*")):
        raise RuntimeError("acceptance environment must not contain a CUDA toolkit")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
        text=True,
    ).strip()
    rows = [row.split(", ") for row in gpu.splitlines()]
    if len(rows) != 1 or "sm" + rows[0][2].replace(".", "") != args.expected_gpu:
        raise RuntimeError("runner GPU does not match the required architecture")
    driver = tuple(map(int, rows[0][1].split(".")))
    if driver < (570, 124, 6):
        raise RuntimeError("release requires NVIDIA driver 570.124.06 or newer")
    values = fusemesh.smoke_test().tolist()
    assert values == [3.0, -2.0, 0.5, 7.0]
    maps = Path("/proc/self/maps").read_text()
    if any(name in maps for name in ("libcudart", "libnvrtc", "libcublas", "/stubs/")):
        raise RuntimeError("unexpected toolkit runtime loaded")
    print(
        json.dumps(
            dict(
                python=platform.python_version(),
                gpu=gpu,
                build=fusemesh.build_info(),
                smoke=values,
                loaded_cuda_libraries=sorted(
                    {
                        line.split()[-1]
                        for line in maps.splitlines()
                        if "libcuda" in line or "libnvidia" in line
                    }
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
