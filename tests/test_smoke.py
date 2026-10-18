import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import fusemesh


def test_metadata():
    info = fusemesh.build_info()
    assert info["nanovdb_revision"] == "e26f5a569201d3236e0c30580572101022d3d754"
    assert info["cuda_version"] == 12080
    assert info["cuda_runtime"] == "static"


def test_negative_device():
    with pytest.raises(ValueError, match="nonnegative"):
        fusemesh.smoke_test(-1)


def test_no_visible_device():
    # Fresh process: CUDA_VISIBLE_DEVICES is consulted at initialization.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import fusemesh; print(fusemesh.build_info()); fusemesh.smoke_test()",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "nanovdb_revision" in result.stdout  # import and metadata remain usable
    assert "RuntimeError: FUSE CUDA:" in result.stderr


@pytest.mark.gpu
def test_gpu_values_and_ownership():
    samples = fusemesh.smoke_test()
    assert samples.dtype == np.float32
    assert samples.shape == (4,)
    np.testing.assert_array_equal(samples, [3, -2, 0.5, 7])
    samples[0] = 99
    np.testing.assert_array_equal(fusemesh.smoke_test(), [3, -2, 0.5, 7])
    assert samples[0] == 99  # subsequent calls do not reuse host result storage


@pytest.mark.gpu
def test_invalid_visible_device():
    with pytest.raises(ValueError, match="visible GPU range"):
        fusemesh.smoke_test(2**30)


@pytest.mark.gpu
def test_concurrent_calls():
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: fusemesh.smoke_test(), range(12)))
    for samples in results:
        np.testing.assert_array_equal(samples, [3, -2, 0.5, 7])
