"""Release checks must reject misleading package metadata and incomplete sets."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_artifacts.py"
spec = importlib.util.spec_from_file_location("check_artifacts", SCRIPT)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def metadata(**changes):
    fields = {
        "Name": "fusemesh",
        "Version": "0.2.0",
        "Requires-Python": ">=3.9",
        "License-Expression": "BSD-3-Clause",
    }
    fields.update(changes)
    return (
        "\n".join(f"{k}: {v}" for k, v in fields.items())
        + "\nRequires-Dist: numpy>=1.23\nRequires-Dist: trimesh>=4.12.2,<5\nRequires-Dist: fast-simplification==0.2.0\nRequires-Dist: rtree==1.4.1\n"
    ).encode()


def test_metadata_valid():
    checks.check_metadata(metadata(), "0.2.0")


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"Name": "other"}, "project name"),
        ({"Version": "0.1.0"}, "version differs"),
        ({"Requires-Python": ">=3.10"}, "Python support"),
        ({"License-Expression": ""}, "project license"),
    ],
)
def test_reject_incompatible_metadata(changes, message):
    with pytest.raises(ValueError, match=message):
        checks.check_metadata(metadata(**changes), "0.2.0")


def test_reject_missing_loader_dependency():
    with pytest.raises(ValueError, match="Trimesh"):
        checks.check_metadata(
            metadata().replace(b"Requires-Dist: trimesh", b"Ignored: trimesh"), "0.2.0"
        )


def test_reject_empty_release_and_mismatched_tag(tmp_path):
    for arguments, message in [
        (["--complete"], "no distribution"),
        (["--tag", "v0.1.0"], "tag differs"),
    ]:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(tmp_path), *arguments], capture_output=True, text=True
        )
        assert result.returncode != 0
        assert message in result.stderr
