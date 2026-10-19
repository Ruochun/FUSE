"""Fail closed on incomplete, misversioned, or toolkit-dependent release artifacts."""

import argparse
from email.parser import BytesParser
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PYTHONS = {"cp39", "cp310", "cp311", "cp312", "cp313", "cp314"}
LICENSES = {
    "LICENSE",
    "THIRD-PARTY-NOTICES.txt",
    "NanoVDB-Apache-2.0.txt",
    "CUDA-12.8-EULA.txt",
    "nanobind-LICENSE.txt",
    "robin-map-LICENSE.txt",
    "CUB-LICENSE.txt",
    "LLVM-Apache-2.0-exception.txt",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_metadata(data, version):
    metadata = BytesParser().parsebytes(data)
    require(metadata["Name"] == "fusemesh", "unexpected project name")
    require(metadata["Version"] == version, "artifact version differs from VERSION")
    require(metadata["Requires-Python"] == ">=3.9", "incorrect Python support metadata")
    require(metadata["License-Expression"] == "BSD-3-Clause", "missing project license")
    dependencies = metadata.get_all("Requires-Dist", [])
    require(any(d.startswith("numpy") for d in dependencies), "missing NumPy dependency")
    require(any(d.startswith("trimesh") for d in dependencies), "missing Trimesh dependency")
    require(
        any(d.startswith("fast-simplification") for d in dependencies),
        "missing simplification dependency",
    )
    require(any(d.startswith("rtree") for d in dependencies), "missing Rtree dependency")
    require("quality" in metadata.get_all("Provides-Extra", []), "missing quality extra")
    require(
        any(d.startswith("manifold3d") and 'extra == "quality"' in d for d in dependencies),
        "missing quality union dependency",
    )


def check_wheel(path, version, portable=True):
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, "wheel checksum error")
        names = archive.namelist()
        metadata = [n for n in names if n.endswith(".dist-info/METADATA")]
        require(len(metadata) == 1, "expected one wheel metadata file")
        check_metadata(archive.read(metadata[0]), version)
        licenses = {Path(n).name for n in names if ".dist-info/licenses/" in n}
        require(LICENSES <= licenses, "missing third-party license notices")
        wheel_metadata = archive.read(metadata[0].replace("METADATA", "WHEEL")).decode()
        tags = re.findall(r"^Tag: (.+)$", wheel_metadata, re.MULTILINE)
        pattern = r"(cp(?:39|310|311|312|313|314))-\1-manylinux_2_28_x86_64"
        if portable:
            require(
                bool(tags) and all(re.fullmatch(pattern, tag) for tag in tags),
                "unexpected wheel tags",
            )
        require(any(tag in path.name for tag in tags), "filename disagrees with wheel tags")
        modules = [n for n in names if n.endswith(".so")]
        require(
            len(modules) == 1 and modules[0].startswith("fusemesh/_native."),
            "unexpected native payload",
        )
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "native.so"
            binary.write_bytes(archive.read(modules[0]))
            dynamic = subprocess.check_output(["readelf", "-d", str(binary)], text=True)
            require(
                not re.search(r"\((?:RPATH|RUNPATH)\)", dynamic),
                "native module has runtime search paths",
            )
            needed = re.findall(r"Shared library: \[(.+?)\]", dynamic)
            allowed = {
                "libdl.so.2",
                "librt.so.1",
                "libpthread.so.0",
                "libstdc++.so.6",
                "libm.so.6",
                "libgcc_s.so.1",
                "libc.so.6",
                "ld-linux-x86-64.so.2",
            }
            require(set(needed) <= allowed, "unexpected shared dependency: " + repr(needed))
            if portable:
                symbols = subprocess.check_output(
                    ["readelf", "--version-info", str(binary)], text=True
                )
                versions = [
                    tuple(map(int, v.split("."))) for v in re.findall(r"GLIBC_([0-9.]+)", symbols)
                ]
                require(
                    all(v <= (2, 28) for v in versions), "glibc requirement exceeds manylinux_2_28"
                )
        return tags[0].split("-")[0]


def check_source(path, version):
    with tarfile.open(path) as archive:
        names = archive.getnames()
        prefix = f"fusemesh-{version}/"
        require(
            all(n.startswith(prefix) and ".." not in Path(n).parts for n in names),
            "invalid source archive paths",
        )
        relative = {n[len(prefix) :] for n in names}
        required = {
            "VERSION",
            "pyproject.toml",
            "CMakeLists.txt",
            "src/cuda/reconstruct.cu",
            "python/fusemesh/__init__.py",
            "external/openvdb/nanovdb/nanovdb/NanoVDB.h",
            "external/openvdb/LICENSE",
            "tests/test_m3.py",
            "tests/test_simplification.py",
            "python/fusemesh/simplification.py",
            "python/fusemesh/quality.py",
            "tests/test_quality.py",
            "docs/quality.md",
            "scripts/check_artifacts.py",
            "scripts/check_no_driver.py",
            "scripts/ci_native_build.sh",
            "scripts/ci_gpu_test.sh",
            "scripts/runtime_evidence.py",
            "requirements/format.txt",
            "requirements/release.txt",
            ".github/workflows/wheels.yml",
            ".github/actionlint.yaml",
            ".cmake-format.yaml",
            ".format_all",
            ".clang-format",
        }
        require(required <= relative, "source archive is incomplete: " + repr(required - relative))
        require(
            not any(".git" in Path(n).parts for n in names), "source archive contains Git internals"
        )
        require(
            archive.extractfile(prefix + "VERSION").read().decode().strip() == version,
            "source VERSION mismatch",
        )
        check_metadata(archive.extractfile(prefix + "PKG-INFO").read(), version)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--python-tag", choices=sorted(PYTHONS))
    parser.add_argument("--tag")
    parser.add_argument("--local", action="store_true", help="allow nonportable development wheels")
    args = parser.parse_args()
    version = (ROOT / "VERSION").read_text().strip()
    if args.tag:
        require(args.tag == "v" + version, "release tag differs from VERSION")
    wheels = sorted(args.directory.glob("*.whl"))
    sources = sorted(args.directory.glob("*.tar.gz"))
    require(bool(wheels or sources), "no distribution artifacts found")
    tags = [check_wheel(p, version, not args.local) for p in wheels]
    for path in sources:
        check_source(path, version)
    if args.source_only:
        require(len(sources) == 1 and not wheels, "expected exactly one source archive")
    if args.python_tag:
        require(tags == [args.python_tag], "expected exactly one wheel for requested Python")
    if args.complete:
        require(not args.local, "local wheels cannot be published")
        require(
            set(tags) == PYTHONS and len(tags) == 6 and len(sources) == 1,
            "release needs six wheels and one source archive",
        )
    print(f"Validated {len(wheels)} wheel(s), {len(sources)} source archive(s), version {version}")


if __name__ == "__main__":
    main()
