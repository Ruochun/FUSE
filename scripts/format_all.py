"""Format first-party sources only; missing tools are errors, never silent skips."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ROOTS = (
    "bindings",
    "include",
    "src",
    "python",
    "tests",
    "benchmarks",
    "examples",
    "scripts",
    "cmake",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    tools = ("clang-format", "black", "cmake-format")
    missing = [tool for tool in tools if not shutil.which(tool)]
    if missing:
        parser.error("missing " + ", ".join(missing) + "; install requirements/format.txt")
    paths = [ROOT / "CMakeLists.txt"]
    for directory in ROOTS:
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and not p.is_symlink())
    cpp, python, cmake = [], [], []
    for path in sorted(paths):
        if "__pycache__" in path.parts:
            continue
        relative = str(path.relative_to(ROOT))
        if path.suffix in (".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh"):
            cpp.append(relative)
        elif path.suffix == ".py":
            python.append(relative)
        elif (
            path.name == "CMakeLists.txt"
            or path.suffix == ".cmake"
            or path.name.endswith(".cmake.in")
        ):
            cmake.append(relative)
    commands = [
        [
            "clang-format",
            "--style=file",
            *(["--dry-run", "--Werror"] if args.check else ["-i"]),
            *cpp,
        ],
        ["black", *(["--check", "--diff"] if args.check else []), *python],
        ["cmake-format", *(["--check"] if args.check else ["-i"]), *cmake],
    ]
    failed = False
    for command in commands:
        failed |= subprocess.run(command, cwd=ROOT).returncode != 0
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
