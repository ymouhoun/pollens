#!/usr/bin/env python3
"""Fail the image build unless llama.cpp embeds every supported GPU target."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile


REQUIRED_SASS = {"90", "100", "120"}
REQUIRED_PTX = {"120"}
ARCH_PATTERN = re.compile(r"(?:sm|compute)_(\d+)")


def locate_wheel(path: Path) -> Path:
    if path.is_file():
        return path
    wheels = sorted(path.glob("llama_cpp_python-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"Expected exactly one llama_cpp_python wheel in {path}, found {len(wheels)}"
        )
    return wheels[0]


def inspect_library(cuobjdump: str, library: Path, option: str) -> set[str]:
    result = subprocess.run(
        [cuobjdump, option, str(library)],
        text=True,
        capture_output=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cuobjdump {option} failed for {library.name}: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return set(ARCH_PATTERN.findall(result.stdout))


def verify(wheel_or_directory: Path) -> None:
    wheel = locate_wheel(wheel_or_directory)
    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    if not Path(cuobjdump).is_file():
        raise RuntimeError("cuobjdump is required for CUDA artifact verification")

    with tempfile.TemporaryDirectory(prefix="llm-enhancer-wheel-") as temp_dir:
        destination = Path(temp_dir)
        with zipfile.ZipFile(wheel) as archive:
            members = [
                name
                for name in archive.namelist()
                if "ggml-cuda" in name and name.endswith(".so")
            ]
            if not members:
                raise RuntimeError(f"No libggml-cuda library found in {wheel.name}")
            for member in members:
                archive.extract(member, destination)

        sass: set[str] = set()
        ptx: set[str] = set()
        for member in members:
            library = destination / member
            sass.update(inspect_library(cuobjdump, library, "--list-elf"))
            ptx.update(inspect_library(cuobjdump, library, "--list-ptx"))

    missing_sass = REQUIRED_SASS - sass
    missing_ptx = REQUIRED_PTX - ptx
    if missing_sass or missing_ptx:
        problems = []
        if missing_sass:
            problems.append("missing SASS: " + ", ".join(f"sm_{x}" for x in sorted(missing_sass)))
        if missing_ptx:
            problems.append("missing PTX: " + ", ".join(f"compute_{x}" for x in sorted(missing_ptx)))
        raise RuntimeError("CUDA wheel verification failed (" + "; ".join(problems) + ")")

    print(
        f"VERIFIED {wheel.name}: SASS={','.join(sorted(sass, key=int))}; "
        f"PTX={','.join(sorted(ptx, key=int))}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    verify(args.artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
