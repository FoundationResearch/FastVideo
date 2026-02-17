#!/usr/bin/env python3
"""Run CuTe qk128 smoke test on Modal serverless GPU.

Usage:
  python -m modal run fastvideo-kernel/dev/modal_cute_qk128_smoke.py

Optional environment variables:
  MODAL_SMOKE_IMAGE_TAG   Container image tag to use
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from textwrap import dedent

import modal

APP_NAME = "fastvideo-cute-qk128-smoke"
DEFAULT_IMAGE = "ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:latest"
GPU_SPEC = "B200:1"


def _resolve_repo_root() -> Path:
    """Find repo root robustly across local and Modal import paths."""
    candidates = [Path(__file__).resolve(), Path.cwd().resolve()]
    for start in candidates:
        for parent in [start, *start.parents]:
            if (parent / "fastvideo-kernel").is_dir():
                return parent
    # Fallback keeps module importable in shallow container paths.
    return Path.cwd().resolve()


REPO_ROOT = _resolve_repo_root()


def _resolve_image_tag() -> str:
    # Keep this standalone from CI-specific IMAGE_VERSION wiring.
    image_tag = os.environ.get("MODAL_SMOKE_IMAGE_TAG")
    if image_tag:
        return image_tag
    image_version = os.environ.get("IMAGE_VERSION")
    if image_version:
        return f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{image_version}"
    return DEFAULT_IMAGE


IMAGE_TAG = _resolve_image_tag()
app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(IMAGE_TAG, add_python="3.12")
    .add_local_dir(str(REPO_ROOT), remote_path="/FastVideo")
)


@app.function(
    gpu=GPU_SPEC,
    image=image,
    timeout=1800,
)
def run_remote_smoke() -> dict:
    """Execute the smoke test remotely and return captured logs."""
    cmd = dedent(
        """
        set -euo pipefail
        source /opt/venv/bin/activate
        cd /FastVideo
        if [ ! -d /FastVideo/fastvideo-kernel/include/flash-attention/flash_attn/cute ]; then
          echo "ERROR: local mount is missing fastvideo-kernel/include/flash-attention/flash_attn/cute"
          echo "Please init submodule locally (or mount with access to that folder) before running modal."
          exit 2
        fi
        if [ ! -f /FastVideo/fastvideo-kernel/dev/test_cute_qk128_smoke.py ]; then
          echo "ERROR: local mount is missing fastvideo-kernel/dev/test_cute_qk128_smoke.py"
          exit 2
        fi
        python -m pip install -e /FastVideo/fastvideo-kernel/include/flash-attention/flash_attn/cute
        python /FastVideo/fastvideo-kernel/dev/test_cute_qk128_smoke.py
        """
    ).strip()

    proc = subprocess.run(
        ["/bin/bash", "-lc", cmd],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "image_tag": IMAGE_TAG,
        "gpu": GPU_SPEC,
        "repo_mode": "mounted_local_fastvideo",
    }


@app.local_entrypoint()
def main() -> None:
    result = run_remote_smoke.remote()
    print("=== Modal CuTe qk128 smoke result ===")
    print(f"image: {result['image_tag']}")
    print(f"gpu: {result['gpu']}")
    print(f"repo mode: {result['repo_mode']}")
    print(f"returncode: {result['returncode']}")
    print("----- stdout -----")
    print(result["stdout"] or "<empty>")
    print("----- stderr -----")
    print(result["stderr"] or "<empty>")
    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])
