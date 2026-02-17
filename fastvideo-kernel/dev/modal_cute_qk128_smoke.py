#!/usr/bin/env python3
"""Run CuTe qk128 smoke test on Modal serverless GPU.

Usage:
  python -m modal run fastvideo-kernel/dev/modal_cute_qk128_smoke.py

Optional environment variables:
  MODAL_SMOKE_IMAGE_TAG   Container image tag to use
  MODAL_SMOKE_REPO_URL    Git repo URL to clone in remote job
  MODAL_SMOKE_COMMIT      Commit SHA/branch to checkout
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
import subprocess
from textwrap import dedent

import modal

APP_NAME = "fastvideo-cute-qk128-smoke"
DEFAULT_IMAGE = "ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:latest"
GPU_SPEC = "B200:1"
DEFAULT_REPO_URL = "https://github.com/hao-ai-lab/FastVideo.git"
DEFAULT_FLASH_ATTN_REPO = "https://github.com/FoundationResearch/flash-attention.git"


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
REPO_URL = os.environ.get("MODAL_SMOKE_REPO_URL", DEFAULT_REPO_URL)
FLASH_ATTN_REPO = os.environ.get(
    "MODAL_SMOKE_FLASH_ATTN_REPO",
    DEFAULT_FLASH_ATTN_REPO,
)


def _resolve_repo_commit() -> str:
    if os.environ.get("MODAL_SMOKE_COMMIT"):
        return os.environ["MODAL_SMOKE_COMMIT"]
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
    except Exception:
        # Fallback keeps behavior deterministic when local git metadata is unavailable.
        return "main"


REPO_COMMIT = _resolve_repo_commit()
LOCAL_SMOKE_SCRIPT = Path(__file__).with_name("test_cute_qk128_smoke.py")
LOCAL_SMOKE_SOURCE_B64 = base64.b64encode(
    LOCAL_SMOKE_SCRIPT.read_bytes()
).decode("ascii")

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(IMAGE_TAG, add_python="3.12")
    .apt_install("git")
)


@app.function(gpu=GPU_SPEC, image=image, timeout=1800)
def run_remote_smoke() -> dict:
    """Execute the smoke test remotely and return captured logs."""
    cmd = dedent(
        f"""
        set -euo pipefail
        source /opt/venv/bin/activate
        rm -rf /FastVideo
        git clone "{REPO_URL}" /FastVideo
        cd /FastVideo
        git checkout "{REPO_COMMIT}"
        git submodule sync --recursive
        git submodule update --init --recursive
        if [ ! -d /FastVideo/fastvideo-kernel/include/flash-attention/flash_attn/cute ]; then
          echo "flash-attention submodule missing, cloning fallback repo..."
          rm -rf /FastVideo/fastvideo-kernel/include/flash-attention
          git clone "{FLASH_ATTN_REPO}" /FastVideo/fastvideo-kernel/include/flash-attention
        fi
        python -m pip install -U pip setuptools wheel
        python -m pip install -e /FastVideo/fastvideo-kernel/include/flash-attention/flash_attn/cute
        SMOKE_SCRIPT=/FastVideo/fastvideo-kernel/dev/test_cute_qk128_smoke.py
        if [ ! -f "$SMOKE_SCRIPT" ]; then
          echo "remote repo missing smoke script, injecting local copy..."
          python - <<'PY'
import base64
from pathlib import Path
content = base64.b64decode("{LOCAL_SMOKE_SOURCE_B64}".encode("ascii"))
Path("/tmp/test_cute_qk128_smoke.py").write_bytes(content)
PY
          SMOKE_SCRIPT=/tmp/test_cute_qk128_smoke.py
        fi
        python "$SMOKE_SCRIPT"
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
        "repo_url": REPO_URL,
        "repo_commit": REPO_COMMIT,
        "flash_attn_repo": FLASH_ATTN_REPO,
    }


@app.local_entrypoint()
def main() -> None:
    result = run_remote_smoke.remote()
    print("=== Modal CuTe qk128 smoke result ===")
    print(f"image: {result['image_tag']}")
    print(f"gpu: {result['gpu']}")
    print(f"repo: {result['repo_url']}")
    print(f"commit: {result['repo_commit']}")
    print(f"flash-attn repo: {result['flash_attn_repo']}")
    print(f"returncode: {result['returncode']}")
    print("----- stdout -----")
    print(result["stdout"] or "<empty>")
    print("----- stderr -----")
    print(result["stderr"] or "<empty>")
    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])
