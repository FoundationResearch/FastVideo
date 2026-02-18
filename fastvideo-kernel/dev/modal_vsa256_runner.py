#!/usr/bin/env python3
"""Run arbitrary VSA256 command on Modal serverless GPU.

Usage (fish):
  set -x MODAL_VSA256_RUN_CMD "pytest fastvideo-kernel/tests/test_vsa256_forward.py -v -s"
  modal run fastvideo-kernel/dev/modal_vsa256_runner.py

  set -x MODAL_VSA256_RUN_CMD "python fastvideo-kernel/benchmarks/bench_vsa256.py --q_seq_lens 49152 --kv_seq_lens 49152 --rep 20 --warmup 5"
  modal run fastvideo-kernel/dev/modal_vsa256_runner.py
"""

from __future__ import annotations

import os
import subprocess
from textwrap import dedent

import modal

APP_NAME = "fastvideo-vsa256-runner"
DEFAULT_IMAGE = "ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:latest"
GPU_SPEC = "B200:1"
DEFAULT_REPO_URL = "git@github.com:FoundationResearch/FastVideo.git"
DEFAULT_BRANCH = "128vsa"
DEFAULT_REPO_DIR = "/cache/FastVideo"
DEFAULT_VOLUME_NAME = "fastvideo-repo-cache"
DEFAULT_SECRET_NAME = "FR-FV"
DEFAULT_FLASH_ATTN_WHEEL = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp310-cp310-linux_x86_64.whl"
# DEFAULT_RUN_CMD = "pytest fastvideo-kernel/tests/test_vsa256_forward.py -v -s"
# DEFAULT_RUN_CMD = "python fastvideo-kernel/benchmarks/bench_vsa256.py --q_seq_lens 49152 --kv_seq_lens 49152 --rep 20 --warmup 5"
# DEFAULT_RUN_CMD = "FASTVIDEO_VSA_256=0 python fastvideo-kernel/benchmarks/bench_triton_vsa64_kernel.py --q_seq_lens 49152 --kv_seq_lens 49152 --rep 20 --warmup 5 --breakdown_rep 20"
# DEFAULT_RUN_CMD = "pytest fastvideo-kernel/tests/test_vsa256_forward_cross.py -v -s"
DEFAULT_RUN_CMD = "python fastvideo-kernel/dev/test_env_read.py"


def _resolve_image_tag() -> str:
    image_tag = os.environ.get("MODAL_SMOKE_IMAGE_TAG")
    if image_tag:
        return image_tag
    image_version = os.environ.get("IMAGE_VERSION")
    if image_version:
        return f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{image_version}"
    return DEFAULT_IMAGE


IMAGE_TAG = _resolve_image_tag()
app = modal.App(APP_NAME)
REPO_URL = os.environ.get("MODAL_SMOKE_REPO_URL", DEFAULT_REPO_URL)
REPO_BRANCH = os.environ.get("MODAL_SMOKE_REPO_BRANCH", DEFAULT_BRANCH)
REPO_DIR = os.environ.get("MODAL_SMOKE_REPO_DIR", DEFAULT_REPO_DIR)
FLASH_ATTN_WHEEL = os.environ.get(
    "MODAL_SMOKE_FLASH_ATTN_WHEEL", DEFAULT_FLASH_ATTN_WHEEL
)
REPO_VOLUME = modal.Volume.from_name(
    os.environ.get("MODAL_SMOKE_VOLUME_NAME", DEFAULT_VOLUME_NAME),
    create_if_missing=True,
)

image = modal.Image.from_registry(IMAGE_TAG, add_python="3.12")


@app.function(
    gpu=GPU_SPEC,
    image=image,
    timeout=7200,
    volumes={"/cache": REPO_VOLUME},
    secrets=[modal.Secret.from_name(DEFAULT_SECRET_NAME)],
)
def run_remote(run_cmd: str) -> dict:
    cmd = dedent(
        """
        set -euo pipefail
        source /opt/venv/bin/activate
        mkdir -p /root/.ssh
        chmod 700 /root/.ssh
        if [ -n "${GITHUB_SSH_KEY:-}" ]; then
          printf "%s\\n" "${GITHUB_SSH_KEY}" > /root/.ssh/id_ed25519
        elif [ -n "${SSH_PRIVATE_KEY:-}" ]; then
          printf "%s\\n" "${SSH_PRIVATE_KEY}" > /root/.ssh/id_ed25519
        else
          echo "ERROR: Secret must provide GITHUB_SSH_KEY (or SSH_PRIVATE_KEY)."
          exit 2
        fi
        chmod 600 /root/.ssh/id_ed25519
        ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null
        chmod 644 /root/.ssh/known_hosts
        export GIT_SSH_COMMAND='ssh -i /root/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes'

        if [ ! -d "${REPO_DIR}/.git" ]; then
          rm -rf "${REPO_DIR}"
          git clone --depth 1 --branch "${REPO_BRANCH}" "${REPO_URL}" "${REPO_DIR}"
        else
          # Volume cache may keep local tracked edits from prior runs.
          # Force a clean tree before switching to latest remote commit.
          git -C "${REPO_DIR}" reset --hard
          git -C "${REPO_DIR}" clean -fd
          git -C "${REPO_DIR}" remote set-url origin "${REPO_URL}"
          git -C "${REPO_DIR}" fetch origin "${REPO_BRANCH}" --depth 1
          git -C "${REPO_DIR}" checkout -B "${REPO_BRANCH}" FETCH_HEAD
          git -C "${REPO_DIR}" clean -fd
        fi

        cd "${REPO_DIR}"
        if [ -f .gitmodules ]; then
          git submodule update --init --recursive
        fi

        python -m pip install --upgrade "${FLASH_ATTN_WHEEL}"
        python -m pip install -e "${REPO_DIR}/fastvideo-kernel/include/flash-attention/flash_attn/cute"

        export FASTVIDEO_VSA_256=1
        export PYTHONPATH="${REPO_DIR}/fastvideo-kernel/python:${PYTHONPATH:-}"
        echo "PYTHONPATH=${PYTHONPATH}"
        eval "${RUN_CMD}"
        """
    ).strip()

    env = os.environ.copy()
    env.update(
        {
            "REPO_URL": REPO_URL,
            "REPO_BRANCH": REPO_BRANCH,
            "REPO_DIR": REPO_DIR,
            "FLASH_ATTN_WHEEL": FLASH_ATTN_WHEEL,
            "RUN_CMD": run_cmd,
        }
    )
    proc = subprocess.run(
        ["/bin/bash", "-lc", cmd],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    REPO_VOLUME.commit()
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "image_tag": IMAGE_TAG,
        "gpu": GPU_SPEC,
        "repo_url": REPO_URL,
        "repo_branch": REPO_BRANCH,
        "run_cmd": run_cmd,
    }


@app.local_entrypoint()
def main() -> None:
    run_cmd = os.environ.get("MODAL_VSA256_RUN_CMD", DEFAULT_RUN_CMD)
    result = run_remote.remote(run_cmd)
    print("=== Modal VSA256 run result ===")
    print(f"image: {result['image_tag']}")
    print(f"gpu: {result['gpu']}")
    print(f"repo: {result['repo_url']} @ {result['repo_branch']}")
    print(f"run_cmd: {result['run_cmd']}")
    print(f"returncode: {result['returncode']}")
    print("----- stdout -----")
    print(result["stdout"] or "<empty>")
    print("----- stderr -----")
    print(result["stderr"] or "<empty>")
    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])
