# SPDX-License-Identifier: Apache-2.0
"""
True end-to-end-ish SP gradient parity test using FastVideo TrainingPipeline code.

This test launches two torchrun jobs (WORLD_SIZE=2):
  - Run A: sp_size=1  (SP disabled; DP=2)
  - Run B: sp_size=2  (2-way SP sharding; DP=1)

Each job runs TrainingPipeline._transformer_forward_and_compute_loss() on a tiny
dummy transformer and a real TrainingBatch, then writes the rank0 gradient to
disk. The parent pytest compares the two gradients.

Run:
  source /home/hao_lab/miniconda3/etc/profile.d/conda.sh
  conda activate alexfv
  pytest -q fastvideo/tests/distributed/test_sp_grad_parity_e2e.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import pytest
import torch


def _run_worker(sp_size: int, out_file: str) -> None:
    env = os.environ.copy()
    env["SP_SIZE"] = str(sp_size)
    env["OUT_FILE"] = out_file
    # Ensure we don't accidentally engage attention backends that expect metadata.
    env.setdefault("FASTVIDEO_ATTENTION_BACKEND", "NONE")

    # Make sure the repo is importable inside the torchrun subprocess.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    env["PYTHONPATH"] = (
        repo_root if "PYTHONPATH" not in env else repo_root + os.pathsep + env["PYTHONPATH"]
    )

    cmd = [
        shutil.which("torchrun") or "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "fastvideo.tests.distributed._sp_grad_worker",
    ]
    subprocess.run(cmd, check=True, env=env)


def test_sp2_grad_matches_sp1_grad_end_to_end():
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires >=2 GPUs")

    with tempfile.TemporaryDirectory() as td:
        out_sp1 = os.path.join(td, "sp1.pt")
        out_sp2 = os.path.join(td, "sp2.pt")

        _run_worker(sp_size=1, out_file=out_sp1)
        _run_worker(sp_size=2, out_file=out_sp2)

        r1 = torch.load(out_sp1, map_location="cpu")
        r2 = torch.load(out_sp2, map_location="cpu")
        g1 = r1["scale_grad"]
        g2 = r2["scale_grad"]

        # Diagnostic prints (show up under pytest -s).
        diff = (g2 - g1).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
        denom = g1.abs().clamp_min(1e-12)
        max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0

        loss1 = float(r1.get("total_loss", 0.0))
        loss2 = float(r2.get("total_loss", 0.0))
        loss_abs = abs(loss2 - loss1)
        loss_rel = loss_abs / max(abs(loss1), 1e-12)

        print(
            f"[sp_grad_parity_e2e] grad diff: max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, max_rel={max_rel:.3e}"
        )
        print(
            f"[sp_grad_parity_e2e] loss diff: sp1={loss1:.8f}, sp2={loss2:.8f}, abs={loss_abs:.3e}, rel={loss_rel:.3e}"
        )

        torch.testing.assert_close(g2, g1, rtol=1e-5, atol=1e-6)


