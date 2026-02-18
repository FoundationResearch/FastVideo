from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path
from typing import Tuple

import pytest
import torch

from .utils import create_full_mask_from_block_mask


def _ensure_local_fastvideo_kernel() -> None:
    repo_python = Path(__file__).resolve().parents[1] / "python"
    repo_python_str = str(repo_python)
    if repo_python_str not in sys.path:
        sys.path.insert(0, repo_python_str)
    loaded = sys.modules.get("fastvideo_kernel")
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", "") or ""
        if repo_python_str not in loaded_file:
            for mod_name in list(sys.modules.keys()):
                if mod_name == "fastvideo_kernel" or mod_name.startswith("fastvideo_kernel."):
                    del sys.modules[mod_name]


def _dense_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    full_mask: torch.Tensor,
) -> torch.Tensor:
    # q,k,v: [B,H,S,D], full_mask: [H,Sq,Skv]
    qf = q.float()
    kf = k.float()
    vf = v.float()
    attn = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    attn = attn.masked_fill(~full_mask.unsqueeze(0), float("-inf"))
    prob = torch.softmax(attn, dim=-1)
    return torch.matmul(prob, vf).to(q.dtype)


def _run_case(
    heads: int,
    head_dim: int,
    q_blocks_256: int,
    kv_blocks_256: int,
    topk_logical: int,
) -> Tuple[float, float]:
    assert torch.cuda.is_available()
    os.environ["FASTVIDEO_VSA_256"] = "1"
    os.environ["FASTVIDEO_VSA_256_BACKEND"] = "cute"
    _ensure_local_fastvideo_kernel()
    import fastvideo_kernel.ops as fv_ops

    importlib.reload(fv_ops)
    video_sparse_attn = fv_ops.video_sparse_attn

    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch = 1
    q_block = 256
    kv_block_logical = 256

    sq = q_blocks_256 * q_block
    skv = kv_blocks_256 * kv_block_logical
    q = torch.randn(batch, heads, sq, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, skv, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, skv, head_dim, device=device, dtype=dtype)

    q_var = torch.full((q_blocks_256,), q_block, dtype=torch.int32, device=device)
    kv_var = torch.full((kv_blocks_256,), kv_block_logical, dtype=torch.int32, device=device)

    q_c = q.view(batch, heads, q_blocks_256, q_block, head_dim)
    k_c = k.view(batch, heads, kv_blocks_256, kv_block_logical, head_dim)
    v_c = v.view(batch, heads, kv_blocks_256, kv_block_logical, head_dim)
    q_c = (q_c.float().sum(dim=3) / q_var.view(1, 1, -1, 1)).to(q.dtype)
    k_c = (k_c.float().sum(dim=3) / kv_var.view(1, 1, -1, 1)).to(k.dtype)
    v_c = (v_c.float().sum(dim=3) / kv_var.view(1, 1, -1, 1)).to(v.dtype)

    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (head_dim**0.5)
    attn = torch.softmax(scores, dim=-1)
    out_c = torch.matmul(attn, v_c)
    out_c = (
        out_c.view(batch, heads, q_blocks_256, 1, head_dim)
        .repeat(1, 1, 1, q_block, 1)
        .view(batch, heads, sq, head_dim)
    )
    topk_idx = torch.topk(scores, topk_logical, dim=-1).indices
    mask_256 = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_idx, True)[0]
    full_mask = create_full_mask_from_block_mask(
        mask_256,
        q_var,
        kv_var,
        device=device,
    )
    out_sparse_ref = _dense_reference(q, k, v, full_mask)
    out_ref = out_c + out_sparse_ref

    out = video_sparse_attn(
        q,
        k,
        v,
        kv_var,
        q_var,
        topk_logical,
        block_size=(4, 8, 8),
        compress_attn_weight=None,
    )

    assert torch.isfinite(out).all().item(), "NaN/Inf found in kernel output"
    diff = (out_ref - out).abs()
    avg_abs = diff.mean().item()
    max_rel = (diff.max() / (out_ref.abs().mean() + 1e-6)).item()
    return avg_abs, max_rel


@pytest.mark.cuda
def test_vsa256_forward_qk_equal() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    avg_abs, max_rel = _run_case(
        heads=8,
        head_dim=128,
        q_blocks_256=8,
        kv_blocks_256=8,
        topk_logical=2,
    )
    print(f"[vsa256 qk_equal] avg_abs={avg_abs:.6e}, max_rel={max_rel:.6e}")
    assert avg_abs < 5e-2
    assert max_rel < 2.0


@pytest.mark.cuda
def test_vsa256_forward_qk_diff() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    avg_abs, max_rel = _run_case(
        heads=8,
        head_dim=128,
        q_blocks_256=8,
        kv_blocks_256=12,
        topk_logical=2,
    )
    print(f"[vsa256 qk_diff] avg_abs={avg_abs:.6e}, max_rel={max_rel:.6e}")
    assert avg_abs < 5e-2
    assert max_rel < 2.0
