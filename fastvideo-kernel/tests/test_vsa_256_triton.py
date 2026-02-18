from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path

import pytest
import torch


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


def _reload_video_sparse_attn_triton():
    os.environ["FASTVIDEO_VSA_256"] = "1"
    os.environ["FASTVIDEO_VSA_256_BACKEND"] = "triton"
    os.environ["FASTVIDEO_VSA_256_TRITON_COMPAT"] = "1"
    os.environ["FASTVIDEO_KERNEL_VSA_FORCE_TRITON"] = "1"
    os.environ["FASTVIDEO_VSA_256_ALLOW_TRITON_FALLBACK"] = "1"
    _ensure_local_fastvideo_kernel()
    import fastvideo_kernel.block_sparse_attn as bs_mod
    import fastvideo_kernel.ops as ops_mod

    importlib.reload(bs_mod)
    importlib.reload(ops_mod)
    return ops_mod.video_sparse_attn


def _torch_vsa256_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_var: torch.Tensor,
    kv_var: torch.Tensor,
    topk_logical: int,
) -> torch.Tensor:
    bsz, heads, _sq, dim = q.shape
    q_blocks = q_var.numel()
    kv_blocks = kv_var.numel()
    q_block = int(q_var[0].item())
    kv_block = int(kv_var[0].item())

    q_c = q.view(bsz, heads, q_blocks, q_block, dim)
    k_c = k.view(bsz, heads, kv_blocks, kv_block, dim)
    v_c = v.view(bsz, heads, kv_blocks, kv_block, dim)
    q_c = (q_c.float().sum(dim=3) / q_var.view(1, 1, -1, 1)).to(q.dtype)
    k_c = (k_c.float().sum(dim=3) / kv_var.view(1, 1, -1, 1)).to(k.dtype)
    v_c = (v_c.float().sum(dim=3) / kv_var.view(1, 1, -1, 1)).to(v.dtype)

    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / math.sqrt(dim)
    attn = torch.softmax(scores, dim=-1)
    out_c = torch.matmul(attn, v_c)
    out_c = (
        out_c.view(bsz, heads, q_blocks, 1, dim)
        .repeat(1, 1, 1, q_block, 1)
        .view_as(q)
    )

    with torch.no_grad():
        topk_idx = torch.topk(scores.detach(), topk_logical, dim=-1).indices
        block_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(
            -1, topk_idx, True
        )
        token_mask = (
            block_mask.repeat_interleave(q_block, dim=2)
            .repeat_interleave(kv_block, dim=3)
            .to(torch.bool)
        )

    qf, kf, vf = q.float(), k.float(), v.float()
    logits = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(dim)
    logits = logits.masked_fill(~token_mask, float("-inf"))
    prob = torch.softmax(logits, dim=-1)
    out_s = torch.matmul(prob, vf).to(q.dtype)
    return out_c + out_s


def test_vsa256_triton_forward_backward_vs_torch_ref() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    bsz, heads, dim = 1, 8, 128
    q_blocks_256, kv_blocks_256 = 8, 12
    q_block = 256
    kv_block = 256
    topk_logical = 2
    sq = q_blocks_256 * q_block
    skv = kv_blocks_256 * kv_block

    q_base = torch.randn(bsz, heads, sq, dim, device=device, dtype=dtype)
    k_base = torch.randn(bsz, heads, skv, dim, device=device, dtype=dtype)
    v_base = torch.randn(bsz, heads, skv, dim, device=device, dtype=dtype)
    grad_out = torch.randn_like(q_base)

    q_var = torch.full((q_blocks_256,), q_block, dtype=torch.int32, device=device)
    kv_var = torch.full((kv_blocks_256,), kv_block, dtype=torch.int32, device=device)

    video_sparse_attn = _reload_video_sparse_attn_triton()

    q = q_base.detach().clone().requires_grad_(True)
    k = k_base.detach().clone().requires_grad_(True)
    v = v_base.detach().clone().requires_grad_(True)
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
    loss = (out * grad_out).sum()
    loss.backward()
    dq, dk, dv = q.grad.detach(), k.grad.detach(), v.grad.detach()

    q_ref = q_base.detach().clone().requires_grad_(True)
    k_ref = k_base.detach().clone().requires_grad_(True)
    v_ref = v_base.detach().clone().requires_grad_(True)
    out_ref = _torch_vsa256_reference(q_ref, k_ref, v_ref, q_var, kv_var, topk_logical)
    loss_ref = (out_ref * grad_out).sum()
    loss_ref.backward()
    dq_ref, dk_ref, dv_ref = q_ref.grad.detach(), k_ref.grad.detach(), v_ref.grad.detach()

    assert torch.isfinite(out).all().item()
    assert torch.isfinite(out_ref).all().item()
    assert torch.isfinite(dq).all().item()
    assert torch.isfinite(dk).all().item()
    assert torch.isfinite(dv).all().item()

    def _metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
        diff = (a - b).abs()
        avg_abs = diff.mean().item()
        max_rel = (diff.max() / (a.abs().mean() + 1e-6)).item()
        return avg_abs, max_rel

    m_out = _metrics(out_ref, out)
    m_dq = _metrics(dq_ref, dq)
    m_dk = _metrics(dk_ref, dk)
    m_dv = _metrics(dv_ref, dv)
    print(
        "[vsa256-triton] "
        f"out(avg_abs={m_out[0]:.6e}, max_rel={m_out[1]:.6e}), "
        f"dq(avg_abs={m_dq[0]:.6e}, max_rel={m_dq[1]:.6e}), "
        f"dk(avg_abs={m_dk[0]:.6e}, max_rel={m_dk[1]:.6e}), "
        f"dv(avg_abs={m_dv[0]:.6e}, max_rel={m_dv[1]:.6e})"
    )

    assert m_out[0] < 1e-3 and m_out[1] < 0.2
    assert m_dq[0] < 2e-2 and m_dq[1] < 0.5
    assert m_dk[0] < 2e-2 and m_dk[1] < 0.5
    assert m_dv[0] < 2e-2 and m_dv[1] < 0.5
