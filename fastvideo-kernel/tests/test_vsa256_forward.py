from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import pytest
import torch

from .utils import create_full_mask_from_block_mask, generate_block_sparse_mask_for_function


def _ensure_flash_attn_importable() -> None:
    kernel_root = Path(__file__).resolve().parents[1]
    flash_attn_repo = kernel_root / "include" / "flash-attention"
    if not flash_attn_repo.exists():
        raise FileNotFoundError(
            f"flash-attention submodule not found: {flash_attn_repo}"
        )
    repo_str = str(flash_attn_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _map_to_index_torch(
    block_map: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # block_map: [B, H, Qb, KVb]
    bsz, nhead, q_blocks, kv_blocks = block_map.shape
    index = torch.zeros(
        (bsz, nhead, q_blocks, kv_blocks), dtype=torch.int32, device=block_map.device
    )
    index_num = torch.zeros(
        (bsz, nhead, q_blocks), dtype=torch.int32, device=block_map.device
    )
    for b in range(bsz):
        for h in range(nhead):
            for q in range(q_blocks):
                row = torch.nonzero(block_map[b, h, q], as_tuple=False).flatten()
                cnt = int(row.numel())
                if cnt > 0:
                    index[b, h, q, :cnt] = row.to(torch.int32)
                index_num[b, h, q] = cnt
    return index, index_num


def _expand_mask_256_to_128(mask_256: torch.Tensor) -> torch.Tensor:
    # mask_256: [H, Qb256, KVb256] -> [H, Qb256, KVb128]
    h, qb, kvb256 = mask_256.shape
    kvb128 = kvb256 * 2
    mask_128 = torch.zeros((h, qb, kvb128), dtype=torch.bool, device=mask_256.device)
    parent_idx = torch.nonzero(mask_256, as_tuple=False)
    if parent_idx.numel() == 0:
        return mask_128
    hh = parent_idx[:, 0]
    qq = parent_idx[:, 1]
    kk = parent_idx[:, 2]
    child0 = 2 * kk
    child1 = child0 + 1
    mask_128[hh, qq, child0] = True
    mask_128[hh, qq, child1] = True
    return mask_128


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
    _ensure_flash_attn_importable()
    from flash_attn.cute import flash_attn_func

    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch = 1
    q_block = 256
    kv_block_logical = 256
    kv_block_kernel = 128

    sq = q_blocks_256 * q_block
    skv = kv_blocks_256 * kv_block_logical
    q = torch.randn(batch, heads, sq, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, skv, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, skv, head_dim, device=device, dtype=dtype)

    mask_256 = generate_block_sparse_mask_for_function(
        heads,
        q_blocks_256,
        kv_blocks_256,
        topk_logical,
        device=device,
    )
    full_mask = create_full_mask_from_block_mask(
        mask_256,
        torch.full((q_blocks_256,), q_block, dtype=torch.int32, device=device),
        torch.full((kv_blocks_256,), kv_block_logical, dtype=torch.int32, device=device),
        device=device,
    )
    out_ref = _dense_reference(q, k, v, full_mask)

    mask_128 = _expand_mask_256_to_128(mask_256).unsqueeze(0)
    mask_idx, mask_cnt = _map_to_index_torch(mask_128)
    full_cnt = torch.zeros_like(mask_cnt)
    full_idx = torch.zeros_like(mask_idx)

    out, _lse = flash_attn_func(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        causal=False,
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=(q_block, kv_block_kernel),
    )
    out = out.transpose(1, 2).contiguous()

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
