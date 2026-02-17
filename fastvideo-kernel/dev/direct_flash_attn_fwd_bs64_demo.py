#!/usr/bin/env python3
"""Direct-call demo: _flash_attn_fwd with n_block_size=64."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


def _ensure_flash_attn_importable() -> None:
    root = Path(__file__).resolve().parents[1]
    flash_attn_repo = root / "include" / "flash-attention"
    if not flash_attn_repo.exists():
        raise FileNotFoundError(
            f"flash-attention submodule not found: {flash_attn_repo}"
        )
    sys.path.insert(0, str(flash_attn_repo))


def build_block_sparse_tensors(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_heads: int,
    block_size: tuple[int, int],
    device: torch.device,
):
    block_m, block_n = block_size
    m_blocks = math.ceil(seqlen_q / block_m)
    n_blocks = math.ceil(seqlen_k / block_n)

    # Simple dense-like sparse pattern: every q block attends all kv blocks.
    mask_block_cnt = torch.full(
        (batch_size, num_heads, m_blocks),
        n_blocks,
        dtype=torch.int32,
        device=device,
    )
    mask_block_idx = torch.arange(
        n_blocks, dtype=torch.int32, device=device
    )[None, None, None, :].expand(batch_size, num_heads, m_blocks, n_blocks)

    # Keep full_* explicit empty tensors to avoid None-path corner cases.
    full_block_cnt = torch.zeros_like(mask_block_cnt)
    full_block_idx = torch.zeros_like(mask_block_idx)
    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    _ensure_flash_attn_importable()
    from flash_attn.cute.interface import _flash_attn_fwd
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch_size = 1
    seqlen = 1024
    num_heads = 8
    head_dim = 128

    # Directly test n_block_size=64 path.
    m_block_size = 128
    n_block_size = 64
    block_size_q = 256 if seqlen > m_block_size else m_block_size
    sparse_block_size = (block_size_q, n_block_size)

    q = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype
    )

    mask_cnt, mask_idx, full_cnt, full_idx = build_block_sparse_tensors(
        batch_size=batch_size,
        seqlen_q=seqlen,
        seqlen_k=seqlen,
        num_heads=num_heads,
        block_size=sparse_block_size,
        device=device,
    )
    block_sparse_tensors = BlockSparseTensorsTorch(
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        block_size=sparse_block_size,
    )

    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        block_sparse_tensors=block_sparse_tensors,
        causal=False,
    )

    print("direct _flash_attn_fwd call success with n_block_size=64")
    print(f"out.shape={tuple(out.shape)}, out.dtype={out.dtype}")
    if lse is None:
        print("lse=None")
    else:
        print(f"lse.shape={tuple(lse.shape)}, lse.dtype={lse.dtype}")


if __name__ == "__main__":
    main()


