#!/usr/bin/env python3
"""Test whether FlashAttention CuTe can run with m_block_size=64 stably."""

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


def _build_dense_blocksparse_tensors(
    batch_size: int,
    num_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
):
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    # Mirror interface.py logic for q_stage on SM100+:
    major, _minor = torch.cuda.get_device_capability()
    q_stage = 2 if (major >= 10 and seqlen_q > m_block_size) else 1
    sparse_block_q = q_stage * m_block_size

    m_blocks = math.ceil(seqlen_q / sparse_block_q)
    n_blocks = math.ceil(seqlen_k / n_block_size)

    mask_block_cnt = torch.full(
        (batch_size, num_heads, m_blocks),
        n_blocks,
        dtype=torch.int32,
        device="cuda",
    )
    mask_block_idx = torch.arange(
        n_blocks, dtype=torch.int32, device="cuda"
    )[None, None, None, :].expand(batch_size, num_heads, m_blocks, n_blocks).contiguous()
    full_block_cnt = torch.zeros_like(mask_block_cnt)
    full_block_idx = torch.zeros_like(mask_block_idx)

    return BlockSparseTensorsTorch(
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        block_size=(sparse_block_q, n_block_size),
    ), q_stage, sparse_block_q


def run_case(seqlen: int, head_dim: int = 128) -> None:
    from flash_attn.cute.interface import _flash_attn_fwd

    batch_size = 1
    num_heads = 8
    m_block_size = 64
    n_block_size = 64
    dtype = torch.bfloat16

    q = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device="cuda", dtype=dtype
    )
    k = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device="cuda", dtype=dtype
    )
    v = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device="cuda", dtype=dtype
    )

    blocksparse_tensors, q_stage, sparse_block_q = _build_dense_blocksparse_tensors(
        batch_size=batch_size,
        num_heads=num_heads,
        seqlen_q=seqlen,
        seqlen_k=seqlen,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
    )

    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        block_sparse_tensors=blocksparse_tensors,
        causal=False,
        return_lse=True,
    )

    out_nan = int(torch.isnan(out).sum().item())
    out_inf = int(torch.isinf(out).sum().item())
    lse_nan = int(torch.isnan(lse).sum().item()) if lse is not None else 0
    lse_inf = int(torch.isinf(lse).sum().item()) if lse is not None else 0

    print(
        f"seqlen={seqlen} | m_block=64 n_block=64 | q_stage={q_stage} "
        f"sparse_block_q={sparse_block_q} | "
        f"out_nan={out_nan} out_inf={out_inf} lse_nan={lse_nan} lse_inf={lse_inf}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    _ensure_flash_attn_importable()

    print("Testing CuTe with m_block_size=64, n_block_size=64")
    for seqlen in (64, 128, 256, 512, 1024, 2048):
        run_case(seqlen)


if __name__ == "__main__":
    main()


