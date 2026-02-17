#!/usr/bin/env python3
"""Minimal demo for calling FlashAttention CuTe block-sparse forward API.

This script uses the local submodule at:
fastvideo-kernel/include/flash-attention
"""

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


def build_causal_block_sparse_tensors(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_heads: int,
    block_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a simple causal block-sparse layout (mask-only path).

    Returns:
        mask_block_cnt: int32 tensor, shape (B, H, M)
        mask_block_idx: int32 tensor, shape (B, H, M, N)
    """
    block_m, block_n = block_size
    m_blocks = math.ceil(seqlen_q / block_m)
    n_blocks = math.ceil(seqlen_k / block_n)

    mask_block_cnt = torch.zeros(
        (batch_size, num_heads, m_blocks),
        dtype=torch.int32,
        device=device,
    )
    mask_block_idx = torch.zeros(
        (batch_size, num_heads, m_blocks, n_blocks),
        dtype=torch.int32,
        device=device,
    )

    for m in range(m_blocks):
        # Causal: query block m can attend to key blocks [0, m].
        valid_n = min(m + 1, n_blocks)
        if valid_n > 0:
            mask_block_idx[:, :, m, :valid_n] = torch.arange(
                valid_n,
                dtype=torch.int32,
                device=device,
            )
        mask_block_cnt[:, :, m] = valid_n

    return mask_block_cnt, mask_block_idx


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for flash_attn.cute demo.")

    _ensure_flash_attn_importable()
    from flash_attn.cute import flash_attn_func

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch_size = 2
    seqlen = 512
    num_heads = 8
    head_dim = 64
    # In flash_attn.cute forward, q_stage can become 2 when seqlen > tile_m(128),
    # so sparse_block_size_q must be a multiple of q_stage * tile_m (i.e. 256).
    block_size_q = 256 if seqlen > 128 else 128
    block_size = (block_size_q, 128)

    q = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype
    )

    mask_block_cnt, mask_block_idx = build_causal_block_sparse_tensors(
        batch_size=batch_size,
        seqlen_q=seqlen,
        seqlen_k=seqlen,
        num_heads=num_heads,
        block_size=block_size,
        device=device,
    )
    # SM100 block-sparse path is more stable when full_* tensors are present.
    # We provide explicit empty full lists (count=0) instead of None.
    full_block_cnt = torch.zeros_like(mask_block_cnt)
    full_block_idx = torch.zeros_like(mask_block_idx)

    out, lse = flash_attn_func(
        q,
        k,
        v,
        causal=False,  # sparsity pattern already encodes causal behavior
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        block_size=block_size,
    )

    print("flash_attn.cute block sparse call success")
    print(f"out.shape={tuple(out.shape)}, out.dtype={out.dtype}")
    if lse is None:
        print("lse=None (this path did not materialize LSE)")
    else:
        print(f"lse.shape={tuple(lse.shape)}, lse.dtype={lse.dtype}")


if __name__ == "__main__":
    main()


