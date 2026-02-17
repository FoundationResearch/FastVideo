#!/usr/bin/env python3
"""Benchmark FlashAttention CuTe block-sparse forward with n_block_size=64."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch

try:
    from triton.testing import do_bench
except Exception as e:  # pragma: no cover
    raise ImportError("This benchmark requires triton (for triton.testing.do_bench).") from e


BLOCK_M = 64
BLOCK_N = 64


def _ensure_flash_attn_importable() -> None:
    root = Path(__file__).resolve().parents[1]
    flash_attn_repo = root / "include" / "flash-attention"
    if not flash_attn_repo.exists():
        raise FileNotFoundError(f"flash-attention submodule not found: {flash_attn_repo}")
    sys.path.insert(0, str(flash_attn_repo))


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark CuTe VSA forward (n_block_size=64)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--head_dim", type=int, default=128, choices=[64, 128])
    p.add_argument("--topk", type=int, default=None, help="KV blocks per Q block (default: ~90%% sparsity)")
    p.add_argument("--q_seq_lens", type=int, nargs="+", default=[49152], help="Q sequence lengths")
    p.add_argument("--kv_seq_lens", type=int, nargs="+", default=None, help="KV sequence lengths (defaults to q_seq_len)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    return p.parse_args()


def bench_ms(fn: Callable[[], object], warmup: int, rep: int) -> float:
    return do_bench(fn, warmup=warmup, rep=rep, quantiles=None)


def flops_sparse_attention(
    bs: int,
    h: int,
    d: int,
    q_len: int,
    topk_blocks: int,
    block_n: int,
) -> float:
    return 4.0 * bs * h * d * q_len * (topk_blocks * block_n)


def create_qkv(
    batch: int,
    heads: int,
    q_len: int,
    kv_len: int,
    d: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # _flash_attn_fwd expects [B, S, H, D]
    q = torch.randn(batch, q_len, heads, d, dtype=dtype, device="cuda")
    k = torch.randn(batch, kv_len, heads, d, dtype=dtype, device="cuda")
    v = torch.randn(batch, kv_len, heads, d, dtype=dtype, device="cuda")
    return q, k, v


def choose_q_sparse_block_size(q_len: int, m_block_size: int = 128) -> int:
    # Keep q_block aligned with q_stage rule on SM100+.
    major, _ = torch.cuda.get_device_capability()
    if major >= 10 and q_len > m_block_size:
        return 2 * m_block_size
    return m_block_size


def build_block_sparse_tensors(
    batch_size: int,
    num_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    q_block_size: int,
    topk: int,
    device: torch.device,
):
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    m_blocks = math.ceil(seqlen_q / q_block_size)
    n_blocks = math.ceil(seqlen_k / BLOCK_N)
    topk = min(max(1, topk), n_blocks)

    mask_block_cnt = torch.full(
        (batch_size, num_heads, m_blocks),
        topk,
        dtype=torch.int32,
        device=device,
    )
    mask_block_idx = torch.zeros(
        (batch_size, num_heads, m_blocks, n_blocks),
        dtype=torch.int32,
        device=device,
    )

    for b in range(batch_size):
        for h in range(num_heads):
            scores = torch.rand(m_blocks, n_blocks, device=device)
            idx = torch.topk(scores, topk, dim=-1).indices.to(torch.int32)
            mask_block_idx[b, h, :, :topk] = idx

    # Explicit empty full tensors are more stable than None on SM100 path.
    full_block_cnt = torch.zeros_like(mask_block_cnt)
    full_block_idx = torch.zeros_like(mask_block_idx)

    return BlockSparseTensorsTorch(
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        block_size=(q_block_size, BLOCK_N),
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark.")

    _ensure_flash_attn_importable()
    from flash_attn.cute.interface import _flash_attn_fwd

    args = parse_arguments()
    set_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    m_block_size = 128
    n_block_size = 64

    bs, h, d = args.batch_size, args.num_heads, args.head_dim
    kv_seq_lens = args.kv_seq_lens if args.kv_seq_lens is not None else args.q_seq_lens
    if len(kv_seq_lens) != len(args.q_seq_lens):
        raise ValueError("kv_seq_lens must have the same number of entries as q_seq_lens (or be omitted).")

    print("CuTe VSA Forward Benchmark (direct _flash_attn_fwd, n_block_size=64)")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"batch={bs}, heads={h}, head_dim={d}, dtype={args.dtype}")
    print(f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, m_block_size={m_block_size}, n_block_size={n_block_size}")

    for q_len, kv_len in zip(args.q_seq_lens, kv_seq_lens):
        if q_len % BLOCK_M != 0 or kv_len % BLOCK_N != 0:
            print(f"[skip] q_len={q_len}, kv_len={kv_len} must be divisible by 64")
            continue

        q_block_size = choose_q_sparse_block_size(q_len, m_block_size=m_block_size)
        num_q_blocks = math.ceil(q_len / q_block_size)
        num_kv_blocks = math.ceil(kv_len / BLOCK_N)
        topk = args.topk if args.topk is not None else max(1, num_kv_blocks // 10)
        topk = min(topk, num_kv_blocks)

        print("\n" + "=" * 80)
        print(
            f"q_len={q_len}, kv_len={kv_len}, q_sparse_block={q_block_size}, "
            f"num_q_blocks={num_q_blocks}, num_kv_blocks={num_kv_blocks}, topk={topk}"
        )

        q, k, v = create_qkv(bs, h, q_len, kv_len, d, dtype)
        block_sparse_tensors = build_block_sparse_tensors(
            batch_size=bs,
            num_heads=h,
            seqlen_q=q_len,
            seqlen_k=kv_len,
            q_block_size=q_block_size,
            topk=topk,
            device=q.device,
        )

        def _fwd():
            return _flash_attn_fwd(
                q,
                k,
                v,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                block_sparse_tensors=block_sparse_tensors,
                causal=False,
            )

        _ = _fwd()
        torch.cuda.synchronize()
        fwd_ms = bench_ms(_fwd, warmup=args.warmup, rep=args.rep)

        flops = flops_sparse_attention(bs, h, d, q_len, topk, BLOCK_N)
        fwd_tflops = flops / fwd_ms * 1e-12 * 1e3
        print(f"fwd(cute, bs64): {fwd_ms:.3f} ms  | {fwd_tflops:.2f} TFLOPs (approx)")


if __name__ == "__main__":
    main()


