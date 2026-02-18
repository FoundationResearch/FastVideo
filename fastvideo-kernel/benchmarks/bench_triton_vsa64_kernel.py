#!/usr/bin/env python3
"""Benchmark Triton VSA 64x64 bare kernel performance.

This benchmark focuses on the Triton sparse forward kernel:
  fastvideo_kernel.triton_kernels.block_sparse_attn_triton.triton_block_sparse_attn_forward

It reports:
- kernel_only latency / TFLOPs
- map_to_index latency
- manual_e2e latency / TFLOPs (mask generation + map_to_index + kernel)
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Callable

import numpy as np
import torch

try:
    from triton.testing import do_bench
except Exception as e:  # pragma: no cover
    raise ImportError("This benchmark requires triton (triton.testing.do_bench).") from e

BLOCK_M = 64
BLOCK_N = 64


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Triton VSA 64x64 bare kernel")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--head_dim", type=int, default=128, choices=[64, 128])
    p.add_argument("--q_seq_lens", type=int, nargs="+", default=[49152])
    p.add_argument("--kv_seq_lens", type=int, nargs="+", default=None)
    p.add_argument("--topk", type=int, default=None, help="KV blocks per Q block")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--breakdown_rep", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    return p.parse_args()


def bench_ms(fn: Callable[[], object], warmup: int, rep: int) -> float:
    return do_bench(fn, warmup=warmup, rep=rep, quantiles=None)


def make_block_map(
    bs: int,
    h: int,
    q_blocks: int,
    kv_blocks: int,
    topk: int,
) -> torch.Tensor:
    scores = torch.rand(bs, h, q_blocks, kv_blocks, device="cuda")
    topk = min(max(1, topk), kv_blocks)
    idx = torch.topk(scores, topk, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def flops_sparse_attention(
    bs: int,
    h: int,
    d: int,
    q_len: int,
    avg_topk_blocks: float,
) -> float:
    # Approx QK^T + PV
    return 4.0 * bs * h * d * q_len * (avg_topk_blocks * BLOCK_N)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    from fastvideo_kernel.triton_kernels.index import map_to_index
    from fastvideo_kernel.triton_kernels.block_sparse_attn_triton import (
        triton_block_sparse_attn_forward,
    )

    args = parse_args()
    set_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    bs, h, d = args.batch_size, args.num_heads, args.head_dim
    kv_seq_lens = args.kv_seq_lens if args.kv_seq_lens is not None else args.q_seq_lens
    if len(kv_seq_lens) != len(args.q_seq_lens):
        raise ValueError("kv_seq_lens must align with q_seq_lens")

    print("Triton VSA Bare Kernel Benchmark (64x64)")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"batch={bs}, heads={h}, head_dim={d}, dtype={args.dtype}")
    print("metrics: kernel_only, map_to_index, manual_e2e")

    for q_len, kv_len in zip(args.q_seq_lens, kv_seq_lens):
        if q_len % BLOCK_M != 0 or kv_len % BLOCK_N != 0:
            print(
                f"[skip] q_len={q_len}, kv_len={kv_len} must be divisible by 64"
            )
            continue

        q_blocks = q_len // BLOCK_M
        kv_blocks = kv_len // BLOCK_N
        topk = args.topk if args.topk is not None else max(1, kv_blocks // 10)
        topk = min(topk, kv_blocks)

        q = torch.randn(bs, h, q_len, d, dtype=dtype, device="cuda")
        k = torch.randn(bs, h, kv_len, d, dtype=dtype, device="cuda")
        v = torch.randn(bs, h, kv_len, d, dtype=dtype, device="cuda")

        block_map = make_block_map(bs, h, q_blocks, kv_blocks, topk)
        idx, num = map_to_index(block_map)
        variable_block_sizes = torch.full(
            (kv_blocks,), BLOCK_N, dtype=torch.int32, device="cuda"
        )

        def _kernel_only():
            return triton_block_sparse_attn_forward(q, k, v, idx, num, variable_block_sizes)

        def _map_to_index_only():
            return map_to_index(block_map)

        def _manual_e2e():
            bm = make_block_map(bs, h, q_blocks, kv_blocks, topk)
            i, n = map_to_index(bm)
            return triton_block_sparse_attn_forward(q, k, v, i, n, variable_block_sizes)

        out, lse = _kernel_only()
        out_finite = torch.isfinite(out).all().item()
        lse_finite = torch.isfinite(lse).all().item()

        avg_topk_blocks = num.float().mean().item()
        flops = flops_sparse_attention(bs, h, d, q_len, avg_topk_blocks)

        kernel_ms = bench_ms(_kernel_only, warmup=args.warmup, rep=args.rep)
        map_to_index_ms = bench_ms(
            _map_to_index_only, warmup=max(1, args.warmup // 2), rep=args.breakdown_rep
        )
        e2e_ms = bench_ms(_manual_e2e, warmup=args.warmup, rep=args.rep)

        kernel_tflops = flops / kernel_ms * 1e-12 * 1e3
        e2e_tflops = flops / e2e_ms * 1e-12 * 1e3

        print("\n" + "=" * 100)
        print(
            f"q_len={q_len}, kv_len={kv_len}, q_blocks={q_blocks}, kv_blocks={kv_blocks}, "
            f"topk={topk}, avg_topk_blocks={avg_topk_blocks:.2f}"
        )
        print(f"finite_check: out={out_finite}, lse={lse_finite}")
        print(f"kernel_only:  {kernel_ms:.3f} ms | {kernel_tflops:.2f} TFLOPs (approx)")
        print(f"map_to_index: {map_to_index_ms:.3f} ms")
        print(f"manual_e2e:   {e2e_ms:.3f} ms | {e2e_tflops:.2f} TFLOPs (approx)")


if __name__ == "__main__":
    main()
