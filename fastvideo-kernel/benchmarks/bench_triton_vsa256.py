#!/usr/bin/env python3
"""Benchmark VSA256 with Triton backend (route-A compatibility)."""

from __future__ import annotations

import argparse
import math
import os
import random

import numpy as np
import torch

try:
    from triton.testing import do_bench
except Exception as e:  # pragma: no cover
    raise ImportError("This benchmark requires triton (triton.testing.do_bench).") from e


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark VSA256 with Triton backend")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--head_dim", type=int, default=128, choices=[64, 128])
    p.add_argument("--q_seq_lens", type=int, nargs="+", default=[49152])
    p.add_argument("--kv_seq_lens", type=int, nargs="+", default=None)
    p.add_argument("--topk_logical", type=int, default=None)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    return p.parse_args()


def bench_ms(fn, warmup: int, rep: int) -> float:
    return do_bench(fn, warmup=warmup, rep=rep, quantiles=None)


def flops_sparse_attention(
    bs: int,
    h: int,
    d: int,
    q_len: int,
    avg_topk_logical: float,
    logical_kv_block: int = 256,
) -> float:
    return 4.0 * bs * h * d * q_len * (avg_topk_logical * logical_kv_block)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    os.environ["FASTVIDEO_VSA_256"] = "1"
    os.environ["FASTVIDEO_VSA_256_BACKEND"] = "triton"
    os.environ["FASTVIDEO_VSA_256_TRITON_COMPAT"] = "1"
    os.environ["FASTVIDEO_KERNEL_VSA_FORCE_TRITON"] = "1"

    from fastvideo_kernel.ops import video_sparse_attn

    args = parse_args()
    set_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    bs, h, d = args.batch_size, args.num_heads, args.head_dim
    kv_seq_lens = args.kv_seq_lens if args.kv_seq_lens is not None else args.q_seq_lens
    if len(kv_seq_lens) != len(args.q_seq_lens):
        raise ValueError("kv_seq_lens must align with q_seq_lens")

    print("VSA256 Triton Benchmark (route-A compatibility)")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"batch={bs}, heads={h}, head_dim={d}, dtype={args.dtype}")

    for q_len, kv_len in zip(args.q_seq_lens, kv_seq_lens):
        if q_len % 256 != 0 or kv_len % 256 != 0:
            print(f"[skip] q_len={q_len}, kv_len={kv_len} must be divisible by 256")
            continue
        q_blocks_256 = q_len // 256
        kv_blocks_256 = kv_len // 256
        topk_logical = args.topk_logical if args.topk_logical is not None else max(
            1, kv_blocks_256 // 10
        )
        topk_logical = min(topk_logical, kv_blocks_256)

        q = torch.randn(bs, h, q_len, d, dtype=dtype, device="cuda")
        k = torch.randn(bs, h, kv_len, d, dtype=dtype, device="cuda")
        v = torch.randn(bs, h, kv_len, d, dtype=dtype, device="cuda")
        q_var = torch.full((q_blocks_256,), 256, dtype=torch.int32, device="cuda")
        kv_var = torch.full((kv_blocks_256,), 256, dtype=torch.int32, device="cuda")

        def _fwd():
            return video_sparse_attn(
                q,
                k,
                v,
                kv_var,
                q_var,
                topk_logical,
                block_size=(4, 8, 8),
                compress_attn_weight=None,
            )

        out = _fwd()
        out_finite = torch.isfinite(out).all().item()
        fwd_ms = bench_ms(_fwd, warmup=args.warmup, rep=args.rep)
        flops = flops_sparse_attention(
            bs, h, d, q_len, float(topk_logical), logical_kv_block=256
        )
        fwd_tflops = flops / fwd_ms * 1e-12 * 1e3

        print("\n" + "=" * 100)
        print(
            f"q_len={q_len}, kv_len={kv_len}, q_blocks_256={q_blocks_256}, "
            f"kv_blocks_256={kv_blocks_256}, topk_logical={topk_logical}"
        )
        print(f"finite_check: out={out_finite}")
        print(f"forward(wrapper_e2e): {fwd_ms:.3f} ms | {fwd_tflops:.2f} TFLOPs (approx)")


if __name__ == "__main__":
    main()
