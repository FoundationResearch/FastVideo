#!/usr/bin/env python3
"""Benchmark VSA-256 forward path.

Semantics:
- Logical sparse layout uses q_block=256, kv_block=256.
- Before kernel launch, each selected kv block i is expanded to
  two kernel blocks (2*i, 2*i+1), so kernel runs with kv_block=128.

Outputs:
- kernel-only time: prebuilt sparse tensors + kernel call
- e2e time: mask generation + 256->128 expansion + index build + kernel call
"""

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
    raise ImportError("This benchmark requires triton (triton.testing.do_bench).") from e

Q_BLOCK = 256
KV_BLOCK_LOGICAL = 256
KV_BLOCK_KERNEL = 128


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark VSA-256 (logical 256, kernel kv128)")
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


def _map_to_index_torch(block_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


def _make_logical_mask(
    bs: int,
    h: int,
    q_blocks_256: int,
    kv_blocks_256: int,
    topk_logical: int,
) -> torch.Tensor:
    scores = torch.rand(bs, h, q_blocks_256, kv_blocks_256, device="cuda")
    topk_logical = min(max(1, topk_logical), kv_blocks_256)
    idx = torch.topk(scores, topk_logical, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def _expand_mask_256_to_128(mask_256: torch.Tensor) -> torch.Tensor:
    bsz, h, qb, kvb256 = mask_256.shape
    kvb128 = kvb256 * 2
    mask_128 = torch.zeros((bsz, h, qb, kvb128), dtype=torch.bool, device=mask_256.device)
    pos = torch.nonzero(mask_256, as_tuple=False)
    if pos.numel() == 0:
        return mask_128
    bb = pos[:, 0]
    hh = pos[:, 1]
    qq = pos[:, 2]
    kk = pos[:, 3]
    child0 = 2 * kk
    child1 = child0 + 1
    mask_128[bb, hh, qq, child0] = True
    mask_128[bb, hh, qq, child1] = True
    return mask_128


def flops_sparse_attention(
    bs: int,
    h: int,
    d: int,
    q_len: int,
    avg_topk_kernel: float,
) -> float:
    return 4.0 * bs * h * d * q_len * (avg_topk_kernel * KV_BLOCK_KERNEL)


def bench_ms(fn: Callable[[], object], warmup: int, rep: int) -> float:
    return do_bench(fn, warmup=warmup, rep=rep, quantiles=None)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    _ensure_flash_attn_importable()
    from flash_attn.cute import flash_attn_func

    args = parse_args()
    set_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    bs, h, d = args.batch_size, args.num_heads, args.head_dim
    kv_seq_lens = args.kv_seq_lens if args.kv_seq_lens is not None else args.q_seq_lens
    if len(kv_seq_lens) != len(args.q_seq_lens):
        raise ValueError("kv_seq_lens must align with q_seq_lens")

    print("VSA256 Benchmark (logical q/kv=256, kernel kv=128)")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"batch={bs}, heads={h}, head_dim={d}, dtype={args.dtype}")

    for q_len, kv_len in zip(args.q_seq_lens, kv_seq_lens):
        if q_len % Q_BLOCK != 0 or kv_len % KV_BLOCK_LOGICAL != 0:
            print(
                f"[skip] q_len={q_len}, kv_len={kv_len} must be divisible by 256"
            )
            continue
        q_blocks_256 = q_len // Q_BLOCK
        kv_blocks_256 = kv_len // KV_BLOCK_LOGICAL
        topk_logical = (
            args.topk_logical
            if args.topk_logical is not None
            else max(1, kv_blocks_256 // 10)
        )
        topk_logical = min(topk_logical, kv_blocks_256)

        q = torch.randn(bs, h, q_len, d, dtype=dtype, device="cuda")
        k = torch.randn(bs, h, kv_len, d, dtype=dtype, device="cuda")
        v = torch.randn(bs, h, kv_len, d, dtype=dtype, device="cuda")
        q_c = q.transpose(1, 2).contiguous()
        k_c = k.transpose(1, 2).contiguous()
        v_c = v.transpose(1, 2).contiguous()

        logical_mask = _make_logical_mask(bs, h, q_blocks_256, kv_blocks_256, topk_logical)
        mask_128 = _expand_mask_256_to_128(logical_mask)
        mask_idx, mask_cnt = _map_to_index_torch(mask_128)
        full_cnt = torch.zeros_like(mask_cnt)
        full_idx = torch.zeros_like(mask_idx)

        def _kernel_only():
            return flash_attn_func(
                q_c,
                k_c,
                v_c,
                causal=False,
                mask_block_cnt=mask_cnt,
                mask_block_idx=mask_idx,
                full_block_cnt=full_cnt,
                full_block_idx=full_idx,
                block_size=(Q_BLOCK, KV_BLOCK_KERNEL),
            )

        def _e2e():
            m256 = _make_logical_mask(bs, h, q_blocks_256, kv_blocks_256, topk_logical)
            m128 = _expand_mask_256_to_128(m256)
            idx, cnt = _map_to_index_torch(m128)
            zcnt = torch.zeros_like(cnt)
            zidx = torch.zeros_like(idx)
            return flash_attn_func(
                q_c,
                k_c,
                v_c,
                causal=False,
                mask_block_cnt=cnt,
                mask_block_idx=idx,
                full_block_cnt=zcnt,
                full_block_idx=zidx,
                block_size=(Q_BLOCK, KV_BLOCK_KERNEL),
            )

        out, lse = _kernel_only()
        out_finite = torch.isfinite(out).all().item()
        lse_finite = True if lse is None else torch.isfinite(lse).all().item()
        avg_topk_kernel = mask_cnt.float().mean().item()
        flops = flops_sparse_attention(bs, h, d, q_len, avg_topk_kernel)

        kernel_ms = bench_ms(_kernel_only, warmup=args.warmup, rep=args.rep)
        e2e_ms = bench_ms(_e2e, warmup=args.warmup, rep=args.rep)
        kernel_tflops = flops / kernel_ms * 1e-12 * 1e3
        e2e_tflops = flops / e2e_ms * 1e-12 * 1e3

        print("\n" + "=" * 100)
        print(
            f"q_len={q_len}, kv_len={kv_len}, q_blocks_256={q_blocks_256}, "
            f"kv_blocks_256={kv_blocks_256}, topk_logical={topk_logical}, "
            f"avg_topk_kernel={avg_topk_kernel:.2f}"
        )
        print(f"finite_check: out={out_finite}, lse={lse_finite}")
        print(f"kernel_only: {kernel_ms:.3f} ms | {kernel_tflops:.2f} TFLOPs (approx)")
        print(f"e2e_total:   {e2e_ms:.3f} ms | {e2e_tflops:.2f} TFLOPs (approx)")


if __name__ == "__main__":
    main()
