#!/usr/bin/env python3
"""
Benchmark FlashAttention CuTe block-sparse route (forward + backward) and report TFLOPs.

Defaults are aligned with fastvideo-kernel/benchmarks/bench_vsa.py:
- batch_size=1
- num_heads=12
- head_dim=128
- q_seq_lens=[49152]
- topk defaults to ~90% sparsity

Difference:
- KV block size uses 128.
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
    raise ImportError("This benchmark requires triton (for triton.testing.do_bench).") from e


BLOCK_N = 128


def _ensure_flash_attn_importable() -> None:
    kernel_root = Path(__file__).resolve().parents[1]
    flash_attn_repo = kernel_root / "include" / "flash-attention"
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
    p = argparse.ArgumentParser(description="Benchmark FlashAttention CuTe block-sparse")
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
    # Approx: QK^T + PV, each is ~2*bs*h*q_len*(topk_blocks*block_n)*d
    return 4.0 * bs * h * d * q_len * (topk_blocks * block_n)


def create_qkv(
    batch: int,
    heads: int,
    q_len: int,
    kv_len: int,
    d: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # flash_attn.cute expects [B, S, H, D]
    q = torch.randn(batch, q_len, heads, d, dtype=dtype, device="cuda")
    k = torch.randn(batch, kv_len, heads, d, dtype=dtype, device="cuda")
    v = torch.randn(batch, kv_len, heads, d, dtype=dtype, device="cuda")
    return q, k, v


def choose_q_block_size(q_len: int) -> int:
    # For SM100 path in flash_attn.cute, q_stage can be 2 when seqlen > tile_m(128),
    # which requires sparse_block_size_q to be a multiple of 256.
    major, _minor = torch.cuda.get_device_capability()
    if major >= 10 and q_len > 128:
        return 256
    return 128


def build_block_sparse_tensors(
    batch_size: int,
    num_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    q_block_size: int,
    topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m_blocks = math.ceil(seqlen_q / q_block_size)
    n_blocks = math.ceil(seqlen_k / BLOCK_N)
    topk = min(max(1, topk), n_blocks)

    # mask_* contains sparse adjacency (q-block -> kv-block list)
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
    # full_* are explicit empty lists; this is more stable than None on SM100 path.
    full_block_cnt = torch.zeros_like(mask_block_cnt)
    full_block_idx = torch.zeros_like(mask_block_idx)

    for b in range(batch_size):
        for h in range(num_heads):
            # Create random top-k blocks for each q-block.
            rand_scores = torch.rand(m_blocks, n_blocks, device=device)
            topk_idx = torch.topk(rand_scores, topk, dim=-1).indices.to(torch.int32)
            mask_block_idx[b, h, :, :topk] = topk_idx

    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    _ensure_flash_attn_importable()
    from flash_attn.cute import flash_attn_func

    args = parse_arguments()
    set_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    bs, h, d = args.batch_size, args.num_heads, args.head_dim
    kv_seq_lens = args.kv_seq_lens if args.kv_seq_lens is not None else args.q_seq_lens
    if len(kv_seq_lens) != len(args.q_seq_lens):
        raise ValueError("kv_seq_lens must have the same number of entries as q_seq_lens (or be omitted).")

    print("VSA Benchmark via FlashAttention CuTe (block-sparse)")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"batch={bs}, heads={h}, head_dim={d}, dtype={args.dtype}, BLOCK_N={BLOCK_N}")
    print("NOTE: timings include Python-side sparse metadata construction only once per case.")

    for q_len, kv_len in zip(args.q_seq_lens, kv_seq_lens):
        q_block_size = choose_q_block_size(q_len)
        num_q_blocks = math.ceil(q_len / q_block_size)
        num_kv_blocks = math.ceil(kv_len / BLOCK_N)
        topk = args.topk if args.topk is not None else max(1, num_kv_blocks // 10)
        topk = min(topk, num_kv_blocks)

        print("\n" + "=" * 80)
        print(
            f"q_len={q_len}, kv_len={kv_len}, q_block={q_block_size}, kv_block={BLOCK_N}, "
            f"num_q_blocks={num_q_blocks}, num_kv_blocks={num_kv_blocks}, topk={topk}"
        )

        q, k, v = create_qkv(bs, h, q_len, kv_len, d, dtype)
        mask_cnt, mask_idx, full_cnt, full_idx = build_block_sparse_tensors(
            batch_size=bs,
            num_heads=h,
            seqlen_q=q_len,
            seqlen_k=kv_len,
            q_block_size=q_block_size,
            topk=topk,
            device=q.device,
        )

        def _fwd():
            return flash_attn_func(
                q,
                k,
                v,
                causal=False,
                mask_block_cnt=mask_cnt,
                mask_block_idx=mask_idx,
                full_block_cnt=full_cnt,
                full_block_idx=full_idx,
                block_size=(q_block_size, BLOCK_N),
            )

        # Trigger first compile before timing.
        _ = _fwd()
        torch.cuda.synchronize()
        fwd_ms = bench_ms(_fwd, warmup=args.warmup, rep=args.rep)

        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        out, _lse = flash_attn_func(
            q_,
            k_,
            v_,
            causal=False,
            mask_block_cnt=mask_cnt,
            mask_block_idx=mask_idx,
            full_block_cnt=full_cnt,
            full_block_idx=full_idx,
            block_size=(q_block_size, BLOCK_N),
        )
        og = torch.randn_like(out)
        loss = (out * og).sum()
        for _ in range(max(1, args.warmup // 2)):
            torch.autograd.grad(loss, (q_, k_, v_), retain_graph=True)
        torch.cuda.synchronize()
        bwd_ms = bench_ms(
            lambda: torch.autograd.grad(loss, (q_, k_, v_), retain_graph=True),
            warmup=0,
            rep=max(5, args.rep // 2),
        )

        flops = flops_sparse_attention(bs, h, d, q_len, topk, BLOCK_N)
        fwd_tflops = flops / fwd_ms * 1e-12 * 1e3
        bwd_tflops = (2.5 * flops) / bwd_ms * 1e-12 * 1e3
        print(f"fwd(cute): {fwd_ms:.3f} ms  | {fwd_tflops:.2f} TFLOPs (approx)")
        print(f"bwd(cute): {bwd_ms:.3f} ms  | {bwd_tflops:.2f} TFLOPs (approx)")


if __name__ == "__main__":
    main()


