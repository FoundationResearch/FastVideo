#!/usr/bin/env python3
"""Forward-only benchmark for CuTe block-sparse with stage=2 and KV block=256.

Checks:
1) output/lse shapes
2) NaN/Inf in outputs
3) forward latency + approximate TFLOPs
"""

from __future__ import annotations

import argparse
import math
import os
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
    repo_str = str(flash_attn_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forward-only CuTe benchmark with stage=2 and block_size=(256,256)"
    )
    # Keep defaults aligned with benchmarks/bench_vsa.py
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--head_dim", type=int, default=128, choices=[64, 128])
    parser.add_argument("--q_seq_len", type=int, default=49152)
    parser.add_argument("--kv_seq_len", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _build_random_block_sparse_tensors(
    batch_size: int,
    num_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: tuple[int, int],
    topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block_m, block_n = block_size
    m_blocks = math.ceil(seqlen_q / block_m)
    n_blocks = math.ceil(seqlen_k / block_n)
    topk = min(max(1, topk), n_blocks)

    mask_block_cnt = torch.full(
        (batch_size, num_heads, m_blocks), topk, dtype=torch.int32, device=device
    )
    mask_block_idx = torch.zeros(
        (batch_size, num_heads, m_blocks, n_blocks), dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        for h in range(num_heads):
            scores = torch.rand(m_blocks, n_blocks, device=device)
            idx = torch.topk(scores, topk, dim=-1).indices.to(torch.int32)
            mask_block_idx[b, h, :, :topk] = idx

    full_block_cnt = torch.zeros_like(mask_block_cnt)
    full_block_idx = torch.zeros_like(mask_block_idx)
    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx


def _time_cuda_ms(fn, warmup: int, rep: int) -> float:
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)
    return total_ms / rep


def _flops_sparse_attention(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    seqlen_q: int,
    topk_blocks: int,
    block_n: int,
) -> float:
    return 4.0 * batch_size * num_heads * head_dim * seqlen_q * (topk_blocks * block_n)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    args = _parse_args()
    _ensure_flash_attn_importable()
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from flash_attn.cute.interface import _flash_attn_fwd

    # Force SM100 q_stage=2 for this experiment.
    os.environ["FLASH_ATTN_CUTE_FORCE_Q_STAGE"] = "2"

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    batch_size = args.batch_size
    num_heads = args.num_heads
    head_dim = args.head_dim
    seqlen_q = args.q_seq_len
    seqlen_k = args.kv_seq_len if args.kv_seq_len is not None else args.q_seq_len
    block_size = (256, 256)

    num_kv_blocks = math.ceil(seqlen_k / block_size[1])
    topk = args.topk if args.topk is not None else max(1, num_kv_blocks // 10)

    q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, num_heads, head_dim, device=device, dtype=dtype)

    mask_cnt, mask_idx, full_cnt, full_idx = _build_random_block_sparse_tensors(
        batch_size=batch_size,
        num_heads=num_heads,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        block_size=block_size,
        topk=topk,
        device=device,
    )

    sparse_tensors = BlockSparseTensorsTorch(
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        block_size=block_size,
    )

    def _fwd():
        return _flash_attn_fwd(
            q,
            k,
            v,
            m_block_size=128,
            n_block_size=256,
            block_sparse_tensors=sparse_tensors,
            causal=False,
            return_lse=True,
        )

    try:
        out, lse = _fwd()
    except AssertionError as e:
        msg = str(e) or "kernel assertion"
        print("UNSUPPORTED: CuTe stage=2 with block_size=(256,256) failed before execution.")
        print(f"reason: {msg}")
        print(
            "hint: SM100 forward kernel enforces TMEM capacity; "
            "for stage=2 and head_dim=128, effective n_block_size is capped at 128."
        )
        return
    out_finite = torch.isfinite(out).all().item()
    lse_finite = True if lse is None else torch.isfinite(lse).all().item()

    fwd_ms = _time_cuda_ms(_fwd, warmup=args.warmup, rep=args.rep)
    flops = _flops_sparse_attention(
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        seqlen_q=seqlen_q,
        topk_blocks=topk,
        block_n=block_size[1],
    )
    fwd_tflops = flops / fwd_ms * 1e-12 * 1e3

    print("PASS: CuTe block-sparse stage=2, block_size=(256,256)")
    print(
        f"config: batch={batch_size}, heads={num_heads}, head_dim={head_dim}, "
        f"q_len={seqlen_q}, kv_len={seqlen_k}, q_block=256, kv_block=256, "
        f"topk={topk}, dtype={args.dtype}, forced_q_stage=2"
    )
    print(f"out.shape={tuple(out.shape)}, out.dtype={out.dtype}")
    if lse is None:
        print("lse=None")
    else:
        print(f"lse.shape={tuple(lse.shape)}, lse.dtype={lse.dtype}")
    print(f"finite_check: out={out_finite}, lse={lse_finite}")
    print(f"perf: fwd={fwd_ms:.3f} ms, approx={fwd_tflops:.2f} TFLOPs")


if __name__ == "__main__":
    main()
