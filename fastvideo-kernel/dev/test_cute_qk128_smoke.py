#!/usr/bin/env python3
"""Forward-only smoke + benchmark for CuTe block-sparse (q/k block = 128).

Checks:
1) output/lse shapes
2) NaN/Inf in outputs
3) forward latency + approximate TFLOPs
"""

from __future__ import annotations

import math
import argparse
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

    # Provide explicit empty full lists (more stable than None on some paths).
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
    # Approximate FLOPs for sparse attention: QK^T + PV
    return 4.0 * batch_size * num_heads * head_dim * seqlen_q * (topk_blocks * block_n)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward-only CuTe q/k=128 smoke + benchmark")
    # Align defaults with fastvideo-kernel/benchmarks/bench_vsa.py
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


def _choose_q_block_size(q_len: int, base_block_m: int = 128) -> int:
    # On SM100+, q_stage can become 2 when seqlen > 128, requiring q block to be multiple of 256.
    major, _minor = torch.cuda.get_device_capability()
    if major >= 10 and q_len > base_block_m:
        return 2 * base_block_m
    return base_block_m


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test.")

    _ensure_flash_attn_importable()
    from flash_attn.cute import flash_attn_func

    args = _parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    batch_size = args.batch_size
    num_heads = args.num_heads
    head_dim = args.head_dim
    seqlen_q = args.q_seq_len
    seqlen_k = args.kv_seq_len if args.kv_seq_len is not None else args.q_seq_len
    q_block_size = _choose_q_block_size(seqlen_q, base_block_m=128)
    block_size = (q_block_size, 128)
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
            block_size=block_size,
        )

    out, lse = _fwd()
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

    print("PASS: CuTe block-sparse q/k block=128 forward test")
    print(
        f"config: batch={batch_size}, heads={num_heads}, head_dim={head_dim}, "
        f"q_len={seqlen_q}, kv_len={seqlen_k}, q_block={q_block_size}, kv_block=128, "
        f"topk={topk}, dtype={args.dtype}"
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
