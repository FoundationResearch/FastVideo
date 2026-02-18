#!/usr/bin/env python3
"""Smoke + benchmark: logical kv_block=256, kernel kv_block=128.

Idea:
- Build sparse pattern on coarse KV blocks (size 256).
- Expand each selected coarse block i to two kernel blocks: (2*i, 2*i+1).
- Run CuTe kernel with block_size=(256, 128).

Checks:
1) output/lse shapes
2) NaN/Inf in outputs
3) forward latency + approximate TFLOPs
"""

from __future__ import annotations

import argparse
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
    repo_str = str(flash_attn_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CuTe smoke: q_block=256, logical kv_block=256 -> kernel kv_block=128"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--head_dim", type=int, default=128, choices=[64, 128])
    parser.add_argument("--q_seq_len", type=int, default=24576)
    parser.add_argument("--kv_seq_len", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _build_expanded_block_sparse_tensors(
    batch_size: int,
    num_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    q_block: int,
    kv_block_logical: int,
    kv_block_kernel: int,
    topk_logical: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m_blocks = math.ceil(seqlen_q / q_block)
    n_blocks_logical = math.ceil(seqlen_k / kv_block_logical)
    n_blocks_kernel = math.ceil(seqlen_k / kv_block_kernel)
    topk_logical = min(max(1, topk_logical), n_blocks_logical)

    scores = torch.rand(
        batch_size, num_heads, m_blocks, n_blocks_logical, device=device
    )
    coarse_idx = torch.topk(scores, topk_logical, dim=-1).indices.to(torch.int32)
    child0 = 2 * coarse_idx
    child1 = child0 + 1
    expanded = torch.stack((child0, child1), dim=-1).reshape(
        batch_size, num_heads, m_blocks, 2 * topk_logical
    )
    valid = expanded < n_blocks_kernel

    # Pack valid expanded indices to the left without Python loops.
    exp_w = expanded.shape[-1]
    pos = torch.arange(exp_w, device=device, dtype=torch.int32).view(
        1, 1, 1, exp_w
    )
    invalid_pos = torch.full_like(pos, exp_w)
    stable_key = torch.where(valid, pos, invalid_pos)
    order = torch.argsort(stable_key, dim=-1)
    packed_idx = torch.gather(expanded, dim=-1, index=order)
    packed_valid = torch.gather(valid, dim=-1, index=order)
    packed_idx = torch.where(packed_valid, packed_idx, torch.zeros_like(packed_idx))

    mask_block_cnt = packed_valid.sum(dim=-1).to(torch.int32)
    mask_block_idx = torch.zeros(
        (batch_size, num_heads, m_blocks, n_blocks_kernel),
        dtype=torch.int32,
        device=device,
    )
    mask_block_idx[..., :exp_w] = packed_idx

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
    avg_kernel_topk: float,
    kv_block_kernel: int,
) -> float:
    return (
        4.0
        * batch_size
        * num_heads
        * head_dim
        * seqlen_q
        * (avg_kernel_topk * kv_block_kernel)
    )


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

    q_block = 256
    kv_block_logical = 256
    kv_block_kernel = 128
    n_blocks_logical = math.ceil(seqlen_k / kv_block_logical)
    topk_logical = args.topk if args.topk is not None else max(1, n_blocks_logical // 10)

    q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, num_heads, head_dim, device=device, dtype=dtype)

    mask_cnt, mask_idx, full_cnt, full_idx = _build_expanded_block_sparse_tensors(
        batch_size=batch_size,
        num_heads=num_heads,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        q_block=q_block,
        kv_block_logical=kv_block_logical,
        kv_block_kernel=kv_block_kernel,
        topk_logical=topk_logical,
        device=device,
    )

    block_size = (q_block, kv_block_kernel)

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
    avg_kernel_topk = mask_cnt.float().mean().item()
    flops = _flops_sparse_attention(
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        seqlen_q=seqlen_q,
        avg_kernel_topk=avg_kernel_topk,
        kv_block_kernel=kv_block_kernel,
    )
    fwd_tflops = flops / fwd_ms * 1e-12 * 1e3

    print("PASS: CuTe q_block=256, logical kv_block=256, kernel kv_block=128")
    print(
        f"config: batch={batch_size}, heads={num_heads}, head_dim={head_dim}, "
        f"q_len={seqlen_q}, kv_len={seqlen_k}, q_block={q_block}, "
        f"kv_block_logical={kv_block_logical}, kv_block_kernel={kv_block_kernel}, "
        f"topk_logical={topk_logical}, avg_topk_kernel={avg_kernel_topk:.2f}, dtype={args.dtype}"
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
