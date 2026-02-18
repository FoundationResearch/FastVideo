#!/usr/bin/env python3
"""Benchmark VSA256 on LTX-realistic latent shape.

Scenario:
- Real latent shape:   T,H,W = 16,34,60
- Padded latent shape: T,H,W = 16,40,64 (spatial padded to multiples of 8)
- Tile size is fixed as (4,8,8), i.e. logical block size = 256 tokens.

This script builds variable_block_sizes from real occupancy per tile and reports:
- full vs padded block counts
- timing for kernel_only / manual_e2e / wrapper_sparse_only / wrapper_e2e
"""

from __future__ import annotations

import argparse
import math
import os
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


T_REAL, H_REAL, W_REAL = 16, 34, 60
T_PAD, H_PAD, W_PAD = 16, 40, 64
T_TILE, H_TILE, W_TILE = 4, 8, 8
BLOCK_256 = T_TILE * H_TILE * W_TILE  # 256
KV_BLOCK_KERNEL = 128


def _ensure_flash_attn_importable() -> None:
    kernel_root = Path(__file__).resolve().parents[1]
    flash_attn_repo = kernel_root / "include" / "flash-attention"
    if not flash_attn_repo.exists():
        raise FileNotFoundError(f"flash-attention submodule not found: {flash_attn_repo}")
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
    p = argparse.ArgumentParser(description="Benchmark VSA256 for LTX real->padded layout")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--head_dim", type=int, default=128, choices=[64, 128])
    p.add_argument("--topk_logical", type=int, default=16)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    return p.parse_args()


def bench_ms(fn: Callable[[], object], warmup: int, rep: int) -> float:
    return do_bench(fn, warmup=warmup, rep=rep, quantiles=None)


def _map_to_index_torch_fallback(
    block_map: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, nhead, q_blocks, kv_blocks = block_map.shape
    index = torch.zeros((bsz, nhead, q_blocks, kv_blocks), dtype=torch.int32, device=block_map.device)
    index_num = torch.zeros((bsz, nhead, q_blocks), dtype=torch.int32, device=block_map.device)
    for b in range(bsz):
        for h in range(nhead):
            for q in range(q_blocks):
                row = torch.nonzero(block_map[b, h, q], as_tuple=False).flatten()
                cnt = int(row.numel())
                if cnt > 0:
                    index[b, h, q, :cnt] = row.to(torch.int32)
                index_num[b, h, q] = cnt
    return index, index_num


def _map_to_index(block_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        from fastvideo_kernel.triton_kernels.index import map_to_index as triton_map_to_index

        return triton_map_to_index(block_map)
    except Exception:
        return _map_to_index_torch_fallback(block_map)


def _expand_mask_256_to_128(mask_256: torch.Tensor) -> torch.Tensor:
    return mask_256.repeat_interleave(2, dim=3)


def _expand_sizes_256_to_128(sizes_256: torch.Tensor) -> torch.Tensor:
    sizes_256 = sizes_256.to(torch.int32)
    child0 = torch.clamp(sizes_256, min=0, max=128)
    child1 = torch.clamp(sizes_256 - 128, min=0, max=128)
    out = torch.empty((sizes_256.numel() * 2,), dtype=torch.int32, device=sizes_256.device)
    out[0::2] = child0
    out[1::2] = child1
    return out


def _make_logical_mask(
    bs: int,
    h: int,
    q_blocks: int,
    kv_blocks: int,
    topk_logical: int,
) -> torch.Tensor:
    scores = torch.rand(bs, h, q_blocks, kv_blocks, device="cuda")
    topk_logical = min(max(1, topk_logical), kv_blocks)
    idx = torch.topk(scores, topk_logical, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def _build_vbs_256_for_ltx2real(device: torch.device) -> torch.Tensor:
    t_tiles = T_PAD // T_TILE
    h_tiles = H_PAD // H_TILE
    w_tiles = W_PAD // W_TILE
    vbs = []
    for _tt in range(t_tiles):
        for hh in range(h_tiles):
            valid_h = min(H_TILE, max(0, H_REAL - hh * H_TILE))
            for ww in range(w_tiles):
                valid_w = min(W_TILE, max(0, W_REAL - ww * W_TILE))
                valid_tokens = T_TILE * valid_h * valid_w
                vbs.append(int(valid_tokens))
    return torch.tensor(vbs, dtype=torch.int32, device=device)


def _vbs_stats(vbs_256: torch.Tensor) -> dict[str, int]:
    full = int((vbs_256 == 256).sum().item())
    half = int((vbs_256 == 128).sum().item())
    quarter = int((vbs_256 == 64).sum().item())
    eighth = int((vbs_256 == 32).sum().item())
    zero = int((vbs_256 == 0).sum().item())
    total = int(vbs_256.numel())
    padded = total - full
    return {
        "total": total,
        "full_256": full,
        "padded_lt_256": padded,
        "vbs_128": half,
        "vbs_64": quarter,
        "vbs_32": eighth,
        "vbs_0": zero,
    }


def flops_sparse_attention(bs: int, h: int, d: int, q_len: int, avg_topk_kernel: float) -> float:
    return 4.0 * bs * h * d * q_len * (avg_topk_kernel * KV_BLOCK_KERNEL)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    _ensure_flash_attn_importable()
    os.environ["FASTVIDEO_VSA_256"] = "1"
    os.environ["FASTVIDEO_VSA_256_BACKEND"] = "cute"

    from flash_attn.cute import flash_attn_func
    from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256
    from fastvideo_kernel.ops import video_sparse_attn

    args = parse_args()
    set_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")

    bs, h, d = args.batch_size, args.num_heads, args.head_dim
    q_len = T_PAD * H_PAD * W_PAD
    kv_len = q_len
    q_blocks_256 = q_len // BLOCK_256
    kv_blocks_256 = q_blocks_256
    topk_logical = min(max(1, args.topk_logical), kv_blocks_256)

    q = torch.randn(bs, h, q_len, d, dtype=dtype, device=device)
    k = torch.randn(bs, h, kv_len, d, dtype=dtype, device=device)
    v = torch.randn(bs, h, kv_len, d, dtype=dtype, device=device)
    q_c = q.transpose(1, 2).contiguous()
    k_c = k.transpose(1, 2).contiguous()
    v_c = v.transpose(1, 2).contiguous()

    logical_mask = _make_logical_mask(bs, h, q_blocks_256, kv_blocks_256, topk_logical)
    mask_128 = _expand_mask_256_to_128(logical_mask)
    mask_idx, mask_cnt = _map_to_index(mask_128)
    full_cnt = torch.zeros_like(mask_cnt)
    full_idx = torch.zeros_like(mask_idx)

    kv_var_256 = _build_vbs_256_for_ltx2real(device)
    q_var_256 = kv_var_256.clone()
    kv_var_128 = _expand_sizes_256_to_128(kv_var_256)
    stats = _vbs_stats(kv_var_256)

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
            block_size=(BLOCK_256, KV_BLOCK_KERNEL),
        )

    def _manual_e2e():
        m256 = _make_logical_mask(bs, h, q_blocks_256, kv_blocks_256, topk_logical)
        m128 = _expand_mask_256_to_128(m256)
        idx, cnt = _map_to_index(m128)
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
            block_size=(BLOCK_256, KV_BLOCK_KERNEL),
        )

    def _wrapper_sparse_only():
        return block_sparse_attn_256(q, k, v, logical_mask, kv_var_256)

    def _wrapper_e2e():
        return video_sparse_attn(
            q,
            k,
            v,
            kv_var_256,
            q_var_256,
            topk_logical,
            block_size=(4, 8, 8),
            compress_attn_weight=None,
        )

    out, lse = _kernel_only()
    out_finite = torch.isfinite(out).all().item()
    lse_finite = True if lse is None else torch.isfinite(lse).all().item()

    avg_topk_kernel = mask_cnt.float().mean().item()
    flops = flops_sparse_attention(bs, h, d, q_len, avg_topk_kernel)

    kernel_ms = bench_ms(_kernel_only, warmup=args.warmup, rep=args.rep)
    manual_e2e_ms = bench_ms(_manual_e2e, warmup=args.warmup, rep=args.rep)
    wrapper_sparse_ms = bench_ms(_wrapper_sparse_only, warmup=args.warmup, rep=args.rep)
    wrapper_e2e_ms = bench_ms(_wrapper_e2e, warmup=args.warmup, rep=args.rep)

    kernel_tflops = flops / kernel_ms * 1e-12 * 1e3
    manual_tflops = flops / manual_e2e_ms * 1e-12 * 1e3
    wrapper_sparse_tflops = flops / wrapper_sparse_ms * 1e-12 * 1e3
    wrapper_e2e_tflops = flops / wrapper_e2e_ms * 1e-12 * 1e3

    print("VSA256 LTX2Real Benchmark (T,H,W: 16x34x60 -> 16x40x64)")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"batch={bs}, heads={h}, head_dim={d}, dtype={args.dtype}")
    print(
        f"seq_len(padded)={q_len}, q_blocks_256={q_blocks_256}, "
        f"kv_blocks_256={kv_blocks_256}, topk_logical={topk_logical}"
    )
    print(
        "vbs_256 distribution: "
        f"total={stats['total']}, full_256={stats['full_256']}, padded_lt_256={stats['padded_lt_256']}, "
        f"vbs_128={stats['vbs_128']}, vbs_64={stats['vbs_64']}, vbs_32={stats['vbs_32']}, vbs_0={stats['vbs_0']}"
    )
    print(f"vbs_128 stats: min={int(kv_var_128.min().item())}, max={int(kv_var_128.max().item())}")
    print(f"finite_check: out={out_finite}, lse={lse_finite}")
    print(f"kernel_only: {kernel_ms:.3f} ms | {kernel_tflops:.2f} TFLOPs (approx)")
    print(f"manual_e2e:  {manual_e2e_ms:.3f} ms | {manual_tflops:.2f} TFLOPs (approx)")
    print(f"wrapper_sparse_only: {wrapper_sparse_ms:.3f} ms | {wrapper_sparse_tflops:.2f} TFLOPs (approx)")
    print(f"wrapper_e2e(video_sparse_attn): {wrapper_e2e_ms:.3f} ms | {wrapper_e2e_tflops:.2f} TFLOPs (approx)")


if __name__ == "__main__":
    main()
