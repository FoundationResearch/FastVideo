#!/usr/bin/env python3
"""Benchmark model-level VSA256 tile fastpath for LTX2-like loop.

Compares:
1) baseline: tile/untile at every attention layer
2) fastpath: tile once before all layers and untile once after all layers
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

try:
    from triton.testing import do_bench
except Exception as e:  # pragma: no cover
    raise ImportError("This benchmark requires triton (triton.testing.do_bench).") from e


T_REAL, H_REAL, W_REAL = 16, 34, 60
T_TILE, H_TILE, W_TILE = 4, 8, 8
BLOCK_256 = T_TILE * H_TILE * W_TILE


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


def bench_ms(fn: Callable[[], object], warmup: int, rep: int) -> float:
    return do_bench(fn, warmup=warmup, rep=rep, quantiles=None)


def _build_vbs_256_for_ltx2real(device: torch.device) -> torch.Tensor:
    t_tiles = (T_REAL + T_TILE - 1) // T_TILE
    h_tiles = (H_REAL + H_TILE - 1) // H_TILE
    w_tiles = (W_REAL + W_TILE - 1) // W_TILE
    vbs = []
    for _tt in range(t_tiles):
        for hh in range(h_tiles):
            valid_h = min(H_TILE, max(0, H_REAL - hh * H_TILE))
            for ww in range(w_tiles):
                valid_w = min(W_TILE, max(0, W_REAL - ww * W_TILE))
                valid_tokens = T_TILE * valid_h * valid_w
                vbs.append(int(valid_tokens))
    return torch.tensor(vbs, dtype=torch.int32, device=device)


def _get_tile_partition_indices_local(
    t: int,
    h: int,
    w: int,
    device: torch.device,
) -> torch.LongTensor:
    indices = torch.arange(t * h * w, device=device, dtype=torch.long).reshape(t, h, w)
    chunks = []
    for tt in range((t + T_TILE - 1) // T_TILE):
        for hh in range((h + H_TILE - 1) // H_TILE):
            for ww in range((w + W_TILE - 1) // W_TILE):
                chunks.append(
                    indices[
                        tt * T_TILE:min(tt * T_TILE + T_TILE, t),
                        hh * H_TILE:min(hh * H_TILE + H_TILE, h),
                        ww * W_TILE:min(ww * W_TILE + W_TILE, w),
                    ].flatten()
                )
    return torch.cat(chunks, dim=0)


def _get_non_pad_index_local(
    variable_block_sizes: torch.LongTensor,
    max_block_size: int,
) -> torch.LongTensor:
    n_win = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts_pad = torch.arange(n_win, device=device) * max_block_size
    index_pad = starts_pad[:, None] + torch.arange(max_block_size, device=device)[None, :]
    index_mask = torch.arange(max_block_size, device=device)[None, :] < variable_block_sizes[:, None]
    return index_pad[index_mask]


def _tile_local(
    x: torch.Tensor,
    padded_seq_len: int,
    tile_partition_indices: torch.LongTensor,
    non_pad_index: torch.LongTensor,
) -> torch.Tensor:
    x_padded = torch.zeros(
        (x.shape[0], padded_seq_len, x.shape[-2], x.shape[-1]),
        device=x.device,
        dtype=x.dtype,
    )
    x_padded[:, non_pad_index] = x[:, tile_partition_indices]
    return x_padded


def _untile_local(
    x: torch.Tensor,
    reverse_tile_partition_indices: torch.LongTensor,
    non_pad_index: torch.LongTensor,
) -> torch.Tensor:
    return x[:, non_pad_index][:, reverse_tile_partition_indices]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model-level LTX2 VSA256 tile fastpath benchmark")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--head_dim", type=int, default=128, choices=[64, 128])
    p.add_argument("--num_layers", type=int, default=16)
    p.add_argument("--topk_logical", type=int, default=16)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    return p.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    _ensure_flash_attn_importable()
    os.environ["FASTVIDEO_VSA_256"] = "1"
    os.environ["FASTVIDEO_VSA_256_BACKEND"] = "cute"

    from fastvideo_kernel.ops import video_sparse_attn_bshd

    args = parse_args()
    set_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")

    bs, h, d = args.batch_size, args.num_heads, args.head_dim
    seq_real = T_REAL * H_REAL * W_REAL
    kv_var_256 = _build_vbs_256_for_ltx2real(device)
    q_var_256 = kv_var_256

    q_blocks_256 = kv_var_256.numel()
    topk_logical = min(max(1, args.topk_logical), q_blocks_256)
    padded_seq_len = q_blocks_256 * BLOCK_256

    tile_partition_indices = _get_tile_partition_indices_local(T_REAL, H_REAL, W_REAL, device)
    reverse_tile_partition_indices = torch.argsort(tile_partition_indices)
    non_pad_index = _get_non_pad_index_local(kv_var_256, BLOCK_256)

    x0 = torch.randn(bs, seq_real, h, d, dtype=dtype, device=device)
    gates = [
        torch.randn(bs, seq_real, h, d, dtype=dtype, device=device)
        for _ in range(args.num_layers)
    ]

    def _attn_on_tiled(x_tiled: torch.Tensor, gate_tiled: torch.Tensor) -> torch.Tensor:
        return video_sparse_attn_bshd(
            x_tiled,
            x_tiled,
            x_tiled,
            kv_var_256,
            q_var_256,
            topk_logical,
            block_size=(4, 8, 8),
            compress_attn_weight=gate_tiled,
        )

    def _baseline_per_layer_tile_untile():
        x = x0
        for i in range(args.num_layers):
            x_t = _tile_local(x, padded_seq_len, tile_partition_indices, non_pad_index)
            g_t = _tile_local(gates[i], padded_seq_len, tile_partition_indices, non_pad_index)
            y_t = _attn_on_tiled(x_t, g_t)
            x = _untile_local(y_t, reverse_tile_partition_indices, non_pad_index)
        return x

    def _fastpath_model_level_tile_untile():
        x_t = _tile_local(x0, padded_seq_len, tile_partition_indices, non_pad_index)
        gates_t = [
            _tile_local(g, padded_seq_len, tile_partition_indices, non_pad_index)
            for g in gates
        ]
        for i in range(args.num_layers):
            x_t = _attn_on_tiled(x_t, gates_t[i])
        x = _untile_local(x_t, reverse_tile_partition_indices, non_pad_index)
        return x

    base_ms = bench_ms(_baseline_per_layer_tile_untile, warmup=args.warmup, rep=args.rep)
    fast_ms = bench_ms(_fastpath_model_level_tile_untile, warmup=args.warmup, rep=args.rep)

    speedup = base_ms / max(1e-6, fast_ms)
    print("LTX2 model-level VSA256 tile fastpath benchmark")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"shape(real): T,H,W={T_REAL},{H_REAL},{W_REAL}, padded_seq={padded_seq_len}")
    print(f"batch={bs}, heads={h}, head_dim={d}, layers={args.num_layers}, topk={topk_logical}")
    print(f"baseline(per-layer tile/untile): {base_ms:.3f} ms")
    print(f"fastpath(model-level tile/untile): {fast_ms:.3f} ms")
    print(f"speedup: {speedup:.3f}x")


if __name__ == "__main__":
    main()

