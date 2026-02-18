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
    p.add_argument("--breakdown_rep", type=int, default=20)
    p.add_argument("--wrapper_breakdown_rep", type=int, default=20)
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


def _upstream_use_vsa_256() -> bool:
    return os.environ.get("FASTVIDEO_VSA_256", "0") == "1"


def _upstream_force_triton() -> bool:
    return os.environ.get("FASTVIDEO_KERNEL_VSA_FORCE_TRITON", "0") == "1"


def _collect_vsa_backend_real_profile(
    fn,
    reset_stats_fn,
    get_stats_fn,
    rep: int,
) -> dict[str, float]:
    reset_stats_fn()
    for _ in range(rep):
        _ = fn()
    torch.cuda.synchronize()
    raw = get_stats_fn()
    calls = max(1, int(raw.get("calls", 0)))
    return {
        "calls": float(calls),
        "coarse_ms_avg": float(raw.get("coarse_ms_total", 0.0)) / calls,
        "sparse_ms_avg": float(raw.get("sparse_ms_total", 0.0)) / calls,
        "fuse_add_ms_avg": float(raw.get("fuse_add_ms_total", 0.0)) / calls,
    }


@torch.no_grad()
def _get_tile_partition_indices_local(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    t, h, w = dit_seq_shape
    ts, hs, ws = tile_size
    indices = torch.arange(t * h * w, device=device, dtype=torch.long).reshape(t, h, w)
    chunks = []
    for tt in range(math.ceil(t / ts)):
        for hh in range(math.ceil(h / hs)):
            for ww in range(math.ceil(w / ws)):
                chunks.append(
                    indices[
                        tt * ts:min(tt * ts + ts, t),
                        hh * hs:min(hh * hs + hs, h),
                        ww * ws:min(ww * ws + ws, w),
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
    num_tiles: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    tile_partition_indices: torch.LongTensor,
    non_pad_index: torch.LongTensor,
) -> torch.Tensor:
    t_padded_size = num_tiles[0] * tile_size[0]
    h_padded_size = num_tiles[1] * tile_size[1]
    w_padded_size = num_tiles[2] * tile_size[2]
    x_padded = torch.zeros(
        (x.shape[0], t_padded_size * h_padded_size * w_padded_size, x.shape[-2], x.shape[-1]),
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


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    _ensure_flash_attn_importable()
    os.environ["FASTVIDEO_VSA_256"] = "1"
    os.environ["FASTVIDEO_VSA_256_BACKEND"] = "cute"
    os.environ.setdefault("FASTVIDEO_VSA_PROFILE", "1")

    from flash_attn.cute import flash_attn_func
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from flash_attn.cute.interface import _flash_attn_fwd
    from fastvideo_kernel.block_sparse_attn_256 import (
        block_sparse_attn_256_bshd,
    )
    from fastvideo_kernel.block_sparse_attn_256 import _expand_vsa256_mask_and_sizes_to_128
    from fastvideo_kernel.block_sparse_attn_cute_fwd import (
        _aggregate_q_block_map,
        _choose_q_sparse_block_size,
        _get_vbs_mask_mod,
        _map_to_index as _map_to_index_cute,
    )
    from fastvideo_kernel.ops import (
        get_vsa_profile_stats,
        reset_vsa_profile_stats,
        video_sparse_attn,
        video_sparse_attn_bshd,
    )

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

    # Upstream true-E2E inputs (unpadded real sequence) for
    # preprocess -> backend.forward(auto-dispatch equivalent) -> postprocess path.
    seq_real = T_REAL * H_REAL * W_REAL
    q_up = torch.randn(bs, seq_real, h, d, dtype=dtype, device=device)
    k_up = torch.randn(bs, seq_real, h, d, dtype=dtype, device=device)
    v_up = torch.randn(bs, seq_real, h, d, dtype=dtype, device=device)
    g_up = torch.randn(bs, seq_real, h, d, dtype=dtype, device=device)
    vsa_tile_size = (4, 8, 8)
    dit_seq_shape = (T_REAL, H_REAL, W_REAL)
    num_tiles = (
        math.ceil(dit_seq_shape[0] / vsa_tile_size[0]),
        math.ceil(dit_seq_shape[1] / vsa_tile_size[1]),
        math.ceil(dit_seq_shape[2] / vsa_tile_size[2]),
    )
    tile_partition_indices = _get_tile_partition_indices_local(
        dit_seq_shape, vsa_tile_size, device
    )
    reverse_tile_partition_indices = torch.argsort(tile_partition_indices)
    non_pad_index = _get_non_pad_index_local(kv_var_256, BLOCK_256)

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
        return block_sparse_attn_256_bshd(q_c, k_c, v_c, logical_mask, kv_var_256)

    def _wrapper_e2e():
        return video_sparse_attn_bshd(
            q_c,
            k_c,
            v_c,
            kv_var_256,
            q_var_256,
            topk_logical,
            block_size=(4, 8, 8),
            compress_attn_weight=None,
        )

    def _upstream_true_e2e():
        qkvg = torch.cat([q_up, k_up, v_up, g_up], dim=0)
        qkvg = _tile_local(
            qkvg,
            num_tiles=num_tiles,
            tile_size=vsa_tile_size,
            tile_partition_indices=tile_partition_indices,
            non_pad_index=non_pad_index,
        )
        q_t, k_t, v_t, g_t = qkvg.chunk(4, dim=0)
        if _upstream_use_vsa_256() and (not _upstream_force_triton()):
            out_t = video_sparse_attn_bshd(
                q_t,
                k_t,
                v_t,
                kv_var_256,
                kv_var_256,
                topk_logical,
                block_size=(4, 8, 8),
                compress_attn_weight=g_t,
            )
        else:
            out_t = video_sparse_attn(
                q_t.transpose(1, 2).contiguous(),
                k_t.transpose(1, 2).contiguous(),
                v_t.transpose(1, 2).contiguous(),
                kv_var_256,
                kv_var_256,
                topk_logical,
                block_size=(4, 8, 8),
                compress_attn_weight=g_t.transpose(1, 2).contiguous(),
            ).transpose(1, 2)
        return _untile_local(
            out_t,
            reverse_tile_partition_indices=reverse_tile_partition_indices,
            non_pad_index=non_pad_index,
        )

    def _upstream_backend_forward(
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        g_t: torch.Tensor,
    ) -> torch.Tensor:
        if _upstream_use_vsa_256() and (not _upstream_force_triton()):
            return video_sparse_attn_bshd(
                q_t,
                k_t,
                v_t,
                kv_var_256,
                kv_var_256,
                topk_logical,
                block_size=(4, 8, 8),
                compress_attn_weight=g_t,
            )
        return video_sparse_attn(
            q_t.transpose(1, 2).contiguous(),
            k_t.transpose(1, 2).contiguous(),
            v_t.transpose(1, 2).contiguous(),
            kv_var_256,
            kv_var_256,
            topk_logical,
            block_size=(4, 8, 8),
            compress_attn_weight=g_t.transpose(1, 2).contiguous(),
        ).transpose(1, 2)

    qkvg_up_once = torch.cat([q_up, k_up, v_up, g_up], dim=0)
    qkvg_tiled_once = _tile_local(
        qkvg_up_once,
        num_tiles=num_tiles,
        tile_size=vsa_tile_size,
        tile_partition_indices=tile_partition_indices,
        non_pad_index=non_pad_index,
    )
    q_up_t, k_up_t, v_up_t, g_up_t = qkvg_tiled_once.chunk(4, dim=0)
    out_up_once = _upstream_backend_forward(q_up_t, k_up_t, v_up_t, g_up_t)

    def _up_cat_qkvg_only():
        return torch.cat([q_up, k_up, v_up, g_up], dim=0)

    def _up_preprocess_tile_only():
        return _tile_local(
            qkvg_up_once,
            num_tiles=num_tiles,
            tile_size=vsa_tile_size,
            tile_partition_indices=tile_partition_indices,
            non_pad_index=non_pad_index,
        )

    def _up_chunk_qkvg_only():
        return qkvg_tiled_once.chunk(4, dim=0)

    def _up_backend_forward_only():
        return _upstream_backend_forward(q_up_t, k_up_t, v_up_t, g_up_t)

    def _up_postprocess_untile_only():
        return _untile_local(
            out_up_once,
            reverse_tile_partition_indices=reverse_tile_partition_indices,
            non_pad_index=non_pad_index,
        )

    logical_mask_fixed = logical_mask
    mask_128_fixed = mask_128
    sizes_256_fixed = kv_var_256

    def _prep_logical_mask_only():
        return _make_logical_mask(bs, h, q_blocks_256, kv_blocks_256, topk_logical)

    def _prep_split_mask_only():
        return _expand_mask_256_to_128(logical_mask_fixed)

    def _prep_split_sizes_only():
        return _expand_sizes_256_to_128(sizes_256_fixed)

    def _prep_index_only():
        return _map_to_index(mask_128_fixed)

    def _prep_kernel_inputs_only():
        m256 = _make_logical_mask(bs, h, q_blocks_256, kv_blocks_256, topk_logical)
        m128 = _expand_mask_256_to_128(m256)
        idx, cnt = _map_to_index(m128)
        zcnt = torch.zeros_like(cnt)
        zidx = torch.zeros_like(idx)
        return idx, cnt, zidx, zcnt

    # Wrapper-internal staged breakdown (block_sparse_attn_256_bshd + cute_fwd_bshd).
    def _wb_expand_256_to_128():
        return _expand_vsa256_mask_and_sizes_to_128(logical_mask, kv_var_256)

    mask_128_wb, sizes_128_wb = _wb_expand_256_to_128()
    q_sparse_candidate = _choose_q_sparse_block_size(q.shape[2], m_block_size=128)
    q_block_size = q.shape[2] // mask_128_wb.shape[2]
    kv_block_size = k.shape[2] // mask_128_wb.shape[3]
    q_sparse_block_size = max(
        q_block_size,
        ((q_sparse_candidate + q_block_size - 1) // q_block_size) * q_block_size,
    )

    def _wb_aggregate_q_sparse_map():
        return _aggregate_q_block_map(
            mask_128_wb, q_sparse_block_size=q_sparse_block_size, q_block_size=q_block_size
        )

    sparse_map_wb = _wb_aggregate_q_sparse_map()

    def _wb_split_full_partial():
        kv_full = (sizes_128_wb == kv_block_size).view(1, 1, 1, -1)
        kv_partial = ((sizes_128_wb > 0) & (sizes_128_wb < kv_block_size)).view(1, 1, 1, -1)
        full_map = sparse_map_wb & kv_full
        mask_map = sparse_map_wb & kv_partial
        return full_map, mask_map

    full_map_wb, mask_map_wb = _wb_split_full_partial()

    def _wb_map_to_index_full():
        return _map_to_index_cute(full_map_wb)

    def _wb_map_to_index_mask():
        return _map_to_index_cute(mask_map_wb)

    full_idx_wb, full_cnt_wb = _wb_map_to_index_full()
    mask_idx_wb, mask_cnt_wb = _wb_map_to_index_mask()
    full_idx_wb = full_idx_wb.to(torch.int32).contiguous()
    full_cnt_wb = full_cnt_wb.to(torch.int32).contiguous()
    mask_idx_wb = mask_idx_wb.to(torch.int32).contiguous()
    mask_cnt_wb = mask_cnt_wb.to(torch.int32).contiguous()

    def _wb_build_sparse_tensors():
        return BlockSparseTensorsTorch(
            full_block_cnt=full_cnt_wb,
            full_block_idx=full_idx_wb,
            mask_block_cnt=mask_cnt_wb,
            mask_block_idx=mask_idx_wb,
            block_size=(q_sparse_block_size, kv_block_size),
        )

    sparse_tensors_wb = _wb_build_sparse_tensors()

    def _wb_layout_transpose():
        # BSHD fast path: layout conversion is bypassed.
        return q_c, k_c, v_c

    q_cute_wb, k_cute_wb, v_cute_wb = _wb_layout_transpose()

    def _wb_flash_fwd_only():
        return _flash_attn_fwd(
            q_cute_wb,
            k_cute_wb,
            v_cute_wb,
            m_block_size=128,
            n_block_size=kv_block_size,
            mask_mod=_get_vbs_mask_mod(kv_block_size),
            block_sparse_tensors=sparse_tensors_wb,
            aux_tensors=[sizes_128_wb],
            causal=False,
            return_lse=True,
        )

    out_wb_once, _lse_wb_once = _wb_flash_fwd_only()

    def _wb_output_transpose():
        # BSHD fast path: output is already in desired layout.
        return out_wb_once

    def _wb_pipeline_replay_total():
        mask_128_local, sizes_128_local = _expand_vsa256_mask_and_sizes_to_128(
            logical_mask, kv_var_256
        )
        q_sparse_candidate_local = _choose_q_sparse_block_size(q.shape[2], m_block_size=128)
        q_block_size_local = q.shape[2] // mask_128_local.shape[2]
        kv_block_size_local = k.shape[2] // mask_128_local.shape[3]
        q_sparse_block_size_local = max(
            q_block_size_local,
            ((q_sparse_candidate_local + q_block_size_local - 1) // q_block_size_local) * q_block_size_local,
        )
        sparse_map_local = _aggregate_q_block_map(
            mask_128_local,
            q_sparse_block_size=q_sparse_block_size_local,
            q_block_size=q_block_size_local,
        )
        kv_full_local = (sizes_128_local == kv_block_size_local).view(1, 1, 1, -1)
        kv_partial_local = ((sizes_128_local > 0) & (sizes_128_local < kv_block_size_local)).view(1, 1, 1, -1)
        full_map_local = sparse_map_local & kv_full_local
        mask_map_local = sparse_map_local & kv_partial_local
        full_idx_local, full_cnt_local = _map_to_index_cute(full_map_local)
        mask_idx_local, mask_cnt_local = _map_to_index_cute(mask_map_local)
        full_idx_local = full_idx_local.to(torch.int32).contiguous()
        full_cnt_local = full_cnt_local.to(torch.int32).contiguous()
        mask_idx_local = mask_idx_local.to(torch.int32).contiguous()
        mask_cnt_local = mask_cnt_local.to(torch.int32).contiguous()
        sparse_tensors_local = BlockSparseTensorsTorch(
            full_block_cnt=full_cnt_local,
            full_block_idx=full_idx_local,
            mask_block_cnt=mask_cnt_local,
            mask_block_idx=mask_idx_local,
            block_size=(q_sparse_block_size_local, kv_block_size_local),
        )
        q_c_local = q_c
        k_c_local = k_c
        v_c_local = v_c
        out_local, _lse_local = _flash_attn_fwd(
            q_c_local,
            k_c_local,
            v_c_local,
            m_block_size=128,
            n_block_size=kv_block_size_local,
            mask_mod=_get_vbs_mask_mod(kv_block_size_local),
            block_sparse_tensors=sparse_tensors_local,
            aux_tensors=[sizes_128_local],
            causal=False,
            return_lse=True,
        )
        return out_local

    def _wb_stage_events_avg(rep: int) -> dict:
        keys = [
            "expand_256_to_128",
            "aggregate_q_sparse_map",
            "split_full_partial_map",
            "map_to_index_full",
            "map_to_index_mask",
            "index_to_int_contiguous",
            "build_sparse_tensors",
            "layout_transpose",
            "flash_attn_fwd_only",
            "output_transpose",
        ]
        accum = {k: 0.0 for k in keys}
        for _ in range(rep):
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t4 = torch.cuda.Event(enable_timing=True)
            t5 = torch.cuda.Event(enable_timing=True)
            t6 = torch.cuda.Event(enable_timing=True)
            t7 = torch.cuda.Event(enable_timing=True)
            t8 = torch.cuda.Event(enable_timing=True)
            t9 = torch.cuda.Event(enable_timing=True)
            t10 = torch.cuda.Event(enable_timing=True)

            t0.record()
            mask_128_local, sizes_128_local = _expand_vsa256_mask_and_sizes_to_128(logical_mask, kv_var_256)
            t1.record()
            q_sparse_candidate_local = _choose_q_sparse_block_size(q.shape[2], m_block_size=128)
            q_block_size_local = q.shape[2] // mask_128_local.shape[2]
            kv_block_size_local = k.shape[2] // mask_128_local.shape[3]
            q_sparse_block_size_local = max(
                q_block_size_local,
                ((q_sparse_candidate_local + q_block_size_local - 1) // q_block_size_local) * q_block_size_local,
            )
            sparse_map_local = _aggregate_q_block_map(
                mask_128_local,
                q_sparse_block_size=q_sparse_block_size_local,
                q_block_size=q_block_size_local,
            )
            t2.record()
            kv_full_local = (sizes_128_local == kv_block_size_local).view(1, 1, 1, -1)
            kv_partial_local = ((sizes_128_local > 0) & (sizes_128_local < kv_block_size_local)).view(1, 1, 1, -1)
            full_map_local = sparse_map_local & kv_full_local
            mask_map_local = sparse_map_local & kv_partial_local
            t3.record()
            full_idx_local, full_cnt_local = _map_to_index_cute(full_map_local)
            t4.record()
            mask_idx_local, mask_cnt_local = _map_to_index_cute(mask_map_local)
            t5.record()
            full_idx_local = full_idx_local.to(torch.int32).contiguous()
            full_cnt_local = full_cnt_local.to(torch.int32).contiguous()
            mask_idx_local = mask_idx_local.to(torch.int32).contiguous()
            mask_cnt_local = mask_cnt_local.to(torch.int32).contiguous()
            t6.record()
            sparse_tensors_local = BlockSparseTensorsTorch(
                full_block_cnt=full_cnt_local,
                full_block_idx=full_idx_local,
                mask_block_cnt=mask_cnt_local,
                mask_block_idx=mask_idx_local,
                block_size=(q_sparse_block_size_local, kv_block_size_local),
            )
            t7.record()
            q_c_local = q_c
            k_c_local = k_c
            v_c_local = v_c
            t8.record()
            out_local, _lse_local = _flash_attn_fwd(
                q_c_local,
                k_c_local,
                v_c_local,
                m_block_size=128,
                n_block_size=kv_block_size_local,
                mask_mod=_get_vbs_mask_mod(kv_block_size_local),
                block_sparse_tensors=sparse_tensors_local,
                aux_tensors=[sizes_128_local],
                causal=False,
                return_lse=True,
            )
            t9.record()
            _ = out_local
            t10.record()
            torch.cuda.synchronize()

            accum["expand_256_to_128"] += t0.elapsed_time(t1)
            accum["aggregate_q_sparse_map"] += t1.elapsed_time(t2)
            accum["split_full_partial_map"] += t2.elapsed_time(t3)
            accum["map_to_index_full"] += t3.elapsed_time(t4)
            accum["map_to_index_mask"] += t4.elapsed_time(t5)
            accum["index_to_int_contiguous"] += t5.elapsed_time(t6)
            accum["build_sparse_tensors"] += t6.elapsed_time(t7)
            accum["layout_transpose"] += t7.elapsed_time(t8)
            accum["flash_attn_fwd_only"] += t8.elapsed_time(t9)
            accum["output_transpose"] += t9.elapsed_time(t10)
        return {k: v / rep for k, v in accum.items()}

    out, lse = _kernel_only()
    out_finite = torch.isfinite(out).all().item()
    lse_finite = True if lse is None else torch.isfinite(lse).all().item()

    avg_topk_kernel = mask_cnt.float().mean().item()
    flops = flops_sparse_attention(bs, h, d, q_len, avg_topk_kernel)

    kernel_ms = bench_ms(_kernel_only, warmup=args.warmup, rep=args.rep)
    manual_e2e_ms = bench_ms(_manual_e2e, warmup=args.warmup, rep=args.rep)
    wrapper_sparse_ms = bench_ms(_wrapper_sparse_only, warmup=args.warmup, rep=args.rep)
    wrapper_e2e_ms = bench_ms(_wrapper_e2e, warmup=args.warmup, rep=args.rep)
    upstream_true_e2e_ms = bench_ms(_upstream_true_e2e, warmup=args.warmup, rep=args.rep)
    prep_logical_ms = bench_ms(
        _prep_logical_mask_only, warmup=max(1, args.warmup // 2), rep=args.breakdown_rep
    )
    prep_split_mask_ms = bench_ms(
        _prep_split_mask_only, warmup=max(1, args.warmup // 2), rep=args.breakdown_rep
    )
    prep_split_sizes_ms = bench_ms(
        _prep_split_sizes_only, warmup=max(1, args.warmup // 2), rep=args.breakdown_rep
    )
    prep_index_ms = bench_ms(
        _prep_index_only, warmup=max(1, args.warmup // 2), rep=args.breakdown_rep
    )
    prep_kernel_inputs_ms = bench_ms(
        _prep_kernel_inputs_only, warmup=max(1, args.warmup // 2), rep=args.breakdown_rep
    )
    wb_expand_ms = bench_ms(
        _wb_expand_256_to_128, warmup=max(1, args.warmup // 2), rep=args.wrapper_breakdown_rep
    )
    wb_aggregate_ms = bench_ms(
        _wb_aggregate_q_sparse_map, warmup=max(1, args.warmup // 2), rep=args.wrapper_breakdown_rep
    )
    wb_split_full_partial_ms = bench_ms(
        _wb_split_full_partial, warmup=max(1, args.warmup // 2), rep=args.wrapper_breakdown_rep
    )
    wb_map_full_ms = bench_ms(
        _wb_map_to_index_full, warmup=max(1, args.warmup // 2), rep=args.wrapper_breakdown_rep
    )
    wb_map_mask_ms = bench_ms(
        _wb_map_to_index_mask, warmup=max(1, args.warmup // 2), rep=args.wrapper_breakdown_rep
    )
    wb_sparse_tensor_ms = bench_ms(
        _wb_build_sparse_tensors, warmup=max(1, args.warmup // 2), rep=args.wrapper_breakdown_rep
    )
    wb_layout_ms = bench_ms(
        _wb_layout_transpose, warmup=max(1, args.warmup // 2), rep=args.wrapper_breakdown_rep
    )
    wb_flash_ms = bench_ms(
        _wb_flash_fwd_only, warmup=max(1, args.warmup // 2), rep=args.wrapper_breakdown_rep
    )
    wb_out_transpose_ms = bench_ms(
        _wb_output_transpose, warmup=max(1, args.warmup // 2), rep=args.wrapper_breakdown_rep
    )
    wb_pipeline_replay_total_ms = bench_ms(
        _wb_pipeline_replay_total, warmup=max(1, args.warmup // 2), rep=args.wrapper_breakdown_rep
    )
    up_cat_ms = bench_ms(
        _up_cat_qkvg_only, warmup=max(1, args.warmup // 2), rep=args.wrapper_breakdown_rep
    )
    up_tile_ms = bench_ms(
        _up_preprocess_tile_only, warmup=max(1, args.warmup // 2), rep=args.wrapper_breakdown_rep
    )
    up_chunk_ms = bench_ms(
        _up_chunk_qkvg_only, warmup=max(1, args.warmup // 2), rep=args.wrapper_breakdown_rep
    )
    up_backend_ms = bench_ms(
        _up_backend_forward_only, warmup=max(1, args.warmup // 2), rep=args.wrapper_breakdown_rep
    )
    up_untile_ms = bench_ms(
        _up_postprocess_untile_only, warmup=max(1, args.warmup // 2), rep=args.wrapper_breakdown_rep
    )
    up_backend_real = _collect_vsa_backend_real_profile(
        _up_backend_forward_only,
        reset_vsa_profile_stats,
        get_vsa_profile_stats,
        rep=max(1, args.wrapper_breakdown_rep),
    )

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
    print(
        f"wrapper_e2e(video_sparse_attn_bshd): {wrapper_e2e_ms:.3f} ms | "
        f"{wrapper_e2e_tflops:.2f} TFLOPs (approx)"
    )
    print(
        f"upstream_true_e2e(preprocess->backend->postprocess): "
        f"{upstream_true_e2e_ms:.3f} ms"
    )
    print("breakdown(prep+kernel):")
    print(f"  prep.logical_mask(topk):         {prep_logical_ms:.3f} ms")
    print(f"  prep.split_kv_mask(256->128):    {prep_split_mask_ms:.3f} ms")
    print(f"  prep.split_kv_sizes(256->128):   {prep_split_sizes_ms:.3f} ms")
    print(f"  prep.map_to_index:               {prep_index_ms:.3f} ms")
    print(f"  prep.kernel_inputs_total:        {prep_kernel_inputs_ms:.3f} ms")
    print(f"  kernel_only:                     {kernel_ms:.3f} ms")
    print("wrapper_internal_breakdown(block_sparse_attn_256_bshd + cute_fwd_bshd):")
    print(f"  wb.expand_256_to_128(mask+sizes): {wb_expand_ms:.3f} ms")
    print(f"  wb.aggregate_q_sparse_map:        {wb_aggregate_ms:.3f} ms")
    print(f"  wb.split_full_partial_map:        {wb_split_full_partial_ms:.3f} ms")
    print(f"  wb.map_to_index(full):            {wb_map_full_ms:.3f} ms")
    print(f"  wb.map_to_index(mask):            {wb_map_mask_ms:.3f} ms")
    print(f"  wb.build_sparse_tensors:          {wb_sparse_tensor_ms:.3f} ms")
    print(f"  wb.layout_transpose:              {wb_layout_ms:.3f} ms")
    print(f"  wb.flash_attn_fwd_only:           {wb_flash_ms:.3f} ms")
    print(f"  wb.output_transpose:              {wb_out_transpose_ms:.3f} ms")
    print(
        f"  wb.pipeline_replay_total:         {wb_pipeline_replay_total_ms:.3f} ms "
        "(single-call staged replay)"
    )
    print("upstream_true_e2e_breakdown(preprocess + backend + postprocess):")
    print(f"  up.cat_qkvg:                      {up_cat_ms:.3f} ms")
    print(f"  up.preprocess_tile:               {up_tile_ms:.3f} ms")
    print(f"  up.chunk_qkvg:                    {up_chunk_ms:.3f} ms")
    print(f"  up.backend_forward:               {up_backend_ms:.3f} ms")
    print(f"  up.backend.coarse_real(avg):      {up_backend_real['coarse_ms_avg']:.3f} ms")
    print(f"  up.backend.sparse_real(avg):      {up_backend_real['sparse_ms_avg']:.3f} ms")
    print(f"  up.backend.fuse_add_real(avg):    {up_backend_real['fuse_add_ms_avg']:.3f} ms")
    print(f"  up.postprocess_untile:            {up_untile_ms:.3f} ms")
    print(f"  up.true_e2e_total:                {upstream_true_e2e_ms:.3f} ms")


if __name__ == "__main__":
    main()
