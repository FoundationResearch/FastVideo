from __future__ import annotations

import os
from typing import Tuple

import torch

from .block_sparse_attn import block_sparse_attn_triton
from .block_sparse_attn_cute_fwd import (
    block_sparse_attn_cute_fwd,
    block_sparse_attn_cute_fwd_bshd,
)

_DISPATCH_PRINTED_256 = False


def _print_dispatch_once(msg: str) -> None:
    global _DISPATCH_PRINTED_256
    if _DISPATCH_PRINTED_256:
        return
    if os.environ.get("FASTVIDEO_VSA_WRAPPER_PRINT", "0") == "1":
        print(f"[fastvideo_kernel.block_sparse_attn_256] dispatch: {msg}")
    _DISPATCH_PRINTED_256 = True


def _resolve_backend() -> str:
    # Performance mode policy:
    # - default: always CuTe
    # - Triton only when explicitly forced
    if os.environ.get("FASTVIDEO_KERNEL_VSA_FORCE_TRITON", "0") == "1":
        return "triton"
    return "cute"


def _expand_vsa256_mask_and_sizes_to_128(
    logical_mask: torch.Tensor,
    logical_kv_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # logical_mask: [B, H, Qb256, KVb256] -> [B, H, Qb256, KVb128]
    expanded_mask = logical_mask.repeat_interleave(2, dim=3)

    logical_kv_sizes = logical_kv_sizes.to(torch.int32)
    child0_size = torch.clamp(logical_kv_sizes, min=0, max=128)
    child1_size = torch.clamp(logical_kv_sizes - 128, min=0, max=128)
    expanded_sizes = torch.empty(
        (logical_kv_sizes.numel() * 2,),
        dtype=torch.int32,
        device=logical_kv_sizes.device,
    )
    expanded_sizes[0::2] = child0_size
    expanded_sizes[1::2] = child1_size
    return expanded_mask, expanded_sizes


def _expand_vsa256_mask_and_sizes_to_64(
    logical_mask: torch.Tensor,
    logical_kv_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Route-A compatibility for Triton64 training:
    # [Q256,KV256] -> [Q64,KV64] via dense 4x4 expansion per logical edge.
    expanded_mask = logical_mask.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
    logical_kv_sizes = logical_kv_sizes.to(torch.int32)
    offsets = torch.tensor([0, 64, 128, 192], dtype=torch.int32, device=logical_kv_sizes.device)
    expanded_sizes = torch.clamp(
        logical_kv_sizes[:, None] - offsets[None, :], min=0, max=64
    ).reshape(-1)
    return expanded_mask, expanded_sizes


def block_sparse_attn_256(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logical_block_map_256: torch.Tensor,
    logical_variable_block_sizes_256: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VSA256 wrapper for sparse branch.

    This wrapper owns VSA256 logical semantics and dispatches to:
    - CuTe path via q256/kv128 expansion
    - Triton path via q64/kv64 compatibility expansion (route A)
    """
    # Performance mode: assume valid CUDA inputs / dtypes from caller.
    if logical_block_map_256.dim() == 3:
        logical_block_map_256 = logical_block_map_256.unsqueeze(0)
    if _resolve_backend() == "triton":
        mask_64, sizes_64 = _expand_vsa256_mask_and_sizes_to_64(
            logical_block_map_256, logical_variable_block_sizes_256
        )
        _print_dispatch_once("triton(q64/kv64 compat)")
        return block_sparse_attn_triton(q, k, v, mask_64, sizes_64)

    mask_128, sizes_128 = _expand_vsa256_mask_and_sizes_to_128(
        logical_block_map_256, logical_variable_block_sizes_256
    )
    out = block_sparse_attn_cute_fwd(q, k, v, mask_128, sizes_128)
    _print_dispatch_once("cute(q256/kv128)")
    return out


def block_sparse_attn_256_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logical_block_map_256: torch.Tensor,
    logical_variable_block_sizes_256: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VSA256 wrapper for [B, S, H, D] layout.

    Default path is CuTe with direct BSHD input to avoid layout round-trips.
    Triton is used only when force-triton is enabled.
    """
    if logical_block_map_256.dim() == 3:
        logical_block_map_256 = logical_block_map_256.unsqueeze(0)
    if _resolve_backend() == "triton":
        out_h, aux = block_sparse_attn_256(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            logical_block_map_256,
            logical_variable_block_sizes_256,
        )
        return out_h.transpose(1, 2).contiguous(), aux

    mask_128, sizes_128 = _expand_vsa256_mask_and_sizes_to_128(
        logical_block_map_256, logical_variable_block_sizes_256
    )
    out = block_sparse_attn_cute_fwd_bshd(q, k, v, mask_128, sizes_128)
    _print_dispatch_once("cute_bshd(q256/kv128)")
    return out
