from __future__ import annotations

import os
from typing import Tuple

import torch

from .block_sparse_attn import block_sparse_attn_triton
from .block_sparse_attn_cute_fwd import block_sparse_attn_cute_fwd

_DISPATCH_PRINTED_256 = False


def _print_dispatch_once(msg: str) -> None:
    global _DISPATCH_PRINTED_256
    if _DISPATCH_PRINTED_256:
        return
    if os.environ.get("FASTVIDEO_VSA_WRAPPER_PRINT", "0") == "1":
        print(f"[fastvideo_kernel.block_sparse_attn_256] dispatch: {msg}")
    _DISPATCH_PRINTED_256 = True


def _resolve_backend() -> str:
    # Allowed: auto | cute | triton
    backend = os.environ.get("FASTVIDEO_VSA_256_BACKEND", "auto").strip().lower()
    if os.environ.get("FASTVIDEO_KERNEL_VSA_FORCE_TRITON", "0") == "1":
        return "triton"
    # Backward compatibility with old route-A flag.
    if (
        backend == "auto"
        and os.environ.get("FASTVIDEO_VSA_256_TRITON_COMPAT", "0") == "1"
    ):
        return "triton"
    if backend not in {"auto", "cute", "triton"}:
        raise ValueError(
            "FASTVIDEO_VSA_256_BACKEND must be one of: auto, cute, triton. "
            f"got {backend}"
        )
    return backend


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
    if logical_block_map_256.dim() == 3:
        logical_block_map_256 = logical_block_map_256.unsqueeze(0)
    if logical_block_map_256.dtype != torch.bool:
        logical_block_map_256 = logical_block_map_256.to(torch.bool)
    if logical_variable_block_sizes_256.dtype != torch.int32:
        logical_variable_block_sizes_256 = logical_variable_block_sizes_256.to(torch.int32)
    if not logical_variable_block_sizes_256.is_cuda:
        logical_variable_block_sizes_256 = logical_variable_block_sizes_256.to(q.device)

    backend = _resolve_backend()
    allow_fallback = os.environ.get("FASTVIDEO_VSA_256_ALLOW_TRITON_FALLBACK", "1") == "1"

    if backend in {"auto", "cute"}:
        try:
            mask_128, sizes_128 = _expand_vsa256_mask_and_sizes_to_128(
                logical_block_map_256, logical_variable_block_sizes_256
            )
            out = block_sparse_attn_cute_fwd(q, k, v, mask_128, sizes_128)
            _print_dispatch_once("cute(q256/kv128)")
            return out
        except Exception as e:
            if backend == "cute" or not allow_fallback:
                raise RuntimeError(
                    "VSA256 CuTe path failed and fallback is disabled. "
                    f"error: {type(e).__name__}: {e}"
                ) from e
            _print_dispatch_once(f"cute(q256/kv128) -> triton64 fallback ({type(e).__name__})")

    mask_64, sizes_64 = _expand_vsa256_mask_and_sizes_to_64(
        logical_block_map_256, logical_variable_block_sizes_256
    )
    _print_dispatch_once("triton(q64/kv64 compat)")
    return block_sparse_attn_triton(q, k, v, mask_64, sizes_64)
