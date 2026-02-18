from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Tuple

import torch


BLOCK_M = 64
BLOCK_N = 64


def _ensure_flash_attn_importable() -> None:
    kernel_root = Path(__file__).resolve().parents[2]
    flash_attn_repo = kernel_root / "include" / "flash-attention"
    if not flash_attn_repo.exists():
        raise FileNotFoundError(
            f"flash-attention submodule not found: {flash_attn_repo}"
        )
    repo_str = str(flash_attn_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    loaded = sys.modules.get("flash_attn")
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", "") or ""
        if repo_str not in loaded_file:
            for mod_name in list(sys.modules.keys()):
                if mod_name == "flash_attn" or mod_name.startswith("flash_attn."):
                    del sys.modules[mod_name]


def _map_to_index(block_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)
    if block_map.dim() != 4:
        raise ValueError(
            "block_map must be [B,H,Q,KV] (or [H,Q,KV]), "
            f"got shape={tuple(block_map.shape)}"
        )
    if block_map.dtype != torch.bool:
        block_map = block_map.to(torch.bool)
    if not block_map.is_cuda:
        raise RuntimeError("block_map must be a CUDA tensor.")
    from fastvideo_kernel.triton_kernels.index import map_to_index as triton_map_to_index

    return triton_map_to_index(block_map)


def _choose_q_sparse_block_size(
    q_len: int,
    m_block_size: int = 128,
) -> int:
    major, _minor = torch.cuda.get_device_capability()
    if major >= 10 and q_len > m_block_size:
        return 2 * m_block_size
    return m_block_size


def _aggregate_q_block_map(
    block_map: torch.Tensor,
    q_sparse_block_size: int,
    q_block_size: int,
) -> torch.Tensor:
    factor = q_sparse_block_size // q_block_size
    if factor <= 0 or q_sparse_block_size % q_block_size != 0:
        raise ValueError(
            "q_sparse_block_size must be a positive multiple of "
            f"q_block_size ({q_block_size}), got {q_sparse_block_size}"
        )
    bsz, nhead, q_blocks, kv_blocks = block_map.shape
    q_blocks_sparse = (q_blocks + factor - 1) // factor
    pad_q = q_blocks_sparse * factor - q_blocks
    if pad_q > 0:
        pad = torch.zeros(
            bsz,
            nhead,
            pad_q,
            kv_blocks,
            dtype=torch.bool,
            device=block_map.device,
        )
        block_map = torch.cat([block_map, pad], dim=2)
    block_map = block_map.view(bsz, nhead, q_blocks_sparse, factor, kv_blocks)
    return block_map.any(dim=3)


def _get_vsa_mask_mod():
    # Kept for compatibility with older benchmark utilities.
    # Current wrapper path relies on block_sparse_tensors directly.
    return None


@functools.lru_cache(maxsize=4)
def _get_vbs_mask_mod(kv_block_size: int):
    """Build a CuTe mask_mod that trims per-KV-block valid tokens.

    aux_tensors[0] must be int32 tensor of shape [kv_blocks], where each value is
    the valid token count in [0, kv_block_size] for that KV block.
    """
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils
    from flash_attn.cute.block_sparsity import fast_sampling

    kv_block_size_const = int(kv_block_size)

    @fast_sampling
    @cute.jit
    def _vbs_mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        del batch, head, m_idx, seqlen_info
        block_size_ssa = utils.scalar_to_ssa(kv_block_size_const, cutlass.Int32)
        zero_ssa = utils.scalar_to_ssa(0, cutlass.Int32)
        kv_blk = n_idx // block_size_ssa
        kv_off = n_idx % block_size_ssa
        kv_sizes = aux_tensors[0]
        valid = utils.scalar_to_ssa(kv_sizes[kv_blk[0]], cutlass.Int32)
        return (valid > zero_ssa) & (kv_off < valid)

    return _vbs_mask_mod


def block_sparse_attn_cute_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CuTe forward-only VSA implementation with n_block_size=64.

    Interface mirrors fastvideo_kernel.block_sparse_attn.block_sparse_attn.
    Inputs q/k/v are expected in [B, H, T, D] layout.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("q, k, v must be CUDA tensors.")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must be 4D tensors [B, H, T, D].")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("Batch size mismatch among q/k/v.")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("Head count mismatch among q/k/v.")
    if q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
        raise ValueError("Head dim mismatch among q/k/v.")
    block_map = block_map.to(torch.bool)
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)
    if block_map.dim() != 4:
        raise ValueError(
            "block_map must be [B,H,Q,KV] (or [H,Q,KV]), "
            f"got shape={tuple(block_map.shape)}"
        )
    if block_map.shape[0] != q.shape[0] or block_map.shape[1] != q.shape[1]:
        raise ValueError(
            "block_map batch/head must match q/k/v. "
            f"got block_map={tuple(block_map.shape[:2])}, q={tuple(q.shape[:2])}"
        )

    q_blocks = block_map.shape[2]
    kv_blocks = block_map.shape[3]
    if q_blocks <= 0 or kv_blocks <= 0:
        raise ValueError("block_map must have positive Q/KV block dimensions.")
    if q.shape[2] % q_blocks != 0 or k.shape[2] % kv_blocks != 0:
        raise ValueError(
            "q_len/kv_len must be divisible by block_map block counts. "
            f"got q_len={q.shape[2]}, kv_len={k.shape[2]}, "
            f"q_blocks={q_blocks}, kv_blocks={kv_blocks}"
        )
    q_block_size = q.shape[2] // q_blocks
    kv_block_size = k.shape[2] // kv_blocks

    if variable_block_sizes.dtype != torch.int32:
        variable_block_sizes = variable_block_sizes.to(torch.int32)
    if not variable_block_sizes.is_cuda:
        variable_block_sizes = variable_block_sizes.to(q.device)
    if variable_block_sizes.numel() != kv_blocks:
        raise ValueError(
            "variable_block_sizes length mismatch: expected "
            f"{kv_blocks}, got {variable_block_sizes.numel()}"
        )
    if torch.any((variable_block_sizes < 0) | (variable_block_sizes > kv_block_size)):
        raise ValueError(
            "variable_block_sizes values must be in "
            f"[0, {kv_block_size}] for this input."
        )

    _ensure_flash_attn_importable()
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from flash_attn.cute.interface import _flash_attn_fwd

    q_sparse_candidate = _choose_q_sparse_block_size(q.shape[2], m_block_size=128)
    q_sparse_block_size = max(
        q_block_size,
        ((q_sparse_candidate + q_block_size - 1) // q_block_size) * q_block_size,
    )
    sparse_map = _aggregate_q_block_map(
        block_map,
        q_sparse_block_size=q_sparse_block_size,
        q_block_size=q_block_size,
    )
    # Split sparse edges into:
    # - full blocks: variable_block_sizes == kv_block_size (no token-level mask needed)
    # - partial blocks: 0 < variable_block_sizes < kv_block_size (needs token-level mask)
    # - zero blocks: variable_block_sizes == 0 (drop)
    kv_full = (variable_block_sizes == kv_block_size).view(1, 1, 1, -1)
    kv_partial = ((variable_block_sizes > 0) & (variable_block_sizes < kv_block_size)).view(1, 1, 1, -1)
    full_map = sparse_map & kv_full
    mask_map = sparse_map & kv_partial

    full_block_idx, full_block_cnt = _map_to_index(full_map)
    mask_block_idx, mask_block_cnt = _map_to_index(mask_map)
    full_block_idx = full_block_idx.to(torch.int32).contiguous()
    full_block_cnt = full_block_cnt.to(torch.int32).contiguous()
    mask_block_idx = mask_block_idx.to(torch.int32).contiguous()
    mask_block_cnt = mask_block_cnt.to(torch.int32).contiguous()

    sparse_tensors = BlockSparseTensorsTorch(
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        block_size=(q_sparse_block_size, kv_block_size),
    )

    # CuTe uses [B, S, H, D].
    q_cute = q.transpose(1, 2).contiguous()
    k_cute = k.transpose(1, 2).contiguous()
    v_cute = v.transpose(1, 2).contiguous()
    use_vbs_mask = bool((variable_block_sizes > 0).any().item() and (variable_block_sizes < kv_block_size).any().item())
    out_cute, lse_cute = _flash_attn_fwd(
        q_cute,
        k_cute,
        v_cute,
        m_block_size=128,
        n_block_size=kv_block_size,
        mask_mod=_get_vbs_mask_mod(kv_block_size) if use_vbs_mask else None,
        block_sparse_tensors=sparse_tensors,
        aux_tensors=[variable_block_sizes] if use_vbs_mask else None,
        causal=False,
        return_lse=True,
    )

    out = out_cute.transpose(1, 2).contiguous()
    # Align with existing wrapper's M shape [B,H,T].
    if lse_cute is None:
        lse = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        lse = lse_cute.transpose(1, 2).contiguous()
    return out, lse


