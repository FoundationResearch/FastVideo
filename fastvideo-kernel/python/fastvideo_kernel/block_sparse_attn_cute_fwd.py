from __future__ import annotations

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
) -> torch.Tensor:
    factor = q_sparse_block_size // BLOCK_M
    if factor <= 0 or q_sparse_block_size % BLOCK_M != 0:
        raise ValueError(
            f"q_sparse_block_size must be a positive multiple of {BLOCK_M}, got {q_sparse_block_size}"
        )
    bsz, nhead, q_blocks_64, kv_blocks_64 = block_map.shape
    q_blocks_sparse = (q_blocks_64 + factor - 1) // factor
    pad_q = q_blocks_sparse * factor - q_blocks_64
    if pad_q > 0:
        pad = torch.zeros(
            bsz,
            nhead,
            pad_q,
            kv_blocks_64,
            dtype=torch.bool,
            device=block_map.device,
        )
        block_map = torch.cat([block_map, pad], dim=2)
    block_map = block_map.view(bsz, nhead, q_blocks_sparse, factor, kv_blocks_64)
    return block_map.any(dim=3)


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
    if q.shape[2] % BLOCK_M != 0 or k.shape[2] % BLOCK_N != 0:
        raise ValueError(
            f"q_len and kv_len must be divisible by {BLOCK_M}/{BLOCK_N} for this wrapper."
        )

    if variable_block_sizes.dtype != torch.int32:
        variable_block_sizes = variable_block_sizes.to(torch.int32)
    if not variable_block_sizes.is_cuda:
        variable_block_sizes = variable_block_sizes.to(q.device)
    if variable_block_sizes.numel() != k.shape[2] // BLOCK_N:
        raise ValueError(
            "variable_block_sizes length mismatch: expected "
            f"{k.shape[2] // BLOCK_N}, got {variable_block_sizes.numel()}"
        )
    if torch.any((variable_block_sizes < 0) | (variable_block_sizes > BLOCK_N)):
        raise ValueError("variable_block_sizes values must be in [0, 64].")
    # Current wrapper does not thread variable block sizes into CuTe mask_mod yet.
    # Keep semantics explicit until mask_mod path is integrated.
    if not torch.all(variable_block_sizes == BLOCK_N):
        raise NotImplementedError(
            "CuTe forward wrapper currently requires all variable_block_sizes == 64."
        )

    _ensure_flash_attn_importable()
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from flash_attn.cute.interface import _flash_attn_fwd

    block_map = block_map.to(torch.bool)
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)
    q_sparse_block_size = _choose_q_sparse_block_size(q.shape[2], m_block_size=128)
    sparse_map = _aggregate_q_block_map(block_map, q_sparse_block_size=q_sparse_block_size)
    mask_block_idx, mask_block_cnt = _map_to_index(sparse_map)
    mask_block_idx = mask_block_idx.to(torch.int32).contiguous()
    mask_block_cnt = mask_block_cnt.to(torch.int32).contiguous()
    full_block_cnt = torch.zeros_like(mask_block_cnt)
    full_block_idx = torch.zeros_like(mask_block_idx)

    sparse_tensors = BlockSparseTensorsTorch(
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        block_size=(q_sparse_block_size, BLOCK_N),
    )

    # CuTe uses [B, S, H, D].
    q_cute = q.transpose(1, 2).contiguous()
    k_cute = k.transpose(1, 2).contiguous()
    v_cute = v.transpose(1, 2).contiguous()
    out_cute, lse_cute = _flash_attn_fwd(
        q_cute,
        k_cute,
        v_cute,
        m_block_size=128,
        n_block_size=BLOCK_N,
        block_sparse_tensors=sparse_tensors,
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


