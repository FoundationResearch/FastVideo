from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest
import torch


T_REAL, H_REAL, W_REAL = 16, 34, 60
T_TILE, H_TILE, W_TILE = 4, 8, 8
BLOCK_256 = T_TILE * H_TILE * W_TILE


def _ensure_local_fastvideo_kernel() -> None:
    repo_python = Path(__file__).resolve().parents[1] / "python"
    repo_python_str = str(repo_python)
    if repo_python_str not in sys.path:
        sys.path.insert(0, repo_python_str)


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
                vbs.append(int(T_TILE * valid_h * valid_w))
    return torch.tensor(vbs, dtype=torch.int32, device=device)


def _get_tile_partition_indices_local(
    t: int,
    h: int,
    w: int,
    device: torch.device,
) -> torch.LongTensor:
    idx = torch.arange(t * h * w, device=device, dtype=torch.long).reshape(t, h, w)
    chunks = []
    for tt in range((t + T_TILE - 1) // T_TILE):
        for hh in range((h + H_TILE - 1) // H_TILE):
            for ww in range((w + W_TILE - 1) // W_TILE):
                chunks.append(
                    idx[
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


def _torch_vsa256_reference_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_var: torch.Tensor,
    kv_var: torch.Tensor,
    topk_logical: int,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    bsz, seq, heads, dim = q.shape
    q_blocks = q_var.numel()
    kv_blocks = kv_var.numel()
    q_block = seq // q_blocks
    kv_block = k.shape[1] // kv_blocks

    q_c = q.view(bsz, q_blocks, q_block, heads, dim)
    k_c = k.view(bsz, kv_blocks, kv_block, heads, dim)
    v_c = v.view(bsz, kv_blocks, kv_block, heads, dim)
    q_c = (q_c.float().sum(dim=2) / q_var.view(1, -1, 1, 1)).to(q.dtype)
    k_c = (k_c.float().sum(dim=2) / kv_var.view(1, -1, 1, 1)).to(k.dtype)
    v_c = (v_c.float().sum(dim=2) / kv_var.view(1, -1, 1, 1)).to(v.dtype)
    q_ch = q_c.permute(0, 2, 1, 3).contiguous()
    k_ch = k_c.permute(0, 2, 1, 3).contiguous()
    v_ch = v_c.permute(0, 2, 1, 3).contiguous()

    scores = torch.matmul(q_ch, k_ch.transpose(-2, -1)) / math.sqrt(dim)
    attn = torch.softmax(scores, dim=-1)
    out_c_ch = torch.matmul(attn, v_ch)
    out_c_blk = out_c_ch.permute(0, 2, 1, 3).contiguous()

    topk_idx = torch.topk(scores, topk_logical, dim=-1).indices
    block_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_idx, True)

    token_idx = torch.arange(kv_block, device=kv_var.device, dtype=torch.int32)
    kv_token_valid_by_block = (token_idx.view(1, -1) < kv_var.view(-1, 1)).to(torch.bool)
    kv_token_valid = kv_token_valid_by_block.reshape(1, 1, 1, kv_blocks * kv_block)
    token_mask = block_mask.repeat_interleave(q_block, dim=2).repeat_interleave(kv_block, dim=3)
    token_mask = token_mask & kv_token_valid

    qf = q.permute(0, 2, 1, 3).float()
    kf = k.permute(0, 2, 1, 3).float()
    vf = v.permute(0, 2, 1, 3).float()
    logits = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(dim)
    logits = logits.masked_fill(~token_mask, float("-inf"))
    prob = torch.softmax(logits, dim=-1)
    out_s = torch.matmul(prob, vf).to(q.dtype).permute(0, 2, 1, 3).contiguous()

    out = out_s
    out_view = out.view(bsz, q_blocks, q_block, heads, dim)
    if gate is not None:
        gate_view = gate.view(bsz, q_blocks, q_block, heads, dim)
        out_view = out_view + out_c_blk.unsqueeze(2) * gate_view
    else:
        out_view = out_view + out_c_blk.unsqueeze(2)
    return out_view.reshape_as(q)


@pytest.mark.cuda
def test_ltx2_model_tile_fastpath_e2e_vs_torch_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(0)
    os.environ["FASTVIDEO_VSA_256"] = "1"
    os.environ["FASTVIDEO_VSA_256_BACKEND"] = "cute"
    _ensure_local_fastvideo_kernel()
    from fastvideo_kernel.ops import video_sparse_attn_bshd

    device = torch.device("cuda")
    dtype = torch.bfloat16
    bsz, heads, dim = 1, 8, 128
    num_layers = 3

    kv_var_256 = _build_vbs_256_for_ltx2real(device)
    q_var_256 = kv_var_256
    q_blocks = kv_var_256.numel()
    topk_logical = min(16, q_blocks)

    seq_real = T_REAL * H_REAL * W_REAL
    padded_seq = q_blocks * BLOCK_256
    tile_partition_indices = _get_tile_partition_indices_local(T_REAL, H_REAL, W_REAL, device)
    reverse_tile_partition_indices = torch.argsort(tile_partition_indices)
    non_pad_index = _get_non_pad_index_local(kv_var_256, BLOCK_256)

    x0 = torch.randn(bsz, seq_real, heads, dim, device=device, dtype=dtype)
    gates = [torch.randn_like(x0) for _ in range(num_layers)]

    # Fastpath e2e: tile once -> multi-layer attention -> untile once.
    x_fast_t = _tile_local(x0, padded_seq, tile_partition_indices, non_pad_index)
    gates_t = [_tile_local(g, padded_seq, tile_partition_indices, non_pad_index) for g in gates]
    for i in range(num_layers):
        x_fast_t = video_sparse_attn_bshd(
            x_fast_t,
            x_fast_t,
            x_fast_t,
            kv_var_256,
            q_var_256,
            topk_logical,
            block_size=(4, 8, 8),
            compress_attn_weight=gates_t[i],
        )
    out_fast = _untile_local(x_fast_t, reverse_tile_partition_indices, non_pad_index)

    # Torch e2e reference under the exact same model-level schedule.
    x_ref_t = _tile_local(x0, padded_seq, tile_partition_indices, non_pad_index)
    for i in range(num_layers):
        x_ref_t = _torch_vsa256_reference_bshd(
            x_ref_t,
            x_ref_t,
            x_ref_t,
            q_var_256,
            kv_var_256,
            topk_logical,
            gates_t[i],
        )
    out_ref = _untile_local(x_ref_t, reverse_tile_partition_indices, non_pad_index)

    assert torch.isfinite(out_fast).all().item(), "NaN/Inf found in fastpath output"
    diff = (out_fast - out_ref).abs()
    avg_abs = diff.mean().item()
    max_rel = (diff.max() / (out_ref.abs().mean() + 1e-6)).item()
    print(
        "[ltx2-model-fastpath-vs-torch] "
        f"layers={num_layers}, avg_abs={avg_abs:.6e}, max_rel={max_rel:.6e}"
    )
    assert avg_abs < 2e-3
    assert max_rel < 0.3

