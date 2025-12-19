import math
import os
import sys

import pytest
import torch

# Match `tests/test_vsa.py` so this file can be run directly via:
#   python tests/test_video_sparse_attn_vs_pytorch.py -sv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.utils import (  # noqa: E402
    create_full_mask_from_block_mask,
    generate_block_sparse_mask_for_function,
)


def _pytorch_masked_attention(q, k, v, full_mask):
    """
    q: [B, H, S_q, D]
    k/v: [B, H, S_kv, D]
    full_mask: [H, S_q, S_kv] bool, True means attendable.
    """
    B, H, S_q, D = q.shape
    S_kv = k.shape[2]
    assert full_mask.shape == (H, S_q, S_kv)

    q_ = q.float()
    k_ = k.float()
    v_ = v.float()
    scores = torch.matmul(q_, k_.transpose(-2, -1)) / math.sqrt(D)  # [B,H,Sq,Skv]
    scores = scores.masked_fill(~full_mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_)  # [B,H,Sq,D]
    return out.to(dtype=q.dtype)


def _vsa_pad(x, non_pad_index, num_blocks, block_size):
    """
    x: [B, H, S, D] (unpadded, packed)
    non_pad_index: indices into [num_blocks * block_size] for valid tokens
    returns: [B, H, num_blocks*block_size, D] padded
    """
    B, H, _, D = x.shape
    padded = torch.zeros((B, H, num_blocks * block_size, D), device=x.device, dtype=x.dtype)
    padded[:, :, non_pad_index, :] = x
    return padded


def _get_non_pad_index(vid_len, n_win, win_size):
    device = vid_len.device
    starts_pad = torch.arange(n_win, device=device) * win_size
    index_pad = starts_pad[:, None] + torch.arange(win_size, device=device)[None, :]
    index_mask = torch.arange(win_size, device=device)[None, :] < vid_len[:, None]
    return index_pad[index_mask]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VSA kernels require CUDA.")
def test_video_sparse_attn_matches_pytorch_qkdiff():
    """
    Like `test_vsa.py`, but compares:
      - PyTorch masked attention (token-level full mask) vs
      - vsa.video_sparse_attn (sparse + compress)

    Covers the q/k different-length case and uses explicit q_variable_block_sizes.
    """
    try:
        from vsa import video_sparse_attn
    except Exception as e:
        pytest.skip(f"VSA extension not available/importable: {e}")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    B, H, D = 1, 4, 64
    BLOCK = 64

    # q/k have different numbers of blocks; both have a partial last block.
    q_variable_block_sizes = torch.tensor([64, 31], device=device, dtype=torch.int32)
    kv_variable_block_sizes = torch.tensor([64, 64, 17], device=device, dtype=torch.int32)
    num_q_blocks = int(q_variable_block_sizes.numel())
    num_kv_blocks = int(kv_variable_block_sizes.numel())
    S_q = int(q_variable_block_sizes.sum().item())
    S_kv = int(kv_variable_block_sizes.sum().item())

    # Build block-level sparse mask and expand to token-level mask for PyTorch ref.
    k_top = min(2, num_kv_blocks)  # sparse connectivity
    block_mask = generate_block_sparse_mask_for_function(H, num_q_blocks, num_kv_blocks, k_top, device=device)
    full_mask = create_full_mask_from_block_mask(
        block_mask, q_variable_block_sizes, kv_variable_block_sizes, device=device
    )

    # Unpadded packed sequences.
    q = torch.randn(B, H, S_q, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S_kv, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S_kv, D, device=device, dtype=dtype)

    # Pad to BLOCK=64 layout for VSA.
    q_non_pad_index = _get_non_pad_index(q_variable_block_sizes, num_q_blocks, BLOCK)
    kv_non_pad_index = _get_non_pad_index(kv_variable_block_sizes, num_kv_blocks, BLOCK)
    q_padded = _vsa_pad(q, q_non_pad_index, num_q_blocks, BLOCK)
    k_padded = _vsa_pad(k, kv_non_pad_index, num_kv_blocks, BLOCK)
    v_padded = _vsa_pad(v, kv_non_pad_index, num_kv_blocks, BLOCK)

    # compress_attn_weight is defined on q_padded tokens; use all-ones for coverage.
    compress_attn_weight = torch.ones_like(q_padded)

    # video_sparse_attn expects a batch dimension in block mask; it uses torch.topk internally.
    out_vsa_padded = video_sparse_attn(
        q_padded,
        k_padded,
        v_padded,
        variable_block_sizes=kv_variable_block_sizes,
        topk=k_top,
        block_size=4,  # 4*4*4 = 64
        compress_attn_weight=compress_attn_weight,
        q_variable_block_sizes=q_variable_block_sizes,
    )
    out_vsa = out_vsa_padded[:, :, q_non_pad_index, :]

    # PyTorch reference on packed sequences.
    out_ref = _pytorch_masked_attention(q, k, v, full_mask)

    # Tolerance: VSA uses bf16 kernels; allow some numerical drift.
    torch.testing.assert_close(out_vsa, out_ref, rtol=5e-2, atol=5e-2)


