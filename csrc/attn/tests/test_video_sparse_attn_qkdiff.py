import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VSA kernels require CUDA.")
def test_video_sparse_attn_supports_qk_diff_length_and_vbs_is_kv_only():
    """
    Regression test:
    - q has a different length than k/v (q_blocks != kv_blocks)
    - variable_block_sizes is KV-side only (len == kv_blocks)
    - If we zero out the compress branch, video_sparse_attn output must match
      the direct block_sparse_attn output with a full block mask.
    """
    # Import inside the test so pytest collection doesn't hard-fail on systems without VSA.
    try:
        from vsa import block_sparse_attn, video_sparse_attn
    except Exception as e:
        pytest.skip(f"VSA extension not available/importable: {e}")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    B, H, D = 1, 2, 64
    block_elements = 64  # VSA uses 64 tokens per tile
    q_blocks = 2
    kv_blocks = 3
    seq_q = q_blocks * block_elements
    seq_kv = kv_blocks * block_elements

    # KV-side variable block sizes: last block is partial to exercise masking.
    kv_vbs = torch.tensor([64, 64, 17], device=device, dtype=torch.int32)

    # Make Q's last tile partially valid to exercise q_variable_block_sizes handling.
    q_vbs = torch.tensor([64, 17], device=device, dtype=torch.int32)
    q = torch.randn(B, H, seq_q, D, device=device, dtype=dtype)
    # Zero-pad the invalid tail of the last Q tile so "average over valid tokens" is meaningful.
    q[:, :, 64 + 17 :, :] = 0
    k = torch.randn(B, H, seq_kv, D, device=device, dtype=dtype)
    v = torch.randn(B, H, seq_kv, D, device=device, dtype=dtype)

    # Full block mask: attend to all KV blocks for every Q block.
    block_mask = torch.ones((B, H, q_blocks, kv_blocks), device=device, dtype=torch.bool)

    # Direct sparse attention (selection branch only).
    out_select, _ = block_sparse_attn(q, k, v, block_mask, kv_vbs)

    # video_sparse_attn: force topk to keep all KV blocks; zero out compress branch.
    compress_weight = torch.zeros_like(q)
    out = video_sparse_attn(
        q=q,
        k=k,
        v=v,
        variable_block_sizes=kv_vbs,
        q_variable_block_sizes=q_vbs,
        topk=kv_blocks,
        block_size=4,  # 4*4*4 = 64 tokens
        compress_attn_weight=compress_weight,
    )

    torch.testing.assert_close(out, out_select, rtol=2e-2, atol=2e-2)


