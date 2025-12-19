import math
from types import SimpleNamespace

import pytest
import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.causal_wanvideo import CausalWanSelfAttention_VSA
from fastvideo.layers.rotary_embedding import _apply_rotary_emb, get_nd_rotary_pos_embed
from fastvideo.attention.backends.video_sparse_attn import (
    VSA_TILE_SIZE,
    VideoSparseAttentionImpl,
    construct_variable_block_sizes,
    get_non_pad_index,
    get_reverse_tile_partition_indices,
    get_tile_partition_indices,
    video_sparse_attn,
)


@pytest.mark.skipif(video_sparse_attn is None, reason="VSA extension (video_sparse_attn) is not installed.")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="VSA kernel requires CUDA.")
def test_causal_vsa_kvcache_two_blocks_matches_one_shot_prefix(monkeypatch):
    """
    Causal Wan (VSA KV-cache) invariant:

    If we run self-attention on two consecutive 4-frame blocks incrementally
    (block0 then block1, updating KV cache), the output of the second call
    (for block1 tokens) should match a "one-shot" reference that uses:
      - Q from block1
      - K/V from prefix = block0 ++ block1
    (i.e. the last 4 frames from an 8-frame prefix computation).

    This test includes RoPE with correct absolute time offsets:
      - block0 uses start_frame=0
      - block1 uses start_frame=T_block_p (4)
    so that the incremental KV cache stores keys at the same positions as the
    one-shot prefix reference.
    """

    # Patch SP group dependency to a trivial world (sp=1) so the unit test can run
    # without initializing distributed parallel state.
    class _DummySPGroup:
        world_size = 1
        rank_in_group = 0

    import fastvideo.attention.backends.video_sparse_attn as vsa_backend

    monkeypatch.setattr(vsa_backend, "get_sp_group", lambda: _DummySPGroup())

    torch.manual_seed(0)
    device = torch.device("cuda")
    # Prefer bf16 (project default autocast dtype) to match common runtime configs.
    dtype = torch.bfloat16

    # Block shape (patch space): (T_block', H', W') with padding in H/W to exercise tiling.
    T_block_p, H_prime, W_prime = 4, 5, 5
    L_block = T_block_p * H_prime * W_prime

    # Attention dims (use supported head_size to be safe with kernel/backends).
    num_heads = 2
    head_dim = 64
    dim = num_heads * head_dim

    # Build block metadata (same logic as CausalWanSelfAttention_VSA._build_block_tiling_metadata)
    dit_seq_shape_block = (T_block_p, H_prime, W_prime)
    num_tiles_block = (
        math.ceil(T_block_p / VSA_TILE_SIZE[0]),
        math.ceil(H_prime / VSA_TILE_SIZE[1]),
        math.ceil(W_prime / VSA_TILE_SIZE[2]),
    )
    tile_partition_indices_block = get_tile_partition_indices(
        dit_seq_shape_block, VSA_TILE_SIZE, device
    )
    reverse_tile_partition_indices_block = get_reverse_tile_partition_indices(
        dit_seq_shape_block, VSA_TILE_SIZE, device
    )
    variable_block_sizes_block = construct_variable_block_sizes(
        dit_seq_shape_block, num_tiles_block, device
    )
    non_pad_index_block = get_non_pad_index(
        variable_block_sizes_block, math.prod(VSA_TILE_SIZE)
    )

    # Two consecutive blocks worth of q/k/v/gate inputs: [B, L_block, H, D]
    B = 1
    q0 = torch.randn(B, L_block, num_heads, head_dim, device=device, dtype=dtype)
    k0 = torch.randn_like(q0)
    v0 = torch.randn_like(q0)
    g0 = torch.randn_like(q0)

    q1 = torch.randn(B, L_block, num_heads, head_dim, device=device, dtype=dtype)
    k1 = torch.randn_like(q1)
    v1 = torch.randn_like(q1)
    g1 = torch.randn_like(q1)

    # Build real RoPE (cos, sin) for each block with correct absolute time offsets.
    # We call `get_nd_rotary_pos_embed` directly with sp_world_size=1 so the unit test
    # does not depend on distributed SP group initialization.
    # Match WanVideo's rope_dim_list recipe:
    #   rope_dim_list = [d - 4*(d//6), 2*(d//6), 2*(d//6)] where d=head_dim.
    d = head_dim
    rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
    assert sum(rope_dim_list) == head_dim
    freqs_cis0 = get_nd_rotary_pos_embed(
        rope_dim_list,
        dit_seq_shape_block,
        theta=10000.0,
        theta_rescale_factor=1.0,
        interpolation_factor=1.0,
        shard_dim=0,
        sp_rank=0,
        sp_world_size=1,
        dtype=torch.float64,
        start_frame=0,
    )
    freqs_cis1 = get_nd_rotary_pos_embed(
        rope_dim_list,
        dit_seq_shape_block,
        theta=10000.0,
        theta_rescale_factor=1.0,
        interpolation_factor=1.0,
        shard_dim=0,
        sp_rank=0,
        sp_world_size=1,
        dtype=torch.float64,
        start_frame=T_block_p,
    )

    # Incremental path: two calls with a KV cache that grows.
    attn = CausalWanSelfAttention_VSA(
        dim=dim,
        num_heads=num_heads,
        kv_cache_policy="extend",
    ).to(device=device)

    kv_cache = {
        "k": torch.empty((B, 0, num_heads, head_dim), device=device, dtype=dtype),
        "v": torch.empty((B, 0, num_heads, head_dim), device=device, dtype=dtype),
        "dit_seq_shape_block": dit_seq_shape_block,
    }

    # Provide forward context with sparsity for topk computation.
    # Keep sparsity=0 => maximum topk (dense over tiles), stable reference.
    with set_forward_context(current_timestep=0, attn_metadata=SimpleNamespace(VSA_sparsity=0.0)):
        out0 = attn(
            q=q0,
            k=k0,
            v=v0,
            gate=g0,
            freqs_cis=freqs_cis0,
            block_mask=None,
            kv_cache=kv_cache,
        )
        out1 = attn(
            q=q1,
            k=k1,
            v=v1,
            gate=g1,
            freqs_cis=freqs_cis1,
            block_mask=None,
            kv_cache=kv_cache,
        )

    # One-shot reference: build tiled prefix (block0 ++ block1) in one go and
    # compute attention for Q from block1 only.
    vsa_impl = VideoSparseAttentionImpl(
        num_heads=num_heads,
        head_size=head_dim,
        causal=False,
        softmax_scale=head_dim**-0.5,
        num_kv_heads=num_heads,
        prefix="test.impl",
    )

    # Apply RoPE exactly like the model does:
    #   roped_query = _apply_rotary_emb(q, cos, sin).type_as(v)
    #   roped_key   = _apply_rotary_emb(k, cos, sin).type_as(v)
    cos0, sin0 = freqs_cis0
    cos1, sin1 = freqs_cis1
    roped_k0 = _apply_rotary_emb(k0, cos0, sin0, is_neox_style=False).type_as(v0)
    roped_k1 = _apply_rotary_emb(k1, cos1, sin1, is_neox_style=False).type_as(v1)
    roped_q1 = _apply_rotary_emb(q1, cos1, sin1, is_neox_style=False).type_as(v1)

    # Tile each block and concatenate along the 1D tiled axis.
    # Note: V is NOT RoPE'd; only Q/K are.
    k0_tiled = vsa_impl.tile(roped_k0, num_tiles_block, tile_partition_indices_block, non_pad_index_block)
    v0_tiled = vsa_impl.tile(v0, num_tiles_block, tile_partition_indices_block, non_pad_index_block)
    k1_tiled = vsa_impl.tile(roped_k1, num_tiles_block, tile_partition_indices_block, non_pad_index_block)
    v1_tiled = vsa_impl.tile(v1, num_tiles_block, tile_partition_indices_block, non_pad_index_block)
    g1_tiled = vsa_impl.tile(g1, num_tiles_block, tile_partition_indices_block, non_pad_index_block)
    q1_tiled = vsa_impl.tile(roped_q1, num_tiles_block, tile_partition_indices_block, non_pad_index_block)

    k_prefix = torch.cat([k0_tiled, k1_tiled], dim=1)
    v_prefix = torch.cat([v0_tiled, v1_tiled], dim=1)
    variable_block_sizes_all = torch.cat([variable_block_sizes_block, variable_block_sizes_block], dim=0)

    total_seq_length_prefix = 2 * T_block_p * H_prime * W_prime
    topk = math.ceil((1.0 - 0.0) * (total_seq_length_prefix / math.prod(VSA_TILE_SIZE)))

    q_in = q1_tiled.transpose(1, 2).contiguous()
    k_in = k_prefix.transpose(1, 2).contiguous()
    v_in = v_prefix.transpose(1, 2).contiguous()
    g_in = g1_tiled.transpose(1, 2).contiguous()

    ref_tiled_out = video_sparse_attn(
        q_in,
        k_in,
        v_in,
        variable_block_sizes=variable_block_sizes_all,
        topk=topk,
        block_size=VSA_TILE_SIZE,
        compress_attn_weight=g_in,
    ).transpose(1, 2)
    ref_out1 = vsa_impl.untile(
        ref_tiled_out, reverse_tile_partition_indices_block, non_pad_index_block
    )

    # Compare second-block outputs (this is the key invariant).
    torch.testing.assert_close(out1, ref_out1, rtol=2e-2, atol=2e-2)

    # Sanity: first block output should match a one-shot prefix of only block0.
    k0_only = k0_tiled
    v0_only = v0_tiled
    vbs0_only = variable_block_sizes_block
    topk0 = math.ceil((T_block_p * H_prime * W_prime) / math.prod(VSA_TILE_SIZE))
    roped_q0 = _apply_rotary_emb(q0, cos0, sin0, is_neox_style=False).type_as(v0)
    ref0_tiled = video_sparse_attn(
        vsa_impl.tile(roped_q0, num_tiles_block, tile_partition_indices_block, non_pad_index_block).transpose(1, 2).contiguous(),
        k0_only.transpose(1, 2).contiguous(),
        v0_only.transpose(1, 2).contiguous(),
        variable_block_sizes=vbs0_only,
        topk=topk0,
        block_size=VSA_TILE_SIZE,
        compress_attn_weight=vsa_impl.tile(g0, num_tiles_block, tile_partition_indices_block, non_pad_index_block).transpose(1, 2).contiguous(),
    ).transpose(1, 2)
    ref_out0 = vsa_impl.untile(ref0_tiled, reverse_tile_partition_indices_block, non_pad_index_block)
    torch.testing.assert_close(out0, ref_out0, rtol=2e-2, atol=2e-2)

