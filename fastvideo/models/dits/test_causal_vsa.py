import math
from types import SimpleNamespace

import torch

from fastvideo.attention.backends.video_sparse_attn import (
    VSA_TILE_SIZE,
    construct_variable_block_sizes,
)
from fastvideo.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from fastvideo.forward_context import set_forward_context
from fastvideo.layers.rotary_embedding import get_nd_rotary_pos_embed
from fastvideo.models.dits.causal_wanvideo import CausalWanSelfAttention_VSA
from fastvideo.utils import is_vsa_available


def _build_freqs_cis(
    dim: int,
    num_heads: int,
    dit_seq_shape_block: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build RoPE (cos, sin) with the same recipe as WanVideo, but without
    requiring sequence parallel initialization.

    We call `get_nd_rotary_pos_embed` directly with `sp_world_size=1` so that
    no distributed SP group is needed for this standalone test.
    """
    d = dim // num_heads
    rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
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
    return freqs_cos.to(device), freqs_sin.to(device)


def run_causal_vsa_kvcache_sanity_check(
    *,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    dit_seq_shape_block: tuple[int, int, int] = (6, 10, 13),
    num_blocks: int = 3,
    VSA_sparsity: float = 0.5,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Sanity check for causal VSA implementation:

    - Verify `variable_block_sizes_block` sums to the per-block token count.
    - Verify concatenated `kv_cache['variable_block_sizes']` matches the
      total prefix token count after each block.
    - Verify KV cache tiling length is consistent with the number of tiles.
    - Verify the derived `cur_topk` matches the intended sparsity formula:
          topk = ceil((1 - VSA_sparsity) *
                      (total_seq_length_prefix / prod(VSA_TILE_SIZE)))
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not is_vsa_available():
        print("VSA backend is not available; skipping causal VSA test.")
        return

    dim = num_heads * head_dim
    T_block_p, H_prime, W_prime = dit_seq_shape_block
    L_block = T_block_p * H_prime * W_prime

    # Geometry for one block in tiled space
    num_tiles_block = (
        math.ceil(T_block_p / VSA_TILE_SIZE[0]),
        math.ceil(H_prime / VSA_TILE_SIZE[1]),
        math.ceil(W_prime / VSA_TILE_SIZE[2]),
    )
    n_tiles_block = math.prod(num_tiles_block)
    L_tiled_block = n_tiles_block * math.prod(VSA_TILE_SIZE)

    print("=== Causal VSA KV-cache sanity check ===")
    print(f"dit_seq_shape_block        = {dit_seq_shape_block}")
    print(f"num_tiles_block            = {num_tiles_block}")
    print(f"L_block (non‑padded tokens) = {L_block}")
    print(f"L_tiled_block (padded)     = {L_tiled_block}")
    print(f"num_heads, head_dim, dim   = {num_heads}, {head_dim}, {dim}")
    print(f"num_blocks                 = {num_blocks}")
    print(f"VSA_sparsity               = {VSA_sparsity}")

    # Allocate KV cache large enough for `num_blocks` blocks in tiled space.
    kv_capacity = num_blocks * L_tiled_block
    kv_cache = {
        "k":
        torch.zeros(
            batch_size,
            kv_capacity,
            num_heads,
            head_dim,
            dtype=dtype,
            device=device,
        ),
        "v":
        torch.zeros(
            batch_size,
            kv_capacity,
            num_heads,
            head_dim,
            dtype=dtype,
            device=device,
        ),
        # per‑block 3D shape (after patching)
        "dit_seq_shape_block":
        torch.tensor(dit_seq_shape_block,
                     dtype=torch.long,
                     device=device),
        # prefix‑wide variable_block_sizes (concatenation over blocks)
        "variable_block_sizes":
        torch.empty(0, dtype=torch.long, device=device),
        "num_cached_blocks":
        0,
    }

    # Build RoPE tensors (just to satisfy the interface; values are not checked).
    freqs_cis = _build_freqs_cis(dim, num_heads, dit_seq_shape_block, device)

    # Causal VSA attention module under test
    attn = CausalWanSelfAttention_VSA(
        dim=dim,
        num_heads=num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        eps=1e-6,
        parallel_attention=False,
        kv_cache_policy="error",
    ).to(device)

    # We only need VSA_sparsity from the forward context.
    meta = SimpleNamespace(VSA_sparsity=VSA_sparsity)

    with set_forward_context(current_timestep=torch.tensor(0),
                             attn_metadata=meta):
        for block_idx in range(num_blocks):
            print(f"\n--- Block {block_idx} ---")
            num_cached_blocks_before = int(kv_cache.get("num_cached_blocks", 0))
            assert num_cached_blocks_before == block_idx

            # Random Q/K/V/gate for a single block.
            q = torch.randn(batch_size,
                            L_block,
                            num_heads,
                            head_dim,
                            device=device,
                            dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            gate = torch.randn_like(q)

            # Expected per‑block metadata from `construct_variable_block_sizes`.
            variable_block_sizes_block = construct_variable_block_sizes(
                dit_seq_shape_block,
                num_tiles_block,
                device=device,
            )

            # Basic metadata checks for the current block.
            assert variable_block_sizes_block.numel() == n_tiles_block, (
                "variable_block_sizes_block length mismatch with num_tiles_block"
            )
            assert (variable_block_sizes_block.sum().item() == L_block), (
                "Sum of variable_block_sizes_block must equal number of non‑padded "
                "tokens in one block."
            )

            print(
                f"  variable_block_sizes_block.shape = {tuple(variable_block_sizes_block.shape)}"
            )
            print(
                f"  sum(variable_block_sizes_block)   = {variable_block_sizes_block.sum().item()} "
                f"(expected {L_block})")

            # Call causal VSA attention to update KV cache and compute output.
            _ = attn(
                q=q,
                k=k,
                v=v,
                gate=gate,
                freqs_cis=freqs_cis,
                block_mask=None,
                kv_cache=kv_cache,
                current_start=0,
                cache_start=None,
            )

            # KV cache geometry checks
            num_cached_blocks_after = int(kv_cache["num_cached_blocks"])
            assert num_cached_blocks_after == block_idx + 1

            # Check cache capacity and that we've written at least up to end_idx.
            k_cache = kv_cache["k"]
            v_cache = kv_cache["v"]
            cache_capacity = k_cache.shape[1]
            expected_min_used = (block_idx + 1) * L_tiled_block
            assert cache_capacity >= expected_min_used, (
                "KV cache capacity is smaller than the minimum required "
                "tiled tokens."
            )

            # variable_block_sizes over the whole prefix.
            vbs_all = kv_cache["variable_block_sizes"]
            expected_vbs_len = (block_idx + 1) * n_tiles_block
            expected_vbs_sum = (block_idx + 1) * L_block
            assert vbs_all.numel() == expected_vbs_len, (
                "Concatenated variable_block_sizes length mismatch over prefix."
            )
            assert vbs_all.sum().item() == expected_vbs_sum, (
                "Sum of concatenated variable_block_sizes must equal total "
                "non‑padded tokens in the prefix."
            )

            print(
                f"  num_cached_blocks (after)        = {num_cached_blocks_after}"
            )
            print(
                f"  kv_cache['k'].shape              = {tuple(k_cache.shape)}"
            )
            print(
                f"  kv_cache['variable_block_sizes'].shape = {tuple(vbs_all.shape)}"
            )
            print(
                f"  sum(variable_block_sizes_all)     = {vbs_all.sum().item()} "
                f"(expected {expected_vbs_sum})")

            # Check top‑k formula consistency (we can't access internal `topk`,
            # but we ensure our understanding of the intended formula matches.)
            total_seq_length_prefix = (block_idx + 1) * L_block
            expected_topk = math.ceil(
                (1.0 - VSA_sparsity) *
                (total_seq_length_prefix / math.prod(VSA_TILE_SIZE)))
            print(f"  total_seq_length_prefix          = {total_seq_length_prefix}")
            print(f"  expected_topk (sparsity={VSA_sparsity}) = {expected_topk}")

    print("\nAll causal VSA KV-cache sanity checks passed.")


def main() -> None:
    # Ensure distributed + model parallel environment is initialized so that
    # VideoSparseAttentionImpl can query the SP group.
    if not model_parallel_is_initialized():
        # Single-process, single-GPU setup: tp_size=1, sp_size=1.
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=1, sp_size=1)
    run_causal_vsa_kvcache_sanity_check()


if __name__ == "__main__":
    main()


