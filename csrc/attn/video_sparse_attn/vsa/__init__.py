import torch
from typing import Tuple
block_sparse_attn=None
import torch
major, minor = torch.cuda.get_device_capability(0)
if major == 9 and minor == 0:# check if H100
    from vsa_cuda import block_sparse_fwd, block_sparse_bwd
    from vsa.block_sparse_wrapper import block_sparse_attn_SM90
    block_sparse_attn = block_sparse_attn_SM90
else:
    from vsa.block_sparse_wrapper import block_sparse_attn_triton
    block_sparse_fwd = None
    block_sparse_bwd = None
    block_sparse_attn = block_sparse_attn_triton

BLOCK_M = 64
BLOCK_N = 64


def torch_attention(q, k, v) -> Tuple[torch.Tensor, torch.Tensor]:
    QK = torch.matmul(q, k.transpose(-2, -1))
    QK /= (q.size(-1)**0.5)

    # Causal mask removed since causal is always false

    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v)
    return output, QK


def video_sparse_attn(
    q,
    k,
    v,
    variable_block_sizes,
    topk,
    block_size,
    compress_attn_weight=None,
    q_variable_block_sizes=None,
):
    """
    q: [batch_size, num_heads, seq_len_q, head_dim]
    k: [batch_size, num_heads, seq_len_kv, head_dim]
    v: [batch_size, num_heads, seq_len_kv, head_dim]
    topk: int
    block_size: int or tuple of 3 ints
    video_shape: tuple of (T, H, W)
    compress_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    select_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    q_variable_block_sizes: Optional[Tensor] of shape [num_q_tiles], giving the number of
        valid (non-padded) tokens in each Q tile. If None, we assume Q uses the same
        variable_block_sizes as KV (backwards compatible) **only when** q and kv have the
        same number of tiles. If q and kv have different numbers of tiles, you must pass
        q_variable_block_sizes explicitly.

    NOTE: We assume q, k, v is zero padded!!
    V1 of sparse attention. Include compress attn and sparse attn branch, use average pooling to compress. 
    Assume q, k, v is flattened in this way: [batch_size, num_heads, T//block_size[0], H//block_size[1], W//block_size[2], block_size[0], block_size[1], block_size[2]]
    """

    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)

    block_elements = block_size[0] * block_size[1] * block_size[2]
    assert block_elements == 64
    assert q.shape[2] % block_elements == 0
    assert k.shape[2] % block_elements == 0
    assert v.shape[2] % block_elements == 0
    assert k.shape == v.shape, "k and v must have the same shape"

    batch_size, num_heads, seq_len_q, head_dim = q.shape
    seq_len_kv = k.shape[2]
    # variable_block_sizes semantics: KV-side per-tile (per 64-token block) valid token counts.
    # Length must match the number of KV tiles.
    assert variable_block_sizes.numel() == (seq_len_kv // block_elements), (
        f"variable_block_sizes must have length seq_len_kv//{block_elements}, "
        f"but got {variable_block_sizes.numel()} vs {seq_len_kv // block_elements}"
    )
    if compress_attn_weight is not None:
        assert compress_attn_weight.shape == q.shape, (
            f"compress_attn_weight must match q shape {tuple(q.shape)}, "
            f"but got {tuple(compress_attn_weight.shape)}"
        )

    # compress attn
    # Q and KV can have different tile counts. For correctness, Q needs its own
    # per-tile valid token counts if it is also padded.
    num_q_tiles = seq_len_q // block_elements
    num_kv_tiles = seq_len_kv // block_elements
    if q_variable_block_sizes is None:
        if num_q_tiles == variable_block_sizes.numel():
            q_variable_block_sizes = variable_block_sizes
        else:
            raise ValueError(
                "q_variable_block_sizes must be provided when q and kv have different "
                f"numbers of tiles (num_q_tiles={num_q_tiles}, num_kv_tiles={num_kv_tiles})."
            )
    else:
        assert q_variable_block_sizes.numel() == num_q_tiles, (
            f"q_variable_block_sizes must have length seq_len_q//{block_elements}, "
            f"but got {q_variable_block_sizes.numel()} vs {num_q_tiles}"
        )

    q_compress = (
        q.view(batch_size, num_heads, num_q_tiles, block_elements, head_dim)
        .float()
        .sum(dim=3)
        / q_variable_block_sizes.view(1, 1, -1, 1)
    ).to(q.dtype)
    k_compress = (
        k.view(batch_size, num_heads, num_kv_tiles, block_elements, head_dim)
        .float()
        .sum(dim=3)
        / variable_block_sizes.view(1, 1, -1, 1)
    ).to(k.dtype)
    v_compress = (
        v.view(batch_size, num_heads, num_kv_tiles, block_elements, head_dim)
        .float()
        .sum(dim=3)
        / variable_block_sizes.view(1, 1, -1, 1)
    ).to(v.dtype)

    output_compress, block_attn_score = torch_attention(q_compress, k_compress,
                                                        v_compress)

    output_compress = output_compress.view(batch_size, num_heads,
                                           seq_len_q // block_elements, 1,
                                           head_dim)
    output_compress = output_compress.repeat(1, 1, 1, block_elements,
                                             1).view(batch_size, num_heads,
                                                     seq_len_q, head_dim)

    topK_indices = torch.topk(block_attn_score, topk, dim=-1).indices
    block_mask = torch.zeros_like(block_attn_score, dtype=torch.bool).scatter_(-1, topK_indices, True)
    output_select, _ = block_sparse_attn(q, k, v, block_mask, variable_block_sizes)

    if compress_attn_weight is not None:
        final_output = output_compress * compress_attn_weight + output_select
    else:
        final_output = output_compress + output_select
    return final_output

