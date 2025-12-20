# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import numpy as np
import os
import torch
import torch.nn as nn

from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
import torch.distributed as dist

import fastvideo.envs as envs
from fastvideo.attention import (DistributedAttention,
                                 DistributedAttention_VSA, LocalAttention)
from fastvideo.attention.backends.video_sparse_attn import (
    VSA_TILE_SIZE,
    construct_variable_block_sizes,
    get_non_pad_index,
    get_reverse_tile_partition_indices,
    get_tile_partition_indices,
    video_sparse_attn,
)
from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionImpl
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.forward_context import get_forward_context
from fastvideo.layers.layernorm import (FP32LayerNorm, LayerNormScaleShift,
                                        RMSNorm, ScaleResidual,
                                        ScaleResidualLayerNormScaleShift)
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import (_apply_rotary_emb,
                                               get_rotary_pos_embed)
from fastvideo.layers.visual_embedding import (PatchEmbed)
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.wanvideo import WanT2VCrossAttention, WanTimeTextImageEmbedding
from fastvideo.platforms import AttentionBackendEnum, current_platform

logger = init_logger(__name__)
class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 local_attn_size: int = -1,
                 sink_size: int = 0,
                 qk_norm=True,
                 eps=1e-6,
                 parallel_attention=False) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.parallel_attention = parallel_attention
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # Scaled dot product attention
        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,
                                          AttentionBackendEnum.TORCH_SDPA))

    def forward(self, 
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                freqs_cis: tuple[torch.Tensor, torch.Tensor],
                block_mask: BlockMask,
                kv_cache: dict | None = None,
                current_start: int = 0,
                cache_start: int | None = None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        if cache_start is None:
            cache_start = current_start

        cos, sin = freqs_cis
        roped_query = _apply_rotary_emb(q, cos, sin, is_neox_style=False).type_as(v)
        roped_key = _apply_rotary_emb(k, cos, sin, is_neox_style=False).type_as(v)

        if kv_cache is None:
            # Padding for flex attention
            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [roped_query,
                    torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                device=q.device, dtype=v.dtype)],
                dim=1
            )

            padded_roped_key = torch.cat(
                [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                        device=k.device, dtype=v.dtype)],
                dim=1
            )

            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype)],
                dim=1
            )

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = q.shape[1]
            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"] = kv_cache["k"].detach()
                kv_cache["v"] = kv_cache["v"].detach()
                # logger.info("kv_cache['k'] is in comp graph: %s", kv_cache["k"].requires_grad or kv_cache["k"].grad_fn is not None)
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            x = self.attn(
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        return x


class CausalWanSelfAttention_VSA(nn.Module):
    """
    Causal self-attention backed by Video Sparse Attention (VSA) with KV cache.
    The KV cache is limited to a fixed size. Should change to dynamic kv cache size in the future if needed.

    This module assumes:
    - Inference-time only (must be called with a KV cache).
    - KV cache stores *tiled* K/V in a 1D layout plus lightweight metadata,
      following `csrc/attn/video_sparse_attn/vsa/understand_kvcache.md`.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        eps: float = 1e-6,
        parallel_attention: bool = False,
        kv_cache_policy: str = "error",  # "error" | "extend"
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.parallel_attention = parallel_attention
        self.kv_cache_policy = kv_cache_policy
        # For code consistency.
        self.backend = AttentionBackendEnum.VIDEO_SPARSE_ATTN

    @staticmethod
    def _ensure_kv_cache_keys(kv_cache: dict) -> None:
        required_keys = {"k", "v", "dit_seq_shape_block"}
        missing = [k for k in required_keys if k not in kv_cache]
        if missing:
            raise ValueError(
                f"CausalWanSelfAttention_VSA expects KV cache to contain keys "
                f"{required_keys}, but missing {missing}. "
                "Please ensure KV cache is initialized following "
                "`understand_kvcache.md`."
            )

    def _build_block_tiling_metadata(
        self,
        kv_cache: dict,
        device: torch.device,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int],
               torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        Build per-block tiling metadata from `dit_seq_shape_block`.

        Returns:
            dit_seq_shape_block: (T_block', H', W')
            num_tiles_block    : (n_t, n_h, n_w)
            tile_partition_indices_block
            reverse_tile_partition_indices_block
            non_pad_index_block
        """
        dit_seq_shape_block: tuple[int, int,
                                   int] = tuple(kv_cache["dit_seq_shape_block"])  # type: ignore[arg-type]
        T_block_p, H_prime, W_prime = dit_seq_shape_block
        num_tiles_block = (
            math.ceil(T_block_p / VSA_TILE_SIZE[0]),
            math.ceil(H_prime / VSA_TILE_SIZE[1]),
            math.ceil(W_prime / VSA_TILE_SIZE[2]),
        )

        variable_block_sizes_block = construct_variable_block_sizes(
            dit_seq_shape_block, num_tiles_block, device
        )
        tile_partition_indices_block = get_tile_partition_indices(
            dit_seq_shape_block, VSA_TILE_SIZE, device
        )
        reverse_tile_partition_indices_block = get_reverse_tile_partition_indices(
            dit_seq_shape_block, VSA_TILE_SIZE, device
        )
        non_pad_index_block = get_non_pad_index(
            variable_block_sizes_block, math.prod(VSA_TILE_SIZE)
        )
        kv_cache.setdefault("variable_block_sizes", torch.empty(0,
                                                                device=device,
                                                                dtype=torch.long))
        return (
            dit_seq_shape_block,
            num_tiles_block,
            tile_partition_indices_block,
            reverse_tile_partition_indices_block,
            non_pad_index_block,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask | None,
        kv_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            q, k, v, gate: [B, L_block, num_heads, head_dim] tensors for the current 4-frame block.
            freqs_cis    : Rotary embeddings (cos, sin).
            block_mask   : Unused for VSA path (causality is enforced by KV cache prefix).
            kv_cache     : Dict with tiled K/V cache and metadata.

        Returns:
            Tensor of shape [B, L_block, num_heads, head_dim] for the current block.
        """
        # raise RuntimeError("Correctly calling VSA Self Attention")
        del block_mask, current_start, cache_start  # Unused in VSA path

        if kv_cache is None:
            raise NotImplementedError(
                "CausalWanSelfAttention_VSA currently supports inference with KV cache only. "
                "Please provide a KV cache dictionary."
            )

        self._ensure_kv_cache_keys(kv_cache)
        device = q.device

        # Apply RoPE on the current block only
        cos, sin = freqs_cis
        roped_query = _apply_rotary_emb(
            q, cos, sin, is_neox_style=False
        ).type_as(v)
        roped_key = _apply_rotary_emb(
            k, cos, sin, is_neox_style=False
        ).type_as(v)

        (
            dit_seq_shape_block,
            num_tiles_block,
            tile_partition_indices_block,
            reverse_tile_partition_indices_block,
            non_pad_index_block,
        ) = self._build_block_tiling_metadata(kv_cache, device) # TODO:为什么这里要重算？我看了里面的计算过程根本不需要重新计算，记下来就行的事情

        T_block_p, H_prime, W_prime = dit_seq_shape_block
        # Causal KV-cache correctness requirement:
        # This implementation appends per-block tiled K/V to form the prefix.
        # That is only equivalent to tiling the whole prefix if the block's
        # patch-time length aligns with VSA's time tile size (4).
        assert (
            T_block_p == VSA_TILE_SIZE[0]
        ), (
            "Causal VSA KV cache assumes each generated block spans exactly "
            f"{VSA_TILE_SIZE[0]} patch-time frames, but got T_block_p={T_block_p}. "
            "If you want variable-length blocks, KV cache tiling must be redesigned "
            "to avoid time-dimension padding per block."
        )

        expected_L_block = T_block_p * H_prime * W_prime
        assert q.shape[1] == expected_L_block, (
            f"VSA causal block token length mismatch: q.shape[1]={q.shape[1]} "
            f"but dit_seq_shape_block implies {expected_L_block} "
            f"(T'={T_block_p}, H'={H_prime}, W'={W_prime})."
        )
        assert k.shape[1] == expected_L_block and v.shape[1] == expected_L_block and gate.shape[1] == expected_L_block, (
            "VSA causal block token length mismatch among q/k/v/gate: "
            f"q={q.shape[1]}, k={k.shape[1]}, v={v.shape[1]}, gate={gate.shape[1]} "
            f"(expected {expected_L_block})."
        )

        # Tile current block's K/V/gate into padded 1D layout
        # Shapes: [B, L_tiled_block, num_heads, head_dim]

        vsa_impl = VideoSparseAttentionImpl(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            num_kv_heads=self.num_heads,
            prefix="causal_vsa.impl",
        )
        # TODO:我不需要给vsa impl传入metadata吗？

        # 这里都是处理当前这个new block的tile，这个没啥问题
        k_tiled_block = vsa_impl.tile(
            roped_key, num_tiles_block, tile_partition_indices_block,
            non_pad_index_block)
        v_tiled_block = vsa_impl.tile(
            v, num_tiles_block, tile_partition_indices_block,
            non_pad_index_block)
        gate_tiled_block = vsa_impl.tile(
            gate, num_tiles_block, tile_partition_indices_block,
            non_pad_index_block)

        B, L_tiled_block, num_heads, head_dim = k_tiled_block.shape
        assert num_heads == self.num_heads and head_dim == self.head_dim

        # Update KV cache in-place without rebinding tensors.
        #
        # IMPORTANT: We index cache slots by *current block index* (derived from `start_frame`)
        # so repeated diffusion timesteps overwrite the same slot, matching the non-VSA
        # causal cache behavior (no "append per timestep").
        k_cache: torch.Tensor = kv_cache["k"]
        v_cache: torch.Tensor = kv_cache["v"]
        # Match non-VSA causal behavior: break the autograd graph through the cache.
        # Use in-place detach to avoid rebinding `kv_cache["k"]` / `kv_cache["v"]`
        # (kv_cache dict wrappers may be shallow-copied by higher-level code).
        #
        # This matters in distillation/training runs where grad is enabled; without this,
        # the cache can accidentally retain references to previous graphs and/or trigger
        # in-place autograd errors.
        try:
            if torch.is_grad_enabled():
                k_cache.detach_()
                v_cache.detach_()
        except Exception:
            # Best-effort: if detach_ isn't possible for some reason, fall back to
            # leaving the cache as-is (may be slower / less safe under autograd).
            pass
        cache_capacity = k_cache.shape[1]

        # `num_cached_blocks` is treated as a tensor state storing the current prefix length
        # in blocks (i.e. number of valid cached blocks). Use in-place updates so shallow-copied
        # kv_cache wrappers still observe changes.
        if "num_cached_blocks" not in kv_cache or not torch.is_tensor(
                kv_cache["num_cached_blocks"]):
            kv_cache["num_cached_blocks"] = torch.tensor(
                [0], dtype=torch.long, device=device)
        cached_blocks_t: torch.Tensor = kv_cache["num_cached_blocks"]
        cached_blocks = int(cached_blocks_t.item())

        cur_block_idx = int(kv_cache.get("_cur_block_idx", 0))
        prefix_blocks = max(cached_blocks, cur_block_idx + 1)

        start_idx = cur_block_idx * L_tiled_block
        end_idx = start_idx + L_tiled_block

        # Handle limited KV cache capacity according to policy:
        # - "error"  : raise if capacity is exceeded (strict, recommended for training).
        # - "extend" : dynamically grow KV cache tensors to accommodate new blocks.
        if end_idx > cache_capacity or (prefix_blocks * L_tiled_block) > cache_capacity:
            if self.kv_cache_policy == "error":
                raise RuntimeError(
                    f"VSA KV cache capacity exceeded: needed {max(end_idx, prefix_blocks * L_tiled_block)}, "
                    f"but cache has length {cache_capacity}. "
                    "Please increase max_num_frames or adjust KV cache sizing, "
                    "or switch KV cache policy to 'extend'."
                )
            elif self.kv_cache_policy == "extend":
                new_capacity = max(prefix_blocks * L_tiled_block, end_idx,
                                   cache_capacity * 2) # TODO: 这里意味着可能不是2的倍数
                if new_capacity <= cache_capacity: # TODO: 这个是不是不可能发生啊？就算发生，那也蕴含着new_capacity等于end_idx了，下面的这个赋值没有意义了吧？
                    new_capacity = end_idx
                # Note: extending capacity requires allocating new tensors and rebinding.
                # This is opt-in via kv_cache_policy="extend".
                new_k = k_cache.new_zeros((k_cache.shape[0], new_capacity,
                                           k_cache.shape[2], k_cache.shape[3]))
                new_v = v_cache.new_zeros((v_cache.shape[0], new_capacity,
                                           v_cache.shape[2], v_cache.shape[3]))
                new_k[:, :cache_capacity] = k_cache
                new_v[:, :cache_capacity] = v_cache
                kv_cache["k"] = new_k
                kv_cache["v"] = new_v
                k_cache = kv_cache["k"]
                v_cache = kv_cache["v"]
                cache_capacity = new_capacity
            else:
                raise ValueError(
                    f"Unsupported KV cache policy: {self.kv_cache_policy}")
        k_cache[:, start_idx:end_idx] = k_tiled_block
        v_cache[:, start_idx:end_idx] = v_tiled_block
        cached_blocks_t.fill_(prefix_blocks)

        # KV variable_block_sizes: store into a preallocated 1D buffer, indexed by block idx.
        # Avoid torch.cat / rebinding so state survives shallow kv_cache wrapper copies.
        variable_block_sizes_block = construct_variable_block_sizes(
            dit_seq_shape_block, num_tiles_block, device
        )
        if "variable_block_sizes" not in kv_cache or kv_cache[
                "variable_block_sizes"].numel() == 0:
            raise RuntimeError(
                "VSA causal KV cache expects a preallocated `variable_block_sizes` buffer. "
                "Please initialize KV cache via `CausalDMDDenosingStage._initialize_kv_cache()` "
                "with VIDEO_SPARSE_ATTN enabled."
            )
        vbs_buf: torch.Tensor = kv_cache["variable_block_sizes"]
        num_tiles_flat_block = int(variable_block_sizes_block.numel())
        start_tile = cur_block_idx * num_tiles_flat_block
        end_tile = start_tile + num_tiles_flat_block
        if end_tile > vbs_buf.numel():
            raise RuntimeError(
                f"VSA variable_block_sizes buffer too small: need {end_tile}, "
                f"but have {vbs_buf.numel()}. Increase KV cache sizing."
            )
        vbs_buf[start_tile:end_tile].copy_(variable_block_sizes_block)
        variable_block_sizes_all = vbs_buf[:prefix_blocks * num_tiles_flat_block]

        # Build prefix-wide sparsity/topk information using the forward context
        forward_ctx = get_forward_context()
        ctx_attn_metadata = forward_ctx.attn_metadata
        # Prefer explicit attribute access so missing metadata doesn't get silently ignored.
        # Keep backward compatibility for call sites that set attn_metadata=None.
        if ctx_attn_metadata is None:
            VSA_sparsity = 0.0
        else:
            VSA_sparsity = float(ctx_attn_metadata.VSA_sparsity)

        total_seq_length_prefix = prefix_blocks * T_block_p * H_prime * W_prime
        cur_topk = math.ceil(
            (1.0 - VSA_sparsity) *
            (total_seq_length_prefix / math.prod(VSA_TILE_SIZE)))

        # Prepare Q/gate for VSA kernel: only current block is tiled,
        # while K/V come from the full prefix cache.
        q_tiled_block = vsa_impl.tile(
            roped_query, num_tiles_block, tile_partition_indices_block,
            non_pad_index_block)

        # K/V prefix: we use all cached tiled tokens up to `prefix_blocks`.
        prefix_end_idx = prefix_blocks * L_tiled_block
        k_prefix = k_cache[:, :prefix_end_idx]
        v_prefix = v_cache[:, :prefix_end_idx]

        # Call the low-level VSA kernel (q/k can have different lengths, but q_variable_block_sizes
        # must be provided to compute q_compress correctly when tile counts differ).
        q_in = q_tiled_block.transpose(1, 2).contiguous()
        k_in = k_prefix.transpose(1, 2).contiguous()
        v_in = v_prefix.transpose(1, 2).contiguous()
        gate_in = gate_tiled_block.transpose(1, 2).contiguous()

        # Debug: optionally log q/k tiled lengths during causal generation.
        # Enable via: FASTVIDEO_DEBUG_CAUSAL_VSA_QKLEN=1
        if os.environ.get("FASTVIDEO_DEBUG_CAUSAL_VSA_QKLEN", "0") == "1":
            q_len = q_in.shape[2]
            k_len = k_in.shape[2]
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                layer_idx = kv_cache.get("_dbg_layer_idx", None) if kv_cache is not None else None
                # Only print layer 0, but print every call (no throttling).
                if layer_idx == 0:
                    dbg_start_frame = kv_cache.get("_dbg_start_frame", None) if kv_cache is not None else None
                    dbg_current_start = kv_cache.get("_dbg_current_start", None) if kv_cache is not None else None
                    dbg_diff_ts = kv_cache.get("_dbg_diff_timestep", None) if kv_cache is not None else None
                    has_ncb = (kv_cache is not None) and ("num_cached_blocks" in kv_cache)
                    has_vbs = (kv_cache is not None) and ("variable_block_sizes" in kv_cache)
                    vbs_len = (int(kv_cache["variable_block_sizes"].numel())
                               if has_vbs else None)
                    kv_id = id(kv_cache) if kv_cache is not None else None
                    k_buf_id = id(kv_cache["k"]) if kv_cache is not None and "k" in kv_cache else None
                    k_buf_ptr = (int(kv_cache["k"].data_ptr())
                                 if kv_cache is not None and "k" in kv_cache else None)
                    logger.info(
                        "causal_vsa layer=%s kv_id=%s k_buf_id=%s k_ptr=%s start_frame=%s current_start=%s diff_t=%s "
                        "q_len=%s (q_tiles=%s) k_len=%s (kv_tiles=%s) "
                        "has_num_cached_blocks=%s num_cached_blocks(before)=%s has_vbs=%s vbs_len=%s cur_topk=%s",
                        layer_idx,
                        kv_id,
                        k_buf_id,
                        k_buf_ptr,
                        dbg_start_frame,
                        dbg_current_start,
                        dbg_diff_ts,
                        q_len,
                        q_len // math.prod(VSA_TILE_SIZE),
                        k_len,
                        k_len // math.prod(VSA_TILE_SIZE),
                        has_ncb,
                        cached_blocks,
                        has_vbs,
                        vbs_len,
                        cur_topk,
                    )

        hidden_tiled_out = video_sparse_attn(
            q_in,
            k_in,
            v_in,
            variable_block_sizes=variable_block_sizes_all,
            q_variable_block_sizes=variable_block_sizes_block,
            topk=cur_topk,
            block_size=VSA_TILE_SIZE,
            compress_attn_weight=gate_in,
        ).transpose(1, 2)

        # Untile back to the current block's original order
        hidden_block = vsa_impl.untile(
            hidden_tiled_out,
            reverse_tile_partition_indices_block,
            non_pad_index_block,
        )
        return hidden_block

class CausalWanTransformerBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 ffn_dim: int,
                 num_heads: int,
                 local_attn_size: int = -1,
                 sink_size: int = 0,
                 qk_norm: str = "rms_norm_across_heads",
                 cross_attn_norm: bool = False,
                 eps: float = 1e-6,
                 added_kv_proj_dim: int | None = None,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = ""):
        super().__init__()

        # 1. Self-attention
        self.norm1 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)

        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        self.attn1 = CausalWanSelfAttention(
            dim,
            num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            eps=eps)
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        self.local_attn_size = local_attn_size
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            print("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32)

        # 2. Cross-attention
        # Only T2V for now
        self.attn2 = WanT2VCrossAttention(dim,
                                            num_heads,
                                            qk_norm=qk_norm,
                                            eps=eps)
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32)

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask,
        kv_cache: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        # hidden_states.shape: [batch_size, seq_length, inner_dim]
        # temb.shape: [batch_size, num_frames, 6, inner_dim]
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        num_frames = temb.shape[1]
        frame_seqlen = hidden_states.shape[1] // num_frames    
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        # assert orig_dtype != torch.float32
        e = self.scale_shift_table + temb
        # e.shape: [batch_size, num_frames, 6, inner_dim]
        assert e.shape == (bs, num_frames, 6, self.hidden_dim)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=2)
        # *_msa.shape: [batch_size, num_frames, 1, inner_dim]
        # assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) *
                        (1 + scale_msa) + shift_msa).flatten(1, 2)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q.forward_native(query)
        if self.norm_k is not None:
            key = self.norm_k.forward_native(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        attn_output = self.attn1(query, key, value, freqs_cis, block_mask, kv_cache, current_start, cache_start)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.tensor([0], device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale)

        # 2. Cross-attention
        attn_output = self.attn2(norm_hidden_states,
                                 context=encoder_hidden_states,
                                 context_lens=None,
                                 crossattn_cache=crossattn_cache)
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)

        return hidden_states


class CausalWanTransformerBlock_VSA(nn.Module):
    """
    Causal Wan transformer block that uses VSA-backed self-attention for the student model.

    Self-attention:
      - Uses `CausalWanSelfAttention_VSA` with a KV cache that stores tiled K/V.
      - Adds a learnable `to_gate_compress` projection, similar to non-causal VSA blocks.

    Cross-attention and MLP parts mirror `CausalWanTransformerBlock`.
    """

    def __init__(self,
                 dim: int,
                 ffn_dim: int,
                 num_heads: int,
                 local_attn_size: int = -1,
                 sink_size: int = 0,
                 qk_norm: str = "rms_norm_across_heads",
                 cross_attn_norm: bool = False,
                 eps: float = 1e-6,
                 added_kv_proj_dim: int | None = None,
                 supported_attention_backends:
                 tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = ""):
        super().__init__()

        # 1. Self-attention (VSA)
        self.norm1 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)
        self.to_gate_compress = ReplicatedLinear(dim, dim, bias=True)

        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        self.attn1 = CausalWanSelfAttention_VSA(
            dim,
            num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        self.local_attn_size = local_attn_size
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            print("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
        )

        # 2. Cross-attention (same as causal block, T2V only)
        self.attn2 = WanT2VCrossAttention(dim,
                                          num_heads,
                                          qk_norm=qk_norm,
                                          eps=eps)
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask,
        kv_cache: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        # hidden_states.shape: [batch_size, seq_length, inner_dim]
        # temb.shape: [batch_size, num_frames, 6, inner_dim]
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        num_frames = temb.shape[1]
        frame_seqlen = hidden_states.shape[1] // num_frames
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype

        e = self.scale_shift_table + temb
        # e.shape: [batch_size, num_frames, 6, inner_dim]
        assert e.shape == (bs, num_frames, 6, self.hidden_dim)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=2)

        # 1. Self-attention (VSA)
        norm_hidden_states = (self.norm1(hidden_states).unflatten(
            dim=1, sizes=(num_frames, frame_seqlen)) *
                               (1 + scale_msa) + shift_msa).flatten(1, 2)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)
        gate_compress, _ = self.to_gate_compress(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q.forward_native(query)
        if self.norm_k is not None:
            key = self.norm_k.forward_native(key)

        query = query.squeeze(1).unflatten(2,
                                           (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2,
                                           (self.num_attention_heads, -1))
        gate_compress = gate_compress.squeeze(1).unflatten(
            2, (self.num_attention_heads, -1))

        attn_output = self.attn1(
            query,
            key,
            value,
            gate_compress,
            freqs_cis,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
        )
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.tensor([0],
                                               device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale)

        # 2. Cross-attention
        attn_output = self.attn2(norm_hidden_states,
                                 context=encoder_hidden_states,
                                 context_lens=None,
                                 crossattn_cache=crossattn_cache)
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ff_output,
                                          c_gate_msa)

        return hidden_states

class CausalWanTransformer3DModel(BaseDiT):
    _fsdp_shard_conditions = WanVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoConfig()._compile_conditions
    _supported_attention_backends = WanVideoConfig(
    )._supported_attention_backends
    param_names_mapping = WanVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoConfig().lora_param_names_mapping

    def __init__(self, config: WanVideoConfig, hf_config: dict[str,
                                                               Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_dim = config.attention_head_dim
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.local_attn_size = config.local_attn_size

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(in_chans=config.in_channels,
                                          embed_dim=inner_dim,
                                          patch_size=config.patch_size,
                                          flatten=False)

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        attn_backend = envs.FASTVIDEO_ATTENTION_BACKEND
        transformer_block_cls = (
            CausalWanTransformerBlock_VSA
            if attn_backend == "VIDEO_SPARSE_ATTN" else CausalWanTransformerBlock
        )
        self.blocks = nn.ModuleList([
            transformer_block_cls(inner_dim,
                                  config.ffn_dim,
                                  config.num_attention_heads,
                                  config.local_attn_size,
                                  config.sink_size,
                                  config.qk_norm,
                                  config.cross_attn_norm,
                                  config.eps,
                                  config.added_kv_proj_dim,
                                  self._supported_attention_backends,
                                  prefix=f"{config.prefix}.blocks.{i}")
            for i in range(config.num_layers)
        ])

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(inner_dim,
                                            norm_type="layer",
                                            eps=config.eps,
                                            elementwise_affine=False,
                                            dtype=torch.float32)
        self.proj_out = nn.Linear(
            inner_dim, config.out_channels * math.prod(config.patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

        # Causal-specific
        self.block_mask = None
        self.num_frame_per_block = config.arch_config.num_frames_per_block
        assert self.num_frame_per_block <= 4 # We're adapting to VSA(4 frames per block)
        self.independent_first_frame = False

        self.__post_init__()

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
                self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
                | None = None,
                kv_cache: dict = None,
                crossattn_cache: dict = None,
                current_start: int = 0,
                cache_start: int = 0,
                start_frame: int = 0,
                **kwargs) -> torch.Tensor:
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)
        """
        if not hasattr(self, "_logged_num_frame_per_block"):
            logger.info(
                "CausalWanTransformer3DModel _forward_inference num_frame_per_block=%s",
                getattr(self, "num_frame_per_block", None),
            )
            self._logged_num_frame_per_block = True
        from fastvideo.platforms import current_platform

        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image,
                      list) and len(encoder_hidden_states_image) > 0:
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        # Debug: print per-call high-level info (frames + offsets) for causal VSA investigations.
        if os.environ.get("FASTVIDEO_DEBUG_CAUSAL_VSA_QKLEN", "0") == "1":
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                try:
                    logger.info(
                        "causal_infer_call start_frame=%s current_start=%s num_frames=%s timestep0=%s",
                        int(start_frame),
                        int(current_start),
                        int(num_frames),
                        int(timestep.flatten()[0].item()),
                    )
                except Exception:
                    pass
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Get rotary embeddings
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (post_patch_num_frames * get_sp_world_size(), post_patch_height,
             post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
            start_frame=start_frame # Assume that start_frame is 0 when kv_cache is None
        )
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos,
                     freqs_sin) if freqs_cos is not None else None

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = torch.stack(
            [torch.tensor(hidden_states[0].shape[1:], dtype=torch.long)])
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states.new_zeros(1, self.text_len - encoder_hidden_states.size(1), encoder_hidden_states.size(2))], dim=1)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
                        timestep.flatten(), encoder_hidden_states, encoder_hidden_states_image)
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size)).unflatten(dim=0, sizes=timestep.shape)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1)

        encoder_hidden_states = encoder_hidden_states.to(
            orig_dtype) if current_platform.is_mps(
            ) else encoder_hidden_states  # cast to orig_dtype for MPS

        assert encoder_hidden_states.dtype == orig_dtype

        # 4. Transformer blocks
        for block_index, block in enumerate(self.blocks):
            # Debug: annotate kv_cache dicts with per-call metadata so lower-level
            # attention modules can print meaningful context.
            if kv_cache is not None and os.environ.get(
                    "FASTVIDEO_DEBUG_CAUSAL_VSA_QKLEN", "0") == "1":
                try:
                    kvd = kv_cache[block_index]
                    kvd["_dbg_layer_idx"] = block_index
                    kvd["_dbg_start_frame"] = int(start_frame)
                    kvd["_dbg_current_start"] = int(current_start)
                    kvd["_dbg_diff_timestep"] = int(timestep.flatten()[0].item())
                    # Provide current block index for VSA KV cache slotting.
                    # This lets VSA overwrite the same slot across diffusion timesteps
                    # and only grow the prefix when `start_frame` advances.
                    try:
                        kvd["_cur_block_idx"] = int(start_frame) // int(
                            self.num_frame_per_block)
                    except Exception:
                        kvd["_cur_block_idx"] = 0
                except Exception:
                    pass
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                causal_kwargs = {
                    "kv_cache": kv_cache[block_index],
                    "current_start": current_start,
                    "cache_start": cache_start,
                    "block_mask": self.block_mask
                }
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states,
                    timestep_proj, freqs_cis,
                    **causal_kwargs)
            else:
                causal_kwargs = {
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": crossattn_cache[block_index],
                    "current_start": current_start,
                    "cache_start": cache_start,
                    "block_mask": self.block_mask
                }
                hidden_states = block(hidden_states, encoder_hidden_states,
                                        timestep_proj, freqs_cis,
                                        **causal_kwargs)

        # 5. Output norm, projection & unpatchify
        temb = temb.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2)
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2,
                                                                    dim=2)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        output = self.unpatchify(hidden_states, grid_sizes)

        return torch.stack(output)

    def _forward_train(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
                | None = None,
                start_frame: int = 0,
                **kwargs) -> torch.Tensor:

        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image,
                      list) and len(encoder_hidden_states_image) > 0:
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        if not hasattr(self, "_logged_num_frame_per_block"):
            logger.info(
                "CausalWanTransformer3DModel _forward_train num_frame_per_block=%s",
                getattr(self, "num_frame_per_block", None),
            )
            self._logged_num_frame_per_block = True
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Get rotary embeddings
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (post_patch_num_frames * get_sp_world_size(), post_patch_height,
             post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
            start_frame=start_frame
        )
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos,
                     freqs_sin) if freqs_cos is not None else None

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device=hidden_states.device,
                num_frames=num_frames,
                frame_seqlen=post_patch_height * post_patch_width,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size
            )

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = torch.stack(
            [torch.tensor(hidden_states[0].shape[1:], dtype=torch.long)])
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states.new_zeros(1, self.text_len - encoder_hidden_states.size(1), encoder_hidden_states.size(2))], dim=1)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
                        timestep.flatten(), encoder_hidden_states, encoder_hidden_states_image)
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size)).unflatten(dim=0, sizes=timestep.shape)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1)

        encoder_hidden_states = encoder_hidden_states.to(
            orig_dtype) if current_platform.is_mps(
            ) else encoder_hidden_states  # cast to orig_dtype for MPS

        assert encoder_hidden_states.dtype == orig_dtype

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states,
                    timestep_proj, freqs_cis,
                    block_mask=self.block_mask)
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states,
                                        timestep_proj, freqs_cis,
                                        block_mask=self.block_mask)

        # 5. Output norm, projection & unpatchify
        temb = temb.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2)
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2,
                                                                    dim=2)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        output = self.unpatchify(hidden_states, grid_sizes)

        return torch.stack(output)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)


    def unpatchify(self, x, grid_sizes):
        r"""


        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,


        Returns:
            Tensor:
                Reconstructed video tensors with shape [B, C_out, F, H / 8, W / 8]
        """

        c = self.out_channels
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = u.permute(6, 0, 3, 1, 4, 2, 5)
            # u = torch.einsum('fhwpqrc->cfphqwr', u.contiguous())
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out