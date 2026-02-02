# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# WAN Action Transformer model integrated with FastVideo

import math
from typing import Any

import torch
import torch.nn as nn

from fastvideo.attention import DistributedAttention
from fastvideo.forward_context import set_forward_context
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.layers.layernorm import (LayerNormScaleShift,
                                        RMSNorm, ScaleResidual,
                                        ScaleResidualLayerNormScaleShift)
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import (_apply_rotary_emb,
                                               get_rotary_pos_embed)
from fastvideo.layers.visual_embedding import (ModulateProjection, PatchEmbed,
                                               TimestepEmbedder,
                                               timestep_embedding)
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.wanvideo import (WanT2VCrossAttention,
                                            WanImageEmbedding)
from fastvideo.models.dits.hyworld.camera_rope import prope_qkv
from fastvideo.platforms import AttentionBackendEnum, current_platform

logger = init_logger(__name__)


class WanActionTimeTextImageEmbedding(nn.Module):
    """
    Embedding module that incorporates action signals in addition to timestep, text, and image embeddings.
    Action embeddings are combined with timestep embeddings before projection.
    """

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        text_embed_dim: int,
        image_embed_dim: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.time_freq_dim = time_freq_dim

        self.time_embedder = TimestepEmbedder(
            dim, frequency_embedding_size=time_freq_dim, act_layer="silu")
        self.time_modulation = ModulateProjection(dim,
                                                  factor=6,
                                                  act_layer="silu")
        self.text_embedder = MLP(text_embed_dim,
                                 dim,
                                 dim,
                                 bias=True,
                                 act_type="gelu_pytorch_tanh") if text_embed_dim > 0 else None

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

        # Action embedder will be initialized via add_discrete_action_parameters()
        self.action_embedder: nn.Module | None = None

    def forward(
        self,
        timestep: torch.Tensor,
        action: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ):
        temb = self.time_embedder(timestep, timestep_seq_len)

        action_emb = timestep_embedding(action.flatten(), self.time_freq_dim)
        action_embedder_dtype = next(iter(self.action_embedder.parameters())).dtype
        if (
            action_emb.dtype != action_embedder_dtype
            and action_embedder_dtype != torch.int8
        ):
            action_emb = action_emb.to(action_embedder_dtype)
        action_emb = self.action_embedder(action_emb).type_as(temb)
        temb = temb + action_emb

        timestep_proj = self.time_modulation(temb)

        if self.text_embedder is not None:
            encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        else:
            encoder_hidden_states = torch.zeros((timestep.shape[0], 0, temb.shape[-1]), 
                                                device=temb.device, dtype=temb.dtype)
        if encoder_hidden_states_image is not None:
            assert self.image_embedder is not None
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanActionSelfAttention(nn.Module):
    """
    Self-attention module with support for:
    - Standard RoPE-based attention
    - Camera PRoPE-based attention (when viewmats and Ks are provided)
    - KV caching for autoregressive generation
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 local_attn_size: int = -1,
                 sink_size: int = 0,
                 qk_norm=True,
                 eps=1e-6) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # Scaled dot product attention (using DistributedAttention for SP support)
        self.attn = DistributedAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,
                                          AttentionBackendEnum.TORCH_SDPA))

        # PRoPE output projection (initialized via add_discrete_action_parameters)
        self.to_out_prope: nn.ModuleList | None = None

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                freqs_cis: tuple[torch.Tensor, torch.Tensor],
                kv_cache: dict | None = None,
                current_start: int = 0,
                cache_start: int | None = None,
                viewmats: torch.Tensor | None = None,
                Ks: torch.Tensor | None = None,
                is_cache: bool = False,
                attention_mask: torch.Tensor | None = None):
        """
        Forward pass with camera PRoPE attention combining standard RoPE and projective positional encoding.
        
        Args:
            q, k, v: Query, key, value tensors [B, L, num_heads, head_dim]
            freqs_cis: RoPE frequency cos/sin tensors
            kv_cache: KV cache dict (may have None values for training)
            current_start: Current position for KV cache
            cache_start: Cache start position
            viewmats: Camera view matrices for PRoPE [B, cameras, 4, 4]
            Ks: Camera intrinsics for PRoPE [B, cameras, 3, 3]
            is_cache: Whether to store to KV cache (for inference)
            attention_mask: Attention mask [B, L] (1 = attend, 0 = mask)
        """
        if cache_start is None:
            cache_start = current_start

        # Apply RoPE manually
        cos, sin = freqs_cis
        query_rope = _apply_rotary_emb(q, cos, sin, is_neox_style=False).type_as(v)
        key_rope = _apply_rotary_emb(k, cos, sin, is_neox_style=False).type_as(v)
        value_rope = v

        # Get PRoPE transformed q, k, v
        query_prope, key_prope, value_prope, apply_fn_o = prope_qkv(
            q.transpose(1, 2),  # [B, num_heads, L, head_dim]
            k.transpose(1, 2),
            v.transpose(1, 2),
            viewmats=viewmats,
            Ks=Ks,
            patches_x=40,  # hardcoded for now
            patches_y=22,
        )
        # PRoPE returns [B, num_heads, L, head_dim], convert to [B, L, num_heads, head_dim]
        query_prope = query_prope.transpose(1, 2)
        key_prope = key_prope.transpose(1, 2)
        value_prope = value_prope.transpose(1, 2)

        # KV cache handling
        if kv_cache is not None:
            cache_key = kv_cache.get("k", None)
            cache_value = kv_cache.get("v", None)

            if cache_value is not None and not is_cache:
                cache_key_rope, cache_key_prope = cache_key.chunk(2, dim=-1)
                cache_value_rope, cache_value_prope = cache_value.chunk(2, dim=-1)

                key_rope = torch.cat([cache_key_rope, key_rope], dim=1)
                value_rope = torch.cat([cache_value_rope, value_rope], dim=1)
                key_prope = torch.cat([cache_key_prope, key_prope], dim=1)
                value_prope = torch.cat([cache_value_prope, value_prope], dim=1)

            if is_cache:
                # Store to cache (update input dict directly)
                kv_cache["k"] = torch.cat([key_rope, key_prope], dim=-1)
                kv_cache["v"] = torch.cat([value_rope, value_prope], dim=-1)

        # Concatenate rope and prope paths (matching original)
        query_all = torch.cat([query_rope, query_prope], dim=0)
        key_all = torch.cat([key_rope, key_prope], dim=0)
        value_all = torch.cat([value_rope, value_prope], dim=0)

        # Check if Q and KV have different sequence lengths (KV cache mode)
        # In this case, use LocalAttention (supports different Q/KV lengths)
        if query_all.shape[1] != key_all.shape[1]:
            # KV cache mode: Q has new tokens only, KV has cached + new tokens
            # Use LocalAttention which supports different Q/KV lengths
            # LocalAttention will use the appropriate backend (SageAttn, FlashAttn, etc.)
            if not hasattr(self, '_kv_cache_attn'):
                from fastvideo.attention import LocalAttention
                self._kv_cache_attn = LocalAttention(
                    num_heads=self.num_heads,
                    head_size=self.head_dim,
                    causal=False,
                    supported_attention_backends=(AttentionBackendEnum.SAGE_ATTN,
                                                  AttentionBackendEnum.FLASH_ATTN,
                                                  AttentionBackendEnum.TORCH_SDPA)
                )
            hidden_states_all = self._kv_cache_attn(query_all, key_all, value_all)
        else:
            # Same sequence length: use DistributedAttention (supports SP)
            # Create default attention mask if not provided
            if attention_mask is None:
                batch_size, seq_len = q.shape[0], q.shape[1]
                attention_mask = torch.ones(batch_size, seq_len, device=q.device, dtype=q.dtype)
        
            if q.dtype == torch.float32:
                from fastvideo.attention.backends.sdpa import SDPAMetadataBuilder
                attn_metadata_builder = SDPAMetadataBuilder
            else:
                from fastvideo.attention.backends.flash_attn import FlashAttnMetadataBuilder
                attn_metadata_builder = FlashAttnMetadataBuilder
            attn_metadata = attn_metadata_builder().build(
                current_timestep=0,
                attn_mask=attention_mask,
            )
            with set_forward_context(current_timestep=0, attn_metadata=attn_metadata):
                hidden_states_all, _ = self.attn(query_all, key_all, value_all, attention_mask=attention_mask)

        hidden_states_rope, hidden_states_prope = hidden_states_all.chunk(2, dim=0)
        hidden_states_prope = apply_fn_o(hidden_states_prope.transpose(1, 2)).transpose(1, 2)

        return hidden_states_rope, hidden_states_prope


class WanActionTransformerBlock(nn.Module):
    """
    Transformer block for WAN Action model with support for:
    - Self-attention with RoPE and camera PRoPE
    - Cross-attention with text/image context
    - Feed-forward network with AdaLN modulation
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
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = ""):
        super().__init__()

        # 1. Self-attention
        self.norm1 = LayerNormScaleShift(dim, norm_type="layer", eps=eps,
                                         elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)
        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        
        self.attn1 = WanActionSelfAttention(
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
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            raise ValueError(f"QK Norm type {qk_norm} not supported")

        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            compute_dtype=torch.float32)

        # 2. Cross-attention (T2V only for now)
        self.attn2 = WanT2VCrossAttention(dim,
                                          num_heads,
                                          qk_norm=qk_norm,
                                          eps=eps)
        # norm3 for FFN input 
        self.norm3 = LayerNormScaleShift(dim, norm_type="layer", eps=eps,
                                         elementwise_affine=False)

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # PRoPE output projection (initialized via add_discrete_action_parameters on the model)
        self.to_out_prope: nn.ModuleList | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        kv_cache: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        is_cache: bool = False,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)

        num_frames = temb.shape[1]
        frame_seqlen = hidden_states.shape[1] // num_frames
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype

        # Cast temb to float32 for scale/shift computation
        e = self.scale_shift_table + temb.float()
        assert e.shape == (bs, num_frames, 6, self.hidden_dim)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(6, dim=2)

        # 1. Self-attention
        norm_hidden_states = self.norm1(
            hidden_states.float(), shift_msa, scale_msa
        ).type_as(hidden_states)
        
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

        # Self-attention with optional camera PRoPE
        attn_output_rope, attn_output_prope = self.attn1(
            query, key, value, freqs_cis,
            kv_cache, current_start, cache_start, viewmats, Ks,
            is_cache=is_cache
        )
        # Combine rope and prope outputs
        attn_output_rope = attn_output_rope.flatten(2)
        attn_output_rope, _ = self.to_out(attn_output_rope)
        attn_output_prope = attn_output_prope.flatten(2)
        attn_output_prope = self.to_out_prope[0](attn_output_prope)
        attn_output = attn_output_rope.squeeze(1) + attn_output_prope.squeeze(1)

        # Self-attention residual + norm in float32
        null_shift = null_scale = torch.zeros(1, device=hidden_states.device, dtype=torch.float32)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states.float(), attn_output.float(), gate_msa, null_shift, null_scale)
        hidden_states = hidden_states.type_as(attn_output)
        norm_hidden_states = norm_hidden_states.type_as(attn_output)

        # 2. Cross-attention
        attn_output = self.attn2(norm_hidden_states.to(orig_dtype),
                                 context=encoder_hidden_states,
                                 context_lens=None,
                                 crossattn_cache=crossattn_cache)
        # Cross-attention residual in bfloat16
        hidden_states = hidden_states + attn_output
        
        # norm3 for FFN input in float32
        norm_hidden_states = self.norm3(
            hidden_states.float(), c_shift_msa, c_scale_msa
        ).type_as(hidden_states)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states.to(orig_dtype))
        hidden_states = self.mlp_residual(hidden_states.float(), ff_output.float(), c_gate_msa)
        hidden_states = hidden_states.to(orig_dtype)  # Cast back to original dtype

        return hidden_states


class WanActionTransformer3DModel(BaseDiT):
    """
    WAN Action Transformer 3D Model for video generation with action conditioning.
    
    Extends the base WAN video model with:
    - Action embedding support for controllable generation
    - camera PRoPE attention for 3D-aware generation
    - KV caching for autoregressive inference
    """
    _fsdp_shard_conditions = WanVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoConfig()._compile_conditions
    _supported_attention_backends = WanVideoConfig()._supported_attention_backends
    param_names_mapping = WanVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoConfig().lora_param_names_mapping

    def __init__(self, config: WanVideoConfig, hf_config: dict[str, Any]) -> None:
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
        self.inner_dim = inner_dim

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(in_chans=config.in_channels,
                                          embed_dim=inner_dim,
                                          patch_size=config.patch_size,
                                          flatten=False)

        # 2. Condition embeddings (with action support)
        self.condition_embedder = WanActionTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList([
            WanActionTransformerBlock(
                inner_dim,
                config.ffn_dim,
                config.num_attention_heads,
                config.local_attn_size,
                config.sink_size,
                config.qk_norm,
                config.cross_attn_norm,
                config.eps,
                config.added_kv_proj_dim,
                supported_attention_backends=self._supported_attention_backends,
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
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

        # Causal-specific
        self.num_frame_per_block = config.arch_config.num_frames_per_block
        assert self.num_frame_per_block <= 3

        self.__post_init__()

    def add_discrete_action_parameters(self):
        """
        Initialize the discrete action embedder and PRoPE output projections.
        Call this method after loading base model weights to add action support.
        """
        # Action embedder
        self.condition_embedder.action_embedder = MLP(
            self.condition_embedder.time_freq_dim,
            self.inner_dim,
            self.inner_dim,
            bias=True,
            act_type="silu"
        )
        # Initialize with zeros for residual-like behavior
        nn.init.zeros_(self.condition_embedder.action_embedder.fc_out.weight)
        if self.condition_embedder.action_embedder.fc_out.bias is not None:
            nn.init.zeros_(self.condition_embedder.action_embedder.fc_out.bias)

        # PRoPE output projections for each block
        for block in self.blocks:
            block.to_out_prope = nn.ModuleList([
                nn.Linear(self.inner_dim, self.inner_dim, bias=True),
            ])
            nn.init.zeros_(block.to_out_prope[0].weight)
            if block.to_out_prope[0].bias is not None:
                nn.init.zeros_(block.to_out_prope[0].bias)

            # Also set the PRoPE projection in the attention module
            block.attn1.to_out_prope = block.to_out_prope

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        action: torch.Tensor | None = None,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int = 0,
        cache_start: int = 0,
        start_frame: int = 0,
        is_cache: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for both training and inference with KV caching.
        
        Args:
            hidden_states: Video latents [B, C, T, H, W]
            encoder_hidden_states: Text embeddings [B, L, D]
            timestep: Timestep tensor
            encoder_hidden_states_image: Optional image embeddings
            action: Action tensor [B, T] for per-frame conditioning
            viewmats: Camera view matrices for PRoPE [B, T, 4, 4]
            Ks: Camera intrinsics for PRoPE [B, T, 3, 3]
            kv_cache: KV cache for autoregressive inference (list of dicts per layer)
            crossattn_cache: Cross-attention cache for inference
            current_start: Current position for KV cache
            cache_start: Cache start position
            start_frame: RoPE offset for new frames in autoregressive mode
            is_cache: If True, populate KV cache and return early (cache-only mode)
        """
        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image, list) and len(encoder_hidden_states_image) > 0:
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Get rotary embeddings
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (post_patch_num_frames * get_sp_world_size(), post_patch_height, post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
            start_frame=start_frame
        )
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = torch.stack([torch.tensor(hidden_states[0].shape[1:], dtype=torch.long)])
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        encoder_hidden_states = torch.cat([
            encoder_hidden_states,
            encoder_hidden_states.new_zeros(1, self.text_len - encoder_hidden_states.size(1), encoder_hidden_states.size(2))
        ], dim=1)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep.flatten(), action, encoder_hidden_states, encoder_hidden_states_image=encoder_hidden_states_image)
        
        # Reshape timestep_proj: [T, 6*dim] -> [B, T, 6, dim]
        # For training: batch_size=1, T=num_frames (diffusion forcing)
        # For inference: batch_size can vary
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size))
        if timestep_proj.shape[0] == post_patch_num_frames and batch_size == 1:
            # Training mode: timestep_proj is [T, 6, dim], add batch dim -> [1, T, 6, dim]
            timestep_proj = timestep_proj.unsqueeze(0)
        else:
            # Inference mode: reshape based on timestep shape
            timestep_proj = timestep_proj.unflatten(dim=0, sizes=timestep.shape)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        encoder_hidden_states = encoder_hidden_states.to(orig_dtype) if current_platform.is_mps() else encoder_hidden_states

        # Transformer blocks
        for block_idx, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, freqs_cis,
                    kv_cache[block_idx] if kv_cache else None,
                    crossattn_cache[block_idx] if crossattn_cache else None,
                    current_start, cache_start,
                    viewmats, Ks, is_cache)
            else:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, freqs_cis,
                    kv_cache[block_idx] if kv_cache else None,
                    crossattn_cache[block_idx] if crossattn_cache else None,
                    current_start, cache_start,
                    viewmats, Ks, is_cache)

        # If cache-only mode, return early
        if is_cache:
            return kv_cache

        # Output norm, projection & unpatchify
        # Reshape temb to match timestep_proj shape: [T, dim] -> [B, T, 1, dim]
        if temb.shape[0] == post_patch_num_frames and batch_size == 1:
            # Training mode: temb is [T, dim] -> [1, T, 1, dim]
            temb = temb.unsqueeze(0).unsqueeze(2)
        else:
            # Inference mode: reshape based on timestep shape
            temb = temb.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2)
        
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2, dim=2)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        output = self.unpatchify(hidden_states, grid_sizes)

        return torch.stack(output)

    def unpatchify(self, x, grid_sizes):
        """
        Reconstruct video tensors from patchified features.
        
        Args:
            x: List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes: Original spatial-temporal grid dimensions before patching

        Returns:
            List of reconstructed video tensors with shape [C_out, F, H, W]
        """
        c = self.out_channels
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = u.permute(6, 0, 3, 1, 4, 2, 5)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
