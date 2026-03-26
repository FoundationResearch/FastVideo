# SPDX-License-Identifier: Apache-2.0
"""MatrixGame causal model plugin (student role with streaming)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.matrixgame.kv_cache import KVCache

from fastvideo.train.models.base import CausalModelBase
from fastvideo.train.models.matrixgame.matrixgame import MatrixGameModel

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import TrainingConfig


@dataclass(slots=True)
class _MatrixGameStreamingCaches:
    kv_cache: list[KVCache]
    crossattn_cache: list[dict[str, Any]]
    kv_cache_mouse: list[KVCache | None]
    kv_cache_keyboard: list[KVCache | None]
    frame_seq_length: int
    local_attn_size: int
    batch_size: int
    dtype: torch.dtype
    device: torch.device


class MatrixGameCausalModel(MatrixGameModel, CausalModelBase):
    """MatrixGame causal model with streaming KV cache support.

    Used as the student in self-forcing distillation.
    """

    _transformer_cls_name: str = "CausalMatrixGameWanModel"

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        flow_shift: float = 5.0,
        enable_gradient_checkpointing_type: str | None = None,
        transformer_override_safetensor: str | None = None,
    ) -> None:
        super().__init__(
            init_from=init_from,
            training_config=training_config,
            trainable=trainable,
            disable_custom_init_weights=disable_custom_init_weights,
            flow_shift=flow_shift,
            enable_gradient_checkpointing_type=(
                enable_gradient_checkpointing_type
            ),
            transformer_override_safetensor=(
                transformer_override_safetensor
            ),
        )
        self._streaming_caches: (
            dict[tuple[int, str], _MatrixGameStreamingCaches]
        ) = {}

    # --- CausalModelBase override: clear_caches ---
    def clear_caches(self, *, cache_tag: str = "pos") -> None:
        key = (id(self), str(cache_tag))
        cached = self._streaming_caches.pop(key, None)
        if cached is not None:
            self._reset_caches(cached)

    # --- CausalModelBase override: predict_noise_streaming ---
    def predict_noise_streaming(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        *,
        conditional: bool,
        cache_tag: str = "pos",
        store_kv: bool = False,
        cur_start_frame: int = 0,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor | None:
        cache_tag = str(cache_tag)
        cur_start_frame = int(cur_start_frame)

        batch_size = int(noisy_latents.shape[0])
        num_frames = int(noisy_latents.shape[1])
        timestep_full = self._ensure_per_frame_timestep(
            timestep=timestep,
            batch_size=batch_size,
            num_frames=num_frames,
            device=noisy_latents.device,
        )

        transformer = self._get_transformer(timestep_full)
        caches = self._get_or_init_streaming_caches(
            cache_tag=cache_tag,
            transformer=transformer,
            noisy_latents=noisy_latents,
        )

        frame_seq_length = int(caches.frame_seq_length)

        # Build input kwargs for the causal generator
        assert batch.image_embeds is not None
        assert batch.image_latents is not None
        image_embeds = batch.image_embeds.to(
            self.device, dtype=torch.bfloat16
        )

        frame_end = cur_start_frame + num_frames
        image_latents_slice = batch.image_latents[
            :, :, cur_start_frame:frame_end, :, :
        ]

        # Slice action conditioning to cover frames up to frame_end
        vae_time_compression_ratio = int(
            getattr(self.transformer, "action_config", {}).get(
                "vae_time_compression_ratio", 4
            )
        )
        action_frame_end = (
            (frame_end - 1) * vae_time_compression_ratio + 1
        )
        keyboard_cond = (
            batch.keyboard_cond[:, :action_frame_end, :]
            if batch.keyboard_cond is not None
            else None
        )
        mouse_cond = (
            batch.mouse_cond[:, :action_frame_end, :]
            if batch.mouse_cond is not None
            else None
        )

        # Build hidden states with I2V concat
        noisy_model_input = self._concat_matrixgame_hidden_states(
            noisy_latents, image_latents_slice
        )

        input_kwargs: dict[str, Any] = {
            "hidden_states": noisy_model_input,
            "encoder_hidden_states": None,
            "timestep": timestep_full,
            "encoder_hidden_states_image": image_embeds,
            "keyboard_cond": keyboard_cond,
            "mouse_cond": mouse_cond,
            "num_frame_per_block": num_frames,
            "kv_cache": caches.kv_cache,
            "kv_cache_mouse": caches.kv_cache_mouse,
            "kv_cache_keyboard": caches.kv_cache_keyboard,
            "crossattn_cache": caches.crossattn_cache,
            "current_start": cur_start_frame * frame_seq_length,
            "start_frame": cur_start_frame,
            "update_kv_cache": not store_kv,
        }

        device_type = self.device.type
        dtype = noisy_latents.dtype

        with (
            torch.autocast(device_type, dtype=dtype),
            set_forward_context(
                current_timestep=batch.timesteps,
                attn_metadata=None,
            ),
        ):
            if store_kv:
                # Store KV only, no output needed
                input_kwargs["update_kv_cache"] = True
                input_kwargs["is_cache"] = True
                with torch.no_grad():
                    _ = transformer(**input_kwargs)
                return None

            pred_noise = transformer(**input_kwargs).permute(
                0, 2, 1, 3, 4
            )
        return pred_noise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_per_frame_timestep(
        self,
        *,
        timestep: torch.Tensor,
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        if timestep.ndim == 0:
            return (
                timestep.view(1, 1)
                .expand(batch_size, num_frames)
                .to(device=device)
            )
        if timestep.ndim == 1:
            if int(timestep.shape[0]) == batch_size:
                return (
                    timestep.view(batch_size, 1)
                    .expand(batch_size, num_frames)
                    .to(device=device)
                )
            raise ValueError(
                "streaming timestep must be scalar, [B], or [B, T]"
                f"; got shape={tuple(timestep.shape)}"
            )
        if timestep.ndim == 2:
            return timestep.to(device=device)
        raise ValueError(
            "streaming timestep must be scalar, [B], or [B, T]"
            f"; got ndim={int(timestep.ndim)}"
        )

    def _get_or_init_streaming_caches(
        self,
        *,
        cache_tag: str,
        transformer: torch.nn.Module,
        noisy_latents: torch.Tensor,
    ) -> _MatrixGameStreamingCaches:
        key = (id(self), cache_tag)
        cached = self._streaming_caches.get(key)

        batch_size = int(noisy_latents.shape[0])
        dtype = noisy_latents.dtype
        device = noisy_latents.device

        frame_seq_length = self._compute_frame_seq_length(
            transformer, noisy_latents
        )
        local_attn_size = self._get_local_attn_size(transformer)

        meta = (
            frame_seq_length,
            local_attn_size,
            batch_size,
            dtype,
            device,
        )

        if cached is not None:
            cached_meta = (
                cached.frame_seq_length,
                cached.local_attn_size,
                cached.batch_size,
                cached.dtype,
                cached.device,
            )
            if cached_meta == meta:
                return cached

        (
            kv_cache,
            crossattn_cache,
            kv_cache_mouse,
            kv_cache_keyboard,
        ) = self._initialize_all_caches(
            transformer=transformer,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            frame_seq_length=frame_seq_length,
            local_attn_size=local_attn_size,
        )

        caches = _MatrixGameStreamingCaches(
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            kv_cache_mouse=kv_cache_mouse,
            kv_cache_keyboard=kv_cache_keyboard,
            frame_seq_length=frame_seq_length,
            local_attn_size=local_attn_size,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        self._streaming_caches[key] = caches
        return caches

    def _compute_frame_seq_length(
        self,
        transformer: torch.nn.Module,
        noisy_latents: torch.Tensor,
    ) -> int:
        latent_height = int(noisy_latents.shape[-2])
        latent_width = int(noisy_latents.shape[-1])

        patch_size = getattr(transformer, "patch_size", None)
        if patch_size is None:
            raise ValueError(
                "Unable to determine transformer.patch_size"
            )

        p_h = int(patch_size[-2])
        p_w = int(patch_size[-1])
        return (latent_height // p_h) * (latent_width // p_w)

    def _get_local_attn_size(
        self, transformer: torch.nn.Module
    ) -> int:
        value = getattr(transformer, "local_attn_size", -1)
        if value is None or value <= 0:
            raise ValueError(
                "MatrixGame causal model requires "
                "transformer.local_attn_size > 0"
            )
        return int(value)

    def _initialize_all_caches(
        self,
        *,
        transformer: torch.nn.Module,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        frame_seq_length: int,
        local_attn_size: int,
    ) -> tuple[
        list[KVCache],
        list[dict[str, Any]],
        list[KVCache | None],
        list[KVCache | None],
    ]:
        num_blocks = len(getattr(transformer, "blocks", []))
        if num_blocks <= 0:
            raise ValueError(
                "Unexpected transformer.blocks for causal streaming"
            )

        num_attention_heads = int(transformer.num_attention_heads)  # type: ignore[attr-defined]
        attention_head_dim = int(transformer.attention_head_dim)  # type: ignore[attr-defined]

        # 1 CLS token + 256 patch tokens = 257
        text_len = 257

        kv_cache_size = local_attn_size * frame_seq_length

        # Main self-attention KV cache
        kv_cache: list[KVCache] = []
        for _ in range(num_blocks):
            kv_cache.append(
                KVCache.zeros(
                    batch_size=batch_size,
                    cache_size=kv_cache_size,
                    num_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                    dtype=dtype,
                    device=device,
                )
            )

        # Cross-attention cache (image tokens)
        crossattn_cache: list[dict[str, Any]] = []
        for _ in range(num_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros(
                        [
                            batch_size,
                            text_len,
                            num_attention_heads,
                            attention_head_dim,
                        ],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [
                            batch_size,
                            text_len,
                            num_attention_heads,
                            attention_head_dim,
                        ],
                        dtype=dtype,
                        device=device,
                    ),
                    "is_init": False,
                }
            )

        # Action module KV caches
        action_config = (
            getattr(transformer, "action_config", {}) or {}
        )
        action_blocks = action_config.get("blocks", [])
        action_heads_num = action_config.get("heads_num", 16)
        mouse_hidden_dim = action_config.get(
            "mouse_hidden_dim", 1024
        )
        keyboard_hidden_dim = action_config.get(
            "keyboard_hidden_dim", 1024
        )

        mouse_head_dim = mouse_hidden_dim // action_heads_num
        keyboard_head_dim = keyboard_hidden_dim // action_heads_num
        action_cache_size = local_attn_size

        kv_cache_mouse: list[KVCache | None] = []
        kv_cache_keyboard: list[KVCache | None] = []
        for block_idx in range(num_blocks):
            if block_idx in action_blocks:
                kv_cache_mouse.append(
                    KVCache.zeros(
                        batch_size=batch_size * frame_seq_length,
                        cache_size=action_cache_size,
                        num_heads=action_heads_num,
                        head_dim=mouse_head_dim,
                        dtype=dtype,
                        device=device,
                    )
                )
                kv_cache_keyboard.append(
                    KVCache.zeros(
                        batch_size=batch_size,
                        cache_size=action_cache_size,
                        num_heads=action_heads_num,
                        head_dim=keyboard_head_dim,
                        dtype=dtype,
                        device=device,
                    )
                )
            else:
                kv_cache_mouse.append(None)
                kv_cache_keyboard.append(None)

        return (
            kv_cache,
            crossattn_cache,
            kv_cache_mouse,
            kv_cache_keyboard,
        )

    def _reset_caches(
        self, caches: _MatrixGameStreamingCaches
    ) -> None:
        for cache in caches.kv_cache:
            cache.length.fill_(0)
            cache.k.zero_()
            cache.v.zero_()

        for cache_dict in caches.crossattn_cache:
            cache_dict["is_init"] = False
            cache_dict["k"].zero_()
            cache_dict["v"].zero_()

        for cache in caches.kv_cache_mouse:
            if cache is not None:
                cache.length.fill_(0)
                cache.k.zero_()
                cache.v.zero_()

        for cache in caches.kv_cache_keyboard:
            if cache is not None:
                cache.length.fill_(0)
                cache.k.zero_()
                cache.v.zero_()
