# SPDX-License-Identifier: Apache-2.0
"""Wan2.1-Fun-InP image-to-video training model plugin (per-role instance).

Adds first-frame image conditioning on top of :class:`WanModel`:

* the 16-channel noisy video latent is concatenated with a 4-channel temporal
  mask and the 16-channel first-frame latent (=> 36 ``in_channels``), and
* the CLIP image features are forwarded as ``encoder_hidden_states_image``.

Unlike the world-model variants (Matrix-Game), text conditioning and
negative-prompt CFG are preserved, so the same class works for both the
fine-tuning and DMD distribution-matching methods.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

import torch

from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v
from fastvideo.distributed import (
    get_sp_group,
    get_world_group,
)
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.training_utils import normalize_dit_input

from fastvideo.train.models.wan.wan import WanModel
from fastvideo.train.utils.dataloader import (
    build_parquet_t2v_train_dataloader, )
from fastvideo.train.utils.moduleloader import (
    load_module_from_path, )


class WanI2VModel(WanModel):
    """Wan2.1-Fun-InP (image-to-video) per-role model for the new trainer."""

    _transformer_cls_name: str = "WanTransformer3DModel"

    def init_preprocessors(self, training_config: Any) -> None:
        self.vae = load_module_from_path(
            model_path=str(training_config.model_path),
            module_type="vae",
            training_config=training_config,
        )
        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self._init_timestep_mechanics()

        # The generic t2v parquet dataloader works for any schema; the i2v
        # schema simply carries the extra clip_feature / first_frame_latent /
        # pil_image columns.
        text_len = (
            training_config.pipeline_config.text_encoder_configs[  # type: ignore[union-attr]
                0].arch_config.text_len)
        self.dataloader = build_parquet_t2v_train_dataloader(
            training_config.data,
            text_len=int(text_len),
            parquet_schema=pyarrow_schema_i2v,
        )
        self.start_step = 0

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        self.ensure_negative_conditioning()
        assert self.training_config is not None
        tc = self.training_config
        dtype = self._get_training_dtype()
        device = self.device

        training_batch = TrainingBatch()
        encoder_hidden_states = raw_batch["text_embedding"]
        encoder_attention_mask = raw_batch["text_attention_mask"]
        infos = raw_batch.get("info_list")

        if latents_source == "zeros":
            batch_size = encoder_hidden_states.shape[0]
            vae_config = (
                tc.pipeline_config.vae_config.arch_config  # type: ignore[union-attr]
            )
            num_channels = vae_config.z_dim
            spatial_compression_ratio = (vae_config.spatial_compression_ratio)
            latent_height = (tc.data.num_height // spatial_compression_ratio)
            latent_width = (tc.data.num_width // spatial_compression_ratio)
            latents = torch.zeros(
                batch_size,
                num_channels,
                tc.data.num_latent_t,
                latent_height,
                latent_width,
                device=device,
                dtype=dtype,
            )
        elif latents_source == "data":
            if "vae_latent" not in raw_batch:
                raise ValueError("vae_latent not found in batch "
                                 "and latents_source='data'")
            latents = raw_batch["vae_latent"][:, :, :tc.data.num_latent_t]
            latents = latents.to(device, dtype=dtype)
        else:
            raise ValueError(f"Unknown latents_source: {latents_source!r}")

        # i2v conditioning fields (extra over the t2v schema).
        clip_feature = raw_batch["clip_feature"].to(device, dtype=dtype)
        first_frame_latent = raw_batch["first_frame_latent"]
        first_frame_latent = first_frame_latent[:, :, :tc.data.num_latent_t]
        first_frame_latent = first_frame_latent.to(device, dtype=dtype)
        pil_image = raw_batch.get("pil_image")
        if pil_image is not None:
            pil_image = pil_image.to(device=device)

        training_batch.latents = latents
        training_batch.encoder_hidden_states = (encoder_hidden_states.to(device, dtype=dtype))
        training_batch.encoder_attention_mask = (encoder_attention_mask.to(device, dtype=dtype))
        training_batch.image_embeds = clip_feature
        training_batch.image_latents = first_frame_latent
        training_batch.preprocessed_image = pil_image
        training_batch.infos = infos

        training_batch.latents = normalize_dit_input("wan", training_batch.latents, self.vae)
        training_batch = self._prepare_dit_inputs(training_batch, generator)
        training_batch = self._build_attention_metadata(training_batch)

        # See WanModel.prepare_batch for why this is a shallow copy.
        training_batch.attn_metadata_vsa = copy.copy(training_batch.attn_metadata)
        if training_batch.attn_metadata is not None:
            training_batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore[attr-defined]

        return training_batch

    def _prepare_dit_inputs(
        self,
        training_batch: TrainingBatch,
        generator: torch.Generator,
    ) -> TrainingBatch:
        # WanModel builds the 16-channel noisy latent + the (negative-)text
        # conditional/unconditional dicts.
        training_batch = super()._prepare_dit_inputs(training_batch, generator)

        image_latents = training_batch.image_latents
        image_embeds = training_batch.image_embeds
        if image_latents is None or image_embeds is None:
            raise RuntimeError("WanI2VModel requires image_latents and image_embeds")

        # mask(4) + first_frame_latent(16) = 20 channels, then concat with the
        # 16-channel noisy latent => 36 in_channels expected by Wan-InP.
        cond_latents = self._build_inp_cond_concat(image_latents)
        training_batch.image_latents = cond_latents
        training_batch.noisy_model_input = torch.cat(
            [training_batch.noisy_model_input, cond_latents],
            dim=1,
        )

        # Keep the text conditioning WanModel set; add the image conditioning to
        # both the conditional and the (negative-prompt) unconditional dicts so
        # CFG / DMD's unconditional pass still sees the first frame.
        if training_batch.conditional_dict is not None:
            training_batch.conditional_dict["encoder_hidden_states_image"] = image_embeds
            training_batch.conditional_dict["image_latents"] = cond_latents
        if training_batch.unconditional_dict is not None:
            training_batch.unconditional_dict["encoder_hidden_states_image"] = image_embeds
            training_batch.unconditional_dict["image_latents"] = cond_latents

        return training_batch

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if text_dict is None:
            raise ValueError("text_dict cannot be None for Wan-InP")
        hidden_states = noise_input.permute(0, 2, 1, 3, 4)
        # A freshly rolled-out latent (e.g. the DMD student output) is
        # 16-channel; re-attach the image conditioning. A pre-concatenated
        # 36-channel input (fine-tuning path) is passed through unchanged.
        if hidden_states.shape[1] == 16:
            cond_latents = text_dict.get("image_latents")
            if cond_latents is None:
                raise RuntimeError("WanI2VModel requires image_latents in the "
                                   "text dict when noise_input has 16 channels")
            num_t = hidden_states.shape[2]
            cond_latents = cond_latents[:, :, :num_t]
            hidden_states = torch.cat([hidden_states, cond_latents], dim=1)
        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": text_dict["encoder_hidden_states"],
            "encoder_attention_mask": text_dict["encoder_attention_mask"],
            "timestep": timestep,
            "encoder_hidden_states_image": (text_dict["encoder_hidden_states_image"]),
            "return_dict": False,
        }

    def _get_uncond_text_dict(
        self,
        batch: TrainingBatch,
        *,
        cfg_uncond: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # Use the negative-prompt dict built by WanModel (augmented with the
        # image conditioning in ``_prepare_dit_inputs``). We do not support the
        # text=zero/keep cfg_uncond policies here because dropping the image
        # conditioning would break the 36-channel concat.
        del cfg_uncond
        uncond = getattr(batch, "unconditional_dict", None)
        if uncond is not None:
            return uncond
        if batch.conditional_dict is None:
            raise RuntimeError("Missing conditional/unconditional dict in "
                               "TrainingBatch")
        return batch.conditional_dict

    def _build_inp_cond_concat(self, image_latents: torch.Tensor) -> torch.Tensor:
        """Build the 20-channel [mask(4), first_frame_latent(16)] conditioning.

        Mirrors the Wan-InP conditioning used by the legacy i2v pipelines and
        by ``MatrixGame2Model``: a binary temporal mask (first frame = 1, rest =
        0) folded into 4 channels via the VAE temporal compression, concatenated
        with the first-frame latent.
        """
        if image_latents.ndim != 5:
            raise ValueError("first_frame_latent must have shape "
                             f"[B, C, T, H, W], got {tuple(image_latents.shape)}")
        if image_latents.shape[1] == 20:
            return image_latents
        if image_latents.shape[1] != 16:
            raise ValueError("WanI2VModel expects first_frame_latent with 16 or "
                             f"20 channels, got {image_latents.shape[1]}")

        temporal_compression_ratio = self._temporal_compression_ratio()
        batch_size, _, num_latent_t, latent_height, latent_width = (image_latents.shape)
        num_frames = (num_latent_t - 1) * temporal_compression_ratio + 1

        mask_lat_size = torch.ones(
            batch_size,
            1,
            num_frames,
            latent_height,
            latent_width,
            device=image_latents.device,
            dtype=image_latents.dtype,
        )
        mask_lat_size[:, :, 1:] = 0
        first_frame_mask = mask_lat_size[:, :, :1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask,
            dim=2,
            repeats=temporal_compression_ratio,
        )
        mask_lat_size = torch.cat(
            [first_frame_mask, mask_lat_size[:, :, 1:]],
            dim=2,
        )
        mask_lat_size = mask_lat_size.view(
            batch_size,
            -1,
            temporal_compression_ratio,
            latent_height,
            latent_width,
        ).transpose(1, 2)
        return torch.cat([mask_lat_size, image_latents], dim=1)

    def _temporal_compression_ratio(self) -> int:
        assert self.training_config is not None
        return int(self.training_config.pipeline_config.vae_config.arch_config.
                   temporal_compression_ratio  # type: ignore[union-attr]
                   )
