# SPDX-License-Identifier: Apache-2.0
"""MatrixGame non-causal model plugin (teacher/critic roles)."""

from __future__ import annotations

import copy
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.distributed import (
    get_sp_group,
    get_world_group,
)
from fastvideo.forward_context import set_forward_context
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.activation_checkpoint import (
    apply_activation_checkpointing,
)
from fastvideo.training.training_utils import (
    compute_density_for_timestep_sampling,
    get_sigmas,
    normalize_dit_input,
    shift_timestep,
)

from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.module_state import apply_trainable
from fastvideo.train.utils.moduleloader import load_module_from_path

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import TrainingConfig


def _remap_keyboard_cond(
    keyboard_cond: torch.Tensor | None,
) -> torch.Tensor | None:
    if keyboard_cond is None or keyboard_cond.shape[-1] != 6:
        return keyboard_cond
    mapped = keyboard_cond.new_zeros(*keyboard_cond.shape[:-1], 23)
    mapped[..., 11:15] = keyboard_cond[..., :4]
    return mapped


def _remap_mouse_cond(
    mouse_cond: torch.Tensor | None,
) -> torch.Tensor | None:
    if mouse_cond is None or mouse_cond.shape[-1] != 2:
        return mouse_cond
    return torch.stack(
        (-mouse_cond[..., 1], mouse_cond[..., 0]), dim=-1
    )


class MatrixGameModel(ModelBase):
    """MatrixGame non-causal model (for teacher/critic roles).

    Uses CLIP image embeddings instead of text conditioning.
    Concatenates [noise, mask(4ch), image_latent(16ch)] as model input.
    """

    _transformer_cls_name: str = "MatrixGameWanModel"

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
        self._init_from = str(init_from)
        self._trainable = bool(trainable)

        self.transformer = self._load_transformer(
            init_from=self._init_from,
            trainable=self._trainable,
            disable_custom_init_weights=disable_custom_init_weights,
            enable_gradient_checkpointing_type=(
                enable_gradient_checkpointing_type
            ),
            training_config=training_config,
            transformer_override_safetensor=(
                transformer_override_safetensor
            ),
        )

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            shift=float(flow_shift)
        )

        # Filled by init_preprocessors (student only).
        self.vae: Any = None
        self.training_config: TrainingConfig = training_config
        self.dataloader: Any = None
        self.start_step: int = 0

        self.world_group: Any = None
        self.sp_group: Any = None

        # Timestep mechanics.
        self.timestep_shift: float = float(flow_shift)
        self.num_train_timestep: int = int(
            self.noise_scheduler.num_train_timesteps
        )
        self.min_timestep: int = 0
        self.max_timestep: int = self.num_train_timestep

    def _load_transformer(
        self,
        *,
        init_from: str,
        trainable: bool,
        disable_custom_init_weights: bool,
        enable_gradient_checkpointing_type: str | None,
        training_config: TrainingConfig,
        transformer_override_safetensor: str | None = None,
    ) -> torch.nn.Module:
        transformer = load_module_from_path(
            model_path=init_from,
            module_type="transformer",
            training_config=training_config,
            disable_custom_init_weights=disable_custom_init_weights,
            override_transformer_cls_name=self._transformer_cls_name,
            transformer_override_safetensor=(
                transformer_override_safetensor
            ),
        )
        transformer = apply_trainable(transformer, trainable=trainable)
        ckpt_type = enable_gradient_checkpointing_type or getattr(
            getattr(training_config, "model", None),
            "enable_gradient_checkpointing_type",
            None,
        )
        if trainable and ckpt_type:
            transformer = apply_activation_checkpointing(
                transformer,
                checkpointing_type=ckpt_type,
            )
        return transformer

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_preprocessors(
        self, training_config: TrainingConfig
    ) -> None:
        self.vae = load_module_from_path(
            model_path=str(training_config.model_path),
            module_type="vae",
            training_config=training_config,
        )

        self.world_group = get_world_group()
        self.sp_group = get_sp_group()

        self._init_timestep_mechanics()

        from fastvideo.dataset.dataloader.schema import (
            pyarrow_schema_matrixgame,
        )
        from fastvideo.train.utils.dataloader import (
            build_parquet_t2v_train_dataloader,
        )

        # MatrixGame uses 257 image tokens (1 CLS + 256 patches)
        # but the dataloader text_len is used for padding; set to 0
        # since MatrixGame has no text.
        self.dataloader = build_parquet_t2v_train_dataloader(
            training_config.data,
            text_len=0,
            parquet_schema=pyarrow_schema_matrixgame,
        )
        self.start_step = 0

    @property
    def num_train_timesteps(self) -> int:
        return int(self.num_train_timestep)

    def shift_and_clamp_timestep(
        self, timestep: torch.Tensor
    ) -> torch.Tensor:
        timestep = shift_timestep(
            timestep,
            self.timestep_shift,
            self.num_train_timestep,
        )
        return timestep.clamp(self.min_timestep, self.max_timestep)

    def on_train_start(self) -> None:
        pass  # No negative prompt for MatrixGame

    # ------------------------------------------------------------------
    # Runtime primitives
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        assert self.training_config is not None
        tc = self.training_config

        dtype = torch.bfloat16
        device = self.device

        training_batch = TrainingBatch()

        # VAE latents
        latents = raw_batch["vae_latent"]
        latents = latents[:, :, : tc.data.num_latent_t]
        latents = latents.to(device, dtype=dtype)
        training_batch.latents = latents

        # CLIP features (image embeddings)
        clip_features = raw_batch["clip_feature"]
        training_batch.image_embeds = clip_features.to(device, dtype=dtype)

        # First-frame latent (16ch)
        image_latents = raw_batch["first_frame_latent"]
        image_latents = image_latents[
            :, :, : tc.data.num_latent_t
        ]
        training_batch.image_latents = image_latents.to(
            device, dtype=dtype
        )

        # No text for MatrixGame
        training_batch.encoder_hidden_states = None
        training_batch.encoder_attention_mask = None

        # Action conditioning
        if (
            "mouse_cond" in raw_batch
            and raw_batch["mouse_cond"].numel() > 0
        ):
            training_batch.mouse_cond = _remap_mouse_cond(
                raw_batch["mouse_cond"]
            ).to(device, dtype=dtype)
        else:
            training_batch.mouse_cond = None

        if (
            "keyboard_cond" in raw_batch
            and raw_batch["keyboard_cond"].numel() > 0
        ):
            training_batch.keyboard_cond = _remap_keyboard_cond(
                raw_batch["keyboard_cond"]
            ).to(device, dtype=dtype)
        else:
            training_batch.keyboard_cond = None

        # PIL image for validation
        if "pil_image" in raw_batch:
            training_batch.preprocessed_image = raw_batch[
                "pil_image"
            ].to(device)

        infos = raw_batch.get("info_list")
        training_batch.infos = infos

        # Normalize and prepare DIT inputs
        training_batch.latents = normalize_dit_input(
            "wan", training_batch.latents, self.vae
        )
        training_batch = self._prepare_dit_inputs(
            training_batch, generator
        )
        training_batch = self._prepare_i2v_concat(training_batch)

        return training_batch

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        b, t = clean_latents.shape[:2]
        noisy = self.noise_scheduler.add_noise(
            clean_latents.flatten(0, 1),
            noise.flatten(0, 1),
            timestep,
        ).unflatten(0, (b, t))
        return noisy

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        device_type = self.device.type
        dtype = noisy_latents.dtype

        if attn_kind == "dense":
            attn_metadata = batch.attn_metadata
        elif attn_kind == "vsa":
            attn_metadata = batch.attn_metadata_vsa
        else:
            raise ValueError(f"Unknown attn_kind: {attn_kind!r}")

        with (
            torch.autocast(device_type, dtype=dtype),
            set_forward_context(
                current_timestep=batch.timesteps,
                attn_metadata=attn_metadata,
            ),
        ):
            input_kwargs = self._build_distill_input_kwargs(
                noisy_latents, timestep, batch
            )
            transformer = self._get_transformer(timestep)
            pred_noise = transformer(**input_kwargs).permute(
                0, 2, 1, 3, 4
            )
        return pred_noise

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        timesteps, attn_metadata = ctx
        with set_forward_context(
            current_timestep=timesteps,
            attn_metadata=attn_metadata,
        ):
            (loss / max(1, int(grad_accum_rounds))).backward()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_timestep_mechanics(self) -> None:
        assert self.training_config is not None
        tc = self.training_config
        self.timestep_shift = float(
            tc.pipeline_config.flow_shift  # type: ignore[union-attr]
        )
        self.num_train_timestep = int(
            self.noise_scheduler.num_train_timesteps
        )
        self.min_timestep = 0
        self.max_timestep = self.num_train_timestep

    def _sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        assert self.training_config is not None
        tc = self.training_config

        u = compute_density_for_timestep_sampling(
            weighting_scheme=tc.model.weighting_scheme,
            batch_size=batch_size,
            generator=generator,
            device=device,
            logit_mean=tc.model.logit_mean,
            logit_std=tc.model.logit_std,
            mode_scale=tc.model.mode_scale,
        )
        indices = (
            u * self.noise_scheduler.config.num_train_timesteps
        ).long()
        return self.noise_scheduler.timesteps[indices.cpu()].to(
            device=device
        )

    def _prepare_dit_inputs(
        self,
        training_batch: TrainingBatch,
        generator: torch.Generator,
    ) -> TrainingBatch:
        assert self.training_config is not None
        tc = self.training_config
        latents = training_batch.latents
        assert isinstance(latents, torch.Tensor)
        batch_size = latents.shape[0]

        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        timesteps = self._sample_timesteps(
            batch_size,
            latents.device,
            generator,
        )
        if int(tc.distributed.sp_size or 1) > 1:
            self.sp_group.broadcast(timesteps, src=0)

        sigmas = get_sigmas(
            self.noise_scheduler,
            latents.device,
            timesteps,
            n_dim=latents.ndim,
            dtype=latents.dtype,
        )
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

        training_batch.noisy_model_input = noisy_model_input
        training_batch.timesteps = timesteps
        training_batch.sigmas = sigmas
        training_batch.noise = noise
        training_batch.raw_latent_shape = latents.shape

        # MatrixGame uses image embeddings, not text
        training_batch.conditional_dict = {
            "encoder_hidden_states_image": (
                training_batch.image_embeds
            ),
        }

        # Permute latents to [B, C, T, H, W]
        training_batch.latents = training_batch.latents.permute(
            0, 2, 1, 3, 4
        )

        # Build attention metadata (None for MatrixGame —
        # it uses its own attention)
        training_batch.attn_metadata = None
        training_batch.attn_metadata_vsa = None

        return training_batch

    def _prepare_i2v_concat(
        self,
        training_batch: TrainingBatch,
    ) -> TrainingBatch:
        """Build mask + image_latent concatenation for I2V input."""
        assert self.training_config is not None
        tc = self.training_config
        image_latents = training_batch.image_latents
        assert isinstance(image_latents, torch.Tensor)

        vae_config = (
            tc.pipeline_config.vae_config.arch_config  # type: ignore[union-attr]
        )
        temporal_compression_ratio = int(
            vae_config.temporal_compression_ratio
        )
        num_latent_t = image_latents.shape[2]
        num_frames = (
            (num_latent_t - 1) * temporal_compression_ratio + 1
        )
        batch_size = image_latents.shape[0]
        latent_height = image_latents.shape[3]
        latent_width = image_latents.shape[4]

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
            [first_frame_mask, mask_lat_size[:, :, 1:]], dim=2
        )
        mask_lat_size = mask_lat_size.view(
            batch_size,
            -1,
            temporal_compression_ratio,
            latent_height,
            latent_width,
        )
        mask_lat_size = mask_lat_size.transpose(1, 2).to(
            image_latents.device, dtype=image_latents.dtype
        )

        # Concatenate [noisy_latents, mask(4ch), image_latent(16ch)]
        assert training_batch.noisy_model_input is not None
        training_batch.noisy_model_input = torch.cat(
            [
                training_batch.noisy_model_input,
                mask_lat_size,
                image_latents,
            ],
            dim=1,
        )

        return training_batch

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
    ) -> dict[str, Any]:
        image_embeds = batch.image_embeds
        assert isinstance(image_embeds, torch.Tensor)

        # noise_input is [B, T, C, H, W].
        # If it already has full channels (noise+mask+image = 36),
        # it came from prepare_batch and is ready to use.
        # If it only has noise channels (16), we need to concat
        # the I2V conditioning.
        in_channels = int(
            getattr(self.transformer, "in_channels", 36)
        )
        if noise_input.shape[2] >= in_channels:
            # Already concatenated (from prepare_batch / finetune)
            noisy_model_input = noise_input.permute(0, 2, 1, 3, 4)
        else:
            # Raw noise only (from distillation rollout)
            assert batch.image_latents is not None
            noisy_model_input = self._concat_matrixgame_hidden_states(
                noise_input, batch.image_latents
            )

        return {
            "hidden_states": noisy_model_input,
            "encoder_hidden_states": None,
            "timestep": timestep,
            "encoder_hidden_states_image": image_embeds.to(
                self.device, dtype=torch.bfloat16
            ),
            "keyboard_cond": batch.keyboard_cond,
            "mouse_cond": batch.mouse_cond,
            "return_dict": False,
        }

    def _concat_matrixgame_hidden_states(
        self,
        noise_input: torch.Tensor,
        image_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Concat noise with image conditioning.

        noise_input: [B, T, C, H, W] (frame-first)
        image_latents: [B, C, T, H, W] (channel-first)
        Returns: [B, T, C_total, H, W] (frame-first for transformer)
        """
        # image_latents is [B, 20, T, H, W] (mask + latent already
        # built) or [B, 16, T, H, W] if raw
        noisy_model_input = torch.cat(
            [noise_input, image_latents.permute(0, 2, 1, 3, 4)], dim=2
        )
        return noisy_model_input.permute(0, 2, 1, 3, 4)

    def _get_transformer(
        self, timestep: torch.Tensor
    ) -> torch.nn.Module:
        return self.transformer
