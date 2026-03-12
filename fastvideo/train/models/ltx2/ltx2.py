# SPDX-License-Identifier: Apache-2.0
"""LTX-2 model plugin (per-role instance).

Supports text-to-video with optional audio stream.  The LTX-2
transformer outputs *denoised* predictions (via internal
``_to_denoised``).  This plugin converts them back to velocity
so that ``predict_noise`` returns flow-matching velocity,
consistent with the ``ModelBase`` contract.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.distributed import (
    get_sp_group,
    get_world_group,
)
from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.ltx2 import VideoLatentShape
from fastvideo.pipelines import TrainingBatch
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.training.activation_checkpoint import (
    apply_activation_checkpointing, )

from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.module_state import apply_trainable
from fastvideo.train.utils.moduleloader import load_module_from_path

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )


class LTX2Model(ModelBase):
    """LTX-2 per-role model: owns transformer + text_encoder."""

    _transformer_cls_name: str = "LTX2Transformer3DModel"

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        enable_gradient_checkpointing_type: str | None = None,
        transformer_override_safetensor: str | None = None,
        first_frame_conditioning_p: float = 0.0,
        with_audio: bool = False,
    ) -> None:
        self._init_from = str(init_from)
        self._trainable = bool(trainable)
        self._first_frame_conditioning_p = float(
            first_frame_conditioning_p)
        self._with_audio = bool(with_audio)

        self.transformer = self._load_transformer(
            init_from=self._init_from,
            trainable=self._trainable,
            disable_custom_init_weights=disable_custom_init_weights,
            enable_gradient_checkpointing_type=(
                enable_gradient_checkpointing_type),
            training_config=training_config,
            transformer_override_safetensor=(
                transformer_override_safetensor),
        )

        # LTX-2 uses continuous sigmas, not a discrete scheduler.
        self.noise_scheduler = _LTX2NoiseScheduler()

        # Filled by init_preprocessors (student only).
        self.vae: Any = None
        self.text_encoder: Any = None
        self.training_config: TrainingConfig = training_config
        self.dataloader: Any = None
        self.start_step: int = 0

        self.world_group: Any = None
        self.sp_group: Any = None

        # Negative-prompt embeddings (video + audio).
        self.negative_video_embeds: torch.Tensor | None = None
        self.negative_audio_embeds: torch.Tensor | None = None
        self.negative_attention_mask: torch.Tensor | None = None

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
            override_transformer_cls_name=(
                self._transformer_cls_name),
            transformer_override_safetensor=(
                transformer_override_safetensor),
        )
        transformer = apply_trainable(
            transformer, trainable=trainable)
        ckpt_type = (
            enable_gradient_checkpointing_type or getattr(
                getattr(training_config, "model", None),
                "enable_gradient_checkpointing_type",
                None,
            ))
        if trainable and ckpt_type:
            # LTX2 wraps transformer blocks in self.model
            if hasattr(transformer, "model"):
                transformer.model = (
                    apply_activation_checkpointing(
                        transformer.model,
                        checkpointing_type=ckpt_type,
                    ))
            else:
                transformer = apply_activation_checkpointing(
                    transformer,
                    checkpointing_type=ckpt_type,
                )
        return transformer

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_preprocessors(
        self,
        training_config: TrainingConfig,
    ) -> None:
        model_path = str(training_config.model_path)

        self.vae = load_module_from_path(
            model_path=model_path,
            module_type="vae",
            training_config=training_config,
        )

        self.text_encoder = load_module_from_path(
            model_path=model_path,
            module_type="text_encoder",
            training_config=training_config,
        )
        self.text_encoder.eval()
        self.text_encoder.to(self.device)

        self.world_group = get_world_group()
        self.sp_group = get_sp_group()

        self._build_dataloader(training_config)

    @property
    def num_train_timesteps(self) -> int:
        # Convention for continuous-sigma schedule.
        return 1000

    def shift_and_clamp_timestep(
        self,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        # LTX-2 timesteps are float sigmas; no shifting needed.
        return timestep

    def on_train_start(self) -> None:
        self.ensure_negative_conditioning()

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
        self.ensure_negative_conditioning()
        assert self.training_config is not None
        tc = self.training_config

        dtype = torch.bfloat16
        device = self.device

        training_batch = TrainingBatch()

        # --- Extract latents & text embeddings ---
        if self._is_precomputed_batch(raw_batch):
            training_batch = self._load_precomputed_batch(
                raw_batch, training_batch, device, dtype, tc)
        else:
            training_batch = self._load_parquet_batch(
                raw_batch, training_batch, device, dtype, tc,
                latents_source)

        # --- Noise & timestep sampling ---
        training_batch = self._prepare_dit_inputs(
            training_batch, generator)

        return training_batch

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        sigma = timestep
        while sigma.ndim < clean_latents.ndim:
            sigma = sigma.unsqueeze(-1)
        return (1.0 - sigma) * clean_latents + sigma * noise

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

        if conditional:
            text_dict = batch.conditional_dict
            if text_dict is None:
                raise RuntimeError(
                    "Missing conditional_dict in TrainingBatch")
        else:
            text_dict = self._get_uncond_text_dict(batch)

        # FineTuneMethod permutes to [B,T,C,H,W]; undo for LTX-2
        # which expects [B,C,T,H,W].
        noisy_latents_bctHW = noisy_latents.permute(0, 2, 1, 3, 4)

        with (
            torch.autocast(device_type, dtype=dtype),
            set_forward_context(
                current_timestep=batch.timesteps,
                attn_metadata=batch.attn_metadata,
            ),
        ):
            input_kwargs = self._build_distill_input_kwargs(
                noisy_latents_bctHW, timestep, text_dict, batch)
            outputs = self.transformer(**input_kwargs)

        # Parse output: (video_denoised, audio_denoised) or just
        # video_denoised.
        if isinstance(outputs, tuple):
            video_denoised, audio_denoised = outputs
        else:
            video_denoised = outputs
            audio_denoised = None

        # Store audio denoised for loss computation in the method.
        if audio_denoised is not None:
            batch._ltx2_audio_denoised = audio_denoised  # type: ignore[attr-defined]

        # Convert denoised -> velocity for ModelBase contract.
        # velocity = (noisy - denoised) / sigma
        sigmas = batch.sigmas
        assert sigmas is not None
        # sigmas is [B,1,1,1,1]; already broadcastable.
        velocity = (
            (noisy_latents_bctHW - video_denoised)
            / sigmas.clamp(min=1e-8)
        )

        # Return in [B,T,C,H,W] to match FineTuneMethod expectation.
        return velocity.permute(0, 2, 1, 3, 4)

    def predict_x0(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        # LTX-2 directly outputs denoised, so we convert:
        # x0 = noisy - velocity * sigma
        pred_velocity = self.predict_noise(
            noisy_latents, timestep, batch,
            conditional=conditional,
            cfg_uncond=cfg_uncond,
            attn_kind=attn_kind,
        )
        sigmas = batch.sigmas
        assert sigmas is not None
        # sigmas is already [B,1,1,1,1]; broadcastable.
        return noisy_latents - pred_velocity * sigmas

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
    # Audio loss helper (called by LTX2-aware methods)
    # ------------------------------------------------------------------

    def compute_audio_loss(
        self,
        batch: TrainingBatch,
    ) -> torch.Tensor | None:
        """Compute audio velocity MSE loss if audio data is present.

        Returns ``None`` if no audio data in this batch.
        """
        audio_denoised = getattr(batch, "_ltx2_audio_denoised", None)
        if audio_denoised is None:
            return None
        if batch.audio_latents is None:
            return None
        if batch.audio_noisy_model_input is None:
            return None

        sigmas = batch.sigmas
        assert sigmas is not None
        # sigmas is [B,1,1,1,1]; squeeze to [B,1,1,1] for audio.
        audio_sigmas = sigmas.squeeze(-1)
        audio_pred_velocity = (
            (batch.audio_noisy_model_input - audio_denoised)
            / audio_sigmas.clamp(min=1e-8)
        )
        audio_target = batch.audio_noise - batch.audio_latents
        return (
            (audio_pred_velocity.float() - audio_target.float())
            ** 2
        ).mean()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_precomputed_batch(
        self,
        raw_batch: dict[str, Any],
    ) -> bool:
        return "latents" in raw_batch and "conditions" in raw_batch

    def _load_precomputed_batch(
        self,
        raw_batch: dict[str, Any],
        training_batch: TrainingBatch,
        device: torch.device,
        dtype: torch.dtype,
        tc: TrainingConfig,
    ) -> TrainingBatch:
        latents = raw_batch["latents"]["latents"].to(
            device, dtype=dtype)
        latents = latents[:, :, :tc.data.num_latent_t]
        conditions = raw_batch["conditions"]

        if ("video_prompt_embeds" in conditions
                and "audio_prompt_embeds" in conditions
                and "prompt_attention_mask" in conditions):
            video_embeds = conditions["video_prompt_embeds"].to(
                device)
            audio_embeds = conditions["audio_prompt_embeds"].to(
                device)
            attention_mask = conditions[
                "prompt_attention_mask"].to(device, dtype=torch.int64)
        else:
            prompt_embeds = conditions["prompt_embeds"].to(device)
            prompt_attention_mask = conditions[
                "prompt_attention_mask"].to(
                    device, dtype=torch.int64)
            video_embeds, audio_embeds, attention_mask = (
                self.text_encoder.run_connectors(
                    prompt_embeds, prompt_attention_mask))

        training_batch.latents = latents
        training_batch.encoder_hidden_states = video_embeds.to(
            device, dtype=dtype)
        training_batch.encoder_attention_mask = attention_mask.to(
            device, dtype=dtype)

        if self._with_audio and "audio_latents" in raw_batch:
            audio_latents = raw_batch["audio_latents"][
                "latents"].to(device, dtype=dtype)
            training_batch.audio_latents = audio_latents
            training_batch.audio_encoder_hidden_states = (
                audio_embeds.to(device, dtype=dtype))
            training_batch.audio_encoder_attention_mask = (
                attention_mask.to(device))

        idxs = raw_batch.get("idx")
        if idxs is not None and torch.is_tensor(idxs):
            training_batch.infos = [
                {"idx": int(i)} for i in idxs.tolist()]
        else:
            training_batch.infos = []
        training_batch.raw_latent_shape = latents.shape
        return training_batch

    def _load_parquet_batch(
        self,
        raw_batch: dict[str, Any],
        training_batch: TrainingBatch,
        device: torch.device,
        dtype: torch.dtype,
        tc: TrainingConfig,
        latents_source: Literal["data", "zeros"],
    ) -> TrainingBatch:
        prompt_embeds = raw_batch["text_embedding"].to(device)
        prompt_attention_mask = raw_batch[
            "text_attention_mask"].to(device, dtype=torch.int64)

        video_embeds, audio_embeds, attention_mask = (
            self.text_encoder.run_connectors(
                prompt_embeds, prompt_attention_mask))

        if latents_source == "zeros":
            batch_size = video_embeds.shape[0]
            vae_cfg = (
                tc.pipeline_config.vae_config.arch_config  # type: ignore[union-attr]
            )
            num_channels = vae_cfg.z_dim
            scr = vae_cfg.spatial_compression_ratio
            latent_h = tc.data.num_height // scr
            latent_w = tc.data.num_width // scr
            latents = torch.zeros(
                batch_size, num_channels, tc.data.num_latent_t,
                latent_h, latent_w,
                device=device, dtype=dtype)
        elif latents_source == "data":
            if "vae_latent" not in raw_batch:
                raise ValueError(
                    "vae_latent not found in batch and "
                    "latents_source='data'")
            latents = raw_batch["vae_latent"]
            latents = latents[:, :, :tc.data.num_latent_t]
            latents = latents.to(device, dtype=dtype)
        else:
            raise ValueError(
                f"Unknown latents_source: {latents_source!r}")

        training_batch.latents = latents
        training_batch.encoder_hidden_states = video_embeds.to(
            device, dtype=dtype)
        training_batch.encoder_attention_mask = attention_mask.to(
            device, dtype=dtype)
        training_batch.infos = []
        training_batch.raw_latent_shape = latents.shape
        return training_batch

    def _prepare_dit_inputs(
        self,
        training_batch: TrainingBatch,
        generator: torch.Generator,
    ) -> TrainingBatch:
        assert self.training_config is not None
        latents = training_batch.latents
        assert isinstance(latents, torch.Tensor)
        batch_size = latents.shape[0]

        # Compute token count for sigma shift.
        video_shape = VideoLatentShape.from_torch_shape(
            latents.shape)
        if hasattr(self.transformer, "patchifier"):
            token_count = (
                self.transformer.patchifier.get_token_count(
                    video_shape))
        else:
            token_count = (
                latents.shape[2] * latents.shape[3]
                * latents.shape[4])

        # Sample sigmas with LTX-2 shifted-sigmoid schedule.
        # _sample_sigmas returns 1-D [B]; we keep a flat copy for
        # building per-token timesteps, then expand to [B,1,1,1,1]
        # for noise mixing and to satisfy FineTuneMethod which
        # broadcasts ``training_batch.sigmas`` against 5-D tensors.
        sigmas_flat = self._sample_sigmas(
            batch_size, token_count,
            latents.device, latents.dtype,
            generator)

        if int(self.training_config.distributed.sp_size or 1) > 1:
            self.sp_group.broadcast(sigmas_flat, src=0)

        noise = torch.randn(
            latents.shape, generator=generator,
            device=latents.device, dtype=latents.dtype)
        sigmas_5d = sigmas_flat.view(-1, 1, 1, 1, 1)
        noisy_model_input = (
            (1.0 - sigmas_5d) * latents + sigmas_5d * noise
        )

        # First-frame conditioning.
        conditioning_mask = None
        if (self._first_frame_conditioning_p > 0
                and torch.rand(
                    1, device=latents.device,
                    generator=generator,
                ).item() < self._first_frame_conditioning_p):
            conditioning_mask = torch.zeros(
                (batch_size, 1, latents.shape[2],
                 latents.shape[3], latents.shape[4]),
                dtype=torch.bool, device=latents.device)
            conditioning_mask[:, :, 0:1] = True
            noisy_model_input = torch.where(
                conditioning_mask, latents, noisy_model_input)

        # Build per-token timesteps [B, token_count].
        if conditioning_mask is None:
            mask_patch = torch.zeros(
                (batch_size, token_count),
                dtype=torch.bool, device=latents.device)
        elif hasattr(self.transformer, "patchifier"):
            mask_patch = (
                self.transformer.patchifier.patchify(
                    conditioning_mask.float()
                ).sum(dim=-1) > 0
            )
        else:
            mask_patch = conditioning_mask[
                :, 0].reshape(batch_size, -1)

        timesteps = torch.where(
            mask_patch,
            torch.zeros_like(mask_patch, dtype=latents.dtype),
            sigmas_flat.view(-1, 1).expand_as(mask_patch).to(
                latents.dtype),
        )

        training_batch.noisy_model_input = noisy_model_input
        training_batch.timesteps = timesteps
        # Store sigmas as [B,1,1,1,1] for FineTuneMethod compat.
        training_batch.sigmas = sigmas_5d
        training_batch.noise = noise
        training_batch.conditioning_mask = conditioning_mask

        # Audio noising.
        if (training_batch.audio_latents is not None):
            audio_latents = training_batch.audio_latents
            audio_noise = torch.randn(
                audio_latents.shape, generator=generator,
                device=audio_latents.device,
                dtype=audio_latents.dtype)
            audio_sigmas_4d = sigmas_flat.view(-1, 1, 1, 1)
            audio_noisy = (
                (1.0 - audio_sigmas_4d) * audio_latents
                + audio_sigmas_4d * audio_noise
            )
            audio_timesteps = sigmas_flat.view(-1, 1).expand(
                batch_size, audio_latents.shape[2])
            training_batch.audio_noisy_model_input = audio_noisy
            training_batch.audio_timesteps = (
                audio_timesteps.to(dtype=audio_latents.dtype))
            training_batch.audio_noise = audio_noise

        # Build conditional dict.
        training_batch.conditional_dict = {
            "encoder_hidden_states": (
                training_batch.encoder_hidden_states),
            "encoder_attention_mask": (
                training_batch.encoder_attention_mask),
        }
        if (training_batch.audio_encoder_hidden_states
                is not None):
            training_batch.conditional_dict.update({
                "audio_encoder_hidden_states": (
                    training_batch.audio_encoder_hidden_states),
                "audio_encoder_attention_mask": (
                    training_batch.audio_encoder_attention_mask),
            })

        # Build unconditional dict from negative embeddings.
        if self.negative_video_embeds is not None:
            neg_v = self.negative_video_embeds
            neg_mask = self.negative_attention_mask
            if neg_v.shape[0] == 1 and batch_size > 1:
                neg_v = neg_v.expand(
                    batch_size, *neg_v.shape[1:]).contiguous()
            if (neg_mask is not None
                    and neg_mask.shape[0] == 1
                    and batch_size > 1):
                neg_mask = neg_mask.expand(
                    batch_size, *neg_mask.shape[1:]).contiguous()
            uncond: dict[str, Any] = {
                "encoder_hidden_states": neg_v,
                "encoder_attention_mask": neg_mask,
            }
            if self.negative_audio_embeds is not None:
                neg_a = self.negative_audio_embeds
                if neg_a.shape[0] == 1 and batch_size > 1:
                    neg_a = neg_a.expand(
                        batch_size, *neg_a.shape[1:]).contiguous()
                uncond["audio_encoder_hidden_states"] = neg_a
                uncond["audio_encoder_attention_mask"] = neg_mask
            training_batch.unconditional_dict = uncond

        # Permute latents to [B,T,C,H,W] for FineTuneMethod.
        training_batch.latents = (
            training_batch.latents.permute(0, 2, 1, 3, 4))
        return training_batch

    def _sample_sigmas(
        self,
        batch_size: int,
        seq_length: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """LTX-2 shifted-sigmoid sigma schedule."""
        min_tokens = 1024
        max_tokens = 4096
        min_shift = 0.95
        max_shift = 2.05
        slope = (max_shift - min_shift) / (
            max_tokens - min_tokens)
        shift = slope * seq_length + (
            min_shift - slope * min_tokens)
        normal_samples = torch.randn(
            (batch_size,), generator=generator,
            device=device) + shift
        return torch.sigmoid(normal_samples).to(
            device=device, dtype=dtype)

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, torch.Tensor] | None,
        batch: TrainingBatch,
    ) -> dict[str, Any]:
        if text_dict is None:
            raise ValueError(
                "text_dict cannot be None for LTX-2")
        kwargs: dict[str, Any] = {
            "hidden_states": noise_input,
            "encoder_hidden_states": (
                text_dict["encoder_hidden_states"]),
            "encoder_attention_mask": (
                text_dict["encoder_attention_mask"]),
            "timestep": timestep,
            "return_dict": False,
        }
        # Audio inputs.
        if batch.audio_noisy_model_input is not None:
            kwargs["audio_hidden_states"] = (
                batch.audio_noisy_model_input)
            kwargs["audio_timestep"] = (
                batch.audio_timesteps)
            audio_enc = text_dict.get(
                "audio_encoder_hidden_states",
                batch.audio_encoder_hidden_states)
            audio_mask = text_dict.get(
                "audio_encoder_attention_mask",
                batch.audio_encoder_attention_mask)
            if audio_enc is not None:
                kwargs["audio_encoder_hidden_states"] = audio_enc
            if audio_mask is not None:
                kwargs["audio_encoder_attention_mask"] = audio_mask
        return kwargs

    def _get_uncond_text_dict(
        self,
        batch: TrainingBatch,
    ) -> dict[str, torch.Tensor]:
        text_dict = getattr(batch, "unconditional_dict", None)
        if text_dict is None:
            raise RuntimeError(
                "Missing unconditional_dict; "
                "ensure_negative_conditioning() may have failed")
        return text_dict

    def ensure_negative_conditioning(self) -> None:
        if self.negative_video_embeds is not None:
            return
        if self.text_encoder is None:
            # Teacher / critic roles don't load text_encoder.
            return

        device = self.device
        dtype = torch.bfloat16
        world_group = self.world_group
        if world_group is None:
            return

        neg_v: torch.Tensor | None = None
        neg_a: torch.Tensor | None = None
        neg_mask: torch.Tensor | None = None

        if world_group.rank_in_group == 0:
            from fastvideo.pipelines.basic.ltx2.ltx2_pipeline import (
                LTX2Pipeline, )
            from fastvideo.train.utils.moduleloader import (
                make_inference_args, )

            tc = self.training_config
            assert tc is not None
            inference_args = make_inference_args(
                tc, model_path=tc.model_path)

            batch_neg = ForwardBatch(
                data_type="video",
                prompt="",
                prompt_embeds=[],
                prompt_attention_mask=[],
            )
            pipeline = LTX2Pipeline.from_pretrained(
                tc.model_path,
                args=inference_args,
                inference_mode=True,
                loaded_modules={
                    "transformer": self.transformer,
                    "text_encoder": self.text_encoder,
                },
                tp_size=tc.distributed.tp_size,
                sp_size=tc.distributed.sp_size,
                num_gpus=tc.distributed.num_gpus,
                pin_cpu_memory=tc.distributed.pin_cpu_memory,
                dit_cpu_offload=True,
            )
            result = pipeline.prompt_encoding_stage(
                batch_neg, inference_args)

            # LTX-2 prompt encoding produces embeddings via
            # run_connectors.
            if hasattr(result, "prompt_embeds") and result.prompt_embeds:
                raw_embeds = result.prompt_embeds[0]
                raw_mask = result.prompt_attention_mask[0]
                neg_v_c, neg_a_c, neg_mask_c = (
                    self.text_encoder.run_connectors(
                        raw_embeds.to(device),
                        raw_mask.to(device, dtype=torch.int64),
                    ))
                neg_v = neg_v_c.to(device=device, dtype=dtype)
                neg_a = neg_a_c.to(device=device, dtype=dtype)
                neg_mask = neg_mask_c.to(
                    device=device, dtype=dtype)

            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Broadcast shapes + data from rank 0.
        self.negative_video_embeds = self._broadcast_tensor(
            neg_v, world_group, device, dtype)
        self.negative_audio_embeds = self._broadcast_tensor(
            neg_a, world_group, device, dtype)
        self.negative_attention_mask = self._broadcast_tensor(
            neg_mask, world_group, device, dtype)

    @staticmethod
    def _broadcast_tensor(
        tensor: torch.Tensor | None,
        world_group: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Broadcast a tensor from rank 0 to all ranks."""
        max_ndim = 8
        meta = torch.zeros((1 + max_ndim,),
                           device=device, dtype=torch.int64)
        if world_group.rank_in_group == 0:
            if tensor is None:
                meta[0] = -1
            else:
                meta[0] = tensor.ndim
                for i, s in enumerate(tensor.shape):
                    meta[1 + i] = s
        world_group.broadcast(meta, src=0)
        ndim = int(meta[0].item())
        if ndim < 0:
            return None
        shape = tuple(int(meta[1 + i].item()) for i in range(ndim))
        if world_group.rank_in_group != 0:
            tensor = torch.empty(shape, device=device, dtype=dtype)
        assert tensor is not None
        world_group.broadcast(tensor, src=0)
        return tensor

    def _build_dataloader(
        self,
        training_config: TrainingConfig,
    ) -> None:
        data_path = training_config.data.data_path
        if self._has_precomputed_pt_data(data_path):
            data_sources = self._get_data_sources(data_path)
            self._with_audio = "audio_latents" in data_sources

            from fastvideo.dataset import (
                build_ltx2_precomputed_dataloader, )

            _dataset, self.dataloader = (
                build_ltx2_precomputed_dataloader(
                    data_path,
                    training_config.data.train_batch_size,
                    num_data_workers=(
                        training_config.data.dataloader_num_workers),
                    data_sources=data_sources,
                    drop_last=True,
                    seed=int(training_config.data.seed or 0),
                ))
        else:
            from fastvideo.dataset.dataloader.schema import (
                pyarrow_schema_text_only, )
            from fastvideo.train.utils.dataloader import (
                build_parquet_t2v_train_dataloader, )

            text_len = int(
                training_config.pipeline_config.text_encoder_configs[  # type: ignore[union-attr]
                    0].arch_config.text_len)
            self.dataloader = build_parquet_t2v_train_dataloader(
                training_config.data,
                text_len=text_len,
                parquet_schema=pyarrow_schema_text_only,
            )

    @staticmethod
    def _has_precomputed_pt_data(data_path: str) -> bool:
        data_root = Path(data_path).expanduser().resolve()
        if (data_root / ".precomputed").exists():
            data_root = data_root / ".precomputed"
        latents_dir = data_root / "latents"
        conditions_dir = data_root / "conditions"
        return (
            latents_dir.exists()
            and conditions_dir.exists()
            and any(latents_dir.rglob("*.pt"))
            and any(conditions_dir.rglob("*.pt"))
        )

    @staticmethod
    def _get_data_sources(data_path: str) -> dict[str, str]:
        data_root = Path(data_path).expanduser().resolve()
        if (data_root / ".precomputed").exists():
            data_root = data_root / ".precomputed"
        sources: dict[str, str] = {
            "latents": "latents",
            "conditions": "conditions",
        }
        audio_dir = data_root / "audio_latents"
        if audio_dir.exists() and any(audio_dir.rglob("*.pt")):
            sources["audio_latents"] = "audio_latents"
        return sources


class _LTX2NoiseScheduler:
    """Minimal scheduler adapter for LTX-2's continuous sigma schedule.

    Provides the interface expected by ``ModelBase`` (specifically
    ``num_train_timesteps`` and ``add_noise``).
    """

    num_train_timesteps: int = 1000

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        while sigma.ndim < clean.ndim:
            sigma = sigma.unsqueeze(-1)
        return (1.0 - sigma) * clean + sigma * noise
