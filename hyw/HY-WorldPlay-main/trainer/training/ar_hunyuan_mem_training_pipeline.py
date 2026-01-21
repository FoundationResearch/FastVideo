# SPDX-License-Identifier: Apache-2.0
import dataclasses
import json
import math
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from typing import Any

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm


from trainer.distributed.parallel_state import (get_sp_parallel_rank,
                                                  get_sp_world_size)

import trainer.envs as envs
from trainer.attention.backends.video_sparse_attn import (
    VideoSparseAttentionMetadataBuilder)
from trainer.configs.sample import SamplingParam
from trainer.dataset import build_ar_camera_hunyuan_w_mem_dataloader
from trainer.dataset.dataloader.schema import pyarrow_schema_t2v
from trainer.dataset.validation_dataset import ValidationDataset
from trainer.distributed import (cleanup_dist_env_and_memory,
                                   get_local_torch_device, get_sp_group,
                                   get_world_group)
from trainer.trainer_args import TrainerArgs, TrainingArgs
from trainer.forward_context import set_forward_context
from trainer.logger import init_logger
from trainer.pipelines import (ComposedPipelineBase, ForwardBatch,
                                 LoRAPipeline, TrainingBatch)
from trainer.training.activation_checkpoint import (
    apply_activation_checkpointing)
from trainer.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    compute_density_for_timestep_sampling, get_scheduler, get_sigmas,
    load_checkpoint, normalize_dit_input, save_checkpoint,
)
from trainer.utils import is_vsa_available, set_random_seed, shallow_asdict
# import muon optimizer
from trainer.training.muon import get_muon_optimizer

import wandb  # isort: skip

vsa_available = is_vsa_available()

logger = init_logger(__name__)


def _get_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_tensor_by_mask(tensor_1, tensor_2, mask, dim):
    assert tensor_1.shape == tensor_2.shape
    # Mask is a 0/1 vector. Choose tensor_2 when the value is 1; otherwise, tensor_1
    masked_indices = torch.nonzero(mask).squeeze(1)
    tmp = tensor_1.clone()
    if dim == 0:
        tmp[masked_indices] = tensor_2[masked_indices]
    elif dim == 1:
        tmp[:, masked_indices] = tensor_2[:, masked_indices]
    elif dim == 2:
        tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]
    return tmp


def _unwrap_module(m):
    """
    Handle common wrappers (FSDP/DDP/torch.compile) without importing their types.

    NOTE: Do NOT unwrap a `.vae` attribute here. Some VAE wrappers may expose a `.vae`
    field that can be `None` (or not the actual decode implementation). Unwrapping it
    can incorrectly turn a valid module into `None`.
    """
    for attr in ("module", "_orig_mod"):
        if hasattr(m, attr):
            m = getattr(m, attr)
    return m


def _resolve_vae_scaling_and_shift(vae):
    """
    Best-effort resolve (scaling_factor, shift_factor) from a variety of VAE implementations.
    Returns:
        (scaling_factor: float, shift_factor: float|None)
    """
    vae_u = _unwrap_module(vae)
    # Try common locations
    scaling = getattr(vae_u, "scaling_factor", None)
    cfg = getattr(vae_u, "config", None)
    if scaling is None and cfg is not None:
        scaling = getattr(cfg, "scaling_factor", None)
    # Some trainers keep scaling in nested config objects
    if scaling is None:
        cfg2 = getattr(vae_u, "vae_config", None)
        if cfg2 is not None:
            scaling = getattr(cfg2, "scaling_factor", None)
    # Some configs may have scaling_factor=0 as default; treat that as unknown
    try:
        if scaling is not None:
            scaling = float(scaling)
            if scaling == 0.0:
                scaling = None
    except Exception:
        scaling = None

    shift = None
    if cfg is not None:
        shift = getattr(cfg, "shift_factor", None)
        try:
            if shift is not None:
                shift = float(shift)
        except Exception:
            shift = None

    # Be strict: visualization must decode correctly.
    if scaling is None:
        raise RuntimeError(
            f"VAE scaling_factor not found (vae={type(vae_u).__name__}). "
            "Expected `vae.scaling_factor` or `vae.config.scaling_factor` to exist."
        )
    return scaling, shift


def _vae_decode_video_frames(vae, latents: torch.Tensor) -> torch.Tensor:
    """
    Decode VAE latents into video frames for visualization.

    Args:
        vae: trainer VAE module (supports .decode and has .scaling_factor or .config.scaling_factor)
        latents: (B,C,T,H,W) in *scaled latent space* (i.e. encoder output multiplied by scaling_factor).

    Returns:
        frames: (B,3,F,H,W) float in [0,1]
    """
    vae_u = _unwrap_module(vae)
    scaling, shift_factor = _resolve_vae_scaling_and_shift(vae_u)
    # Match HY-WorldPlay pipeline behavior when possible: divide by scaling_factor, optionally add shift_factor.
    z = latents / scaling
    if shift_factor is not None:
        z = z + shift_factor

    # Decode on the VAE's device to avoid device mismatch (CPU/GPU offload variants).
    try:
        vae_device = next(vae_u.parameters()).device
    except StopIteration:
        vae_device = z.device
    z = z.to(device=vae_device)

    dec = vae_u.decode(z)
    # Some VAE impls return tuple/list; some return tensor directly.
    if isinstance(dec, (tuple, list)):
        dec = dec[0]
    # diffusers-style DecoderOutput
    if hasattr(dec, "sample"):
        dec = dec.sample
    frames = (dec / 2 + 0.5).clamp(0, 1)
    return frames

class TrainingPipeline(LoRAPipeline, ABC):
    """
    A pipeline for training a model. All training pipelines should inherit from this class.
    All reusable components and code should be implemented in this class.
    """
    _required_config_modules = ["scheduler", "transformer"]
    validation_pipeline: ComposedPipelineBase
    train_dataloader: StatefulDataLoader
    train_loader_iter: Iterator[dict[str, Any]]
    current_epoch: int = 0

    def __init__(
            self,
            model_path: str,
            trainer_args: TrainingArgs,
            required_config_modules: list[str] | None = None,
            loaded_modules: dict[str, torch.nn.Module] | None = None) -> None:
        trainer_args.inference_mode = False
        self.lora_training = trainer_args.lora_training
        if self.lora_training and trainer_args.lora_rank is None:
            raise ValueError("lora rank must be set when using lora training")

        set_random_seed(trainer_args.seed)  # for lora param init
        super().__init__(model_path, trainer_args, required_config_modules,
                         loaded_modules)  # type: ignore

    def create_pipeline_stages(self, trainer_args: TrainerArgs):
        raise RuntimeError(
            "create_pipeline_stages should not be called for training pipeline")

    def set_schemas(self) -> None:
        self.train_dataset_schema = pyarrow_schema_t2v

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing training pipeline...")
        self.device = get_local_torch_device()
        self.training_args = training_args
        world_group = get_world_group()
        self.world_size = world_group.world_size
        self.global_rank = world_group.rank
        self.sp_group = get_sp_group()
        self.rank_in_sp_group = self.sp_group.rank_in_group
        self.sp_world_size = self.sp_group.world_size
        self.local_rank = world_group.local_rank
        self.transformer = self.get_module("transformer")
        self.seed = training_args.seed
        self.set_schemas()
        self.action = training_args.action
        # add the causal option
        self.causal = training_args.causal
        self.train_time_shift = training_args.train_time_shift

        # Set random seeds for deterministic training
        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed)
        self.transformer.train()
        if training_args.enable_gradient_checkpointing_type is not None:
            self.transformer = apply_activation_checkpointing(
                self.transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)

        self.set_trainable()
        params_to_optimize = self.transformer.parameters()
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, params_to_optimize))
        self.optimizer = get_muon_optimizer(
            model=self.transformer,
            lr=training_args.learning_rate,                      # Learning rate
            weight_decay=training_args.weight_decay,  # Weight decay
            adamw_betas=(0.9, 0.999),   # AdamW betas for 1D parameters
            adamw_eps=1e-8,        # AdamW epsilon
        )


        self.init_steps = 0
        logger.info("optimizer: %s", self.optimizer)

        self.lr_scheduler = get_scheduler(
            training_args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.max_train_steps,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            min_lr_ratio=training_args.min_lr_ratio,
            last_epoch=self.init_steps - 1,
        )

        self.train_dataset, self.train_dataloader = build_ar_camera_hunyuan_w_mem_dataloader(
            json_path=training_args.json_path,
            causal=training_args.causal,
            window_frames=training_args.window_frames,
            batch_size=training_args.train_batch_size,
            num_data_workers=training_args.dataloader_num_workers,
            drop_last=True,
            drop_first_row=False,
            seed=self.seed,
            cfg_rate=training_args.training_cfg_rate,
            i2v_rate=training_args.i2v_rate,
        )

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) /
            training_args.gradient_accumulation_steps * training_args.sp_size /
            training_args.train_sp_batch_size)
        self.num_train_epochs = math.ceil(training_args.max_train_steps /
                                          self.num_update_steps_per_epoch)

        # TODO(will): is there a cleaner way to track epochs?
        self.current_epoch = 0

        if self.global_rank == 0:
            project = training_args.tracker_project_name or "trainer"
            wandb_config = dataclasses.asdict(training_args)
            # If user already ran `wandb login`, allow empty key and fall back to stored credentials.
            wandb.login(key=training_args.wandb_key or None)
            wandb.init(
                config=wandb_config,
                name=training_args.wandb_run_name,
                entity=training_args.wandb_entity,
                project=project,
            )

    @abstractmethod
    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        raise NotImplementedError(
            "Training pipelines must implement this method")

    def _should_log_train_videos(self, step: int) -> bool:
        every = int(getattr(self.training_args, "train_video_log_steps", 0) or 0)
        return every > 0 and (step % every) == 0

    def _get_train_vis_vae(self) -> torch.nn.Module:
        """
        Training pipeline does not load the VAE by default (`_required_config_modules` excludes it).
        For strict train-time visualization, we lazily load a VAE decoder from the pretrained model.
        """
        if hasattr(self, "_train_vis_vae") and self._train_vis_vae is not None:
            return self._train_vis_vae

        model_root = getattr(self.training_args, "pretrained_model_name_or_path", None) or getattr(self.training_args, "model_path", None)
        if not model_root:
            raise RuntimeError("Cannot load visualization VAE: pretrained_model_name_or_path/model_path is empty.")
        vae_dir = os.path.join(str(model_root), "vae")
        if not os.path.isdir(vae_dir):
            raise RuntimeError(f"Cannot load visualization VAE: missing vae directory: {vae_dir}")

        # Use the hyvideo VAE implementation (matches latent encoding used by our precompute script).
        from hyvideo.models.autoencoders.hunyuanvideo_15_vae_w_cache import AutoencoderKLConv3D  # type: ignore

        device = get_local_torch_device()
        vis_vae = AutoencoderKLConv3D.from_pretrained(vae_dir, torch_dtype=torch.float16)
        vis_vae = vis_vae.to(device).eval()
        for p in vis_vae.parameters():
            p.requires_grad_(False)

        self._train_vis_vae = vis_vae
        logger.info("Loaded visualization VAE for train video logging from %s", vae_dir)
        return self._train_vis_vae

    @torch.no_grad()
    def _log_train_videos_to_wandb(self, training_batch: TrainingBatch, step: int) -> None:
        """
        Log gt/noisy/pred decoded videos to wandb.

        Fast path for multi-GPU debugging (no inter-rank collectives):
        - Each rank loads its own VAE decoder (cached after first use).
        - At each log event we select `vis_rank = log_idx % world_size` (round-robin).
          Only `vis_rank` decodes & writes the MP4 (from its own local batch).
        - Rank0 logs the MP4 to WandB exactly once, by polling the filesystem for the expected output file.
        """
        if not self._should_log_train_videos(step):
            return
        if self.training_args.sp_size > 1:
            # SP shards tokens/latents; reconstructing full video for logging is non-trivial.
            logger.warning(
                "Skipping train video logging because sp_size > 1 (sp_size=%s).",
                self.training_args.sp_size,
            )
            return

        if training_batch.latents is None or training_batch.noisy_model_input is None or training_batch.noise is None:
            return
        if training_batch.model_pred is None:
            return

        # -----------------------------
        # Choose visualization rank (round-robin).
        # -----------------------------
        world_size = int(getattr(self, "world_size", 1) or 1)
        rank = int(getattr(self, "global_rank", 0) or 0)
        every = int(getattr(self.training_args, "train_video_log_steps", 0) or 0)
        log_idx = (int(step) // max(1, every)) if every > 0 else int(step)
        vis_rank = int(log_idx % max(1, world_size))

        max_samples = int(getattr(self.training_args, "train_video_log_max_samples", 1) or 1)
        max_samples = max(1, max_samples)
        fps = int(getattr(self.training_args, "train_video_log_fps", 25) or 25)

        # Keep training output dir tidy: write VAE-decoded previews under a dedicated subfolder.
        vis_dir = os.path.join(self.training_args.output_dir, "vae_during_training")
        os.makedirs(vis_dir, exist_ok=True)
        triptych_path = os.path.join(vis_dir, f"train_step_{step:06d}_rank{vis_rank}_gt_noisy_x0hat.mp4")
        stats_path = os.path.join(vis_dir, f"train_step_{step:06d}_rank{vis_rank}_stats.json")

        # Load VAE on every rank (cached) so that when it's a rank's turn it can decode immediately.
        # NOTE: training pipeline does not necessarily load VAE; use a dedicated VAE for visualization.
        _ = self.get_module("vae", None) or self._get_train_vis_vae()

        # -----------------------------
        # Decode only on vis_rank, using that rank's local batch.
        # -----------------------------
        if rank == vis_rank:
            device = get_local_torch_device()
            vae = self.get_module("vae", None) or self._get_train_vis_vae()

            # Build x0_hat latents for visualization (does NOT affect training).
            if self.training_args.precondition_outputs:
                x0_hat_latents = training_batch.model_pred
            else:
                x0_hat_latents = training_batch.noise - training_batch.model_pred

            gt_latents = training_batch.latents[:max_samples].to(device)
            noisy_latents = training_batch.noisy_model_input[:max_samples].to(device)
            x0_hat_latents = x0_hat_latents[:max_samples].to(device)

            # Local scalar stats for caption/overlay (written to a sidecar JSON for rank0 to read).
            t_mean = float("nan")
            s_mean = float("nan")
            if training_batch.timesteps is not None:
                t_mean = float(training_batch.timesteps.detach().float().mean().cpu().item())
            if training_batch.sigmas is not None:
                s_mean = float(training_batch.sigmas.detach().float().mean().cpu().item())
            loss0 = float(getattr(training_batch, "total_loss", 0.0) or 0.0)
            gn0 = float(getattr(training_batch, "grad_norm", 0.0) or 0.0)
            try:
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"vis_rank": vis_rank, "t_mean": t_mean, "sigma_mean": s_mean, "loss": loss0, "grad_norm": gn0},
                        f,
                        indent=2,
                    )
            except Exception:
                pass

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                gt_frames = _vae_decode_video_frames(vae, gt_latents)
                noisy_frames = _vae_decode_video_frames(vae, noisy_latents)
                x0_hat_frames = _vae_decode_video_frames(vae, x0_hat_latents)

            def _to_uint8_video(frames01: torch.Tensor) -> np.ndarray:
                # (B,3,T,H,W) -> (T,H,W,3) uint8 (only first sample)
                x = frames01[0].detach().cpu()
                x = (x * 255.0).clamp(0, 255).to(torch.uint8)
                return x.permute(1, 2, 3, 0).numpy()

            gt_vid = _to_uint8_video(gt_frames)
            noisy_vid = _to_uint8_video(noisy_frames)
            x0_hat_vid = _to_uint8_video(x0_hat_frames)

            # Add timestep/sigma overlay for clarity (compact for low-res videos).
            overlay_lines = [f"step={step}"]
            if not math.isnan(t_mean) and not math.isnan(s_mean):
                overlay_lines.append(f"t={t_mean:.0f}  σ={s_mean:.3f}")
            elif not math.isnan(t_mean):
                overlay_lines.append(f"t={t_mean:.0f}")
            elif not math.isnan(s_mean):
                overlay_lines.append(f"σ={s_mean:.3f}")
            overlay = "\n".join(overlay_lines)

            def _draw_overlay(frames: np.ndarray) -> np.ndarray:
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                out = frames.copy()
                for i in range(out.shape[0]):
                    im = Image.fromarray(out[i])
                    draw = ImageDraw.Draw(im)
                    hud_h = 34
                    draw.rectangle([(0, 0), (im.size[0], hud_h)], fill=(0, 0, 0))
                    draw.multiline_text((4, 2), overlay, fill=(255, 255, 255), font=font, spacing=0)
                    out[i] = np.asarray(im)
                return out

            gt_vid = _draw_overlay(gt_vid)
            noisy_vid = _draw_overlay(noisy_vid)
            x0_hat_vid = _draw_overlay(x0_hat_vid)

            # Combine into one side-by-side (GT | noisy | x0_hat) so wandb shows a single video panel.
            T = min(gt_vid.shape[0], noisy_vid.shape[0], x0_hat_vid.shape[0])
            gt_vid = gt_vid[:T]
            noisy_vid = noisy_vid[:T]
            x0_hat_vid = x0_hat_vid[:T]
            triptych = np.concatenate([gt_vid, noisy_vid, x0_hat_vid], axis=2)
            imageio.mimsave(triptych_path, triptych, fps=fps, format="mp4")

        if self.global_rank == 0:
            # Rank0 logs exactly once to WandB (avoid duplicate panels).
            # Caption uses stats written by vis_rank (sidecar JSON). Avoid collectives by polling filesystem.
            t_mean = float("nan")
            s_mean = float("nan")
            loss0 = float("nan")
            gn0 = float("nan")
            # Wait up to ~120s for the mp4 (and stats) to appear.
            timeout_s = float(os.environ.get("TRAIN_VIDEO_LOG_TIMEOUT_S", "120"))
            t0 = time.time()
            while (not os.path.exists(triptych_path)) and (time.time() - t0 < timeout_s):
                time.sleep(0.2)
            # Stats is optional; try best-effort.
            if os.path.exists(stats_path):
                try:
                    st = json.loads(open(stats_path, "r", encoding="utf-8").read())
                    t_mean = float(st.get("t_mean", t_mean))
                    s_mean = float(st.get("sigma_mean", s_mean))
                    loss0 = float(st.get("loss", loss0))
                    gn0 = float(st.get("grad_norm", gn0))
                except Exception:
                    pass

            if not os.path.exists(triptych_path):
                logger.warning(
                    "Train video preview not found within timeout (step=%s, vis_rank=%s, path=%s). Skipping wandb video log.",
                    step,
                    vis_rank,
                    triptych_path,
                )
                return

            overlay_lines = [f"step={step}"]
            if not math.isnan(t_mean) and not math.isnan(s_mean):
                overlay_lines.append(f"t={t_mean:.0f}  σ={s_mean:.3f}")
            elif not math.isnan(t_mean):
                overlay_lines.append(f"t={t_mean:.0f}")
            elif not math.isnan(s_mean):
                overlay_lines.append(f"σ={s_mean:.3f}")
            overlay = "  ".join(overlay_lines)
            caption = f"{overlay}  vis_rank={vis_rank}  loss={loss0:.6f} grad_norm={gn0:.3f}"
            wandb.log(
                {
                    "train_video_gt_noisy_x0hat": wandb.Video(triptych_path, caption=caption),
                },
                step=step,
            )

    def _prepare_training(self, training_batch: TrainingBatch) -> TrainingBatch:
        self.transformer.train()
        self.optimizer.zero_grad()
        training_batch.total_loss = 0.0
        return training_batch

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            # Reset iterator for next epoch
            self.train_loader_iter = iter(self.train_dataloader)
            # Get first batch of new epoch
            batch = next(self.train_loader_iter)

        latents = batch["latent"]
        prompt_embed = batch["prompt_embed"]

        if self.action:
            w2c = batch['w2c']
            intrinsic = batch['intrinsic']
            action = batch['action']

            video_path = batch['video_path']
            image_cond = batch['image_cond']
            vision_states = batch['vision_states']
            prompt_mask = batch['prompt_mask']
            byt5_text_states = batch['byt5_text_states']
            byt5_text_mask = batch['byt5_text_mask']
            # add an indicator for memory training
            select_window_out_flag = batch['select_window_out_flag']
            i2v_mask = batch['i2v_mask']
        else:
            video_path = batch['path']

        training_batch.latents = latents.to(get_local_torch_device(),
                                            dtype=torch.bfloat16)
        training_batch.prompt_embed = prompt_embed.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        if self.action:
            training_batch.w2c = w2c.to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.intrinsic = intrinsic.to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.action = action.to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.video_path = video_path[0]

            training_batch.image_cond = image_cond.to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.vision_states = vision_states.to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.prompt_mask = prompt_mask.to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.byt5_text_states = byt5_text_states.to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.byt5_text_mask = byt5_text_mask.to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.select_window_out_flag = select_window_out_flag[0]
            training_batch.i2v_mask = i2v_mask.to(
                get_local_torch_device(), dtype=torch.bfloat16)    # i2v mask only works for memory training
        else:
            training_batch.video_path = video_path[0]

        return training_batch

    def _normalize_dit_input(self,
                             training_batch: TrainingBatch) -> TrainingBatch:
        # TODO(will): support other models
        training_batch.latents = normalize_dit_input('wan',
                                                     training_batch.latents,
                                                     self.get_module("vae"))
        return training_batch

    def timestep_transform(self, t, shift=1.0, num_timesteps=1000.0):
        t = t / num_timesteps
        t = shift * t / (1 + (shift - 1) * t)
        t = t * num_timesteps
        return t

    def _prepare_ar_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        latents = training_batch.latents
        batch_size = latents.shape[0]
        latent_t = latents.shape[2]
        latent_h = latents.shape[3]
        latent_w = latents.shape[4]
        noise = torch.randn(latents.shape,
                            generator=self.noise_gen_cuda,
                            device=latents.device,
                            dtype=latents.dtype)

        # add a parameter: chunk_latent_num means number of latent in one chunk
        chunk_latent_num = 4
        first_chunk_num = 4
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.training_args.weighting_scheme,
            batch_size=batch_size * ((latent_t - first_chunk_num) // chunk_latent_num + 1),
            generator=self.noise_random_generator,
            logit_mean=self.training_args.logit_mean,
            logit_std=self.training_args.logit_std,
            mode_scale=self.training_args.mode_scale,
        )
        u = u.reshape(batch_size, -1)
        if first_chunk_num == 1:
            first_u = u[:, :first_chunk_num]
            rest_u = u[:, first_chunk_num:]
            # Replicate the rest timesteps chunk_latent_num times
            rest_u = rest_u.unsqueeze(-1).repeat_interleave(chunk_latent_num, dim=-1).reshape(batch_size, -1)
            # Concatenate the first and rest sigmas
            u = torch.cat([first_u, rest_u], dim=1).reshape(-1)
        else:
            u = u.unsqueeze(-1).repeat_interleave(chunk_latent_num, dim=-1).reshape(batch_size, -1).reshape(-1)
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        indices = (self.noise_scheduler.config.num_train_timesteps - self.timestep_transform(indices, self.train_time_shift)).long()
        # TO DO: change the noise, for outside window, only add noise to the outside part
        if training_batch.select_window_out_flag == 1:    # select as memory training
            for i in range(0, indices.shape[0] - 4, 4):
                rand_val = torch.randint(500, 985, (1, ), device=latents.device)
                indices[i:i + 4] = rand_val
        
        timesteps = self.noise_scheduler.timesteps[indices].to(device=self.device)
        if self.training_args.sp_size > 1:
            # Make sure that the timesteps are the same across all sp processes.
            sp_group = get_sp_group()
            sp_group.broadcast(timesteps, src=0)

        sigmas = get_sigmas(
            self.noise_scheduler,
            latents.device,
            timesteps,
            n_dim=latents.ndim,
            dtype=latents.dtype,
        )
        sigmas = rearrange(sigmas, '(B D) C T H W -> B C (D T) H W', D=latent_t)
        noisy_model_input = (1.0 -
                             sigmas) * training_batch.latents + sigmas * noise
        training_batch.noisy_model_input = noisy_model_input
        training_batch.timesteps = timesteps
        training_batch.sigmas = sigmas
        training_batch.noise = noise
        training_batch.raw_latent_shape = training_batch.latents.shape

        return training_batch

    def _build_attention_metadata(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        latents_shape = training_batch.raw_latent_shape
        patch_size = self.training_args.pipeline_config.dit_config.patch_size
        current_vsa_sparsity = training_batch.current_vsa_sparsity
        assert latents_shape is not None
        assert training_batch.timesteps is not None
        if vsa_available and envs.TRAINER_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN":
            training_batch.attn_metadata = VideoSparseAttentionMetadataBuilder(  # type: ignore
            ).build(  # type: ignore
                raw_latent_shape=latents_shape[2:5],
                current_timestep=training_batch.timesteps,
                patch_size=patch_size,
                VSA_sparsity=current_vsa_sparsity,
                device=get_local_torch_device())
        else:
            training_batch.attn_metadata = None

        return training_batch

    def _build_rope_idx(self,
                        training_batch: TrainingBatch) -> TrainingBatch:
        rank_in_sp_group = get_sp_parallel_rank()
        per_sp_seq_length = training_batch.latents.shape[2] * training_batch.per_seq_length

        training_batch.current_start = per_sp_seq_length * rank_in_sp_group
        training_batch.current_end = per_sp_seq_length * rank_in_sp_group + per_sp_seq_length
        return training_batch

    def _prepare_cond_latents(self, task_type, cond_latents, latents, multitask_mask):
        """
        Prepare conditional latents and mask for multitask training.

        Args:
            task_type: Type of task ("i2v" or "t2v").
            cond_latents: Conditional latents tensor.
            latents: Main latents tensor.
            multitask_mask: Multitask mask tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - latents_concat: Concatenated conditional latents.
                - mask_concat: Concatenated mask tensor.
        """
        latents_concat = None
        mask_concat = None

        if cond_latents is not None and task_type == 'i2v':
            latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
            latents_concat[:, :, 1:, :, :] = 0.0
        else:
            latents_concat = torch.zeros(latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3],
                                         latents.shape[4]).to(latents.device)

        mask_zeros = torch.zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_ones = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_concat = merge_tensor_by_mask(mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2).to(
            device=latents.device)

        cond_latents = torch.concat([latents_concat, mask_concat], dim=1)

        return cond_latents

    def get_task_mask(self, task_type, latent_target_length):
        if task_type == "t2v":
            mask = torch.zeros(latent_target_length)
        elif task_type == "i2v":
            mask = torch.zeros(latent_target_length)
            mask[0] = 1.0
        else:
            raise ValueError(f"{task_type} is not supported !")
        return mask

    def _build_input_kwargs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        extra_kwargs = {
            "byt5_text_states": training_batch.byt5_text_states,
            "byt5_text_mask": training_batch.byt5_text_mask,
        }

        multitask_mask = self.get_task_mask("i2v", training_batch.noisy_model_input.shape[2]).to(self.device)
        cond_latents = self._prepare_cond_latents(
            "i2v", training_batch.image_cond, training_batch.noisy_model_input, multitask_mask
        )

        latents_concat = torch.concat([training_batch.noisy_model_input, cond_latents], dim=1)
        training_batch.input_kwargs = {
            "hidden_states":
            latents_concat,
            "timestep":
            training_batch.timesteps.to(get_local_torch_device(),
                                        dtype=torch.bfloat16),
            "timestep_txt": torch.tensor(0).unsqueeze(0).to(get_local_torch_device(),
                                        dtype=torch.bfloat16), # for ar model, we set txt timestep to 0
            "text_states":
                training_batch.prompt_embed,
            "text_states_2": None,
            "encoder_attention_mask": training_batch.prompt_mask,
            "timestep_r": None,
            "vision_states": training_batch.vision_states,
            "mask_type": "i2v",
            "guidance": None,
            "extra_kwargs": extra_kwargs,

            "viewmats": training_batch.w2c,
            "Ks": training_batch.intrinsic,
            "action": training_batch.action.reshape(-1),
            "return_dict": False,

        }
        return training_batch

    def _transformer_forward_and_compute_loss(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        if vsa_available and envs.TRAINER_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN":
            assert training_batch.attn_metadata is not None
        else:
            assert training_batch.attn_metadata is None
        input_kwargs = training_batch.input_kwargs

        with set_forward_context(
                current_timestep=training_batch.current_timestep,
                attn_metadata=training_batch.attn_metadata):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                model_pred = self.transformer(**input_kwargs)[0]

            if self.training_args.precondition_outputs:
                assert training_batch.sigmas is not None
                model_pred = training_batch.noisy_model_input - model_pred * training_batch.sigmas
            # Stash model output for optional visualization (detach to avoid autograd retention)
            if self._should_log_train_videos(training_batch.current_timestep):
                training_batch.model_pred = model_pred.detach()
            assert training_batch.latents is not None
            assert training_batch.noise is not None
            target = training_batch.latents if self.training_args.precondition_outputs else training_batch.noise - training_batch.latents
            i2v_mask = training_batch.i2v_mask
            if training_batch.select_window_out_flag == 1 and self.causal:
                i2v_mask[:,:,:-4,...] = 0 # only compute the last chunk for outside window training 
            assert model_pred.shape == target.shape, f"model_pred.shape: {model_pred.shape}, target.shape: {target.shape}"

            diff = (model_pred.float() * i2v_mask - target.float() * i2v_mask) ** 2
            loss = diff.sum() / max(i2v_mask.sum(), 1) / self.training_args.gradient_accumulation_steps

            loss.backward()
            avg_loss = loss.detach().clone()

        dist.all_reduce(avg_loss, op=dist.ReduceOp.MAX)
        training_batch.total_loss += avg_loss.item()

        return training_batch

    def _clip_grad_norm(self, training_batch: TrainingBatch) -> TrainingBatch:
        max_grad_norm = self.training_args.max_grad_norm

        # TODO(will): perhaps move this into transformer api so that we can do
        # the following:
        # grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        if max_grad_norm is not None:
            model_parts = [self.transformer]
            grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for m in model_parts for p in m.parameters()],
                max_grad_norm,
                foreach=None,
            )
            assert grad_norm is not float('nan') or grad_norm is not float(
                'inf')
            grad_norm = grad_norm.item() if grad_norm is not None else 0.0
        else:
            grad_norm = 0.0
        training_batch.grad_norm = grad_norm
        return training_batch

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        training_batch = self._prepare_training(training_batch)

        for _ in range(self.training_args.gradient_accumulation_steps):
            training_batch = self._get_next_batch(training_batch)

            training_batch = self._prepare_ar_dit_inputs(training_batch)

            training_batch = self._build_input_kwargs(training_batch)

            training_batch = self._transformer_forward_and_compute_loss(
                training_batch)

        training_batch = self._clip_grad_norm(training_batch)
        grad_norm = torch.tensor(training_batch.grad_norm).to(get_local_torch_device())
        dist.all_reduce(grad_norm, op=dist.ReduceOp.MAX)
        training_batch.grad_norm = grad_norm.item()

        if self.global_rank == 0 and training_batch.grad_norm >= 10.0:
            print(self.global_rank, training_batch.grad_norm, training_batch.current_timestep, training_batch.video_path)

        # NOTE:
        # `clip_grad_norm_...` returns the *pre-clipping* total norm. Even when max_grad_norm is small
        # (e.g. 1.0), this value can stay >10 and would cause us to skip optimizer.step() forever.
        # Rely on clipping for stability; only skip the step for non-finite norms.
        if not torch.isfinite(torch.tensor(training_batch.grad_norm)):
            if self.global_rank == 0:
                logger.warning(
                    "Non-finite grad_norm=%s at step=%s, skipping optimizer step.",
                    training_batch.grad_norm,
                    training_batch.current_timestep,
                )
        else:
            self.optimizer.step()
            self.lr_scheduler.step()

        training_batch.total_loss = training_batch.total_loss
        training_batch.grad_norm = training_batch.grad_norm
        return training_batch

    def _resume_from_checkpoint(self) -> None:
        logger.info("Loading checkpoint from %s",
                    self.training_args.resume_from_checkpoint)
        resumed_step = load_checkpoint(
            self.transformer, self.global_rank,
            self.training_args.resume_from_checkpoint, self.optimizer,
            self.train_dataloader, self.lr_scheduler,
            self.noise_random_generator)
        if resumed_step > 0:
            self.init_steps = resumed_step
            logger.info("Successfully resumed from step %s", resumed_step)
        else:
            logger.warning("Failed to load checkpoint, starting from step 0")
            self.init_steps = 0

    def train(self) -> None:
        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed + self.global_rank)
        logger.info('rank: %s: start training',
                    self.global_rank,
                    local_main_process_only=False)
        if not self.post_init_called:
            self.post_init()
        num_trainable_params = _get_trainable_params(self.transformer)
        logger.info("Starting training with %s B trainable parameters",
                    round(num_trainable_params / 1e9, 3))

        # Set random seeds for deterministic training
        self.noise_random_generator = torch.Generator(device="cpu").manual_seed(
            self.seed)
        self.noise_gen_cuda = torch.Generator(device="cuda").manual_seed(
            self.seed)
        self.validation_random_generator = torch.Generator(
            device="cpu").manual_seed(self.seed)
        logger.info("Initialized random seeds with seed: %s", self.seed)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler()

        if self.training_args.resume_from_checkpoint:
            self._resume_from_checkpoint()

        self.train_loader_iter = iter(self.train_dataloader)

        step_times: deque[float] = deque(maxlen=100)

        self._log_training_info()

        # Train!
        progress_bar = tqdm(
            range(0, self.training_args.max_train_steps),
            initial=self.init_steps,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=self.local_rank > 0,
        )

        for step in range(self.init_steps + 1,
                          self.training_args.max_train_steps + 1):

            self.train_dataset.update_max_frames(step)

            start_time = time.perf_counter()
            if vsa_available:
                vsa_sparsity = self.training_args.VSA_sparsity
                vsa_decay_rate = self.training_args.VSA_decay_rate
                vsa_decay_interval_steps = self.training_args.VSA_decay_interval_steps
                current_decay_times = min(step // vsa_decay_interval_steps,
                                          vsa_sparsity // vsa_decay_rate)
                current_vsa_sparsity = current_decay_times * vsa_decay_rate
            else:
                current_vsa_sparsity = 0.0

            training_batch = TrainingBatch()
            training_batch.current_timestep = step
            training_batch.current_vsa_sparsity = current_vsa_sparsity
            training_batch = self.train_one_step(training_batch)

            loss = training_batch.total_loss
            grad_norm = training_batch.grad_norm

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": grad_norm,
            })
            progress_bar.update(1)
            if self.global_rank == 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                        "vsa_sparsity": current_vsa_sparsity,
                    },
                    step=step,
                )
                # Optional: log decoded train-time videos (gt/noisy/pred)
                self._log_train_videos_to_wandb(training_batch, step)

            if step % self.training_args.checkpointing_steps == 0:
                save_checkpoint(self.transformer, self.global_rank,
                                self.training_args.output_dir, step,
                                self.optimizer, self.train_dataloader,
                                self.lr_scheduler, self.noise_random_generator)
                self.transformer.train()
                self.sp_group.barrier()

        wandb.finish()
        save_checkpoint(self.transformer, self.global_rank,
                        self.training_args.output_dir,
                        self.training_args.max_train_steps, self.optimizer,
                        self.train_dataloader, self.lr_scheduler,
                        self.noise_random_generator)

        if get_sp_group():
            cleanup_dist_env_and_memory()

    def _log_training_info(self) -> None:
        total_batch_size = (self.world_size *
                            self.training_args.gradient_accumulation_steps /
                            self.training_args.sp_size *
                            self.training_args.train_sp_batch_size)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %s", len(self.train_dataset))
        logger.info("  Dataloader size = %s", len(self.train_dataloader))
        logger.info("  Num Epochs = %s", self.num_train_epochs)
        logger.info("  Resume training from step %s",
                    self.init_steps)  # type: ignore
        logger.info("  Instantaneous batch size per device = %s",
                    self.training_args.train_batch_size)
        logger.info(
            "  Total train batch size (w. data & sequence parallel, accumulation) = %s",
            total_batch_size)
        logger.info("  Gradient Accumulation steps = %s",
                    self.training_args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %s",
                    self.training_args.max_train_steps)
        logger.info("  Total training parameters per FSDP shard = %s B",
                    round(_get_trainable_params(self.transformer) / 1e9, 3))
        # print dtype
        logger.info("  Master weight dtype: %s",
                    self.transformer.parameters().__next__().dtype)

        gpu_memory_usage = torch.cuda.memory_allocated() / 1024**2
        logger.info("GPU memory usage before train_one_step: %s MB",
                    gpu_memory_usage)
        logger.info("VSA validation sparsity: %s",
                    self.training_args.VSA_sparsity)

    def _prepare_validation_batch(self, sampling_param: SamplingParam,
                                  training_args: TrainingArgs,
                                  validation_batch: dict[str, Any],
                                  num_inference_steps: int) -> ForwardBatch:
        sampling_param.prompt = validation_batch['prompt']
        sampling_param.height = training_args.num_height
        sampling_param.width = training_args.num_width
        sampling_param.num_inference_steps = num_inference_steps
        sampling_param.data_type = "video"
        assert self.seed is not None
        sampling_param.seed = self.seed

        latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                        sampling_param.height // 8, sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]
        temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = (training_args.num_latent_t -
                      1) * temporal_compression_factor + 1
        sampling_param.num_frames = num_frames
        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            latents=None,
            generator=self.validation_random_generator,
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
        )

        return batch

    @torch.no_grad()
    def _log_validation(self, transformer, training_args, global_step) -> None:
        """
        Generate a validation video and log it to wandb to check the quality during training.
        """
        training_args.inference_mode = True
        training_args.dit_cpu_offload = True
        if not training_args.log_validation:
            return
        if self.validation_pipeline is None:
            raise ValueError("Validation pipeline is not set")

        logger.info("Starting validation")

        # Create sampling parameters if not provided
        sampling_param = SamplingParam.from_pretrained(training_args.model_path)

        # Prepare validation prompts
        logger.info('rank: %s: trainer_args.validation_dataset_file: %s',
                    self.global_rank,
                    training_args.validation_dataset_file,
                    local_main_process_only=False)
        validation_dataset = ValidationDataset(
            training_args.validation_dataset_file)
        validation_dataloader = DataLoader(validation_dataset,
                                           batch_size=None,
                                           num_workers=0)

        transformer.eval()

        validation_steps = training_args.validation_sampling_steps.split(",")
        validation_steps = [int(step) for step in validation_steps]
        validation_steps = [step for step in validation_steps if step > 0]
        # Log validation results for this step
        world_group = get_world_group()
        num_sp_groups = world_group.world_size // self.sp_group.world_size

        # Process each validation prompt for each validation step
        for num_inference_steps in validation_steps:
            logger.info("rank: %s: num_inference_steps: %s",
                        self.global_rank,
                        num_inference_steps,
                        local_main_process_only=False)
            step_videos: list[np.ndarray] = []
            step_captions: list[str] = []

            for validation_batch in validation_dataloader:
                batch = self._prepare_validation_batch(sampling_param,
                                                       training_args,
                                                       validation_batch,
                                                       num_inference_steps)
                logger.info("rank: %s: rank_in_sp_group: %s, batch.prompt: %s",
                            self.global_rank,
                            self.rank_in_sp_group,
                            batch.prompt,
                            local_main_process_only=False)

                assert batch.prompt is not None and isinstance(
                    batch.prompt, str)
                step_captions.append(batch.prompt)

                # Run validation inference
                output_batch = self.validation_pipeline.forward(
                    batch, training_args)
                samples = output_batch.output

                if self.rank_in_sp_group != 0:
                    continue

                # Process outputs
                video = rearrange(samples, "b c t h w -> t b c h w")
                frames = []
                for x in video:
                    x = torchvision.utils.make_grid(x, nrow=6)
                    x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    frames.append((x * 255).numpy().astype(np.uint8))
                step_videos.append(frames)

            # Only sp_group leaders (rank_in_sp_group == 0) need to send their
            # results to global rank 0
            if self.rank_in_sp_group == 0:
                if self.global_rank == 0:
                    # Global rank 0 collects results from all sp_group leaders
                    all_videos = step_videos  # Start with own results
                    all_captions = step_captions

                    # Receive from other sp_group leaders
                    for sp_group_idx in range(1, num_sp_groups):
                        src_rank = sp_group_idx * self.sp_world_size  # Global rank of other sp_group leaders
                        recv_videos = world_group.recv_object(src=src_rank)
                        recv_captions = world_group.recv_object(src=src_rank)
                        all_videos.extend(recv_videos)
                        all_captions.extend(recv_captions)

                    video_filenames = []
                    for i, (video, caption) in enumerate(
                            zip(all_videos, all_captions, strict=True)):
                        os.makedirs(training_args.output_dir, exist_ok=True)
                        filename = os.path.join(
                            training_args.output_dir,
                            f"validation_step_{global_step}_inference_steps_{num_inference_steps}_video_{i}.mp4"
                        )
                        imageio.mimsave(filename, video, fps=sampling_param.fps)
                        video_filenames.append(filename)

                    logs = {
                        f"validation_videos_{num_inference_steps}_steps": [
                            wandb.Video(filename, caption=caption)
                            for filename, caption in zip(
                                video_filenames, all_captions, strict=True)
                        ]
                    }
                    wandb.log(logs, step=global_step)
                else:
                    # Other sp_group leaders send their results to global rank 0
                    world_group.send_object(step_videos, dst=0)
                    world_group.send_object(step_captions, dst=0)

        # Re-enable gradients for training
        training_args.inference_mode = False
        transformer.train()
