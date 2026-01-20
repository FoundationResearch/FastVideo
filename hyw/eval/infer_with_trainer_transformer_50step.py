from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import imageio
import numpy as np
import torch

from hyw.eval.path_utils import find_repo_root, resolve_data_path


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_video_rgb(path: Path) -> np.ndarray:
    """Returns (T,H,W,3) uint8."""
    reader = imageio.get_reader(str(path))
    frames = []
    for frame in reader:
        frames.append(frame[:, :, :3])
    reader.close()
    return np.stack(frames, axis=0).astype(np.uint8)


def _write_video_rgb(frames: np.ndarray, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps, format="mp4")


def _pose_frame_json_to_viewmats_Ks(pose_json: Dict, latent_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert per-frame pose.json to latent-aligned (w2c, K).
    Sample at frame indices: 0,4,8,... (latent boundaries).
    """
    viewmats = []
    Ks = []
    for i in range(latent_num):
        frame_idx = i * 4
        entry = pose_json[str(frame_idx)]
        w2c = np.array(entry["w2c"], dtype=np.float32)
        K = np.array(entry["K"], dtype=np.float32)

        # Normalize K (match HY pipeline conventions)
        K_norm = K.copy()
        K_norm[0, 0] /= (K_norm[0, 2] * 2.0)
        K_norm[1, 1] /= (K_norm[1, 2] * 2.0)
        K_norm[0, 2] = 0.5
        K_norm[1, 2] = 0.5

        viewmats.append(w2c)
        Ks.append(K_norm)
    return torch.tensor(np.stack(viewmats, axis=0)), torch.tensor(np.stack(Ks, axis=0))


def _frame_action_to_labels(action_json: Dict, latent_num: int) -> torch.Tensor:
    """Frame-level action.json -> per-latent discrete labels (0..80)."""
    mapping = {
        (0, 0, 0, 0): 0,
        (1, 0, 0, 0): 1,
        (0, 1, 0, 0): 2,
        (0, 0, 1, 0): 3,
        (0, 0, 0, 1): 4,
        (1, 0, 1, 0): 5,
        (1, 0, 0, 1): 6,
        (0, 1, 1, 0): 7,
        (0, 1, 0, 1): 8,
    }
    move_map = {"": (0, 0, 0, 0), "W": (1, 0, 0, 0), "S": (0, 1, 0, 0), "D": (0, 0, 1, 0), "A": (0, 0, 0, 1)}
    rot_map = {"": (0, 0, 0, 0), "LR": (1, 0, 0, 0), "LL": (0, 1, 0, 0), "LU": (0, 0, 1, 0), "LD": (0, 0, 0, 1)}

    labels = [0]
    for i in range(1, latent_num):
        frame_idx = i * 4
        entry = action_json.get(str(frame_idx), {"move_action": "", "view_action": ""})
        move = entry.get("move_action", "")
        view = entry.get("view_action", "")
        trans_label = mapping[move_map.get(move, (0, 0, 0, 0))]
        rot_label = mapping[rot_map.get(view, (0, 0, 0, 0))]
        labels.append(trans_label * 9 + rot_label)
    return torch.tensor(labels, dtype=torch.float32)


def _load_latent_pt(path: Path) -> Dict[str, torch.Tensor]:
    # `weights_only` is only available in newer torch; keep backward-compatible.
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    out: Dict[str, torch.Tensor] = {}
    out["latents"] = payload["latent"]
    out["prompt_embeds"] = payload["prompt_embeds"]
    out["prompt_mask"] = payload["prompt_mask"]
    out["image_cond"] = payload["image_cond"]
    out["vision_states"] = payload["vision_states"]
    out["byt5_text_states"] = payload["byt5_text_states"]
    out["byt5_text_mask"] = payload["byt5_text_mask"]
    return out


def _decode_video_latents_to_uint8(vae, latents: torch.Tensor) -> np.ndarray:
    """
    latents: (1,C,T,H,W) -> (T,H',W',3) uint8
    """
    cfg = getattr(vae, "config", None)
    scaling = getattr(cfg, "scaling_factor", None)
    if scaling is None:
        scaling = getattr(vae, "scaling_factor", None)
    if scaling is None:
        raise RuntimeError("VAE scaling_factor not found")
    scaling = float(scaling)

    shift = getattr(cfg, "shift_factor", None)
    shift = float(shift) if shift is not None else None

    # Match dtype/device expected by VAE weights to avoid conv dtype mismatches.
    try:
        vae_param = next(vae.parameters())
        vae_dtype = vae_param.dtype
        vae_device = vae_param.device
    except StopIteration:
        vae_dtype = torch.float16
        vae_device = latents.device

    z = latents.to(device=vae_device, dtype=torch.float32) / scaling
    if shift is not None:
        z = z + shift
    z = z.to(dtype=vae_dtype)

    with torch.no_grad():
        # VAE is typically fp16; enable autocast for stability/perf.
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=(vae_device.type == "cuda" and vae_dtype in (torch.float16, torch.bfloat16)))
            if hasattr(torch, "autocast")
            else None
        )
        if autocast_ctx is None:
            dec = vae.decode(z)
        else:
            with autocast_ctx:
                dec = vae.decode(z)
        frames = dec.sample if hasattr(dec, "sample") else dec
    # (1,3,T,H,W) -> (T,H,W,3)
    frames = frames.detach().float().cpu()
    frames = (frames / 2.0 + 0.5).clamp(0, 1)
    frames = (frames[0].permute(1, 2, 3, 0).numpy() * 255.0).round().astype(np.uint8)
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description="50-step denoising using HY-WorldPlay trainer transformer (diagnostic).")
    p.add_argument("--train_json", type=str, required=True)
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--action_ckpt", type=str, required=True, help="Action safetensors ckpt used to init action params.")
    p.add_argument("--finetuned_ckpt", type=str, default=None, help="Optional ckpt dir with transformer/diffusion_pytorch_model.safetensors")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--flow_shift", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = p.parse_args()

    repo_root = find_repo_root(Path(__file__))
    hyworld_root = (repo_root / "hyw" / "HY-WorldPlay-main").resolve()
    if str(hyworld_root) not in os.sys.path:
        os.sys.path.insert(0, str(hyworld_root))

    device = torch.device(args.device)
    compute_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # Initialize HY-WorldPlay distributed + model-parallel globals for trainer transformer.
    if device.type == "cuda":
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29500 + (os.getpid() % 1000)))
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        from trainer.distributed.parallel_state import (  # type: ignore
            maybe_init_distributed_environment_and_model_parallel,
            model_parallel_is_initialized,
        )

        if not model_parallel_is_initialized():
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1, distributed_init_method="env://")

    items = json.loads(Path(args.train_json).expanduser().read_text(encoding="utf-8"))
    item = items[int(args.sample_idx)]

    latent_pt_path = resolve_data_path(item["latent_path"], repo_root)
    pose_path = resolve_data_path(item["pose_path"], repo_root)
    action_path = resolve_data_path(item["action_path"], repo_root)
    video_path = resolve_data_path(item.get("video_path", ""), repo_root) if "video_path" in item else None

    latent_pt = _load_latent_pt(latent_pt_path)
    x0 = latent_pt["latents"]  # (1,C,T,H,W)
    B, C, T, H, W = x0.shape

    pose_json = _load_json(pose_path)
    action_json = _load_json(action_path)
    viewmats, Ks = _pose_frame_json_to_viewmats_Ks(pose_json, T)
    action_labels = _frame_action_to_labels(action_json, T)

    # Text / vision cond
    prompt_embeds = latent_pt["prompt_embeds"].to(device=device, dtype=torch.float32)
    prompt_mask = latent_pt["prompt_mask"].to(device=device)
    vision_states = latent_pt["vision_states"].to(device=device, dtype=torch.float32)
    byt5_text_states = latent_pt["byt5_text_states"].to(device=device, dtype=torch.float32)
    byt5_text_mask = latent_pt["byt5_text_mask"].to(device=device)
    extra_kwargs = {"byt5_text_states": byt5_text_states, "byt5_text_mask": byt5_text_mask}

    viewmats = viewmats.unsqueeze(0).to(device=device, dtype=torch.float32)
    Ks = Ks.unsqueeze(0).to(device=device, dtype=torch.float32)
    action_vec = action_labels.reshape(-1).to(device=device, dtype=torch.float32)

    # Prepare i2v cond_latents = [cond_latents_repeat (C), mask (1)]
    image_cond = latent_pt["image_cond"].to(device=device, dtype=torch.float32)  # (1,C,1,H,W)
    cond_latents = image_cond.repeat(1, 1, T, 1, 1)
    # Match HY pipeline: only frame-0 is conditioned; others are zero.
    if T > 1:
        cond_latents[:, :, 1:, :, :] = 0.0
    mask = torch.zeros((B, 1, T, H, W), device=device, dtype=torch.float32)
    mask[:, :, 0:1, :, :] = 1.0
    cond_latents_plus_mask = torch.cat([cond_latents, mask], dim=1)  # (1,C+1,T,H,W)

    # Load trainer transformer
    base_transformer_dir = str(Path(args.model_path) / "transformer" / "480p_i2v")
    from trainer.models.hyvideo.models.transformers.ar_action_hunyuanvideo_1_5_transformer import (  # type: ignore
        ARHunyuanVideo_1_5_DiffusionTransformer as TrainerTransformer,
    )
    from safetensors.torch import load_file

    trainer_model = TrainerTransformer.from_pretrained(base_transformer_dir, local_attn_size=-1, sink_size=0)
    trainer_model.add_discrete_action_parameters()
    trainer_model.load_state_dict(load_file(args.action_ckpt), strict=True)

    if args.finetuned_ckpt:
        ft_path = Path(args.finetuned_ckpt) / "transformer" / "diffusion_pytorch_model.safetensors"
        trainer_model.load_state_dict(load_file(str(ft_path)), strict=False)

    trainer_model = trainer_model.to(device=device, dtype=torch.float32).eval()

    # Scheduler (match HY pipeline)
    from hyvideo.schedulers.scheduling_flow_match_discrete import (  # type: ignore
        FlowMatchDiscreteScheduler,
    )

    scheduler = FlowMatchDiscreteScheduler(shift=float(args.flow_shift), reverse=True, solver="euler")
    scheduler.set_timesteps(int(args.num_inference_steps), device=device)

    # Init latents (noise) in fp32, compute under autocast
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    latents = torch.randn((B, C, T, H, W), device=device, dtype=torch.float32)

    timesteps = scheduler.timesteps
    timestep_txt = torch.tensor(0.0, device=device, dtype=torch.float32).unsqueeze(0)

    autocast_enabled = (device.type == "cuda") and (compute_dtype in (torch.bfloat16, torch.float16))
    autocast_ctx = torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=autocast_enabled)

    try:
        with torch.no_grad(), autocast_ctx:
            for t in timesteps:
                t_in = torch.full((T,), t, device=device, dtype=timesteps.dtype)
                latents_concat = torch.cat([latents, cond_latents_plus_mask], dim=1)  # (1,2C+1,T,H,W)
                noise_pred = trainer_model(
                    hidden_states=latents_concat,
                    timestep=t_in,
                    timestep_txt=timestep_txt,
                    text_states=prompt_embeds,
                    text_states_2=None,
                    encoder_attention_mask=prompt_mask,
                    timestep_r=None,
                    vision_states=vision_states,
                    return_dict=False,
                    guidance=None,
                    mask_type="i2v",
                    extra_kwargs=extra_kwargs,
                    action=action_vec,
                    viewmats=viewmats.to(dtype=compute_dtype if autocast_enabled else torch.float32),
                    Ks=Ks.to(dtype=compute_dtype if autocast_enabled else torch.float32),
                )[0]
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    # Decode and write videos
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from hyvideo.models.autoencoders.hunyuanvideo_15_vae_w_cache import (  # type: ignore
        AutoencoderKLConv3D,
    )

    vae = AutoencoderKLConv3D.from_pretrained(str(Path(args.model_path) / "vae"), torch_dtype=torch.float16)
    vae = vae.to(device=device, dtype=torch.float16).eval()

    gt_frames = _decode_video_latents_to_uint8(vae, x0.to(device=device, dtype=torch.float32))
    pred_frames = _decode_video_latents_to_uint8(vae, latents.to(device=device, dtype=torch.float32))

    # Optional: load raw rgb video for reference if present
    raw_frames = None
    if video_path is not None and str(video_path) != "":
        try:
            raw_frames = _read_video_rgb(Path(video_path))
        except Exception:
            raw_frames = None

    # Side-by-side: GT | pred
    h = min(gt_frames.shape[1], pred_frames.shape[1])
    w = min(gt_frames.shape[2], pred_frames.shape[2])
    gt_crop = gt_frames[:, :h, :w]
    pred_crop = pred_frames[:, :h, :w]
    side = np.concatenate([gt_crop, pred_crop], axis=2)

    _write_video_rgb(gt_frames, out_dir / "gt_from_latent.mp4", fps=int(args.fps))
    _write_video_rgb(pred_frames, out_dir / "pred_trainer_50step.mp4", fps=int(args.fps))
    _write_video_rgb(side, out_dir / "side_by_side_gt_vs_trainer_pred.mp4", fps=int(args.fps))
    if raw_frames is not None:
        _write_video_rgb(raw_frames, out_dir / "raw_video.mp4", fps=int(args.fps))

    print(f"[DONE] wrote to: {out_dir}")


if __name__ == "__main__":
    main()


