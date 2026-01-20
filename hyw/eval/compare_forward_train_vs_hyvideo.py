from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from hyw.eval.path_utils import find_repo_root, resolve_data_path


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pose_frame_json_to_viewmats_Ks(pose_json: Dict, latent_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert our per-frame pose.json to latent-aligned (w2c, K) lists.
    Sample at frame indices: 0,4,8,... (latent boundaries). Normalize K to match hyvideo generate.
    """
    viewmats = []
    Ks = []
    for i in range(latent_num):
        frame_idx = i * 4
        entry = pose_json[str(frame_idx)]
        w2c = np.array(entry["w2c"], dtype=np.float32)
        K = np.array(entry["K"], dtype=np.float32)

        K_norm = K.copy()
        K_norm[0, 0] /= (K_norm[0, 2] * 2.0)
        K_norm[1, 1] /= (K_norm[1, 2] * 2.0)
        K_norm[0, 2] = 0.5
        K_norm[1, 2] = 0.5

        viewmats.append(w2c)
        Ks.append(K_norm)
    return torch.tensor(np.stack(viewmats, axis=0)), torch.tensor(np.stack(Ks, axis=0))


def _frame_action_to_labels(action_json: Dict, latent_num: int) -> torch.Tensor:
    """
    Convert our frame-level action.json to HY discrete labels per latent (0..80).
    We sample actions at 0,4,8,... ; action[0] is forced to 0.
    """
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
    # normalize key naming a bit
    out: Dict[str, torch.Tensor] = {}
    out["latents"] = payload["latent"]
    out["prompt_embeds"] = payload["prompt_embeds"]
    out["prompt_mask"] = payload["prompt_mask"]
    out["image_cond"] = payload["image_cond"]
    out["vision_states"] = payload["vision_states"]
    out["byt5_text_states"] = payload["byt5_text_states"]
    out["byt5_text_mask"] = payload["byt5_text_mask"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Compare trainer-transformer vs hyvideo-transformer forward_bi outputs.")
    p.add_argument("--train_json", type=str, required=True, help="HY-WorldPlay training json (latent_path/pose_path/action_path).")
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--model_path", type=str, required=True, help="Base model root (e.g. HunyuanVideo-1.5).")
    p.add_argument("--action_ckpt", type=str, required=True, help="Action safetensors ckpt (for action params init).")
    p.add_argument("--finetuned_ckpt", type=str, default=None, help="Optional checkpoint dir with transformer/diffusion_pytorch_model.safetensors")
    p.add_argument("--t_value", type=float, default=500.0, help="Scalar timestep value fed to both models (applied to all latent frames).")
    p.add_argument("--sigma", type=float, default=0.5, help="Noise mixing sigma for constructing x_t = (1-s)*x0 + s*eps.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Compute dtype for both models/inputs. auto=bf16 on cuda else fp32.",
    )
    args = p.parse_args()

    repo_root = find_repo_root(Path(__file__))
    hyworld_root = (repo_root / "hyw" / "HY-WorldPlay-main").resolve()
    if str(hyworld_root) not in os.sys.path:
        os.sys.path.insert(0, str(hyworld_root))

    device = torch.device(args.device)
    # In training, HY-WorldPlay often uses `--dit-precision fp32` with AMP for compute.
    # For a robust forward equivalence diagnostic, keep *weights* in fp32 and optionally
    # use autocast for compute (bf16/fp16) on CUDA.
    param_dtype = torch.float32
    if args.dtype == "auto":
        compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    elif args.dtype == "bf16":
        compute_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    # Trainer-side transformer depends on HY-WorldPlay distributed + model-parallel globals
    # (e.g. get_sp_world_size()), which are normally initialized under torchrun.
    # For this standalone diagnostic, initialize a single-process (world_size=1, tp=1, sp=1) setup.
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

    latent_pt = _load_latent_pt(latent_pt_path)
    x0 = latent_pt["latents"].to(device=device, dtype=param_dtype)  # (1,C,T,H,W)
    B, _, T, _, _ = x0.shape

    # Build conditioning tensors
    prompt_embeds = latent_pt["prompt_embeds"].to(device=device, dtype=param_dtype)
    prompt_mask = latent_pt["prompt_mask"].to(device=device)
    image_cond = latent_pt["image_cond"].to(device=device, dtype=param_dtype)  # (1,C,1,H,W)
    vision_states = latent_pt["vision_states"].to(device=device, dtype=param_dtype)
    byt5_text_states = latent_pt["byt5_text_states"].to(device=device, dtype=param_dtype)
    byt5_text_mask = latent_pt["byt5_text_mask"].to(device=device)

    pose_json = _load_json(pose_path)
    action_json = _load_json(action_path)
    viewmats, Ks = _pose_frame_json_to_viewmats_Ks(pose_json, T)
    action_labels = _frame_action_to_labels(action_json, T)
    viewmats = viewmats.unsqueeze(0).to(device=device, dtype=param_dtype)  # (1,T,4,4)
    Ks = Ks.unsqueeze(0).to(device=device, dtype=param_dtype)
    action_vec = action_labels.reshape(-1).to(device=device, dtype=param_dtype)  # (T,)

    # Construct x_t for a controlled comparison
    # Some torch versions do not support `generator=` for randn_like; rely on global RNG for compatibility.
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    eps = torch.randn(x0.shape, device=device, dtype=torch.float32).to(dtype=param_dtype)
    sigma = float(args.sigma)
    x_t = (1.0 - sigma) * x0 + sigma * eps
    cond_latents = image_cond.repeat(1, 1, x0.shape[2], 1, 1)
    # Trainer's PatchEmbed in concat_condition mode expects: [noisy_latents, cond_latents, cond_mask] -> (B, 2*C+1, T, H, W)
    # Where C is VAE latent channels (e.g. 32 for HunyuanVideo-1.5 480p).
    cond_mask = torch.ones((B, 1, x0.shape[2], x0.shape[3], x0.shape[4]), device=device, dtype=param_dtype)
    hidden_states = torch.cat([x_t, cond_latents, cond_mask], dim=1)

    timestep = torch.full((B * T,), float(args.t_value), device=device, dtype=param_dtype)
    timestep_txt = torch.tensor(0, device=device, dtype=param_dtype).unsqueeze(0)

    extra_kwargs = {"byt5_text_states": byt5_text_states, "byt5_text_mask": byt5_text_mask}

    # --- Load models ---
    base_transformer_dir = str(Path(args.model_path) / "transformer" / "480p_i2v")

    # trainer transformer implementation (the one used during training)
    from trainer.models.hyvideo.models.transformers.ar_action_hunyuanvideo_1_5_transformer import (  # type: ignore
        ARHunyuanVideo_1_5_DiffusionTransformer as TrainerTransformer,
    )
    trainer_model = TrainerTransformer.from_pretrained(base_transformer_dir, local_attn_size=-1, sink_size=0)
    trainer_model.add_discrete_action_parameters()
    from safetensors.torch import load_file

    trainer_model.load_state_dict(load_file(args.action_ckpt), strict=True)

    # hyvideo transformer implementation (the one used in pipeline/eval)
    from hyvideo.models.transformers.worldplay_1_5_transformer import (  # type: ignore
        HunyuanVideo_1_5_DiffusionTransformer as HyvideoTransformer,
    )
    hyvideo_model = HyvideoTransformer.from_pretrained(base_transformer_dir, torch_dtype=param_dtype, low_cpu_mem_usage=True)
    hyvideo_model.add_action_parameters()
    hyvideo_model.load_state_dict(load_file(args.action_ckpt), strict=True)

    # Optional finetuned weights
    if args.finetuned_ckpt:
        ft_path = Path(args.finetuned_ckpt) / "transformer" / "diffusion_pytorch_model.safetensors"
        ft_sd = load_file(str(ft_path))
        m1 = trainer_model.load_state_dict(ft_sd, strict=False)
        m2 = hyvideo_model.load_state_dict(ft_sd, strict=False)
        print(f"[trainer] missing={len(m1.missing_keys)} unexpected={len(m1.unexpected_keys)}")
        print(f"[hyvideo] missing={len(m2.missing_keys)} unexpected={len(m2.unexpected_keys)}")

    # Keep weights in fp32 for robustness; use autocast for compute if requested.
    trainer_model = trainer_model.to(device=device, dtype=param_dtype).eval()
    hyvideo_model = hyvideo_model.to(device=device, dtype=param_dtype).eval()

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=compute_dtype)
        if device.type == "cuda" and compute_dtype in (torch.bfloat16, torch.float16)
        else torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False)
    )

    try:
        with torch.no_grad(), autocast_ctx:
            out_tr = trainer_model(
                hidden_states=hidden_states,
                timestep=timestep,
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
                viewmats=viewmats,
                Ks=Ks,
            )[0]

            out_hv = hyvideo_model.forward_bi(
                hidden_states=hidden_states,
                timestep=timestep,
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
                viewmats=viewmats,
                Ks=Ks,
            )[0]
    finally:
        # Avoid ProcessGroup warnings/leaks in one-off diagnostics.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    diff = (out_tr - out_hv).abs()
    print(f"[OUT] shape={tuple(out_tr.shape)} dtype={out_tr.dtype}")
    print(f"[DIFF] max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g} rms={(diff.float().pow(2).mean().sqrt().item()):.6g}")


if __name__ == "__main__":
    main()


