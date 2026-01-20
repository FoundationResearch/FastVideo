from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

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


def _tensor_stats(x: torch.Tensor) -> str:
    xf = x.detach().float()
    return f"shape={tuple(x.shape)} dtype={x.dtype} min={xf.min().item():.6g} mean={xf.mean().item():.6g} max={xf.max().item():.6g} std={xf.std().item():.6g}"


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> str:
    d = (a - b).detach().float()
    return f"max_abs={d.abs().max().item():.6g} mean_abs={d.abs().mean().item():.6g} rms={(d.pow(2).mean().sqrt().item()):.6g}"


def _print_stage(name: str, a: Optional[torch.Tensor], b: Optional[torch.Tensor], threshold: float) -> bool:
    """
    Returns True if this stage diverges beyond threshold (max_abs).
    """
    print(f"\n=== {name} ===")
    if a is None or b is None:
        print(f"[WARN] one side is None (a={a is None}, b={b is None})")
        return True
    print(f"[trainer] {_tensor_stats(a)}")
    print(f"[hyvideo] {_tensor_stats(b)}")
    if a.shape != b.shape:
        print("[DIFF] shape mismatch -> treat as divergence")
        return True
    ds = (a - b).detach().float()
    max_abs = ds.abs().max().item()
    print(f"[DIFF] {_diff_stats(a, b)}")
    return max_abs > threshold


def main() -> None:
    p = argparse.ArgumentParser(description="Step-by-step compare trainer-transformer forward vs hyvideo forward_bi.")
    p.add_argument("--train_json", type=str, required=True, help="HY-WorldPlay training json (latent_path/pose_path/action_path).")
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--model_path", type=str, required=True, help="Base model root (e.g. HunyuanVideo-1.5).")
    p.add_argument("--action_ckpt", type=str, required=True, help="Action safetensors ckpt (for action params init).")
    p.add_argument("--finetuned_ckpt", type=str, default=None, help="Optional checkpoint dir with transformer/diffusion_pytorch_model.safetensors")
    p.add_argument("--t_value", type=float, default=500.0, help="Scalar timestep value fed to both models (applied to all latent frames).")
    p.add_argument("--sigma", type=float, default=0.5, help="Noise mixing sigma for constructing x_t = (1-s)*x0 + s*eps.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--diff_threshold", type=float, default=1e-4, help="Stop when max_abs diff exceeds this.")
    p.add_argument("--max_blocks", type=int, default=4, help="How many double_blocks to step through before stopping.")
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

    # timestep shapes:
    # - trainer forward expects timestep shape aligns with SP chunking (they later chunk t along dim=0)
    # - hyvideo forward_bi expects timestep shape (B*T,) then broadcasts to per-token internally
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
            # --- Stage 0: inputs ---
            _print_stage("hidden_states input", hidden_states, hidden_states, args.diff_threshold)

            # --- Stage 1: patch embed ---
            img_tr = trainer_model.img_in(hidden_states)
            img_hv = hyvideo_model.img_in(hidden_states)
            diverged = _print_stage("img_in (PatchEmbed output)", img_tr, img_hv, args.diff_threshold)
            if diverged:
                print("[STOP] diverged at img_in")
                return

            # --- Stage 2: time embeddings (pre-broadcast) ---
            # trainer uses vec_txt/time_in for text separately
            vec_txt_tr = trainer_model.time_in(timestep_txt)
            vec_txt_hv = hyvideo_model.time_in(timestep_txt)
            diverged = _print_stage("vec_txt = time_in(timestep_txt)", vec_txt_tr, vec_txt_hv, args.diff_threshold)
            if diverged:
                print("[STOP] diverged at vec_txt")
                return

            vec_tr = trainer_model.time_in(timestep)
            vec_hv = hyvideo_model.time_in(timestep)
            diverged = _print_stage("vec = time_in(timestep)", vec_tr, vec_hv, args.diff_threshold)
            if diverged:
                print("[STOP] diverged at vec")
                return

            # --- Stage 3: action injection ---
            vec_tr2 = vec_tr + trainer_model.action_in(action_vec)
            vec_hv2 = vec_hv + hyvideo_model.action_in(action_vec)
            diverged = _print_stage("vec += action_in(action)", vec_tr2, vec_hv2, args.diff_threshold)
            if diverged:
                print("[STOP] diverged at action_in")
                return

            # --- Stage 4: text projection/refiner + reorder (ByT5 + vision) ---
            # We run both models' get_text_and_mask pipelines as they are implemented.
            txt_tr = prompt_embeds
            mask_tr = prompt_mask
            if trainer_model.text_projection == "linear":
                txt_tr = trainer_model.txt_in(txt_tr)
            else:
                txt_tr = trainer_model.txt_in(txt_tr, timestep_txt, mask_tr if trainer_model.use_attention_mask else None)
            txt_hv, mask_hv, _ = hyvideo_model.get_text_and_mask(
                encoder_attention_mask=prompt_mask,
                text_states=prompt_embeds,
                timestep_txt=timestep_txt,
                extra_kwargs=extra_kwargs,
                vision_states=vision_states,
                mask_type="i2v",
            )
            diverged = _print_stage("txt after txt_in (trainer manual vs hyvideo get_text_and_mask)", txt_tr, txt_hv, args.diff_threshold)
            if diverged:
                print("[STOP] diverged at txt_in/get_text_and_mask")
                return

            # trainer continues to apply byt5/vision reorder in forward; replicate minimally via get_text_and_mask-equivalent pieces
            if trainer_model.glyph_byT5_v2:
                byt5_txt_tr = trainer_model.byt5_in(extra_kwargs["byt5_text_states"])
                txt_tr, mask_tr = trainer_model.reorder_txt_token(byt5_txt_tr, txt_tr, extra_kwargs["byt5_text_mask"], mask_tr, zero_feat=True)
            if trainer_model.vision_in is not None and vision_states is not None:
                extra_enc_tr = trainer_model.vision_in(vision_states)
                extra_attn_mask_tr = torch.ones((extra_enc_tr.shape[0], extra_enc_tr.shape[1]), dtype=mask_tr.dtype, device=mask_tr.device)
                txt_tr, mask_tr = trainer_model.reorder_txt_token(extra_enc_tr, txt_tr, extra_attn_mask_tr, mask_tr)

            # Compare with hyvideo get_text_and_mask output already includes those merges
            diverged = _print_stage("txt after (ByT5+vision) reorder", txt_tr, txt_hv, args.diff_threshold)
            if diverged:
                print("[STOP] diverged at reorder_txt_token")
                return

            # --- Stage 5: masking strategy before blocks differs between the two implementations ---
            # trainer: masks out padding tokens and passes text_mask=None into blocks.
            txt_tr_masked = txt_tr[mask_tr.bool().to(txt_tr.device)].unsqueeze(0)
            # hyvideo forward_bi: does NOT pre-mask txt; it passes text_mask into block(...) for masking.
            txt_hv_unmasked = txt_hv
            print("\n=== txt masking strategy (this is a known divergence point) ===")
            print(f"[trainer] txt masked: {_tensor_stats(txt_tr_masked)}")
            print(f"[hyvideo] txt unmasked: {_tensor_stats(txt_hv_unmasked)}")
            print("[NOTE] shapes differ here by design (trainer pre-masks; hyvideo keeps padding + provides text_mask).")

            # --- Stage 6: run a few double_blocks and compare outputs ---
            # Prepare freqs (both use same helper)
            tt = hidden_states.shape[2] // trainer_model.patch_size[0]
            th = hidden_states.shape[3] // trainer_model.patch_size[1]
            tw = hidden_states.shape[4] // trainer_model.patch_size[2]
            freqs_cos_tr, freqs_sin_tr = trainer_model.get_rotary_pos_embed((tt, th, tw))
            freqs_cos_hv, freqs_sin_hv = hyvideo_model.get_rotary_pos_embed((tt, th, tw))
            _print_stage("freqs_cos", freqs_cos_tr, freqs_cos_hv, args.diff_threshold)
            _print_stage("freqs_sin", freqs_sin_tr, freqs_sin_hv, args.diff_threshold)

            freqs_cis_tr = (freqs_cos_tr, freqs_sin_tr)
            freqs_cis_hv = (freqs_cos_hv, freqs_sin_hv)

            # hyvideo forward_bi broadcasts vec to token length before blocks; trainer does not.
            # We compare the *inputs into block 0* as they are actually used:
            img0_tr = img_tr
            txt0_tr = txt_tr_masked
            vec0_tr = vec_tr2
            vec_txt0_tr = vec_txt_tr

            img0_hv = img_hv
            # forward_bi uses get_text_and_mask -> txt_hv, mask_hv and passes text_mask into blocks.
            txt0_hv = txt_hv
            vec0_hv = vec_hv2
            vec_txt0_hv = vec_txt_hv

            # Run blocks
            n = min(args.max_blocks, len(trainer_model.double_blocks), len(hyvideo_model.double_blocks))
            for i in range(n):
                # trainer block: expects (img, txt, vec_txt, vec, freqs_cis, text_mask=None, ...)
                trainer_block = trainer_model.double_blocks[i]
                img0_tr, txt0_tr = trainer_block(
                    img=img0_tr,
                    txt=txt0_tr,
                    vec_txt=vec_txt0_tr,
                    vec=vec0_tr,
                    freqs_cis=freqs_cis_tr,
                    text_mask=None,  # trainer pre-masked tokens
                    attn_param=trainer_model.attn_param,
                    is_flash=False,
                    block_idx=i,
                    viewmats=viewmats,
                    Ks=Ks,
                )

                # hyvideo block: signature includes inference flags; forward_bi sets bi_inference=True when viewmats is not None
                hy_block = hyvideo_model.double_blocks[i]
                img0_hv, txt0_hv = hy_block(
                    bi_inference=True,
                    ar_txt_inference=False,
                    ar_vision_inference=False,
                    img=img0_hv,
                    txt=txt0_hv,
                    vec_txt=vec_txt0_hv,
                    vec=vec0_hv,
                    freqs_cis=freqs_cis_hv,
                    text_mask=mask_hv,
                    attn_param=hyvideo_model.attn_param,
                    is_flash=False,
                    block_idx=i,
                    viewmats=viewmats,
                    Ks=Ks,
                )

                print(f"\n--- after double_block[{i}] ---")
                if img0_tr.shape == img0_hv.shape:
                    print(f"[img diff] {_diff_stats(img0_tr, img0_hv)}")
                else:
                    print(f"[img diff] shape mismatch trainer={tuple(img0_tr.shape)} hyvideo={tuple(img0_hv.shape)}")
                    break
                # For txt, shapes will likely differ due to masking strategy; report both stats only.
                print(f"[trainer txt] {_tensor_stats(txt0_tr)}")
                print(f"[hyvideo txt] {_tensor_stats(txt0_hv)}")
                if img0_tr.shape == img0_hv.shape and (img0_tr - img0_hv).detach().float().abs().max().item() > args.diff_threshold:
                    print(f"[STOP] first divergence beyond threshold at double_block[{i}] (img stream)")
                    return

            # Final outputs (call full forwards just to show end diff)
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
            print("\n=== final output ===")
            print(f"[OUT trainer] {_tensor_stats(out_tr)}")
            print(f"[OUT hyvideo] {_tensor_stats(out_hv)}")
            print(f"[DIFF out] {_diff_stats(out_tr, out_hv)}")
    finally:
        # Avoid ProcessGroup warnings/leaks in one-off diagnostics.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    return


if __name__ == "__main__":
    main()


