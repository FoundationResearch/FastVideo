from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import imageio
import numpy as np
import torch
from PIL import Image

from hyw.eval.metrics import compute_video_metrics
from hyw.eval.path_utils import find_repo_root, resolve_data_path


def _load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def _read_video_rgb(path: Path) -> np.ndarray:
    """
    Returns (T,H,W,3) uint8.
    """
    reader = imageio.get_reader(str(path))
    frames = []
    for frame in reader:
        # imageio returns RGB already for most backends
        frames.append(frame[:, :, :3])
    reader.close()
    return np.stack(frames, axis=0).astype(np.uint8)


def _write_video_rgb(frames: np.ndarray, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames, fps=fps)


def _make_side_by_side(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    T = min(gt.shape[0], pred.shape[0])
    gt = gt[:T]
    pred = pred[:T]
    if gt.shape[1:3] != pred.shape[1:3]:
        # naive: center-crop to min size
        h = min(gt.shape[1], pred.shape[1])
        w = min(gt.shape[2], pred.shape[2])
        gt = gt[:, :h, :w, :]
        pred = pred[:, :h, :w, :]
    return np.concatenate([gt, pred], axis=2)


def _frame_action_to_labels(action_json: Dict, latent_num: int) -> torch.Tensor:
    """
    Convert our frame-level (move_action, view_action) to HY action labels (0..80) for each latent.
    We sample action at frame indices: 0,4,8,... (latent boundaries). action[0] is forced to 0.
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

    rot_map = {
        "": (0, 0, 0, 0),
        "LR": (1, 0, 0, 0),  # yaw right
        "LL": (0, 1, 0, 0),  # yaw left
        "LU": (0, 0, 1, 0),  # pitch up
        "LD": (0, 0, 0, 1),  # pitch down
    }

    labels: List[int] = []
    labels.append(0)
    for i in range(1, latent_num):
        frame_idx = i * 4
        entry = action_json.get(str(frame_idx), {"move_action": "", "view_action": ""})
        move = str(entry.get("move_action", "") or "")
        view = str(entry.get("view_action", "") or "")

        # Match official HY-WorldPlay dataset parsing:
        # - Allow diagonal moves like "WA"/"WD"/"SA"/"SD"
        # - Disallow contradictory moves like "WS" or "AD" (treated as no-move)
        trans_one_hot = [0, 0, 0, 0]  # [W, S, D, A]
        if ("W" in move) and ("S" not in move):
            trans_one_hot[0] = 1
        if ("S" in move) and ("W" not in move):
            trans_one_hot[1] = 1
        if ("D" in move) and ("A" not in move):
            trans_one_hot[2] = 1
        if ("A" in move) and ("D" not in move):
            trans_one_hot[3] = 1

        trans_label = mapping[tuple(trans_one_hot)]
        rot_label = mapping[rot_map.get(view, (0, 0, 0, 0))]
        labels.append(trans_label * 9 + rot_label)
    return torch.tensor(labels, dtype=torch.long)


def _pose_frame_json_to_viewmats_Ks(pose_json: Dict, latent_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert our per-frame pose.json to latent-aligned (w2c, K) lists.

    We sample pose at frame indices: 0,4,8,... (latent boundaries).
    Ks is normalized to match HY generate.py behavior (fx,fy normalized; cx,cy=0.5).
    """
    viewmats = []
    Ks = []
    for i in range(latent_num):
        frame_idx = i * 4
        entry = pose_json[str(frame_idx)]

        w2c = np.array(entry["w2c"], dtype=np.float32)
        K = np.array(entry["K"], dtype=np.float32)

        # normalize K like hyvideo/generate.py
        K_norm = K.copy()
        K_norm[0, 0] /= (K_norm[0, 2] * 2.0)
        K_norm[1, 1] /= (K_norm[1, 2] * 2.0)
        K_norm[0, 2] = 0.5
        K_norm[1, 2] = 0.5

        viewmats.append(w2c)
        Ks.append(K_norm)

    return torch.tensor(np.stack(viewmats, axis=0)), torch.tensor(np.stack(Ks, axis=0))


def _infer_video_length_for_latents(latent_num: int) -> int:
    # HY uses temporal compression with overlap: F = 4*(L-1)+1
    return 4 * (latent_num - 1) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_manifest", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--action_ckpt",
        type=str,
        default=None,
        help=(
            "Optional action safetensors (used by HY-WorldPlay create_pipeline to strict-load action params). "
            "Set to 'none' to RANDOM-INIT action params for evaluation."
        ),
    )
    parser.add_argument("--finetuned_ckpt", type=str, default=None, help="e.g. outputs/.../checkpoint-50")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument(
        "--model_type",
        type=str,
        default="ar",
        choices=["ar", "bi"],
        help="Which transformer denoise path to use inside HY-WorldPlay pipeline. "
        "'ar' uses kv-cache + ar_vision_inference; 'bi' uses bi_inference (joint img+txt attention, closer to training).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="CFG scale. Set to 1.0 to disable CFG (aligns with our training cfg_rate=0 precompute).",
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=None,
        help="Optional scheduler flow shift override. If None, use pipeline default.",
    )
    parser.add_argument(
        "--print_weight_delta",
        action="store_true",
        help="Before loading finetuned_ckpt, compute and print max(|ft-base|) over overlapping transformer params. "
        "This can be expensive for large checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__))
    hyworld_root = (repo_root / "hyw" / "HY-WorldPlay-main").resolve()
    if str(hyworld_root) not in os.sys.path:
        os.sys.path.insert(0, str(hyworld_root))

    # Initialize infer_state (required by HY-WorldPlay pipeline)
    from argparse import Namespace
    from hyvideo.commons.infer_state import get_infer_state, initialize_infer_state

    if get_infer_state() is None:
        initialize_infer_state(
            Namespace(
                enable_torch_compile=False,
                use_sageattn=False,
                sage_blocks_range="0-53",
                use_vae_parallel=False,
                use_fp8_gemm=False,
                quant_type="fp8-per-block",
                include_patterns="double_blocks",
            )
        )

    from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.raw_manifest).read_text())

    action_ckpt = args.action_ckpt
    if action_ckpt is not None and str(action_ckpt).strip().lower() in {"", "none", "null"}:
        action_ckpt = None

    # Create pipeline from base MODEL_PATH.
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version="480p_i2v",
        enable_offloading=False,
        enable_group_offloading=False,
        create_sr_pipeline=False,
        transformer_dtype=torch.bfloat16,
        action_ckpt=action_ckpt,
    )

    # Optionally load fine-tuned checkpoint weights into transformer.
    if args.finetuned_ckpt:
        ckpt_dir = Path(args.finetuned_ckpt)
        safetensors_path = ckpt_dir / "transformer" / "diffusion_pytorch_model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"Missing finetuned safetensors: {safetensors_path}")
        from safetensors.torch import load_file

        sd = load_file(str(safetensors_path))
        # Sanity check: ensure this checkpoint is actually compatible with the inference transformer.
        model_keys = set(pipe.transformer.state_dict().keys())
        sd_keys = set(sd.keys())
        overlap = len(model_keys.intersection(sd_keys))
        overlap_ratio = overlap / max(1, len(sd_keys))
        print(
            f"[INFO] finetuned_ckpt key overlap: {overlap}/{len(sd_keys)} = {overlap_ratio:.3f} "
            f"(model_keys={len(model_keys)})"
        )
        if overlap_ratio < 0.5:
            raise RuntimeError(
                "finetuned_ckpt seems incompatible with the inference transformer "
                f"(overlap_ratio={overlap_ratio:.3f}). "
                "This usually means the checkpoint was saved for a different class/architecture."
            )

        if args.print_weight_delta:
            # Compute max(|ft-base|) over overlapping tensors.
            # NOTE: This may be slow and will move finetuned tensors to the model device one-by-one.
            with torch.no_grad():
                base_sd = pipe.transformer.state_dict()
                max_abs = 0.0
                max_key = None
                max_base_abs = 0.0
                max_base_key = None
                compared = 0
                skipped = 0
                for k, ft_v in sd.items():
                    base_v = base_sd.get(k)
                    if base_v is None:
                        skipped += 1
                        continue
                    # Some entries may be non-floating; handle them conservatively.
                    if not torch.is_tensor(ft_v) or not torch.is_tensor(base_v):
                        skipped += 1
                        continue
                    try:
                        ft_t = ft_v.to(device=base_v.device, dtype=base_v.dtype, non_blocking=True)
                        diff = (ft_t - base_v).abs()
                        dmax = float(diff.max().item())
                        bmax = float(base_v.abs().max().item())
                        compared += 1
                        if dmax > max_abs:
                            max_abs = dmax
                            max_key = k
                        if bmax > max_base_abs:
                            max_base_abs = bmax
                            max_base_key = k
                    except Exception:
                        skipped += 1
                        continue
                ratio = (max_abs / max_base_abs) if max_base_abs > 0 else float("inf")
                print(
                    f"[INFO] max(|ft-base|)/max(|base|) over {compared} tensors = {ratio:.6g} "
                    f"(max_diff={max_abs:.6g} @ {max_key}; max_base={max_base_abs:.6g} @ {max_base_key}; skipped={skipped})"
                )

        missing, unexpected = pipe.transformer.load_state_dict(sd, strict=False)
        if missing:
            print(f"[WARN] Missing keys when loading finetuned ckpt: {len(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected keys when loading finetuned ckpt: {len(unexpected)}")

    device = torch.device(args.device)
    pipe = pipe.to(device)

    all_rows = []
    for idx, item in enumerate(manifest[: args.max_samples]):
        sample_id = item.get("id", f"sample_{idx:05d}")

        video_path = resolve_data_path(item["video_path"], repo_root)
        pose_path = resolve_data_path(item["pose_path"], repo_root)
        action_path = resolve_data_path(item["action_path"], repo_root)

        if not video_path.exists():
            raise FileNotFoundError(f"video_path not found: {video_path}")
        if not pose_path.exists():
            raise FileNotFoundError(f"pose_path not found: {pose_path}")
        if not action_path.exists():
            raise FileNotFoundError(f"action_path not found: {action_path}")

        meta = item.get("meta", {})
        fps = int(meta.get("fps", 12))
        prompt = item.get("text", "a colored circle")

        gt_frames = _read_video_rgb(video_path)
        pose_json = _load_json(pose_path)
        action_json = _load_json(action_path)

        # Use 16 latents for our test setup: latent_num = floor((F-1)/4)+1 (but HY expects F=4*(L-1)+1).
        latent_num = (min(int(meta.get("num_frames", gt_frames.shape[0])), gt_frames.shape[0]) - 1) // 4 + 1
        # Force to nearest valid: keep latent_num as-is, but use HY-aligned video_length.
        video_length = _infer_video_length_for_latents(latent_num)
        gt_frames = gt_frames[:video_length]

        # Reference image = first GT frame
        ref_img = Image.fromarray(gt_frames[0])

        viewmats, Ks = _pose_frame_json_to_viewmats_Ks(pose_json, latent_num)
        action_labels = _frame_action_to_labels(action_json, latent_num)

        # Run generation
        torch.manual_seed(args.seed)
        with torch.no_grad():
            out = pipe(
                enable_sr=False,
                prompt=prompt,
                aspect_ratio="1:1",
                num_inference_steps=int(args.num_inference_steps),
                sr_num_inference_steps=None,
                video_length=int(video_length),
                guidance_scale=float(args.guidance_scale),
                negative_prompt=None,
                seed=int(args.seed),
                output_type="pt",
                prompt_rewrite=False,
                return_pre_sr_video=False,
                viewmats=viewmats.unsqueeze(0).to(device),
                Ks=Ks.unsqueeze(0).to(device),
                action=action_labels.unsqueeze(0).to(device),
                few_step=False,
                chunk_latent_frames=4,
                model_type=str(args.model_type),
                user_height=int(meta.get("height", 256)),
                user_width=int(meta.get("width", 256)),
                reference_image=ref_img,
                flow_shift=(float(args.flow_shift) if args.flow_shift is not None else None),
            )

        pred = out.videos
        if pred.ndim == 5:
            pred = pred[0]  # (C,F,H,W)
        pred = (pred * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        pred = np.transpose(pred, (1, 2, 3, 0))  # (F,H,W,C)

        # Save videos
        sample_out = out_dir / sample_id
        _write_video_rgb(gt_frames, sample_out / "gt.mp4", fps=fps)
        _write_video_rgb(pred, sample_out / "pred.mp4", fps=fps)
        _write_video_rgb(_make_side_by_side(gt_frames, pred), sample_out / "side_by_side.mp4", fps=fps)

        m = compute_video_metrics(gt_frames, pred)
        row = {
            "id": sample_id,
            "video_length_eval": int(min(gt_frames.shape[0], pred.shape[0])),
            "l1": m.l1,
            "mse": m.mse,
            "psnr": m.psnr,
            "gt_path": str(video_path),
            "pred_path": str(sample_out / "pred.mp4"),
            "side_by_side_path": str(sample_out / "side_by_side.mp4"),
        }
        all_rows.append(row)

        with (out_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

        print(f"[{idx+1}/{min(len(manifest), args.max_samples)}] {sample_id} PSNR={m.psnr:.2f} L1={m.l1:.4f}")

    # Summary
    if all_rows:
        avg = {
            "avg_l1": float(np.mean([r["l1"] for r in all_rows])),
            "avg_mse": float(np.mean([r["mse"] for r in all_rows])),
            "avg_psnr": float(np.mean([r["psnr"] for r in all_rows])),
            "n": len(all_rows),
        }
        (out_dir / "summary.json").write_text(json.dumps(avg, indent=2))
        print("Saved:", out_dir / "summary.json")


if __name__ == "__main__":
    main()


