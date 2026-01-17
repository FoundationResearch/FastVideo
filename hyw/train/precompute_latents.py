# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image


def _find_repo_root(start: Path) -> Path:
    """
    Find FastVideo repo root by searching upwards for pyproject.toml.
    This makes the script location-independent (e.g. moved under hyw/train/).
    """
    cur = start.resolve()
    for _ in range(10):
        if (cur / "pyproject.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise RuntimeError(f"Could not locate repo root from: {start}")


def _read_video_frames(video_path: str, num_frames: int) -> np.ndarray:
    """Return uint8 RGB frames: (F,H,W,3)."""
    frames: list[np.ndarray] = []
    with imageio.get_reader(video_path) as r:
        for frame in r:
            frames.append(np.asarray(frame, dtype=np.uint8))
            if len(frames) >= num_frames:
                break
    if not frames:
        raise RuntimeError(f"Empty video: {video_path}")
    return np.stack(frames, axis=0)


def _to_vae_input(frames: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    frames: (F,H,W,3) uint8
    return: (1,3,F,H,W) float in [-1,1]
    """
    x = torch.from_numpy(frames).to(device=device)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # (1,3,F,H,W)
    x = x.to(dtype=torch.float32) / 255.0
    x = x * 2.0 - 1.0
    return x.to(dtype=dtype)


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_manifest", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--action_ckpt", type=str, required=True)
    p.add_argument("--transformer_version", type=str, default="480p_i2v")
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    args = p.parse_args()

    device = torch.device(args.device)
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    items = json.loads(Path(args.raw_manifest).read_text(encoding="utf-8"))
    if args.max_samples and args.max_samples > 0:
        items = items[: args.max_samples]

    # Import HY-WorldPlay pipeline WITHOUT modifying it.
    repo_root = _find_repo_root(Path(__file__).resolve())
    hyworld_root = (repo_root / "hyw" / "HY-WorldPlay-main").resolve()
    sys.path.insert(0, str(hyworld_root))

    from hyvideo.pipelines.worldplay_video_pipeline import (  # type: ignore
        HunyuanVideo_1_5_Pipeline,
    )

    # HY-WorldPlay expects a global "infer_state" to be initialized (official generate.py does this).
    # If we don't, create_pipeline() will call get_infer_state() and crash on None.
    from argparse import Namespace

    from hyvideo.commons.infer_state import (  # type: ignore
        get_infer_state,
        initialize_infer_state,
    )

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

    # Load pipeline once (heavy).
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=args.transformer_version,
        enable_offloading=False,
        enable_group_offloading=False,
        create_sr_pipeline=False,
        action_ckpt=args.action_ckpt,
    )
    pipe = pipe.to(device)

    vae_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    for i, it in enumerate(items):
        sample_dir = out_root / f"sample_{i:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_pt = sample_dir / "latent.pt"
        if out_pt.exists():
            continue

        video_path = str(it["video_path"])
        prompt = str(it.get("text") or it.get("caption") or "a colored circle")

        # Read frames
        num_frames = int(it.get("meta", {}).get("num_frames", 64))
        frames = _read_video_frames(video_path, num_frames=num_frames)

        # image_cond: first frame as PIL, let pipeline do transform+encode
        first_frame = Image.fromarray(frames[0])
        height, width = int(frames.shape[1]), int(frames.shape[2])
        image_cond = pipe.get_image_condition_latents("i2v", first_frame, height, width)  # (1,C,1,H',W')
        image_cond = image_cond.to(device=device, dtype=vae_dtype)

        # latents: VAE encode full video
        x = _to_vae_input(frames, device=device, dtype=vae_dtype)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            latents = pipe.vae.encode(x).latent_dist.mode()
            latents.mul_(pipe.vae.config.scaling_factor)

        # prompt embeddings (positive only)
        prompt_embeds, _, prompt_mask, _ = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            clip_skip=None,
            data_type="video",
        )
        if prompt_mask is None:
            # match trainer expectations (it uses encoder_attention_mask)
            prompt_mask = torch.ones(prompt_embeds.shape[:2], device=device, dtype=torch.int64)

        # byT5 embeddings (positive only) — avoid pipeline CFG duplication
        if getattr(pipe.config, "glyph_byT5_v2", False):
            byt5_text_states, byt5_text_mask = pipe._process_single_byt5_prompt(prompt, device)  # pylint: disable=protected-access
        else:
            byt5_text_states = torch.zeros((1, 1, 1), device=device, dtype=prompt_embeds.dtype)
            byt5_text_mask = torch.ones((1, 1), device=device, dtype=torch.int64)

        # vision_states: SigLIP features from reference image (align train with inference).
        #
        # NOTE:
        # We do NOT call `pipe._prepare_vision_states(...)` here because that helper touches
        # `pipe.do_classifier_free_guidance`, which depends on runtime sampling state set inside `pipe(...)`.
        # For precompute we always run w/o CFG, so we can directly reproduce the same preprocessing:
        # resize+center-crop to the closest bucket, then `vision_encoder.encode_images`.
        if pipe.vision_encoder is None:
            raise RuntimeError(
                "vision_encoder is unavailable, but vision_states alignment (scheme A) requires it. "
                "Please ensure the vision encoder checkpoint is downloaded and accessible."
            )
        from hyvideo.utils.data_utils import resize_and_center_crop  # type: ignore

        target_resolution = pipe.ideal_resolution or "480p"
        bucket_h, bucket_w = pipe.get_closest_resolution_given_reference_image(
            first_frame, target_resolution
        )
        input_image_np = resize_and_center_crop(
            np.array(first_frame.convert("RGB")),
            target_width=int(bucket_w),
            target_height=int(bucket_h),
        )
        vision_out = pipe.vision_encoder.encode_images(input_image_np)
        vision_states = vision_out.last_hidden_state.to(
            device=device, dtype=prompt_embeds.dtype
        )

        payload = {
            # dataset expects list-like tensors and then indexes [0]
            "latent": latents.detach().to("cpu"),
            "prompt_embeds": prompt_embeds.detach().to("cpu"),
            "prompt_mask": prompt_mask.detach().to("cpu"),
            "image_cond": image_cond.detach().to("cpu"),
            "vision_states": vision_states.detach().to("cpu"),
            "byt5_text_states": byt5_text_states.detach().to("cpu"),
            "byt5_text_mask": byt5_text_mask.detach().to("cpu"),
        }
        torch.save(payload, out_pt)
        print(f"[{i+1}/{len(items)}] wrote {out_pt}")

    print(f"[OK] done. out_root={out_root}")


if __name__ == "__main__":
    main()


