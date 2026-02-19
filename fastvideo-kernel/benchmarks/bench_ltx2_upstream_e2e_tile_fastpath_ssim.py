#!/usr/bin/env python3
"""Compare LTX2 upstream e2e latency: old tile vs one-tile fastpath.

Also computes frame-wise SSIM (or MS-SSIM) between the two generated outputs.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ms_ssim, ssim


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_root


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LTX2 upstream e2e benchmark: tile old way vs tile fastpath + SSIM"
    )
    p.add_argument("--model_path", type=str, default="Davids048/LTX2-Base-Diffusers")
    p.add_argument(
        "--prompt",
        type=str,
        default="A cinematic shot of clouds moving over a mountain range.",
    )
    p.add_argument("--num_frames", type=int, default=121)
    p.add_argument("--height", type=int, default=1088)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--num_inference_steps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--rep", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vsa_sparsity", type=float, default=0.7)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument(
        "--output_path",
        type=str,
        default="outputs_video/bench_ltx2_upstream_e2e_tile_fastpath_ssim",
    )
    p.add_argument(
        "--save_video",
        action="store_true",
        help="Save generated videos for both modes (disabled by default).",
    )
    p.add_argument(
        "--use_ms_ssim",
        action="store_true",
        help="Use MS-SSIM instead of SSIM.",
    )
    return p.parse_args()


def _generate_once(
    generator,
    args: argparse.Namespace,
    mode_name: str,
    save_video: bool,
) -> dict:
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_out = str(out_dir / mode_name)
    return generator.generate_video(
        prompt=args.prompt,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        output_path=mode_out,
        save_video=save_video,
        return_frames=False,
        seed=args.seed,
    )


def _measure_mode(
    generator,
    args: argparse.Namespace,
    mode_name: str,
    tile_fastpath: bool,
) -> tuple[dict[str, float], dict]:
    os.environ["FASTVIDEO_LTX2_TILE_FASTPATH"] = "1" if tile_fastpath else "0"

    for _ in range(args.warmup):
        _ = _generate_once(generator, args, mode_name, save_video=False)
        torch.cuda.synchronize()

    times_ms: list[float] = []
    last_out: dict | None = None
    for _ in range(args.rep):
        t0 = time.perf_counter()
        last_out = _generate_once(generator, args, mode_name, save_video=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    # Optional video save is done once after timing to avoid I/O skewing e2e latency.
    if args.save_video:
        _ = _generate_once(generator, args, mode_name, save_video=True)
        torch.cuda.synchronize()

    assert last_out is not None and "samples" in last_out
    stats = {
        "avg": float(np.mean(times_ms)),
        "p50": float(np.percentile(times_ms, 50)),
        "p90": float(np.percentile(times_ms, 90)),
        "min": float(np.min(times_ms)),
    }
    return stats, last_out


def _compute_video_ssim(
    samples_a: torch.Tensor,
    samples_b: torch.Tensor,
    use_ms_ssim: bool,
) -> dict[str, float]:
    # Compare the first sample in batch; shape after slicing: [C, T, H, W].
    a = samples_a[0].detach().float().clamp(0.0, 1.0)
    b = samples_b[0].detach().float().clamp(0.0, 1.0)

    # Align shapes defensively in case of edge-case frame/size adjustment.
    t = min(a.shape[1], b.shape[1])
    h = min(a.shape[2], b.shape[2])
    w = min(a.shape[3], b.shape[3])
    a = a[:, :t, :h, :w]
    b = b[:, :t, :h, :w]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = a.to(device=device)
    b = b.to(device=device)

    vals: list[float] = []
    with torch.no_grad():
        for i in range(t):
            # [1, C, H, W]
            fa = a[:, i].unsqueeze(0)
            fb = b[:, i].unsqueeze(0)
            if use_ms_ssim:
                v = ms_ssim(fa, fb, data_range=1.0)
            else:
                v = ssim(fa, fb, data_range=1.0)
            vals.append(float(v.item()))

    vals_np = np.asarray(vals, dtype=np.float64)
    min_idx = int(np.argmin(vals_np))
    max_idx = int(np.argmax(vals_np))
    return {
        "mean": float(vals_np.mean()),
        "min": float(vals_np[min_idx]),
        "max": float(vals_np[max_idx]),
        "min_frame_idx": float(min_idx),
        "max_frame_idx": float(max_idx),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    _ensure_repo_on_path()
    args = parse_args()
    set_seed(args.seed)

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
    os.environ.setdefault("FASTVIDEO_VSA_256", "1")
    os.environ.setdefault("FASTVIDEO_VSA_256_BACKEND", "cute")
    # Force dispatch logs to print on every generate call (no dedup).
    os.environ.setdefault("FASTVIDEO_VSA_DISPATCH_LOG_ALWAYS", "1")

    from fastvideo import VideoGenerator

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    generator = VideoGenerator.from_pretrained(
        args.model_path,
        num_gpus=args.num_gpus,
        torch_dtype=torch_dtype,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        VSA_sparsity=args.vsa_sparsity,
    )

    old_stats, old_out = _measure_mode(
        generator=generator,
        args=args,
        mode_name="tile_old_way",
        tile_fastpath=False,
    )
    fast_stats, fast_out = _measure_mode(
        generator=generator,
        args=args,
        mode_name="tile_fastpath",
        tile_fastpath=True,
    )

    ssim_stats = _compute_video_ssim(
        old_out["samples"],
        fast_out["samples"],
        use_ms_ssim=args.use_ms_ssim,
    )

    speedup = old_stats["avg"] / max(1e-6, fast_stats["avg"])
    metric_name = "MS-SSIM" if args.use_ms_ssim else "SSIM"

    print("LTX2 upstream e2e: tile old way vs tile fastpath")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"model_path: {args.model_path}")
    print(
        f"frames={args.num_frames}, hw={args.height}x{args.width}, "
        f"steps={args.num_inference_steps}, dtype={args.dtype}, "
        f"num_gpus={args.num_gpus}, vsa_sparsity={args.vsa_sparsity}"
    )
    print("mode: tile_old_way (FASTVIDEO_LTX2_TILE_FASTPATH=0)")
    print(
        f"  avg: {old_stats['avg']:.3f} ms, p50: {old_stats['p50']:.3f} ms, "
        f"p90: {old_stats['p90']:.3f} ms, min: {old_stats['min']:.3f} ms"
    )
    print("mode: tile_fastpath (FASTVIDEO_LTX2_TILE_FASTPATH=1)")
    print(
        f"  avg: {fast_stats['avg']:.3f} ms, p50: {fast_stats['p50']:.3f} ms, "
        f"p90: {fast_stats['p90']:.3f} ms, min: {fast_stats['min']:.3f} ms"
    )
    print(f"speedup(old_avg / fast_avg): {speedup:.3f}x")
    print(
        f"{metric_name}(old vs fast): mean={ssim_stats['mean']:.6f}, "
        f"min={ssim_stats['min']:.6f}@frame{int(ssim_stats['min_frame_idx'])}, "
        f"max={ssim_stats['max']:.6f}@frame{int(ssim_stats['max_frame_idx'])}"
    )

    generator.shutdown()


if __name__ == "__main__":
    main()

