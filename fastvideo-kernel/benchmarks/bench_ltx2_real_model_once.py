#!/usr/bin/env python3
"""Benchmark real LTX2 inference latency via VideoGenerator."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


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
        description="Real LTX2 one-call benchmark through VideoGenerator"
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="Davids048/LTX2-Base-Diffusers",
    )
    p.add_argument("--prompt", type=str, default="A cat walking on grass, cinematic.")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--height", type=int, default=34)
    p.add_argument("--width", type=int, default=60)
    p.add_argument("--num_inference_steps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--rep", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Maps to VideoGenerator torch_dtype.",
    )
    p.add_argument(
        "--num_layers",
        type=int,
        default=48,
        help="Kept only for CLI compatibility; unused in pretrained path.",
    )
    p.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Pass through to VideoGenerator.from_pretrained",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="outputs_video/bench_ltx2_real_model_once",
    )
    return p.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    _ensure_repo_on_path()
    os.environ.setdefault("FASTVIDEO_VSA_256", "1")
    os.environ.setdefault("FASTVIDEO_VSA_256_BACKEND", "cute")
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")

    from fastvideo import VideoGenerator

    args = parse_args()
    set_seed(args.seed)
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    generator = VideoGenerator.from_pretrained(
        args.model_path,
        num_gpus=args.num_gpus,
        torch_dtype=torch_dtype,
        use_fsdp_inference=False,
    )

    def _generate_once():
        return generator.generate_video(
            prompt=args.prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            output_path=args.output_path,
            save_video=False,
            return_frames=False,
            seed=args.seed,
        )

    # Warmup runs
    for _ in range(args.warmup):
        _ = _generate_once()
        torch.cuda.synchronize()

    # Timed runs
    times_ms = []
    last_out = None
    for _ in range(args.rep):
        t0 = time.perf_counter()
        last_out = _generate_once()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    finite_ok = False
    if isinstance(last_out, dict) and "samples" in last_out:
        finite_ok = torch.isfinite(last_out["samples"]).all().item()

    avg_ms = float(np.mean(times_ms))
    p50_ms = float(np.percentile(times_ms, 50))
    p90_ms = float(np.percentile(times_ms, 90))
    min_ms = float(np.min(times_ms))

    print("LTX2 real benchmark via VideoGenerator.generate_video (single call)")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"model_path: {args.model_path}")
    print(
        f"frames={args.num_frames}, hw={args.height}x{args.width}, "
        f"steps={args.num_inference_steps}, dtype={args.dtype}, num_gpus={args.num_gpus}"
    )
    print(f"finite_check(samples): {finite_ok}")
    print(f"latency_avg: {avg_ms:.3f} ms")
    print(f"latency_p50: {p50_ms:.3f} ms")
    print(f"latency_p90: {p90_ms:.3f} ms")
    print(f"latency_min: {min_ms:.3f} ms")
    if args.num_layers != 48:
        print(
            f"note: --num_layers={args.num_layers} ignored in pretrained path "
            "(kept for CLI compatibility)."
        )
    generator.shutdown()


if __name__ == "__main__":
    main()

