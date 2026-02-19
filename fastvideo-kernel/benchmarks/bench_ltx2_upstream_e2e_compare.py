#!/usr/bin/env python3
"""Compare upstream e2e latency: full attention vs VSA."""

from __future__ import annotations

import argparse
import gc
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
        description="Compare upstream e2e latency: FLASH_ATTN vs VIDEO_SPARSE_ATTN"
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="Davids048/LTX2-Base-Diffusers",
    )
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
        default="outputs_video/bench_ltx2_upstream_e2e_compare",
    )
    return p.parse_args()


def _measure_backend(
    backend: str,
    args: argparse.Namespace,
    torch_dtype: torch.dtype,
) -> dict[str, float]:
    from fastvideo import VideoGenerator

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = backend
    # Allow external override; default remains upstream per-layer tile/untile.
    os.environ.setdefault("FASTVIDEO_LTX2_TILE_FASTPATH", "0")
    os.environ.setdefault("FASTVIDEO_VSA_256", "1")
    os.environ.setdefault("FASTVIDEO_VSA_256_BACKEND", "cute")

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

    for _ in range(args.warmup):
        _ = _generate_once()
        torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(args.rep):
        t0 = time.perf_counter()
        _ = _generate_once()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    generator.shutdown()
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "avg": float(np.mean(times_ms)),
        "p50": float(np.percentile(times_ms, 50)),
        "p90": float(np.percentile(times_ms, 90)),
        "min": float(np.min(times_ms)),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    _ensure_repo_on_path()
    args = parse_args()
    set_seed(args.seed)
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    full = _measure_backend("FLASH_ATTN", args, torch_dtype)
    vsa = _measure_backend("VIDEO_SPARSE_ATTN", args, torch_dtype)
    speedup = full["avg"] / max(1e-6, vsa["avg"])

    print("LTX2 upstream e2e backend comparison (per-layer tile/untile)")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"model_path: {args.model_path}")
    print(
        f"frames={args.num_frames}, hw={args.height}x{args.width}, "
        f"steps={args.num_inference_steps}, dtype={args.dtype}, "
        f"num_gpus={args.num_gpus}, vsa_sparsity={args.vsa_sparsity}"
    )
    print("FULL attention backend: FLASH_ATTN")
    print(f"  avg: {full['avg']:.3f} ms, p50: {full['p50']:.3f} ms, p90: {full['p90']:.3f} ms, min: {full['min']:.3f} ms")
    print("VSA attention backend: VIDEO_SPARSE_ATTN")
    print(f"  avg: {vsa['avg']:.3f} ms, p50: {vsa['p50']:.3f} ms, p90: {vsa['p90']:.3f} ms, min: {vsa['min']:.3f} ms")
    print(f"speedup(full_avg / vsa_avg): {speedup:.3f}x")


if __name__ == "__main__":
    main()

