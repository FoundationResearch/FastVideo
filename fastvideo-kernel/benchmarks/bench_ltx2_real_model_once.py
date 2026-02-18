#!/usr/bin/env python3
"""Benchmark real LTX2 DiT single forward latency (no breakdown)."""

from __future__ import annotations

import argparse
import importlib.machinery
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch

try:
    from triton.testing import do_bench
except Exception as e:  # pragma: no cover
    raise ImportError("This benchmark requires triton (triton.testing.do_bench).") from e


def _install_torchvision_stub_if_needed() -> None:
    # Some runtime images have mismatched torchvision/torch builds.
    # LTX2 import path does not require torchvision features in this benchmark.
    if "torchvision" in sys.modules:
        return
    tv = types.ModuleType("torchvision")
    tv_utils = types.ModuleType("torchvision.utils")
    # diffusers checks importlib.util.find_spec("torchvision"), so __spec__ must exist.
    tv.__spec__ = importlib.machinery.ModuleSpec(
        "torchvision",
        loader=None,
        is_package=True,
    )
    tv.__path__ = []
    tv_utils.__spec__ = importlib.machinery.ModuleSpec(
        "torchvision.utils",
        loader=None,
        is_package=False,
    )

    def _dummy_make_grid(*args, **kwargs):  # noqa: ANN001, ANN003
        if args:
            return args[0]
        return None

    tv_utils.make_grid = _dummy_make_grid
    tv.utils = tv_utils
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.utils"] = tv_utils


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
    p = argparse.ArgumentParser(description="Real LTX2 model-level one-call benchmark")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--height", type=int, default=34)
    p.add_argument("--width", type=int, default=60)
    p.add_argument("--text_tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--rep", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--num_layers", type=int, default=48)
    return p.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    _ensure_repo_on_path()
    _install_torchvision_stub_if_needed()
    os.environ.setdefault("FASTVIDEO_VSA_256", "1")
    os.environ.setdefault("FASTVIDEO_VSA_256_BACKEND", "cute")

    from fastvideo.configs.models.dits.ltx2 import LTX2VideoConfig
    import fastvideo.models.dits.ltx2 as ltx2_mod

    # Run this benchmark in single-process mode without distributed init.
    ltx2_mod.get_sp_world_size = lambda: 1
    ltx2_mod.get_sp_parallel_rank = lambda: 0

    args = parse_args()
    set_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")

    cfg = LTX2VideoConfig()
    cfg.arch_config.num_layers = args.num_layers
    model = ltx2_mod.LTX2Transformer3DModel(cfg, hf_config={}).to(device=device, dtype=dtype)
    model.eval()

    bsz = args.batch_size
    c = cfg.arch_config.num_channels_latents
    t, h, w = args.num_frames, args.height, args.width
    seq_len = t * h * w
    caption_channels = cfg.arch_config.caption_channels

    hidden_states = torch.randn(bsz, c, t, h, w, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(
        bsz, args.text_tokens, caption_channels, device=device, dtype=dtype
    )
    timestep = torch.ones((bsz, seq_len), device=device, dtype=torch.float32)

    def _forward_once():
        out = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=None,
            guidance=None,
            audio_hidden_states=None,
            audio_encoder_hidden_states=None,
            audio_timestep=None,
            audio_encoder_attention_mask=None,
            skip_cross_modal_attn=True,
        )
        return out

    with torch.no_grad():
        out = _forward_once()
        if isinstance(out, tuple):
            finite_ok = all(
                (x is None) or torch.isfinite(x).all().item() for x in out
            )
        else:
            finite_ok = torch.isfinite(out).all().item()
        ms = do_bench(_forward_once, warmup=args.warmup, rep=args.rep, quantiles=None)

    print("LTX2 real model-level benchmark (single DiT forward)")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(
        f"batch={bsz}, latent_shape=[{c},{t},{h},{w}], "
        f"text_tokens={args.text_tokens}, num_layers={args.num_layers}, dtype={args.dtype}"
    )
    print(f"finite_check: {finite_ok}")
    print(f"real_model_forward_once: {ms:.3f} ms")


if __name__ == "__main__":
    main()

