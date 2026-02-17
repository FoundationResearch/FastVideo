#!/usr/bin/env python3
"""Reusable debug suite for FlashAttention CuTe m=64/n=64 forward path."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def _ensure_flash_attn_importable() -> None:
    root = Path(__file__).resolve().parents[1]
    flash_attn_repo = root / "include" / "flash-attention"
    if not flash_attn_repo.exists():
        raise FileNotFoundError(f"flash-attention submodule not found: {flash_attn_repo}")
    sys.path.insert(0, str(flash_attn_repo))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ref_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q,k,v: [B, S, H, D]
    d = q.shape[-1]
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3).float()
    p = torch.softmax(torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(d), dim=-1)
    return torch.matmul(p, vh).permute(0, 2, 1, 3).to(q.dtype)


def run_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    force_q_stage1: bool,
    env_overrides: Dict[str, str],
) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_attn.cute.interface import _flash_attn_fwd

    os.environ["FLASH_ATTN_CUTE_FORCE_Q_STAGE_1"] = "1" if force_q_stage1 else "0"
    for key, value in env_overrides.items():
        os.environ[key] = value

    _flash_attn_fwd.compile_cache.clear()
    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        m_block_size=64,
        n_block_size=64,
        block_sparse_tensors=None,
        causal=False,
        return_lse=True,
    )
    return out, lse


def calc_metrics(out: torch.Tensor, ref: torch.Tensor, lse: torch.Tensor) -> Dict[str, float]:
    diff = (out - ref).abs()
    rel = diff / (ref.abs() + 1e-6)
    row_norm = out[0, :, 0, :].float().norm(dim=-1)
    zero_rows = int((row_norm == 0).sum().item()) if out.shape[0] == 1 and out.shape[2] >= 1 else -1
    return {
        "avg_abs": float(diff.mean()),
        "max_rel": float(rel.max()),
        "out_nan": int(torch.isnan(out).sum().item()),
        "out_inf": int(torch.isinf(out).sum().item()),
        "lse_nan": int(torch.isnan(lse).sum().item()),
        "lse_inf": int(torch.isinf(lse).sum().item()),
        "zero_rows_b1h0": zero_rows,
    }


def build_qkv(
    batch: int,
    seqlen: int,
    heads: int,
    dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=dtype)
    return q, k, v


def cmd_single(args: argparse.Namespace) -> None:
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    q, k, v = build_qkv(args.batch, args.seqlen, args.heads, args.dim, dtype)
    ref = ref_attn(q, k, v)
    env = {
        "FLASH_ATTN_CUTE_M64_GUARD_SSCALE": str(args.guard_sscale),
        "FLASH_ATTN_CUTE_M64_SSCALE_THREAD_THRESHOLD": str(args.sscale_thr),
        "FLASH_ATTN_CUTE_M64_ZERO_INACTIVE_P": str(args.zero_inactive_p),
        "FLASH_ATTN_CUTE_M64_ZERO_MODE": args.zero_mode,
        "FLASH_ATTN_CUTE_M64_ZERO_THREAD_THRESHOLD": str(args.p_thr),
        "FLASH_ATTN_CUTE_M64_ZERO_THREAD_LOWER": str(args.p_lower),
        "FLASH_ATTN_CUTE_M64_ZERO_THREAD_UPPER": str(args.p_upper),
        "FLASH_ATTN_CUTE_M64_MASK_INVALID_S_COORD": str(args.mask_invalid_s_coord),
    }
    out, lse = run_flash(q, k, v, force_q_stage1=bool(args.force_q_stage1), env_overrides=env)
    metrics = calc_metrics(out, ref, lse)
    print("single config:")
    print(env)
    print(metrics)


def cmd_sweep(args: argparse.Namespace) -> None:
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    q, k, v = build_qkv(args.batch, args.seqlen, args.heads, args.dim, dtype)
    ref = ref_attn(q, k, v)

    s_values = [int(x) for x in args.sscale_thr_list.split(",")]
    p_values = [int(x) for x in args.p_thr_list.split(",")]
    print(f"sweep seqlen={args.seqlen}, heads={args.heads}, dim={args.dim}, dtype={args.dtype}")
    for s_thr in s_values:
        for p_thr in p_values:
            env = {
                "FLASH_ATTN_CUTE_M64_GUARD_SSCALE": "1",
                "FLASH_ATTN_CUTE_M64_SSCALE_THREAD_THRESHOLD": str(s_thr),
                "FLASH_ATTN_CUTE_M64_ZERO_INACTIVE_P": "1",
                "FLASH_ATTN_CUTE_M64_ZERO_MODE": "thread",
                "FLASH_ATTN_CUTE_M64_ZERO_THREAD_THRESHOLD": str(p_thr),
                "FLASH_ATTN_CUTE_M64_ZERO_THREAD_LOWER": "-1",
                "FLASH_ATTN_CUTE_M64_ZERO_THREAD_UPPER": "-1",
                "FLASH_ATTN_CUTE_M64_MASK_INVALID_S_COORD": "0",
            }
            out, lse = run_flash(q, k, v, force_q_stage1=True, env_overrides=env)
            m = calc_metrics(out, ref, lse)
            print(
                f"s{s_thr}_p{p_thr} "
                f"avg={m['avg_abs']:.6e} max_rel={m['max_rel']:.6e} "
                f"zero={m['zero_rows_b1h0']} out_nan={m['out_nan']} out_inf={m['out_inf']} "
                f"lse_inf={m['lse_inf']}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CuTe m64 debug suite")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--batch", type=int, default=1)
    common.add_argument("--seqlen", type=int, default=64)
    common.add_argument("--heads", type=int, default=1)
    common.add_argument("--dim", type=int, default=128)
    common.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    common.add_argument("--seed", type=int, default=42)

    p_single = sub.add_parser("single", parents=[common], help="Run one config")
    p_single.add_argument("--force_q_stage1", type=int, default=1, choices=[0, 1])
    p_single.add_argument("--guard_sscale", type=int, default=1, choices=[0, 1])
    p_single.add_argument("--sscale_thr", type=int, default=64)
    p_single.add_argument("--zero_inactive_p", type=int, default=1, choices=[0, 1])
    p_single.add_argument("--zero_mode", type=str, default="thread", choices=["thread", "coord"])
    p_single.add_argument("--p_thr", type=int, default=64)
    p_single.add_argument("--p_lower", type=int, default=-1)
    p_single.add_argument("--p_upper", type=int, default=-1)
    p_single.add_argument("--mask_invalid_s_coord", type=int, default=0, choices=[0, 1])

    p_sweep = sub.add_parser("sweep", parents=[common], help="Sweep sscale/p thresholds")
    p_sweep.add_argument("--sscale_thr_list", type=str, default="64,96,112,128")
    p_sweep.add_argument("--p_thr_list", type=str, default="64,96,112,128")

    return parser


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _ensure_flash_attn_importable()
    args = build_parser().parse_args()
    set_seed(args.seed)

    if args.cmd == "single":
        cmd_single(args)
    elif args.cmd == "sweep":
        cmd_sweep(args)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()


