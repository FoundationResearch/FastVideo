#!/usr/bin/env python3
"""
Compare two .safetensors checkpoints (AR vs BI) and report how different they are.

Usage:
  python hyw/research/ar_bi_diff.py \
    --a /path/to/ar/diffusion_pytorch_model.safetensors \
    --b /path/to/bi/diffusion_pytorch_model.safetensors
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class TensorDiffStat:
    key: str
    shape_a: Tuple[int, ...]
    shape_b: Tuple[int, ...]
    dtype_a: str
    dtype_b: str
    numel: int
    max_abs: float
    mean_abs: float
    rmse: float
    rel_rmse_vs_a: float


def _fmt_num(n: float) -> str:
    if n == 0.0:
        return "0"
    a = abs(n)
    if a >= 1e4 or a < 1e-3:
        return f"{n:.4e}"
    return f"{n:.6f}".rstrip("0").rstrip(".")


def compare_safetensors(
    path_a: str,
    path_b: str,
    *,
    topk: int = 50,
    eps: float = 1e-12,
) -> Tuple[Dict[str, object], List[TensorDiffStat]]:
    import torch
    from safetensors import safe_open

    # Some safetensors versions don't expose get_shape/get_dtype on the file handle.
    # We therefore do key-level set diff first, and then detect shape/dtype mismatches
    # lazily when actually loading tensors for comparison.
    with safe_open(path_a, framework="pt", device="cpu") as fa, safe_open(
        path_b, framework="pt", device="cpu"
    ) as fb:
        keys_a = list(fa.keys())
        keys_b = list(fb.keys())

    set_a, set_b = set(keys_a), set(keys_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    common = sorted(set_a & set_b)

    shape_mismatch: List[str] = []
    dtype_mismatch: List[str] = []
    comparable: List[str] = []

    per_tensor: List[TensorDiffStat] = []

    # global accumulators (float64 for stable sums)
    total_numel = 0
    sum_abs = 0.0
    sum_sq = 0.0
    sum_sq_a = 0.0
    global_max_abs = 0.0

    with safe_open(path_a, framework="pt", device="cpu") as fa, safe_open(path_b, framework="pt", device="cpu") as fb:
        for k in common:
            ta = fa.get_tensor(k)
            tb = fb.get_tensor(k)
            if tuple(ta.shape) != tuple(tb.shape):
                shape_mismatch.append(k)
                continue
            if str(ta.dtype) != str(tb.dtype):
                dtype_mismatch.append(k)

            # stats in float32
            da = ta.detach().to(dtype=torch.float32)
            db = tb.detach().to(dtype=torch.float32)
            diff = (da - db).reshape(-1)
            a_flat = da.reshape(-1)

            numel = diff.numel()
            if numel == 0:
                continue

            absdiff = diff.abs()
            max_abs = float(absdiff.max().item())
            mean_abs = float(absdiff.mean().item())
            rmse = float(torch.mean(diff * diff).sqrt().item())

            rms_a = float(torch.mean(a_flat * a_flat).sqrt().item())
            rel = rmse / (rms_a + eps)

            per_tensor.append(
                TensorDiffStat(
                    key=k,
                    shape_a=tuple(ta.shape),
                    shape_b=tuple(tb.shape),
                    dtype_a=str(ta.dtype),
                    dtype_b=str(tb.dtype),
                    numel=int(numel),
                    max_abs=max_abs,
                    mean_abs=mean_abs,
                    rmse=rmse,
                    rel_rmse_vs_a=rel,
                )
            )

            total_numel += int(numel)
            sum_abs += float(absdiff.sum().item())
            sum_sq += float(torch.sum(diff * diff).item())
            sum_sq_a += float(torch.sum(a_flat * a_flat).item())
            global_max_abs = max(global_max_abs, max_abs)

    # sort by relative rmse, then rmse, then max_abs
    per_tensor_sorted = sorted(
        per_tensor,
        key=lambda s: (s.rel_rmse_vs_a, s.rmse, s.max_abs),
        reverse=True,
    )

    global_rmse = math.sqrt(sum_sq / max(total_numel, 1))
    global_mean_abs = sum_abs / max(total_numel, 1)
    global_rms_a = math.sqrt(sum_sq_a / max(total_numel, 1))
    global_rel_rmse = global_rmse / (global_rms_a + eps)

    summary: Dict[str, object] = {
        "path_a": path_a,
        "path_b": path_b,
        "num_keys_a": len(keys_a),
        "num_keys_b": len(keys_b),
        "num_common_keys": len(common),
        "num_only_a": len(only_a),
        "num_only_b": len(only_b),
        "num_shape_mismatch": len(shape_mismatch),
        "num_dtype_mismatch": len(dtype_mismatch),
        "num_compared_tensors": len(per_tensor),
        "total_compared_numel": int(total_numel),
        "global": {
            "mean_abs": float(global_mean_abs),
            "rmse": float(global_rmse),
            "rel_rmse_vs_a": float(global_rel_rmse),
            "max_abs": float(global_max_abs),
        },
        "only_in_a": only_a[: min(50, len(only_a))],
        "only_in_b": only_b[: min(50, len(only_b))],
        "shape_mismatch_examples": shape_mismatch[: min(50, len(shape_mismatch))],
        "dtype_mismatch_examples": dtype_mismatch[: min(50, len(dtype_mismatch))],
        "topk": topk,
    }

    return summary, per_tensor_sorted[:topk]


def _print_report(summary: Dict[str, object], top: List[TensorDiffStat]) -> None:
    g = summary["global"]
    print("=== safetensors diff report ===")
    print(f"A: {summary['path_a']}")
    print(f"B: {summary['path_b']}")
    print("")
    print(
        "keys:",
        f"A={summary['num_keys_a']}",
        f"B={summary['num_keys_b']}",
        f"common={summary['num_common_keys']}",
        f"onlyA={summary['num_only_a']}",
        f"onlyB={summary['num_only_b']}",
        f"shape_mismatch={summary['num_shape_mismatch']}",
        f"dtype_mismatch={summary['num_dtype_mismatch']}",
    )
    print(
        "compared:",
        f"tensors={summary['num_compared_tensors']}",
        f"numel={summary['total_compared_numel']}",
    )
    print(
        "global:",
        f"mean_abs={_fmt_num(g['mean_abs'])}",
        f"rmse={_fmt_num(g['rmse'])}",
        f"rel_rmse_vs_a={_fmt_num(g['rel_rmse_vs_a'])}",
        f"max_abs={_fmt_num(g['max_abs'])}",
    )
    print("")
    if summary["num_only_a"] > 0:
        print("only in A (first 50):")
        for k in summary["only_in_a"]:
            print("  -", k)
        print("")
    if summary["num_only_b"] > 0:
        print("only in B (first 50):")
        for k in summary["only_in_b"]:
            print("  -", k)
        print("")
    if summary["num_shape_mismatch"] > 0:
        print("shape mismatch examples (first 50):")
        for k in summary["shape_mismatch_examples"]:
            print("  -", k)
        print("")
    if summary["num_dtype_mismatch"] > 0:
        print("dtype mismatch examples (first 50):")
        for k in summary["dtype_mismatch_examples"]:
            print("  -", k)
        print("")
    print(f"top {summary['topk']} tensors by relative RMSE:")
    for s in top:
        print(
            f"- {s.key} | shape={tuple(s.shape_a)} | dtypeA={s.dtype_a} dtypeB={s.dtype_b} | "
            f"rel_rmse={_fmt_num(s.rel_rmse_vs_a)} rmse={_fmt_num(s.rmse)} "
            f"mean_abs={_fmt_num(s.mean_abs)} max_abs={_fmt_num(s.max_abs)} | "
            f"numel={s.numel}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a",
        type=str,
        default="/mnt/weka/home/hao.zhang/alex/weights/tencent/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors",
        help="Path to model A safetensors (default: AR).",
    )
    parser.add_argument(
        "--b",
        type=str,
        default="/mnt/weka/home/hao.zhang/alex/weights/tencent/HY-WorldPlay/bidirectional_model/diffusion_pytorch_model.safetensors",
        help="Path to model B safetensors (default: BI).",
    )
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    try:
        summary, top = compare_safetensors(args.a, args.b, topk=args.topk)
    except ModuleNotFoundError as e:
        if "safetensors" in str(e):
            raise SystemExit(
                "Missing dependency 'safetensors'. Install with: pip install safetensors"
            ) from e
        raise
    _print_report(summary, top)


if __name__ == "__main__":
    main()


