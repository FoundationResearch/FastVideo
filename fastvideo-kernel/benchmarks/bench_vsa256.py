#!/usr/bin/env python3
"""Benchmark VSA-256 forward path.

Semantics:
- Logical sparse layout uses q_block=256, kv_block=256.
- Before kernel launch, each selected kv block i is expanded to
  two kernel blocks (2*i, 2*i+1), so kernel runs with kv_block=128.

Outputs:
- kernel-only time: prebuilt sparse tensors + kernel call
- e2e time: mask generation + 256->128 expansion + index build + kernel call
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch

try:
    from triton.testing import do_bench
except Exception as e:  # pragma: no cover
    raise ImportError("This benchmark requires triton (triton.testing.do_bench).") from e

Q_BLOCK = 256
KV_BLOCK_LOGICAL = 256
KV_BLOCK_KERNEL = 128


def _ensure_flash_attn_importable() -> None:
    kernel_root = Path(__file__).resolve().parents[1]
    flash_attn_repo = kernel_root / "include" / "flash-attention"
    if not flash_attn_repo.exists():
        raise FileNotFoundError(
            f"flash-attention submodule not found: {flash_attn_repo}"
        )
    repo_str = str(flash_attn_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark VSA-256 (logical 256, kernel kv128)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--head_dim", type=int, default=128, choices=[64, 128])
    p.add_argument("--q_seq_lens", type=int, nargs="+", default=[49152])
    p.add_argument("--kv_seq_lens", type=int, nargs="+", default=None)
    p.add_argument("--topk_logical", type=int, default=None)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--breakdown_rep", type=int, default=20)
    p.add_argument("--vbs_min", type=int, default=16)
    p.add_argument("--vbs_max", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    return p.parse_args()


def _map_to_index_torch_fallback(
    block_map: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, nhead, q_blocks, kv_blocks = block_map.shape
    index = torch.zeros(
        (bsz, nhead, q_blocks, kv_blocks), dtype=torch.int32, device=block_map.device
    )
    index_num = torch.zeros(
        (bsz, nhead, q_blocks), dtype=torch.int32, device=block_map.device
    )
    for b in range(bsz):
        for h in range(nhead):
            for q in range(q_blocks):
                row = torch.nonzero(block_map[b, h, q], as_tuple=False).flatten()
                cnt = int(row.numel())
                if cnt > 0:
                    index[b, h, q, :cnt] = row.to(torch.int32)
                index_num[b, h, q] = cnt
    return index, index_num


def _map_to_index(block_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Prefer Triton implementation to reflect real wrapper prep cost.
    try:
        from fastvideo_kernel.triton_kernels.index import map_to_index as triton_map_to_index

        return triton_map_to_index(block_map)
    except Exception:
        return _map_to_index_torch_fallback(block_map)


def _make_logical_mask(
    bs: int,
    h: int,
    q_blocks_256: int,
    kv_blocks_256: int,
    topk_logical: int,
) -> torch.Tensor:
    scores = torch.rand(bs, h, q_blocks_256, kv_blocks_256, device="cuda")
    topk_logical = min(max(1, topk_logical), kv_blocks_256)
    idx = torch.topk(scores, topk_logical, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def _expand_mask_256_to_128(mask_256: torch.Tensor) -> torch.Tensor:
    return mask_256.repeat_interleave(2, dim=3)


def _expand_sizes_256_to_128(sizes_256: torch.Tensor) -> torch.Tensor:
    sizes_256 = sizes_256.to(torch.int32)
    child0 = torch.clamp(sizes_256, min=0, max=128)
    child1 = torch.clamp(sizes_256 - 128, min=0, max=128)
    out = torch.empty((sizes_256.numel() * 2,), dtype=torch.int32, device=sizes_256.device)
    out[0::2] = child0
    out[1::2] = child1
    return out


def flops_sparse_attention(
    bs: int,
    h: int,
    d: int,
    q_len: int,
    avg_topk_kernel: float,
) -> float:
    return 4.0 * bs * h * d * q_len * (avg_topk_kernel * KV_BLOCK_KERNEL)


def bench_ms(fn: Callable[[], object], warmup: int, rep: int) -> float:
    return do_bench(fn, warmup=warmup, rep=rep, quantiles=None)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    _ensure_flash_attn_importable()
    # Enable full VSA256 path.
    os.environ["FASTVIDEO_VSA_256"] = "1"
    # This benchmark targets the q256/kv128 CuTe path.
    os.environ["FASTVIDEO_VSA_256_BACKEND"] = "cute"
    from flash_attn.cute import flash_attn_func
    from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256
    from fastvideo_kernel.ops import video_sparse_attn

    args = parse_args()
    if not (1 <= args.vbs_min <= args.vbs_max <= KV_BLOCK_LOGICAL):
        raise ValueError(
            f"Require 1 <= vbs_min <= vbs_max <= {KV_BLOCK_LOGICAL}, "
            f"got vbs_min={args.vbs_min}, vbs_max={args.vbs_max}"
        )
    set_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    bs, h, d = args.batch_size, args.num_heads, args.head_dim
    kv_seq_lens = args.kv_seq_lens if args.kv_seq_lens is not None else args.q_seq_lens
    if len(kv_seq_lens) != len(args.q_seq_lens):
        raise ValueError("kv_seq_lens must align with q_seq_lens")

    print("VSA256 Benchmark (logical q/kv=256, kernel kv=128)")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"batch={bs}, heads={h}, head_dim={d}, dtype={args.dtype}")
    print(f"kv variable_block_sizes: random int in [{args.vbs_min}, {args.vbs_max}]")

    for q_len, kv_len in zip(args.q_seq_lens, kv_seq_lens):
        if q_len % Q_BLOCK != 0 or kv_len % KV_BLOCK_LOGICAL != 0:
            print(
                f"[skip] q_len={q_len}, kv_len={kv_len} must be divisible by 256"
            )
            continue
        q_blocks_256 = q_len // Q_BLOCK
        kv_blocks_256 = kv_len // KV_BLOCK_LOGICAL
        topk_logical = (
            args.topk_logical
            if args.topk_logical is not None
            else max(1, kv_blocks_256 // 10)
        )
        topk_logical = min(topk_logical, kv_blocks_256)

        q = torch.randn(bs, h, q_len, d, dtype=dtype, device="cuda")
        k = torch.randn(bs, h, kv_len, d, dtype=dtype, device="cuda")
        v = torch.randn(bs, h, kv_len, d, dtype=dtype, device="cuda")
        q_c = q.transpose(1, 2).contiguous()
        k_c = k.transpose(1, 2).contiguous()
        v_c = v.transpose(1, 2).contiguous()

        logical_mask = _make_logical_mask(bs, h, q_blocks_256, kv_blocks_256, topk_logical)
        mask_128 = _expand_mask_256_to_128(logical_mask)
        mask_idx, mask_cnt = _map_to_index(mask_128)
        full_cnt = torch.zeros_like(mask_cnt)
        full_idx = torch.zeros_like(mask_idx)
        variable_block_sizes_256 = torch.randint(
            args.vbs_min,
            args.vbs_max + 1,
            (kv_blocks_256,),
            dtype=torch.int32,
            device="cuda",
        )
        q_variable_block_sizes_256 = torch.full(
            (q_blocks_256,), Q_BLOCK, dtype=torch.int32, device="cuda"
        )
        variable_block_sizes_128 = _expand_sizes_256_to_128(variable_block_sizes_256)

        def _kernel_only():
            return flash_attn_func(
                q_c,
                k_c,
                v_c,
                causal=False,
                mask_block_cnt=mask_cnt,
                mask_block_idx=mask_idx,
                full_block_cnt=full_cnt,
                full_block_idx=full_idx,
                block_size=(Q_BLOCK, KV_BLOCK_KERNEL),
            )

        def _e2e():
            m256 = _make_logical_mask(bs, h, q_blocks_256, kv_blocks_256, topk_logical)
            m128 = _expand_mask_256_to_128(m256)
            idx, cnt = _map_to_index(m128)
            zcnt = torch.zeros_like(cnt)
            zidx = torch.zeros_like(idx)
            return flash_attn_func(
                q_c,
                k_c,
                v_c,
                causal=False,
                mask_block_cnt=cnt,
                mask_block_idx=idx,
                full_block_cnt=zcnt,
                full_block_idx=zidx,
                block_size=(Q_BLOCK, KV_BLOCK_KERNEL),
            )

        logical_mask_fixed = logical_mask
        mask_128_fixed = mask_128
        sizes_256_fixed = variable_block_sizes_256

        def _prep_logical_mask_only():
            return _make_logical_mask(bs, h, q_blocks_256, kv_blocks_256, topk_logical)

        def _prep_split_mask_only():
            return _expand_mask_256_to_128(logical_mask_fixed)

        def _prep_split_sizes_only():
            return _expand_sizes_256_to_128(sizes_256_fixed)

        def _prep_index_only():
            return _map_to_index(mask_128_fixed)

        def _prep_kernel_inputs_only():
            m256 = _make_logical_mask(bs, h, q_blocks_256, kv_blocks_256, topk_logical)
            m128 = _expand_mask_256_to_128(m256)
            idx, cnt = _map_to_index(m128)
            zcnt = torch.zeros_like(cnt)
            zidx = torch.zeros_like(idx)
            return idx, cnt, zidx, zcnt

        def _wrapper_sparse_only():
            return block_sparse_attn_256(
                q,
                k,
                v,
                logical_mask,
                variable_block_sizes_256,
            )

        def _wrapper_e2e():
            return video_sparse_attn(
                q,
                k,
                v,
                variable_block_sizes_256,
                q_variable_block_sizes_256,
                topk_logical,
                block_size=(4, 8, 8),
                compress_attn_weight=None,
            )

        out, lse = _kernel_only()
        out_finite = torch.isfinite(out).all().item()
        lse_finite = True if lse is None else torch.isfinite(lse).all().item()
        avg_topk_kernel = mask_cnt.float().mean().item()
        flops = flops_sparse_attention(bs, h, d, q_len, avg_topk_kernel)

        kernel_ms = bench_ms(_kernel_only, warmup=args.warmup, rep=args.rep)
        e2e_ms = bench_ms(_e2e, warmup=args.warmup, rep=args.rep)
        wrapper_sparse_ms = bench_ms(_wrapper_sparse_only, warmup=args.warmup, rep=args.rep)
        wrapper_e2e_ms = bench_ms(_wrapper_e2e, warmup=args.warmup, rep=args.rep)
        prep_logical_ms = bench_ms(
            _prep_logical_mask_only, warmup=max(1, args.warmup // 2), rep=args.breakdown_rep
        )
        prep_split_mask_ms = bench_ms(
            _prep_split_mask_only, warmup=max(1, args.warmup // 2), rep=args.breakdown_rep
        )
        prep_split_sizes_ms = bench_ms(
            _prep_split_sizes_only, warmup=max(1, args.warmup // 2), rep=args.breakdown_rep
        )
        prep_index_ms = bench_ms(
            _prep_index_only, warmup=max(1, args.warmup // 2), rep=args.breakdown_rep
        )
        prep_kernel_inputs_ms = bench_ms(
            _prep_kernel_inputs_only, warmup=max(1, args.warmup // 2), rep=args.breakdown_rep
        )
        kernel_tflops = flops / kernel_ms * 1e-12 * 1e3
        e2e_tflops = flops / e2e_ms * 1e-12 * 1e3
        wrapper_sparse_tflops = flops / wrapper_sparse_ms * 1e-12 * 1e3
        wrapper_e2e_tflops = flops / wrapper_e2e_ms * 1e-12 * 1e3

        print("\n" + "=" * 100)
        print(
            f"q_len={q_len}, kv_len={kv_len}, q_blocks_256={q_blocks_256}, "
            f"kv_blocks_256={kv_blocks_256}, topk_logical={topk_logical}, "
            f"avg_topk_kernel={avg_topk_kernel:.2f}"
        )
        print(
            "vbs_256 stats: "
            f"min={int(variable_block_sizes_256.min().item())}, "
            f"max={int(variable_block_sizes_256.max().item())}, "
            f"mean={float(variable_block_sizes_256.float().mean().item()):.2f}"
        )
        print(f"finite_check: out={out_finite}, lse={lse_finite}")
        print(f"kernel_only: {kernel_ms:.3f} ms | {kernel_tflops:.2f} TFLOPs (approx)")
        print(f"manual_e2e:  {e2e_ms:.3f} ms | {e2e_tflops:.2f} TFLOPs (approx)")
        print(
            f"wrapper_sparse_only: {wrapper_sparse_ms:.3f} ms | "
            f"{wrapper_sparse_tflops:.2f} TFLOPs (approx)"
        )
        print(
            f"wrapper_e2e(video_sparse_attn): {wrapper_e2e_ms:.3f} ms | "
            f"{wrapper_e2e_tflops:.2f} TFLOPs (approx)"
        )
        print("breakdown(prep+kernel):")
        print(f"  prep.logical_mask(topk):         {prep_logical_ms:.3f} ms")
        print(f"  prep.split_kv_mask(256->128):    {prep_split_mask_ms:.3f} ms")
        print(f"  prep.split_kv_sizes(256->128):   {prep_split_sizes_ms:.3f} ms")
        print(f"  prep.map_to_index:               {prep_index_ms:.3f} ms")
        print(f"  prep.kernel_inputs_total:        {prep_kernel_inputs_ms:.3f} ms")
        print(f"  kernel_only:                     {kernel_ms:.3f} ms")
        print(
            "note: wrapper_* includes CuTe token-level vbs mask_mod overhead; "
            "manual kernel/e2e path does not."
        )


if __name__ == "__main__":
    main()
