#!/usr/bin/env python3
"""
Benchmark VSA *wrapper* performance (forward + backward) and report TFLOPs.

This script benchmarks the autograd-enabled wrapper:
  - fastvideo_kernel.block_sparse_attn.block_sparse_attn

So measured time includes wrapper overhead (map->index conversion, dispatch) plus kernel time.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Tuple, Callable

import numpy as np
import torch

try:
    from triton.testing import do_bench
except Exception as e:  # pragma: no cover
    raise ImportError("This benchmark requires triton (for triton.testing.do_bench).") from e


BLOCK_M = 64
BLOCK_N = 64


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark FastVideo VSA block-sparse attention")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--head_dim", type=int, default=128, choices=[64, 128])
    p.add_argument("--topk", type=int, default=None, help="KV blocks per Q block (default: ~90%% sparsity)")
    p.add_argument("--q_seq_lens", type=int, nargs="+", default=[49152], help="Q sequence lengths (must be /64)")
    p.add_argument("--kv_seq_lens", type=int, nargs="+", default=None, help="KV sequence lengths (defaults to q_seq_len)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--force_triton", action="store_true", help="Force wrapper to use Triton path (if supported by shapes).")
    p.add_argument("--skip_backward", action="store_true", help="Benchmark forward only and skip backward timing.")
    p.add_argument("--breakdown", action="store_true", help="Print detailed forward time breakdown for wrapper path.")
    p.add_argument("--breakdown_rep", type=int, default=10, help="Repetitions for breakdown timing.")
    return p.parse_args()


def create_qkv(batch: int, heads: int, q_len: int, kv_len: int, d: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch, heads, q_len, d, dtype=dtype, device="cuda")
    k = torch.randn(batch, heads, kv_len, d, dtype=dtype, device="cuda")
    v = torch.randn(batch, heads, kv_len, d, dtype=dtype, device="cuda")
    return q, k, v


def make_block_map(bs: int, h: int, num_q_blocks: int, num_kv_blocks: int, topk: int) -> torch.Tensor:
    # block_map: [bs, h, num_q_blocks, num_kv_blocks] bool
    scores = torch.rand(bs, h, num_q_blocks, num_kv_blocks, device="cuda")
    topk = min(max(1, topk), num_kv_blocks)
    idx = torch.topk(scores, topk, dim=-1).indices
    block_map = torch.zeros(bs, h, num_q_blocks, num_kv_blocks, dtype=torch.bool, device="cuda")
    block_map.scatter_(-1, idx, True)
    return block_map


def flops_sparse_attention(bs: int, h: int, d: int, q_len: int, topk_blocks: int, block_n: int) -> float:
    # Approx: QK^T + PV, each is ~2*bs*h*q_len*(topk_blocks*block_n)*d
    return 4.0 * bs * h * d * q_len * (topk_blocks * block_n)


def bench_ms(fn: Callable[[], object], warmup: int, rep: int) -> float:
    return do_bench(fn, warmup=warmup, rep=rep, quantiles=None)


def _time_cuda_ms(fn: Callable[[], object], rep: int = 10, warmup: int = 2) -> float:
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)
    return total_ms / rep


def _print_breakdown_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    rep: int,
) -> None:
    from fastvideo_kernel.block_sparse_attn_cute_fwd import (
        _aggregate_q_block_map,
        _choose_q_sparse_block_size,
        _ensure_flash_attn_importable,
        _map_to_index,
        _get_vsa_mask_mod,
        BLOCK_N as CUTE_BLOCK_N,
    )

    _ensure_flash_attn_importable()
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from flash_attn.cute.interface import _flash_attn_fwd

    block_map_bool = block_map.to(torch.bool)
    if block_map_bool.dim() == 3:
        block_map_bool = block_map_bool.unsqueeze(0)
    q_sparse_block_size = _choose_q_sparse_block_size(q.shape[2], m_block_size=128)

    def _preprocess():
        sparse_map = _aggregate_q_block_map(
            block_map_bool, q_sparse_block_size=q_sparse_block_size
        )
        mask_block_idx, mask_block_cnt = _map_to_index(sparse_map)
        mask_block_idx = mask_block_idx.to(torch.int32).contiguous()
        mask_block_cnt = mask_block_cnt.to(torch.int32).contiguous()
        full_block_cnt = torch.zeros_like(mask_block_cnt)
        full_block_idx = torch.zeros_like(mask_block_idx)
        return BlockSparseTensorsTorch(
            full_block_cnt=full_block_cnt,
            full_block_idx=full_block_idx,
            mask_block_cnt=mask_block_cnt,
            mask_block_idx=mask_block_idx,
            block_size=(q_sparse_block_size, CUTE_BLOCK_N),
        )

    sparse_tensors = _preprocess()
    aux_tensors = [block_map_bool.contiguous()]
    mask_mod = _get_vsa_mask_mod()

    def _transpose_in():
        return (
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
        )

    q_cute, k_cute, v_cute = _transpose_in()

    def _kernel():
        return _flash_attn_fwd(
            q_cute,
            k_cute,
            v_cute,
            m_block_size=128,
            n_block_size=CUTE_BLOCK_N,
            mask_mod=mask_mod,
            block_sparse_tensors=sparse_tensors,
            causal=False,
            return_lse=True,
            aux_tensors=aux_tensors,
        )

    out_cute, _lse_cute = _kernel()

    def _transpose_out():
        return out_cute.transpose(1, 2).contiguous()

    preprocess_ms = _time_cuda_ms(_preprocess, rep=rep)
    transpose_in_ms = _time_cuda_ms(_transpose_in, rep=rep)
    kernel_ms = _time_cuda_ms(_kernel, rep=rep)
    transpose_out_ms = _time_cuda_ms(_transpose_out, rep=rep)
    total_ms = preprocess_ms + transpose_in_ms + kernel_ms + transpose_out_ms
    print("breakdown(cute_fwd):")
    print(
        f"  preprocess(map/index+sparse_tensors): {preprocess_ms:.3f} ms ({preprocess_ms / total_ms * 100:.1f}%)"
    )
    print(
        f"  transpose_in([B,H,T,D]->[B,T,H,D]): {transpose_in_ms:.3f} ms ({transpose_in_ms / total_ms * 100:.1f}%)"
    )
    print(f"  kernel(_flash_attn_fwd): {kernel_ms:.3f} ms ({kernel_ms / total_ms * 100:.1f}%)")
    print(
        f"  transpose_out([B,T,H,D]->[B,H,T,D]): {transpose_out_ms:.3f} ms ({transpose_out_ms / total_ms * 100:.1f}%)"
    )
    print(f"  subtotal: {total_ms:.3f} ms")


def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if args.force_triton:
        os.environ["FASTVIDEO_KERNEL_VSA_FORCE_TRITON"] = "1"

    from fastvideo_kernel.block_sparse_attn import block_sparse_attn

    bs, h, d = args.batch_size, args.num_heads, args.head_dim
    kv_seq_lens = args.kv_seq_lens
    if kv_seq_lens is None:
        kv_seq_lens = args.q_seq_lens
    if len(kv_seq_lens) != len(args.q_seq_lens):
        raise ValueError("kv_seq_lens must have the same number of entries as q_seq_lens (or be omitted).")

    print("VSA Block-Sparse Attention Benchmark (WRAPPER)")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"batch={bs}, heads={h}, head_dim={d}, dtype={args.dtype}")
    print(f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
    print("NOTE: timings include wrapper overhead (map->index + dispatch).")
    if args.force_triton:
        print("dispatch: forced Triton (FASTVIDEO_KERNEL_VSA_FORCE_TRITON=1)")
    else:
        print("dispatch: SM90 if available, else Triton")

    for q_len, kv_len in zip(args.q_seq_lens, kv_seq_lens):
        if q_len % BLOCK_M != 0 or kv_len % BLOCK_N != 0:
            print(f"[skip] q_len={q_len}, kv_len={kv_len} must be divisible by 64")
            continue

        num_q_blocks = q_len // BLOCK_M
        num_kv_blocks = kv_len // BLOCK_N
        topk = args.topk if args.topk is not None else max(1, num_kv_blocks // 10)
        topk = min(topk, num_kv_blocks)

        print("\n" + "=" * 80)
        print(f"q_len={q_len}, kv_len={kv_len}, num_q_blocks={num_q_blocks}, num_kv_blocks={num_kv_blocks}, topk={topk}")

        q, k, v = create_qkv(bs, h, q_len, kv_len, d, dtype)
        block_map = make_block_map(bs, h, num_q_blocks, num_kv_blocks, topk)

        # Variable block sizes: default full blocks (64 tokens per KV block)
        variable_block_sizes = torch.full((num_kv_blocks,), BLOCK_N, dtype=torch.int32, device="cuda")

        def _fwd():
            return block_sparse_attn(q, k, v, block_map, variable_block_sizes)

        fwd_ms = bench_ms(_fwd, warmup=args.warmup, rep=args.rep)

        flops = flops_sparse_attention(bs, h, d, q_len, topk, BLOCK_N)
        fwd_tflops = flops / fwd_ms * 1e-12 * 1e3

        print(f"fwd(wrapper): {fwd_ms:.3f} ms  | {fwd_tflops:.2f} TFLOPs (approx)")
        if args.breakdown and not args.force_triton:
            _print_breakdown_cute(
                q=q,
                k=k,
                v=v,
                block_map=block_map,
                variable_block_sizes=variable_block_sizes,
                rep=max(1, args.breakdown_rep),
            )
        if args.skip_backward:
            print("bwd(wrapper): skipped (--skip_backward)")
            continue

        # Backward benchmark (wrapper autograd). We build the graph once, then repeatedly run backward
        # on the retained graph so bwd timing excludes the forward compute.
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        o_, _aux_ = block_sparse_attn(q_, k_, v_, block_map, variable_block_sizes)
        og = torch.randn_like(o_)
        loss = (o_ * og).sum()

        for _ in range(max(1, args.warmup // 2)):
            torch.autograd.grad(loss, (q_, k_, v_), retain_graph=True)
        torch.cuda.synchronize()

        bwd_ms = bench_ms(
            lambda: torch.autograd.grad(loss, (q_, k_, v_), retain_graph=True),
            warmup=0,
            rep=max(5, args.rep // 2),
        )
        # Rough backward multiplier (attention backward typically ~2-3x forward)
        bwd_tflops = (2.5 * flops) / bwd_ms * 1e-12 * 1e3
        print(f"bwd(wrapper): {bwd_ms:.3f} ms  | {bwd_tflops:.2f} TFLOPs (approx)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    main()


