# SPDX-License-Identifier: Apache-2.0
"""
Minimal CUDA workload for Nsight Systems (nsys) profiling.

Use this to verify nsys works or to profile a tiny kernel without loading any model.

  nsys profile -o minimal --trace=cuda,nvtx python scripts/profile_nsys_minimal.py

Or: ./scripts/profile_nsys_hyworld.sh
"""

import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("No CUDA; run on a GPU node for nsys to capture CUDA activity.")
        return

    # Small random workload so nsys has something to trace
    n = 4096
    a = torch.randn(n, n, device=device, dtype=torch.float32)
    b = torch.randn(n, n, device=device, dtype=torch.float32)
    for _ in range(10):
        c = torch.mm(a, b)
        torch.cuda.synchronize()
    print("Done. nsys trace will contain the matmul kernels.")


if __name__ == "__main__":
    main()
