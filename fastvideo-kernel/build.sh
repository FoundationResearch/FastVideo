#!/bin/bash
set -ex

# Simple build script wrapping uv/pip
# Usage:
#   ./build.sh                 # local dev build (CMake AUTO toggles)
#   ./build.sh --h100          # build SM90/H100 kernels (old TK)
#   ./build.sh --b200          # build SM100/B200 kernels (new TK at include/tk_b200)
#   ./build.sh --both          # build both SM90 and SM100 kernels into one extension (larger wheel)
#   ./build.sh --rocm          # ROCm build (no CUDA kernels)

echo "Building fastvideo-kernel..."

# Ensure submodules are initialized if needed (tk)
git submodule update --init --recursive

# Install build dependencies
pip install scikit-build-core cmake ninja

GPU_BACKEND=CUDA
MODE="auto"
for arg in "$@"; do
    case "$arg" in
        --rocm)
            GPU_BACKEND=ROCM
            ;;
        --h100)
            MODE="h100"
            ;;
        --b200)
            MODE="b200"
            ;;
        --both)
            MODE="both"
            ;;
    esac
done

if [[ "${GPU_BACKEND}" == "CUDA" ]]; then
    case "${MODE}" in
        h100)
            # SM90a (H100) kernels (old TK)
            : "${TORCH_CUDA_ARCH_LIST:=9.0a}"
            export TORCH_CUDA_ARCH_LIST
            export CMAKE_ARGS="${CMAKE_ARGS:-} -DFASTVIDEO_KERNEL_BUILD_TK=ON -DFASTVIDEO_KERNEL_BUILD_TK_SM100=OFF -DCMAKE_CUDA_ARCHITECTURES=90a"
            ;;
        b200)
            # SM100 (B200) kernels (new TK)
            : "${TORCH_CUDA_ARCH_LIST:=10.0}"
            export TORCH_CUDA_ARCH_LIST
            export CMAKE_ARGS="${CMAKE_ARGS:-} -DFASTVIDEO_KERNEL_BUILD_TK=OFF -DFASTVIDEO_KERNEL_BUILD_TK_SM100=ON -DCMAKE_CUDA_ARCHITECTURES=100"
            ;;
        both)
            # Build both SM90a and SM100 kernels into one extension.
            : "${TORCH_CUDA_ARCH_LIST:=9.0a;10.0}"
            export TORCH_CUDA_ARCH_LIST
            export CMAKE_ARGS="${CMAKE_ARGS:-} -DFASTVIDEO_KERNEL_BUILD_TK=ON -DFASTVIDEO_KERNEL_BUILD_TK_SM100=ON -DCMAKE_CUDA_ARCHITECTURES=90a;100"
            ;;
        auto)
            # Let CMake decide (AUTO), but do not override user env.
            ;;
        *)
            echo "Unknown mode: ${MODE}" >&2
            exit 1
            ;;
    esac
fi

export CMAKE_ARGS="${CMAKE_ARGS:-} -DGPU_BACKEND=${GPU_BACKEND}"

echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST:-<unset>}"
echo "CMAKE_ARGS: ${CMAKE_ARGS:-<unset>}"
echo "GPU_BACKEND: ${GPU_BACKEND:-<unset>}"
# Build and install
# Use -v for verbose output
pip install . -v --no-build-isolation
