#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Profile with Nsight Systems (nsys).
# - No args: minimal CUDA workload (no model).
# - First arg "hyworld": HYWorld inference (basic_hyworld + hyworld_denoising DIT).
# - First arg = output name for minimal run; second arg after "hyworld" = output name.
#
# Prereqs: nsys in PATH (CUDA toolkit or Nsight Systems CLI).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_SCRIPT="examples/inference/basic/profile_hyworld_nsys.py"
OUTPUT_NAME="${2:-dit_profile-4gpu}"
echo "Profiling HYWorld (basic_hyworld -> hyworld_denoising DIT)..."

OUTPUT_NAME="${OUTPUT_NAME%.nsys-rep}"

if ! command -v nsys &>/dev/null; then
  echo "Error: nsys not found. Add CUDA toolkit bin to PATH."
  exit 1
fi

echo "Output: ${OUTPUT_NAME}.nsys-rep"
echo ""

nsys profile \
  -o "${OUTPUT_NAME}" \
  --trace=cuda,nvtx \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --stats=true \
  python "$PYTHON_SCRIPT"

echo ""
echo "Done. Open ${OUTPUT_NAME}.nsys-rep in Nsight Systems GUI; zoom into the denoising phase for DIT kernels."
