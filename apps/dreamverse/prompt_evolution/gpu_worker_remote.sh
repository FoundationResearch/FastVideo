#!/bin/bash
# Wrapper run on a remote slurm node via `srun --jobid=<id> --overlap ... bash gpu_worker_remote.sh <work.json>`.
# CUDA_VISIBLE_DEVICES is passed in by srun --export. Everything (code, conda env, cuda
# libs, the work file) lives on the shared NFS mount, so the node just runs it.
# args: $1 = work json (shared path), $2 = gpu index to pin on this node
set -e
REPO=/home/hal-alex/workspace/FastVideo
cd "$REPO"
source apps/dreamverse/prompt_evolution/env.local.sh
# pin to ONE gpu here (slurm gres binding leaves all 4 visible; this overrides it)
export CUDA_VISIBLE_DEVICES="${2:-0}"
export METRIC_DEVICE=cuda:0
export PYTHONPATH="$REPO"
export ENABLE_TORCH_COMPILE=0
export LD_LIBRARY_PATH=/home/shared-bin/cuda-12.9/lib64:$LD_LIBRARY_PATH
echo "[remote] $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES work=$1"
exec /home/hal-alex/miniconda3/envs/alexfvi/bin/python \
    apps/dreamverse/prompt_evolution/gpu_worker.py --work "$1"
