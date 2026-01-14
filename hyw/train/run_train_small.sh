#!/usr/bin/env bash
set -euo pipefail

# Small-scale HY-WorldPlay training on the synthetic circle dataset.
# Run from anywhere:
#   bash hyw/train/run_train_small.sh
# Or override:
#   MODEL_PATH=... ACTION_CKPT=... bash hyw/train/run_train_small.sh

export CUDA_VISIBLE_DEVICES=0

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HYWORLD_ROOT="${REPO_ROOT}/hyw/HY-WorldPlay-main"

# --- Paths (defaults) ---
# These defaults match the typical output printed by hyw/HY-WorldPlay-main/download_models.py
: "${MODEL_PATH:=/mnt/fast-disks/hao_lab/alex/weights/tencent/HunyuanVideo-1.5}"  # same as hyvideo/generate.py --model_path
: "${ACTION_CKPT:=/mnt/fast-disks/hao_lab/alex/weights/tencent/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors}"  # a .safetensors FILE
TRANSFORMER_DIR="${MODEL_PATH}/transformer/480p_i2v"      # transformer weights dir
AR_ACTION_CKPT="${ACTION_CKPT}"                           # trainer expects a safetensors FILE here (it uses load_file())

TRAIN_JSON="${REPO_ROOT}/hyw/data/sythcircle_v0_modelinput/sythcircle_v0_train_for_hyworld.json"
OUT_DIR="${REPO_ROOT}/hyw/outputs/hyworld_sythcircle_small"

# 1 GPU smoke test
NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=29611

if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi
if [ ! -f "${AR_ACTION_CKPT}" ]; then
  echo "ERROR: ACTION_CKPT (safetensors) not found: ${AR_ACTION_CKPT}" >&2
  exit 1
fi
if [ ! -d "${TRANSFORMER_DIR}" ]; then
  echo "ERROR: TRANSFORMER_DIR not found: ${TRANSFORMER_DIR}" >&2
  exit 1
fi
if [ ! -f "${TRAIN_JSON}" ]; then
  echo "ERROR: TRAIN_JSON not found: ${TRAIN_JSON}" >&2
  echo "Hint: run hyw/train/make_training_json.py first (see hyw/train/README.md)." >&2
  exit 1
fi

cd "${HYWORLD_ROOT}"

# NOTE: Do NOT insert comment-only lines inside a backslash-continued command.
# In bash, a line that begins with '#' will end the continued command unless the
# previous line escapes the newline AND this line also escapes it. To avoid
# accidentally dropping required args, keep comments outside the torchrun arg list.

torchrun \
  --master_port=${MASTER_PORT} \
  --nproc_per_node=${NUM_GPUS} \
  --nnodes 1 \
  trainer/training/ar_hunyuan_w_mem_training_pipeline.py \
  --data-path "${REPO_ROOT}/hyw/data" \
  --dataloader-num-workers 0 \
  --num-height 256 \
  --num-width 256 \
  --num-frames 24 \
  --train-batch-size 1 \
  --num-latent-t 6 \
  --pretrained-model-name-or-path "${MODEL_PATH}" \
  --output-dir "${OUT_DIR}" \
  --mode finetuning \
  --workload-type i2v \
  --num-gpus ${NUM_GPUS} \
  --sp-size 1 \
  --tp-size 1 \
  --hsdp-replicate-dim 1 \
  --hsdp-shard-dim ${NUM_GPUS} \
  --cls-name "HunyuanTransformer3DARActionModel" \
  --load-from-dir "${TRANSFORMER_DIR}" \
  --ar-action-load-from-dir "${AR_ACTION_CKPT}" \
  --model-path "${MODEL_PATH}" \
  --inference-mode False \
  --json-path "${TRAIN_JSON}" \
  --causal \
  --action \
  --i2v-rate 0.2 \
  --train-time-shift 3.0 \
  --window-frames 6 \
  --max-train-steps 50 \
  --train-sp-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-5 \
  --mixed-precision "bf16" \
  --checkpointing-steps 25 \
  --weight-decay 1e-4 \
  --max-grad-norm 1.0 \
  --checkpoints-total-limit 2 \
  --training-cfg-rate 0.0 \
  --not-apply-cfg-solver \
  --dit-precision "fp32" \
  --num-euler-timesteps 50 \
  --ema-start-step 0


