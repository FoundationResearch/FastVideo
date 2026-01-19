#!/usr/bin/env bash
set -euo pipefail

# Small-scale HY-WorldPlay training on the synthetic circle dataset.
# Run from anywhere:
#   bash hyw/train/run_train_small.sh
# Or override:
#   MODEL_PATH=... ACTION_CKPT=... bash hyw/train/run_train_small.sh

export WANDB_MODE=${WANDB_MODE:-online}
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HYWORLD_ROOT="${REPO_ROOT}/hyw/HY-WorldPlay-main"

# --- Paths (defaults) ---
# These defaults match the typical output printed by hyw/HY-WorldPlay-main/download_models.py
: "${MODEL_PATH:=~/alex/weights/tencent/HunyuanVideo-1.5}"  # same as hyvideo/generate.py --model_path
: "${ACTION_CKPT:=~/alex/weights/tencent/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors}"  # a .safetensors FILE
# Expand "~" manually since parameter expansion doesn't expand it.
MODEL_PATH="${MODEL_PATH/#\~/$HOME}"
ACTION_CKPT="${ACTION_CKPT/#\~/$HOME}"

# If you want to train the transformer from scratch (random init), set:
#   TRANSFORMER_FROM_SCRATCH=true
# (Note: trainer's StoreBoolean only accepts 'true'/'false', not 0/1.)
: "${TRANSFORMER_FROM_SCRATCH:=false}"

# WandB online (optional):
#   export WANDB_MODE=online
#   export WANDB_API_KEY=...   # optional if you already ran `wandb login`
#   export WANDB_ENTITY=...
#   export WANDB_PROJECT=...
#   export WANDB_RUN_NAME=...  # you can override this per-run
: "${WANDB_API_KEY:=}"
: "${WANDB_ENTITY:=alexzms-ucsd}"
: "${WANDB_PROJECT:=hyw-official}"
: "${WANDB_RUN_NAME:=hyworld_sythcircle_small}"  # feel free to override
TRANSFORMER_DIR="${MODEL_PATH}/transformer/480p_i2v"      # transformer weights dir
AR_ACTION_CKPT="${ACTION_CKPT}"                           # trainer expects a safetensors FILE here (it uses load_file())

# Train-time wandb video logging (official debug hook)
# - Default: log every 100 steps (rank0 only), 1 sample, fps=25.
# - Set TRAIN_VIDEO_LOG_STEPS=0 to disable.
: "${TRAIN_VIDEO_LOG_STEPS:=100}"
: "${TRAIN_VIDEO_LOG_FPS:=25}"
: "${TRAIN_VIDEO_LOG_MAX_SAMPLES:=1}"

# Training data and output (can be overridden via env vars for different datasets like sythball)
: "${TRAIN_JSON:=${REPO_ROOT}/hyw/data/sythcircle_v1_125f_modelinput/sythcircle_v1_125f_train_for_hyworld.json}"
: "${OUT_DIR:=${REPO_ROOT}/hyw/outputs/hyworld_sythcircle_small}"
# Short-video/one-chunk overrides:
#   - WorldPlay uses F = 4*(L-1)+1 where L is #latent frames
#   - One chunk for AR corresponds to L=4 => F=13 frames
: "${NUM_FRAMES:=125}"
: "${NUM_LATENT_T:=32}"
: "${WINDOW_FRAMES:=16}"
: "${NUM_HEIGHT:=256}"
: "${NUM_WIDTH:=256}"
# Expand "~" in TRAIN_JSON and OUT_DIR
TRAIN_JSON="${TRAIN_JSON/#\~/$HOME}"
OUT_DIR="${OUT_DIR/#\~/$HOME}"

# Resume (optional):
#   RESUME_CKPT=hyw/outputs/hyworld_sythcircle_small/checkpoint-250 bash hyw/train/run_train_small.sh
# Note: pass the checkpoint directory (the one that contains `distributed_checkpoint/`).
: "${RESUME_CKPT:=}"

# GPU config (allow overrides for small/debug runs)
# - For tiny datasets (e.g. 1 sample), using 8 GPUs with drop_last=True can lead to 0 batches.
#   In that case, set NUM_GPUS=1 and CUDA_VISIBLE_DEVICES=0.
NUM_GPUS=${NUM_GPUS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_PORT=${MASTER_PORT:-29611}

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
if [[ -n "${RESUME_CKPT}" ]]; then
  # Resolve relative paths against repo root for convenience.
  if [[ "${RESUME_CKPT}" != /* ]]; then
    RESUME_CKPT="${REPO_ROOT}/${RESUME_CKPT}"
  fi
  if [ ! -d "${RESUME_CKPT}" ]; then
    echo "ERROR: RESUME_CKPT does not exist: ${RESUME_CKPT}" >&2
    exit 1
  fi
  if [ ! -d "${RESUME_CKPT}/distributed_checkpoint" ]; then
    echo "ERROR: RESUME_CKPT is missing distributed_checkpoint/: ${RESUME_CKPT}" >&2
    exit 1
  fi
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
  --num-height "${NUM_HEIGHT}" \
  --num-width "${NUM_WIDTH}" \
  --num-frames "${NUM_FRAMES}" \
  --train-batch-size 1 \
  --num-latent-t "${NUM_LATENT_T}" \
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
  --transformer-from-scratch "${TRANSFORMER_FROM_SCRATCH}" \
  --model-path "${MODEL_PATH}" \
  --inference-mode False \
  --json-path "${TRAIN_JSON}" \
  --causal \
  --action \
  --i2v-rate 0.2 \
  --train-time-shift 3.0 \
  --window-frames "${WINDOW_FRAMES}" \
  --max-train-steps 1000 \
  --train-sp-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 2e-5 \
  --mixed-precision "bf16" \
  --checkpointing-steps 100 \
  ${RESUME_CKPT:+--resume-from-checkpoint "${RESUME_CKPT}"} \
  --weight-decay 1e-4 \
  --max-grad-norm 1.0 \
  --checkpoints-total-limit 2 \
  --training-cfg-rate 0.0 \
  --not-apply-cfg-solver \
  --dit-precision "fp32" \
  --num-euler-timesteps 50 \
  --ema-start-step 0 \
  --train-video-log-steps "${TRAIN_VIDEO_LOG_STEPS}" \
  --train-video-log-fps "${TRAIN_VIDEO_LOG_FPS}" \
  --train-video-log-max-samples "${TRAIN_VIDEO_LOG_MAX_SAMPLES}" \
  ${WANDB_API_KEY:+--wandb-key "${WANDB_API_KEY}"} \
  ${WANDB_ENTITY:+--wandb-entity "${WANDB_ENTITY}"} \
  ${WANDB_PROJECT:+--tracker-project-name "${WANDB_PROJECT}"} \
  ${WANDB_RUN_NAME:+--wandb-run-name "${WANDB_RUN_NAME}"}
