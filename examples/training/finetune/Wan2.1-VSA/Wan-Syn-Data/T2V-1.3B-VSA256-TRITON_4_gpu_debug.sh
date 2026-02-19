#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Wan 2.1 T2V 1.3B debug (4 GPU)
# VSA256 + forced Triton route
# -----------------------------

# Basic env
export WANDB_MODE="${WANDB_MODE:-online}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER}_$$}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NODE_RANK="${NODE_RANK:-0}"
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"

# Attention / VSA route
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
export FASTVIDEO_VSA_256=1
export FASTVIDEO_KERNEL_VSA_FORCE_TRITON=1
export FASTVIDEO_VSA_256_BACKEND=triton
export FASTVIDEO_VSA_WRAPPER_PRINT="${FASTVIDEO_VSA_WRAPPER_PRINT:-1}"
export FASTVIDEO_VSA_DISPATCH_LOG="${FASTVIDEO_VSA_DISPATCH_LOG:-1}"

# Required external env (set these before launch)
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"

# Config
NUM_GPUS="${NUM_GPUS:-4}"
MODEL_PATH="${MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
DATA_DIR="${DATA_DIR:-data/Wan-Syn_77x448x832_600k}"
VALIDATION_DATASET_FILE="${VALIDATION_DATASET_FILE:-examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_4.json}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/wan_t2v_finetune_VSA256_triton_debug}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NODE_RANK=${NODE_RANK}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_DIR=${DATA_DIR}"
echo "VALIDATION_DATASET_FILE=${VALIDATION_DATASET_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"

parallel_args=(
  --num_gpus "${NUM_GPUS}"
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim "${NUM_GPUS}"
  --hsdp_shard_dim 1
)

model_args=(
  --model_path "${MODEL_PATH}"
  --pretrained_model_name_or_path "${MODEL_PATH}"
)

dataset_args=(
  --data_path "${DATA_DIR}"
  --dataloader_num_workers 4
)

training_args=(
  --tracker_project_name wan_t2v_VSA256_triton_debug
  --output_dir "${OUTPUT_DIR}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 20
  --num_height 448
  --num_width 832
  --num_frames 77
  --enable_gradient_checkpointing_type full
)

validation_args=(
  --log_validation
  --validation_dataset_file "${VALIDATION_DATASET_FILE}"
  --validation_steps 100
  --validation_sampling_steps "50"
  --validation_guidance_scale "5.0"
)

optimizer_args=(
  --learning_rate 1e-6
  --mixed_precision bf16
  --weight_only_checkpointing_steps 200
  --training_state_checkpointing_steps 200
  --weight_decay 0.01
  --max_grad_norm 1.0
)

misc_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.1
  --dit_precision fp32
  --ema_start_step 0
  --flow_shift 1
  --seed 1000
)

vsa_args=(
  --VSA_decay_rate 0.03
  --VSA_decay_interval_steps 50
  --VSA_sparsity 0.9
)

torchrun \
  --nnodes 1 \
  --nproc_per_node "${NUM_GPUS}" \
  --node_rank "${NODE_RANK}" \
  --rdzv_backend c10d \
  --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  fastvideo/training/wan_training_pipeline.py \
  "${parallel_args[@]}" \
  "${model_args[@]}" \
  "${dataset_args[@]}" \
  "${training_args[@]}" \
  "${optimizer_args[@]}" \
  "${validation_args[@]}" \
  "${misc_args[@]}" \
  "${vsa_args[@]}"

