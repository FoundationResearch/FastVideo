#!/bin/bash
# DMD few-step i2v distillation of Wan2.1-Fun-1.3B-InP on the mixkit i2v dataset.
# Distills the base Fun-InP (50-step) into a 3-step student. Three models:
#   student (transformer, trainable) + teacher (real_score, frozen)
#   + critic (fake_score, trainable), all initialized from the base Fun-InP.
# 4-GPU config matches the proven i2v finetune layout. Run from the FastVideo/ repo root.

source ../nanoVideo/env.sh
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_NAME="dmd-i2v-mixkit"
export TOKENIZERS_PARALLELISM=false
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN

MODEL_PATH="weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers"
REAL_SCORE_MODEL_PATH="weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers"
FAKE_SCORE_MODEL_PATH="weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers"
DATA_DIR="data/mixkit_processed_i2v_1_3b_inp/combined_parquet_dataset/"
VALIDATION_DATASET_FILE="examples/training/finetune/Wan2.1-Fun-1.3B-InP/mixkit/validation.json"
NUM_GPUS=4

training_args=(
  --tracker_project_name "fastvideo-i2v-distill"
  --output_dir "data/mixkit_processed_i2v_1_3b_inp/outputs/wan_i2v_dmd"
  --max_train_steps 2000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 8
  --num_height 480
  --num_width 832
  --num_frames 77
  --enable_gradient_checkpointing_type "full"
)

# 4 GPUs: mirror the T2V DMD layout (no SP — DMD rollout is incompatible with
# sequence-parallel temporal splitting; pure data parallel via HSDP replicate).
parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 4
  --hsdp_shard_dim 1
)

# student + teacher(real_score) + critic(fake_score), all from base Fun-InP
model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
  --real_score_model_path $REAL_SCORE_MODEL_PATH
  --fake_score_model_path $FAKE_SCORE_MODEL_PATH
)

dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

# DMD validation runs few-step (3) inference of the student
validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 200
  --validation_sampling_steps "3"
  --validation_guidance_scale "6.0"
)

optimizer_args=(
  --learning_rate 2e-6
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 500
  --weight_only_checkpointing_steps 500
  --weight_decay 0.01
  --max_grad_norm 1.0
)

miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "fp32"
  --ema_start_step 0
  --flow_shift 8
  --seed 1000
)

# DMD few-step distillation (3 denoising steps)
dmd_args=(
  --dmd_denoising_steps '1000,757,522'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --generator_update_interval 5
  --real_score_guidance_scale 3.5
)

torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/wan_i2v_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}"
