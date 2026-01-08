#!/bin/bash
#SBATCH --job-name=t2v
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=dmd_t2v_output/t2v_%j.out
#SBATCH --error=dmd_t2v_output/t2v_%j.err
#SBATCH --exclusive

# Basic Info
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
# different cache dir for different processes
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
export MASTER_PORT=29505
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="50632ebd88ffd970521cec9ab4a1a2d7e85bfc45"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# W&B run name: {gitcommit_id_前7位}-{vsa_sparsity}-{num_height}
GIT_COMMIT_SHORT="$(git -C "$(dirname "$0")/../.." rev-parse --short=7 HEAD 2>/dev/null || echo nogit)"
# Remember to change those numbers below - no auto sync exists!
VSA_SPARSITY="0.72"
NUM_HEIGHT="480"
export WANDB_NAME="${GIT_COMMIT_SHORT}-${VSA_SPARSITY}(0.72,0.03,25)-${NUM_HEIGHT}"
# export WANDB_NAME="test251220-2"
TEST_MODE=false

if [ "$TEST_MODE" = "true" ]; then
    echo "⚠️  TEST MODE ACTIVATED: WandB is DISABLED, debug output is shown."
    export FASTVIDEO_DEBUG_CAUSAL_VSA_QKLEN=1
    export WANDB_MODE="disabled"  # 核心：这行代码会彻底切断上传
else
    echo "✅  PRODUCTION MODE: WandB is ONLINE."
    export WANDB_MODE="online"
fi

# Configs
NUM_GPUS=2

# Model paths for Self-Forcing DMD distillation:
GENERATOR_MODEL_PATH="wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
REAL_SCORE_MODEL_PATH="Wan-AI/Wan2.1-T2V-14B-Diffusers"  # Teacher model
FAKE_SCORE_MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # Critic modelc

DATA_DIR="/home/hao_lab/alex/datas/VSA_SF/datas/train/mixkit-64_processed"
VALIDATION_DATASET_FILE="/home/hao_lab/alex/datas/VSA_SF/datas/valid/validation_64.json"
export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_VISIBLE_DEVICES=6,7
# IP=[MASTER NODE IP]

training_args=(
  --tracker_project_name alex_VSA_SFwan_t2v_distill_self_forcing_dmd  
  --wandb_run_name "$WANDB_NAME"
  --output_dir "/home/hao_lab/alex/datas/VSA_SF/outputs_VSA_backend"
  # IMPORTANT: force the student/generator transformer to the *causal* architecture so that
  # KV cache + start_frame RoPE offsets are actually used.
  # --override_transformer_cls_name "CausalWanTransformer3DModel"
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 24
  --num_height 480
  --num_width 832
  --enable_gradient_checkpointing_type "full"
  --log_visualization
  --simulate_generator_forward
  --num_frames 81
  --num_frame_per_block 4  # Frame generation block size for self-forcing (VSA 4-4-4)
  --enable_gradient_masking
  --gradient_mask_last_n_frames 21
)

parallel_args=(
  --num_gpus $NUM_GPUS # 64
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1 # 64
  --hsdp_shard_dim $NUM_GPUS
)

model_args=(
  --model_path $GENERATOR_MODEL_PATH  # TODO: check if you can remove this in this script
  --pretrained_model_name_or_path $GENERATOR_MODEL_PATH
  --real_score_model_path $REAL_SCORE_MODEL_PATH
  --fake_score_model_path $FAKE_SCORE_MODEL_PATH
)

dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 50
  --validation_sampling_steps "4"
  --validation_guidance_scale "6.0" # not used for dmd inference
)

optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 500
  --weight_only_checkpointing_steps 500
  --weight_decay 0.01
  --betas '0.0,0.999'
  --max_grad_norm 1.0
)

miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "fp32"
  --flow_shift 5
  --seed 1000
  --use_ema True
  --ema_decay 0.99
  --ema_start_step 100
  --init_weights_from_safetensors /home/hao_lab/alex/datas/ODE_INIT/conversion/model.safetensors
)

dmd_args=(
  --dmd_denoising_steps '1000,750,500,250'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --dfake_gen_update_ratio 5
  --real_score_guidance_scale 3.0
  --fake_score_learning_rate 8e-6
  --fake_score_betas '0.0,0.999'
  --warp_denoising_step
  # Attention backend overrides: student=VSA, teacher/critic=Flash
  --student_attention_backend VIDEO_SPARSE_ATTN
  --teacher_attention_backend FLASH_ATTN
  --critic_attention_backend FLASH_ATTN
)

self_forcing_args=(
  --independent_first_frame False  # Whether to treat first frame independently
  --same_step_across_blocks True  # Whether to use same denoising step across all blocks
  --last_step_only False  # Whether to only use the last denoising step
  --context_noise 0  # Amount of noise to add during context caching (0 = no noise)
)

vsa_args=(
  --VSA_sparsity 0.72
  --VSA_decay_rate 0.03
  --VSA_decay_interval_steps 25
)

torchrun \
--nnodes 1 \
--master_port $MASTER_PORT \
--nproc_per_node $NUM_GPUS \
    fastvideo/training/wan_self_forcing_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}" \
    "${self_forcing_args[@]}" \
    "${vsa_args[@]}"
