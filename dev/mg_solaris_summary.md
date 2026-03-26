# Matrix Game Solaris Training Infrastructure — Merge Plan

## Summary of Kaiqin's Code Changes

Kaiqin's `FastVideo_clean` repo (`/mnt/weka/home/hao.zhang/kaiqin/FastVideo_clean`) implements
Matrix Game Solaris training via the **old training infrastructure** (`fastvideo/training/` —
monolithic pipeline scripts with argparse). The key components are:

### 1. KV Cache (`fastvideo/models/dits/matrixgame/kv_cache.py`)
- Refactored `KVCache` dataclass with sliding-window update/get_view semantics
- `KVCacheDict` bundles 3 caches: main self-attention, mouse action, keyboard action
- `attend_with_kv_cache()` unified interface for cache-aware attention
- **Impact:** Self-forcing training + inference only (not standard finetuning)

### 2. Diffusion Forcing Scheduler (`fastvideo/models/schedulers/scheduling_diffusion_forcing.py`)
- New `DiffusionForcingScheduler` with sigma-based flow matching
- `add_noise_high()` for boundary-timestep constrained corruption
- Used in the self-forcing distillation pipeline for MatrixGame

### 3. Self-Forcing Distillation Pipeline (`fastvideo/training/matrixgame_self_forcing_distillation_pipeline.py`)
- ~1200 lines, extends `SelfForcingDistillationPipeline`
- 3-model DMD: generator (causal student), real_score (teacher), fake_score (critic)
- Block-wise rollout with `num_frame_per_block=3`, KV cache management per block
- Diagonal denoising warmup strategy
- Gradient masking (only last 21 frames get gradients)
- Action conditioning (mouse/keyboard) threaded through forward passes

### 4. Standard Training Pipeline (`fastvideo/training/matrixgame_training_pipeline.py`)
- Supervised I2V finetuning for MatrixGame
- Dataset: parquet with VAE latents, CLIP features, first-frame latent (16ch), mouse/keyboard actions
- Mouse remap `(-y, x)`, keyboard remap `6D → 23D`
- Concatenates `[noise, mask(4ch), image_latent(16ch)]` as model input

### 5. Dataset Schema
`pyarrow_schema_matrixgame` with action conditioning fields:
mouse_cond, keyboard_cond, pil_image, first_frame_latent, clip_feature

### 6. Known Issues
- `--enable_gradient_checkpointing_type "full"` crashes due to action KV cache
  incompatibility with gradient recomputation. No fix yet.

---

## Data & Model Paths

| Resource | Path |
|----------|------|
| Code | `/mnt/weka/home/hao.zhang/kaiqin/FastVideo_clean` |
| Models | `/mnt/weka/home/hao.zhang/kaiqin/mg_models/` |
| VPT train | `/mnt/weka/home/hao.zhang/kaiqin/solaris/datasets/vpt/vpt/train_81f` |
| VPT test | `/mnt/weka/home/hao.zhang/kaiqin/solaris/datasets/vpt/vpt/test_81f` |
| Solaris train | `/mnt/weka/home/hao.zhang/kaiqin/solaris/datasets/train_81f` |
| Solaris test | `/mnt/weka/home/hao.zhang/kaiqin/solaris/datasets/eval` and `train_81f_test` |

Key model checkpoints:
- `Solaris-SF-30K-8K-18K` — Self-Forcing causal student
- `Solaris-Causal-30K-8K-18K` — Causal student (alternative)
- `Solaris-30K-8K-18K` — Non-causal teacher/critic

---

## Execution Plan

### Phase 1: Sync Model-Level Code
- Copy updated `kv_cache.py` into `fastvideo/models/dits/matrixgame/`
- Copy `scheduling_diffusion_forcing.py` into `fastvideo/models/schedulers/`

### Phase 2: Model Wrappers (`fastvideo/train/models/matrixgame/`)
- `matrixgame.py` — wraps `MatrixGameWanModel` (non-causal, for teacher/critic)
  - `prepare_batch()`: handle CLIP features, first-frame latent, mouse/keyboard action remap
  - `predict_noise()`: pass action conditioning through transformer
  - `init_preprocessors()`: load CLIP encoder + VAE + MatrixGame dataloader
  - Model input: `[noise, mask(4ch), image_latent(16ch)]` concatenation
- `matrixgame_causal.py` — wraps causal variant (for student)
  - `predict_noise_streaming()`: KV cache management (main + mouse + keyboard via KVCacheDict)

### Phase 3: Method Layer Updates
- Update `SelfForcingMethod` for diagonal denoising warmup config
- Add `DiffusionForcingScheduler` as alternative scheduler option

### Phase 4: YAML Configs + Launch
- `examples/train/configs/matrixgame_self_forcing_solaris.yaml`
- `examples/train/configs/matrixgame_finetune_solaris.yaml`

Launch command:
```bash
torchrun --nnodes=1 --nproc_per_node=8 \
  -m fastvideo.train.entrypoint.train \
  --config examples/train/configs/matrixgame_self_forcing_solaris.yaml
```

### Phase 5: Validation Pipeline
- Wire `MatrixGameCausalDMDPipeline` as validation callback target

---

## Execution Order

| Step | Description | Effort |
|------|------------|--------|
| 1 | Sync KV cache + DiffusionForcingScheduler | Small |
| 2 | Create `fastvideo/train/models/matrixgame/` model wrappers | Medium-Large |
| 3 | Update `SelfForcingMethod` for diagonal denoising warmup | Small |
| 4 | Create YAML configs | Small |
| 5 | Test finetuning (simpler path first) | Medium |
| 6 | Test self-forcing distillation | Medium |
| 7 | Wire validation pipeline | Small |
