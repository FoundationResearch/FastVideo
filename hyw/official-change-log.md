## Changes to `HY-WorldPlay-main/` (local fork)

### 2026-01-19 — Log train-time decoded videos (gt/noisy/pred) to WandB for debugging train vs eval

- **Files**:
  - `hyw/HY-WorldPlay-main/trainer/trainer_args.py`
  - `hyw/HY-WorldPlay-main/trainer/training/ar_hunyuan_mem_training_pipeline.py`
  - `hyw/HY-WorldPlay-main/trainer/pipelines/pipeline_batch_info.py`
- **What**:
  - Add CLI flags:
    - `--train-video-log-steps` (int, default 0 disables)
    - `--train-video-log-fps` (int, default 25)
    - `--train-video-log-max-samples` (int, default 1)
  - During training (rank0 only), every `train_video_log_steps` steps decode and log 3 videos to WandB:
    - `train_video_gt`: decoded `training_batch.latents`
    - `train_video_noisy`: decoded `training_batch.noisy_model_input`
    - `train_video_pred`: decoded predicted \(x_0\) latents computed from the model output
  - Store `TrainingBatch.model_pred` for visualization when logging is enabled.
- **Why**:
  - Make it easy to visually compare train-time forward behavior vs eval/infer behavior and quickly spot mismatches (CFG, timestep schedule, conditioning usage, etc.).

- **Follow-up (same day)**:
  - Make train-time visualization robust across different trainer VAE implementations:
    - resolve scaling/shift factors best-effort (fallback to scaling=1.0 with warning)
    - if decode fails, skip logging instead of crashing training
  - Fix a bug where the VAE wrapper could be incorrectly unwrapped into `None` (causing `'NoneType' object has no attribute 'decode'`).
  - Change behavior to be **strict** for debugging: require a real VAE module + scaling factor; do not silently skip video logging.
  - Fix VAE=None in training visualization by lazily loading a dedicated VAE decoder from `${pretrained_model_name_or_path}/vae` (rank0 only).
  - Log a single side-by-side video panel to WandB (`train_video_gt_noisy_x0hat`) instead of 3 separate videos, for easier visual comparison. The third panel is always \(x_0\) estimate:
    - `precondition_outputs=True`: use model output directly as \(x_0\_hat\)
    - else: compute \(x_0\_hat = \epsilon - (\epsilon - x_0)\_hat\)
  - Overlay sampled diffusion timestep/sigma statistics onto the training preview video (top-left) and include them in the WandB caption for easier debugging. Updated to a compact two-line HUD (`step` on line 1, `t_mean` and `sigma_mean` on line 2) to fit low-res previews.

### 2026-01-14 — Fix out-of-range `current_frame_idx` during training (synthetic 125f / latent_T=32)

- **File**: `hyw/HY-WorldPlay-main/trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`
- **Problem observed**: Training frequently printed errors like
  - `The current frame index must be within the valid range of w2c_list and must be at least 3.36, 32`
  - where `len(w2c_list)=32` (latent-aligned poses), but `current_frame_idx` could become 32/36 due to an unsafe sampling range.
- **Root cause**:
  - `w2c_list`/`intrinsic_list` are constructed at **latent resolution** (`len == latent.shape[1]`).
  - The original sampling computed `max_index = latent_T - (window_frames - memory_frames)`. When `memory_frames > window_frames` (default `memory_frames=20`, common `window_frames=16`), `max_index` can exceed `latent_T`, and using `randint` (inclusive) can produce out-of-range indices.
- **Fix**:
  - Replace the `current_frame_idx` sampling with a **safe, chunk-aligned latent index** selection:
    - `current_frame_idx ∈ {window_frames, window_frames+4, ..., latent_T-4}`
  - If the sequence is too short to satisfy this, the code **falls back to the in-window path** (`select_window_out_flag=0`) instead of throwing and retrying.

### 2026-01-15 — Add "transformer from scratch" option (random init) for trainer

- **Files**:
  - `hyw/HY-WorldPlay-main/trainer/trainer_args.py`
  - `hyw/HY-WorldPlay-main/trainer/models/loader/fsdp_load.py`
  - `hyw/HY-WorldPlay-main/trainer/models/loader/component_loader.py`
- **What**:
  - Added CLI flag `--transformer-from-scratch` (bool).
  - When enabled, the transformer is created via `model_cls.from_config(model_cls.load_config(load_from_dir))` (random init) instead of `from_pretrained(load_from_dir)`.
- **Why**:
  - Enable experiments where the diffusion transformer is trained without starting from the pretrained transformer weights, while still reusing the same model architecture/config from `--load-from-dir`.

### 2026-01-15 — Allow WandB login via stored credentials (no explicit key required)

- **File**: `hyw/HY-WorldPlay-main/trainer/training/ar_hunyuan_mem_training_pipeline.py`
- **What**: change `wandb.login(key=training_args.wandb_key)` to `wandb.login(key=training_args.wandb_key or None)`
- **Why**: after running `wandb login`, users often don't want to pass an API key on every run; passing an empty string can fail, while `None` properly falls back to stored credentials / env.

### 2026-01-15 — Fix "no learning" bug: optimizer step was skipped when unclipped grad norm is large

- **File**: `hyw/HY-WorldPlay-main/trainer/training/ar_hunyuan_mem_training_pipeline.py`
- **Problem observed**: training could run for many steps with no visible change in outputs / loss stays high.
- **Root cause**:
  - The code did `if grad_norm < 10: optimizer.step()`, but `grad_norm` is the *pre-clipping* norm returned by `clip_grad_norm_...`.
  - With `--max-grad-norm 1.0`, gradients are clipped but the returned pre-clipping norm can remain >10, causing the optimizer step to be skipped forever.
- **Fix**:
  - Always perform `optimizer.step()`/`lr_scheduler.step()` when grad norm is finite; rely on clipping for stability.
  - Only skip the step for non-finite norms.

### 2026-01-16 — Fix action.json being ignored: dataset now checks file existence instead of magic path substring

- **File**: `hyw/HY-WorldPlay-main/trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`
- **Problem observed**: Custom datasets with explicit `action.json` files had their action labels ignored, causing the model to learn incorrect action→motion mappings.
- **Root cause**:
  - The original code checked `if 'latent_dataset_w_action' in latent_pt_path` to decide whether to use `action.json`.
  - If the `latent_path` didn't contain this magic substring, the code fell back to **on-the-fly action computation** from w2c matrices.
  - On-the-fly computation often produces **incorrect/inconsistent** action labels compared to ground truth (observed 58% mismatch rate):
    - Multiple directions can activate simultaneously (e.g., `LLLU` instead of `LL`)
    - Movement direction detection can be completely wrong (e.g., `D` detected instead of `A`)
- **Fix**:
  - Replace the magic-substring check with: `action_path = json_data.get("action_path"); use_action_json = action_path and os.path.exists(action_path)`
  - Now any dataset that provides `action_path` in its training JSON will correctly use the ground-truth action labels.
