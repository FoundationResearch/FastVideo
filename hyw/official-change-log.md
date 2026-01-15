## Changes to `HY-WorldPlay-main/` (local fork)

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

