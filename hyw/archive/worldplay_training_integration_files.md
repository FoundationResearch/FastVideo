## 目标

把 `HY-WorldPlay-main` 的 **AR action + memory trainer**（`trainer/training/ar_hunyuan_w_mem_training_pipeline.py`）映射为 FastVideo 的训练体系，实现：

- FastVideo 内新增一个训练 pipeline（继承 `fastvideo/training/training_pipeline.py::TrainingPipeline`）
- 新增对应的 parquet 数据 schema + preprocess 产数
- `torchrun ...` 训练可跑、可保存 ckpt、可 resume，且验证能用同一套 WorldPlay inference pipeline 做采样

---

## 接入点总览（FastVideo 训练现状）

- **训练框架基类**：`fastvideo/training/training_pipeline.py::TrainingPipeline`
  - 负责：optimizer/lr_scheduler、dataloader、日志、ckpt、分布式/SP 切分等通用逻辑
- **现有样例**：
  - `fastvideo/training/wan_training_pipeline.py`
  - `fastvideo/training/wan_i2v_training_pipeline.py`
  - `fastvideo/training/ode_causal_pipeline.py`
- **数据加载**：
  - schema 在 `fastvideo/dataset/dataloader/schema.py`
  - record creator 在 `fastvideo/dataset/dataloader/record_schema.py`
  - dataset decode/collate 在 `fastvideo/dataset/utils.py` 等

---

## 需要新增/修改的文件（Training）

### 1) 新增 parquet schema：WorldPlay 训练所需字段

HY-WorldPlay trainer 的样本结构（来自 `HY-WorldPlay-main/trainer/README.md`）包括：

- `latent`（视频 VAE latent）
- `prompt_embeds` + `prompt_mask`
- `image_cond`（i2v 第一帧 latent 或 cond latent）
- `vision_states`（SigLIP 特征）
- `byt5_text_states` + `byt5_text_mask`（可选：若你在 FastVideo 侧仍用 byT5）
- **WorldPlay 专用**：`viewmats`, `Ks`, `action_labels`

对应 FastVideo 的 schema 设计（当前统一用 `*_bytes + *_shape + *_dtype`）：

- **修改文件**：`fastvideo/dataset/dataloader/schema.py`
  - **新增**：`pyarrow_schema_worldplay_i2v`（建议命名）
  - **字段建议（最小可训集合）**：
    - `vae_latent_bytes/shape/dtype`（[C, T_latent, H_latent, W_latent]）
    - `text_embedding_bytes/shape/dtype`
    - `text_attention_mask_bytes/shape/dtype`（当前 schema 里缺这个；WorldPlay 训练需要 mask）
    - `vision_states_bytes/shape/dtype`（SigLIP 输出）
    - `first_frame_latent_bytes/shape/dtype`（image_cond）
    - `viewmats_bytes/shape/dtype`（[L, 4, 4] 或 [T_latent, 4, 4]）
    - `Ks_bytes/shape/dtype`（[L, 3, 3]）
    - `action_labels_bytes/shape/dtype`（[L]）
    - 以及 metadata：`caption`, `width/height/num_frames/fps` 等

> 注意：当前 FastVideo `pyarrow_schema_i2v` 已有 `clip_feature_bytes` 与 `first_frame_latent_bytes`，但 worldplay 的 `vision_states` 并不一定等价于 CLIP feature；建议新增独立字段避免混淆。

---

### 2) 新增 record creator：把 PreprocessBatch 写成 schema 对应的 parquet record

- **修改文件**：`fastvideo/dataset/dataloader/record_schema.py`
  - **新增函数**：`worldplay_i2v_record_creator(batch: PreprocessBatch) -> list[dict[str, Any]]`
  - **行为**：
    - 类似 `i2v_record_creator`，但额外写入：
      - `vision_states_bytes/...`
      - `viewmats_bytes/...`
      - `Ks_bytes/...`
      - `action_labels_bytes/...`
      - `text_attention_mask_bytes/...`（若 schema 增加）

---

### 3) 扩展 preprocess pipeline：产出训练数据（latents + 条件）

FastVideo 目前的 preprocess 入口是 `fastvideo/pipelines/preprocess/v1_preprocess.py`，按 `--preprocess_task` 分发到不同 preprocess pipeline。

- **新增文件（推荐）**：`fastvideo/pipelines/preprocess/preprocess_pipeline_worldplay_i2v.py`
  - **做什么**：
    - 读取原始数据（video/image + caption + trajectory/action log）
    - 计算：
      - VAE latent（视频）
      - 第一帧 cond latent（i2v）
      - text embedding + attention mask
      - vision_states（SigLIP）
      - viewmats/Ks/action_labels（由 trajectory/pose/action log 生成）
    - 调 `ParquetDatasetWriter` 写出 `pyarrow_schema_worldplay_i2v`

- **修改文件**：`fastvideo/pipelines/preprocess/v1_preprocess.py`
  - **新增 preprocess_task**：
    - `choices` 增加 `"worldplay_i2v"`
    - 分支里新增：
      - `elif args.preprocess_task == "worldplay_i2v": PreprocessPipeline = PreprocessPipeline_WorldPlay_I2V`

- **可能需要修改**：
  - `fastvideo/pipelines/pipeline_batch_info.py` 里的 `PreprocessBatch`（如果目前不含 viewmats/Ks/action_labels/attention_mask）
  - `fastvideo/models/vision_utils.py` 增加 SigLIP 对齐的 resize/crop
  - 新增 `fastvideo/utils/worldplay_pose.py`（与 inference 共用，生成 viewmats/Ks/action_labels）

---

### 4) 新增训练 pipeline：WorldPlay AR action+memory

- **新增文件**：`fastvideo/training/worldplay_ar_action_mem_training_pipeline.py`
  - **继承**：`TrainingPipeline`
  - **必须实现/覆盖**：
    - `set_schemas()`：将 `self.train_dataset_schema = pyarrow_schema_worldplay_i2v`
    - `_get_next_batch()`：
      - 从 dataloader batch 取出：
        - `vae_latent`
        - `text_embedding` + `text_attention_mask`
        - `vision_states`
        - `first_frame_latent`
        - `viewmats/Ks/action_labels`
      - move to device + dtype（bf16/fp16）
      - 组织成 transformer forward 所需格式
    - `initialize_validation_pipeline()`：
      - 复用 inference pipeline 做定期采样验证（推荐直接 `build_pipeline` 用你实现的 WorldPlay pipeline）
    - `train_step()`（或当前 FastVideo 训练 pipeline 的等效函数）：
      - 选择 timestep（对齐 HY 的 `train_time_shift` / FlowMatch scheduler shift）
      - 计算 noise_pred / velocity_pred
      - 计算 loss（MSE + weighting_scheme 等）
      - memory 训练逻辑：
        - window_frames（latent window）
        - 按需采样/构造 memory context（可先做简化版：仅窗口内 causal + action 条件；再逐步加入“reconstituted memory”）

- **修改文件（参数层）**：`fastvideo/fastvideo_args.py`
  - `TrainingArgs` 增加/对齐 HY 训练脚本参数（如果 FastVideo 当前没有）：
    - `window_frames: int`
    - `train_time_shift: float`
    - `i2v_rate: float`
    - `use_memory: bool`
    - （可选）`ar_action_load_from_dir`（用于从已有 action 模型继续训）
  - 并在 `TrainingArgs.add_cli_args` 增加对应 CLI 参数

> 说明：FastVideo 的 `TrainingArgs` 已经有大量字段；这里的重点是把 HY-WorldPlay 训练脚本里真正用到的语义（window_frames/train_time_shift/i2v_rate/use_memory）接进来，并确保 SP 约束能被检查。

---

### 5) 训练入口脚本（examples/scripts）

- **新增脚本（推荐）**：`examples/training/run_worldplay_ar_action_mem.sh`
  - 参考 `HY-WorldPlay-main/scripts/training/hyvideo15/run_ar_hunyuan_action_mem.sh`
  - 以及 FastVideo 已有 `examples/training/*`
  - 内容包括：
    - `torchrun --nproc_per_node=$NUM_GPUS fastvideo/training/worldplay_ar_action_mem_training_pipeline.py ...`
    - 传入：
      - `--data-path`（parquet 目录）
      - `--sp-size` / `--hsdp-*` / batch size
      - `--window-frames`（并 assert `window_frames % sp_size == 0`）
      - `--worldplay-action-ckpt`（如果训练需要加载）

---

### 6) 测试与校验（防止 schema/shape silently wrong）

- **新增单测（建议最小）**：
  - `fastvideo/tests/training/worldplay/test_worldplay_batch_shapes.py`
    - 构造一个假的 dataloader batch（或读一条 parquet）
    - assert：
      - `viewmats.shape == [B, L, 4, 4]`
      - `Ks.shape == [B, L, 3, 3]`
      - `action_labels.shape == [B, L]`
      - `vae_latent` 的 T_latent 与 L 对齐（`L == (num_frames-1)//4+1`）

---

## 与 inference 的共用修改（建议复用同一套工具）

为了避免 “训练/推理 viewmats/Ks/action 定义不一致”，建议：

- `fastvideo/utils/worldplay_pose.py` 作为唯一实现
- preprocess 与 inference 都调用它

---

## 验收清单（Training）

- **数据侧**
  - preprocess 输出的 parquet 能被 dataloader 正常 decode（无空字段/shape 错误）
- **训练侧**
  - 4 GPU（或你指定的 GPU 数）稳定跑 `N=1000` steps
  - loss 有下降趋势
  - checkpoint 可 resume
- **验证侧**
  - 每隔 `validation_steps` 用 inference pipeline 采样一段视频成功保存


