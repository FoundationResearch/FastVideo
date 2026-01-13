## 背景 & 目标

本文件规划：将 `hyw/HY-WorldPlay-main` 的 **inference**（WorldPlay i2v + action + memory streaming）与 **trainer**（AR action+memory 训练框架）以“FastVideo 原生方式”集成进 `fastvideo/` 代码库。

- **核心结论（读代码后的架构定位）**
  - HY-WorldPlay 的推理端主要在 `hyvideo/generate.py` + `hyvideo/pipelines/worldplay_video_pipeline.py`：Diffusers 风格 pipeline，支持 **chunked rollout（AR/BI）**、**memory frame selection**、**pose(viewmats/Ks) + action label 条件**、可选 SR。
  - HY-WorldPlay 的训练端主要在 `trainer/training/ar_hunyuan_w_mem_training_pipeline.py`（由 `scripts/training/hyvideo15/run_ar_hunyuan_action_mem.sh` 启动）：训练依赖“预处理后的样本”json（latent/prompt_embeds/image_cond/vision_states/byt5 等），并通过 SP/TP/HSDP 做分布式。
  - FastVideo 本身已经有非常接近的框架与接口：`fastvideo/entrypoints`、`fastvideo/configs/pipelines`、`fastvideo/pipelines`、`fastvideo/workflow/preprocess`、`fastvideo/training`，并且已经支持 **streaming_step(keyboard_action, mouse_action)**（见 `fastvideo/pipelines/stages/matrixgame_denoising.py`）。
  - 因此“集成 trainer”最佳路线不是复制一套 `trainer/`，而是把 WorldPlay 的 **action+memory+camera 条件**作为 FastVideo 的 **新 pipeline/model variant** 接入现有 registry/训练框架。

- **目标**
  - 在 FastVideo 中新增一个可用的 `WorldPlay(HY-World 1.5)` pipeline：支持 i2v、pose/trajectory 控制、action 条件、memory 一致性、chunk streaming。
  - 在 FastVideo 中新增对应训练 pipeline：AR action + memory（与 WorldPlay-8B 训练脚本对齐），并复用 FastVideo 的 preprocess/workflow、分布式、ckpt、logging。

- **非目标（第一阶段明确不做）**
  - RL 后训练（WorldCompass）与 distillation（Context Forcing）完整复现（可以作为后续扩展 milestone）。
  - WAN-5B lite pipeline 的全量对齐（先以 HunyuanVideo-1.5 backbone 为主，WAN 作为后续）。

---

## 代码阅读摘记（集成需要覆盖的关键能力）

### Inference（HY-WorldPlay）

- **入口**
  - `hyvideo/generate.py`: CLI、pose string/json → `viewmats/Ks/action`，然后调用 pipeline。
  - `hyvideo/pipelines/worldplay_video_pipeline.py`:
    - `HunyuanVideo_1_5_Pipeline.create_pipeline(...)`：加载 transformer/vae/scheduler/text&vision encoders，加载 action ckpt（safetensors）并 `add_action_parameters()`。
    - `__call__(...)`：准备 prompt embed、byt5 embed、vision states、cond latents（i2v）与 mask，然后走 `ar_rollout` / `bi_rollout`。
    - `ar_rollout/bi_rollout`：chunk 生成；关键是 **selected_frame_indices** 的 memory reconstitution（`select_aligned_memory_frames(...)`），并把 `viewmats/Ks/action` 传入 transformer。

### Trainer（HY-WorldPlay）

- **训练启动**
  - `scripts/training/hyvideo15/run_ar_hunyuan_action_mem.sh`：`torchrun trainer/training/ar_hunyuan_w_mem_training_pipeline.py ...`
- **数据假设**
  - `trainer/README.md` 明确：训练样本是预处理后的 json（latent/prompt_embeds/image_cond/vision_states/prompt_mask/byt5_text_states/byt5_text_mask），思路与 FastVideo preprocess 类似。
- **分布式/并行**
  - 参数里强依赖 `--sp_size`, `--hsdp_*`，训练 batch size 计算基于 SP group。

---

## 集成总体策略（推荐）

### Strategy A（推荐）：把 WorldPlay 作为 FastVideo 的“新 pipeline + 新 transformer variant”接入

- 复用 FastVideo 现有：
  - CLI/VideoGenerator 接口
  - Pipeline registry（`fastvideo/configs/pipelines/registry.py`）
  - preprocess/workflow（输出 parquet/json）
  - training pipelines（`fastvideo/training/*`）与分布式封装
  - streaming executor（`streaming_reset/streaming_step`）
- 新增/改造：
  - 一个 WorldPlay pipeline config（支持 i2v + action + memory + camera）
  - 一个 WorldPlay denoising stage / transformer forward 接口（需要接收 `viewmats/Ks/action`，并支持 memory selected frames 逻辑）
  - 一个训练 pipeline（AR action+memory）与对应 dataset schema

### Strategy B（备选/过渡）：先做“adapter wrapper”直接调用 `hyvideo` 推理

用于快速拿到“能跑通的结果”和权重/行为验证，但最终仍建议迁移到 Strategy A，才能享受 FastVideo 的 kernel/STA/VSA/分布式能力。

---

## Milestones（分步落地 & 验收标准）

> 每个 milestone 都要求：能复现一个端到端可运行流程，并且有明确可验收产物（CLI、单测或示例脚本、可比对输出）。

### Milestone 0：对齐依赖与工程边界（1–3 天）

- **工作内容**
  - 梳理 WorldPlay 必需依赖（byT5 glyph、SigLIP、Qwen2.5-VL、action ckpt safetensors、FlowMatchDiscreteScheduler 等）与 FastVideo 现有依赖的差异。
  - 明确哪些能力复用 FastVideo（建议优先复用：text encoders、VAE、attention backend、distributed init）。
  - 明确 license 边界（HY-WorldPlay 代码带 Tencent Hunyuan License；FastVideo 是 Apache-2.0，需在文档中注明“可选组件/外部依赖/用户自担”）。
- **产物**
  - `docs/inference/` or `docs/design/` 中新增一页：WorldPlay 集成说明（依赖/权重路径/许可提示）。
- **验收**
  - 文档中明确：最小可运行所需模型文件列表 & 目录结构。

### Milestone 1：WorldPlay Offline Inference MVP（能生成视频）（3–7 天）

- **目标**
  - FastVideo 内部提供一个“WorldPlay i2v 推理”pipeline，输入：image + prompt + pose(json/string) → 输出：mp4。
- **工作内容**
  - 增加 pipeline config：`HYWorldPlay15I2V480PConfig`（命名建议）：
    - 基于 `fastvideo/configs/pipelines/hunyuan15.py` 的 text encoder 配置（Qwen2.5-VL + byT5）；
    - 增加 vision encoder（SigLIP）配置；
    - 增加 action 条件与 memory 参数（chunk_latent_frames、memory_frames、temporal_context_size、pred_latent_size 等）。
  - 增加 inference 参数：
    - pose 输入（string/json path）
    - action_ckpt 路径（safetensors）
  - 先以 Strategy B 方式做 adapter（可选）：用最少改动把 `hyvideo/pipelines/worldplay_video_pipeline.py` 包一层进 `VideoGenerator`，确保权重与行为正确。
- **产物**
  - `examples/inference/` 新增 `worldplay_i2v.py`（或 shell）示例。
  - `docs/inference/` 新增 WorldPlay 推理示例（包含 pose string 示例）。
- **验收**
  - 单卡可跑：给定 image + pose `w-31`（或 json），稳定生成 `gen.mp4`。
  - 输出分辨率/帧数规则与 WorldPlay 一致（`(num_frames-1)%4==0`，latent 数要求对齐）。

### Milestone 2：Streaming Inference（接入 FastVideo 的 streaming_step）（5–10 天）

- **目标**
  - 像 MatrixGame 一样支持：
    - `streaming_reset(initial_frame/image/prompt/seed/...)`
    - `streaming_step(keyboard_action, mouse_action)` 每步生成下一 chunk（例如 16 帧）。
- **工作内容**
  - 把 WorldPlay 的 action 表达接到 FastVideo 的 streaming 输入：
    - 方案 1：保持 WorldPlay action label（`action_one_label`）作为内部状态，由 keyboard/mouse 驱动 pose 更新→`viewmats/Ks`→action label；
    - 方案 2：直接把 keyboard/mouse 作为条件输入（需要改 transformer conditioning，与原权重不一致，风险更高，不建议第一版做）。
  - 实现“相机轨迹状态机”：
    - 维护当前 camera extrinsic/intrinsic；
    - keyboard/mouse → delta motion/rotation；
    - 生成下一 chunk 所需的 `viewmats/Ks/action`（与 `hyvideo/generate.py` 的逻辑对齐）。
  - 实现 memory reconstitution（selected history frames）：
    - 复用 WorldPlay 的 `select_aligned_memory_frames` 思路（可先直接移植算法，后续再优化）。
- **产物**
  - `fastvideo/pipelines/basic/worldplay/...`（新 pipeline）或在现有 composed pipeline 中新增 stage。
  - `demo/` 或 `comfyui/` 增加一个 WorldPlay streaming demo（可选）。
- **验收**
  - 在 FastVideo streaming executor 下连续 step N 次不崩溃、显存不持续泄漏、输出视频连续可视。
  - 和离线推理结果在同 seed/同轨迹下“行为一致”（允许像素级不完全一致，但结构一致）。

### Milestone 3：数据预处理（生成训练所需样本）（3–7 天）

- **目标**
  - 在 FastVideo preprocess/workflow 里新增 WorldPlay 数据 schema：可批量产出训练所需字段（对齐 `trainer/README.md` 的 json 结构）。
- **工作内容**
  - 扩展 preprocess pipeline：
    - 输入原始视频/图像/文本/轨迹（pose 或 action log）
    - 输出训练样本字段：
      - `latent`（VAE encode 后的视频 latent）
      - `prompt_embeds` + `prompt_mask`
      - `byt5_text_states` + `byt5_text_mask`
      - `image_cond`（i2v image latent，可选）
      - `vision_states`（SigLIP 特征）
      - `viewmats/Ks/action`（WorldPlay 专用条件，训练必须）
  - 选择存储格式：
    - FastVideo 现状偏 parquet schema（`fastvideo/dataset/dataloader/schema.py`），建议新增对应 schema，避免大量 json 小文件 IO。
- **产物**
  - `scripts/preprocess/worldplay_preprocess.sh`（或 python 示例）
  - 新的 dataset schema + dataloader 支持
- **验收**
  - 用少量数据生成可训练的 parquet/manifest，并能被 dataloader 正常读出（shape/ dtype 全对）。

### Milestone 4：AR Action+Memory 训练 pipeline（对齐 WorldPlay 训练脚本）（7–14 天）

- **目标**
  - 在 FastVideo 内提供 `worldplay_ar_action_mem_training_pipeline.py`（命名建议），可用 `torchrun` 训练、保存 checkpoint、可 resume。
- **工作内容**
  - 对齐 HY-WorldPlay 训练参数语义：
    - `--window_frames`（latent window）
    - `--sp_size`（要求 `window_frames % sp_size == 0`）
    - `--train_time_shift`（对应 flow_shift/shift）
    - `--i2v_rate`、`--training_cfg_rate` 等
  - loss/weighting：
    - 对齐 FlowMatch/ Euler scheduler 的训练形式（参考 FastVideo 现有 flow-match / self-forcing / ode 相关实现）
  - validation：
    - 用 streaming/offline pipeline 做周期性采样验证（与训练同 backbone，避免再实现一套）。
- **产物**
  - 新训练 pipeline 文件 + 配套脚本（参考 `examples/training/` 风格）
  - 文档：如何从 preprocess 输出直接开训
- **验收**
  - 4 GPU/8 GPU（至少一种）能稳定跑满若干 step（如 1k step），loss 正常下降；checkpoint 可 resume；验证样例能生成视频。

### Milestone 5：性能/工程化（可选，持续迭代）

- **目标**
  - 让 WorldPlay 在 FastVideo 的 kernel/attention 体系下跑到“可用的实时/准实时”区间。
- **工作内容**
  - attention backend 选择（FLASH_ATTN / SDPA / VSA / STA）
  - offload 策略（layerwise/group offload）
  - quant（fp8 gemm / weight-only 等）：对齐 FastVideo 现有量化入口（`FastVideoArgs` 已有 override/quant 字段）。
- **验收**
  - 给定固定硬件（如 A100/H100/L40S）给出稳定的 latency & memory profiling（脚本+结果）。

---

## 风险点 & 缓解

- **权重/条件不一致风险**
  - WorldPlay 的 action 条件是 “pose→relative motion→action label” 的定义；若改成直接喂 keyboard/mouse，很容易与权重训练时分布不一致。
  - 缓解：第一版严格复刻 pose/action label 生成逻辑；keyboard/mouse 只是驱动 pose 状态机。

- **分布式并行语义差异（SP/HSDP）**
  - HY-WorldPlay trainer 和 FastVideo 的 distributed 抽象虽相似，但细节（group 划分、通信 op）可能不同。
  - 缓解：优先复用 FastVideo 的 parallel_state，并把 WorldPlay 训练的 shape 约束映射为 FastVideo 的训练 batch 组织方式。

- **数据 IO**
  - WorldPlay README 用 json 描述样本，但 FastVideo 更偏 parquet；直接用 json 会 IO 成瓶颈。
  - 缓解：以 parquet schema 落地为主，保留 json 导入作为兼容层。

---

## TODO（可立即开工的下一步）

- [ ] 选定 Strategy A/B（建议：M1 用 B 快速验权重；M2 起迁回 A）。
- [ ] 确认 WorldPlay 训练/推理的最小字段集合（尤其是 `viewmats/Ks/action` 的 shape 与 dtype）。
- [ ] 定义 FastVideo 内部的 WorldPlay pipeline 接口：offline + streaming 两种调用路径共用同一套 denoising stage。

