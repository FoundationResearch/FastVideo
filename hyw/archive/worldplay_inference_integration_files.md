## 目标

把 `HY-WorldPlay-main` 的 **WorldPlay inference**（i2v + pose(viewmats/Ks) + action + memory-chunk rollout）以 **FastVideo 原生 pipeline** 的方式接入，能通过：

- `fastvideo generate ...` 走通离线推理（offline）
- `fastvideo/entrypoints/streaming_generator.py`（或 MultiprocExecutor streaming API）走通 streaming（reset/step）

> 下面按“需要改哪些文件、怎么改”逐条细化。默认采用 **Strategy A（推荐）**：在 FastVideo 内新增一个 WorldPlay pipeline 与 stage；不建议直接复制 `trainer/` 框架。

---

## 接入点总览（FastVideo 现有机制）

- **pipeline 类解析入口**：`fastvideo/pipelines/__init__.py::build_pipeline`
  - 从 diffusers `model_index.json` 读取 `_class_name`
  - 去 `fastvideo/pipelines/pipeline_registry.py` 根据 `_class_name` 找到具体 pipeline class（按 `fastvideo/pipelines/basic/<arch>/...` 扫描 `EntryClass`）
- **用户入口**：
  - Python：`fastvideo/entrypoints/video_generator.py::VideoGenerator.generate_video`
  - CLI：`fastvideo/entrypoints/cli/generate.py`（把 CLI 参数拆成 init_args + generation_args，其中 generation_args 来自 `SamplingParam`）
- **state 承载**：`fastvideo/pipelines/pipeline_batch_info.py::ForwardBatch`
- **推理核心**：`fastvideo/pipelines/stages/denoising.py::DenoisingStage`（通用 denoise loop）
- **streaming 模式已存在**：`fastvideo/pipelines/stages/matrixgame_denoising.py`（参考实现 `streaming_reset/streaming_step`）

---

## 需要新增/修改的文件（Inference）

### 1) 新增 WorldPlay pipeline（组合 stages）

- **新增目录**：`fastvideo/pipelines/basic/worldplay/`
  - **新增文件**：`fastvideo/pipelines/basic/worldplay/worldplay_hy15_pipeline.py`
    - **做什么**：定义一个新的 `ComposedPipelineBase` pipeline，例如 `WorldPlayHunyuan15Pipeline`
    - **怎么做**：
      - stages 顺序建议：
        1. `InputValidationStage`
        2. `TextEncodingStage`（复用现有：Qwen2.5-VL + T5/byT5）
        3. `ConditioningStage`
        4. `TimestepPreparationStage`
        5. `LatentPreparationStage`
        6. `WorldPlayVisionEncodingStage`（替换/扩展现有 `Hy15ImageEncodingStage`，真正跑 SigLIP/vision encoder）
        7. `WorldPlayPoseActionStage`（把 `pose` → `viewmats/Ks/action_label` 张量）
        8. `WorldPlayDenoisingStage`（核心：chunk rollout + memory reconstitution）
        9. `DecodingStage`
      - 文件末尾必须暴露 `EntryClass = WorldPlayHunyuan15Pipeline`，否则 registry 扫描不到

> 参考现有 `fastvideo/pipelines/basic/hunyuan15/hunyuan15_pipeline.py` 与 `fastvideo/pipelines/basic/matrixgame/matrixgame_causal_dmd_pipeline.py` 的写法。

---

### 2) 注册 pipeline name → architecture folder

- **修改文件**：`fastvideo/pipelines/pipeline_registry.py`
  - **改什么**：在 `_PIPELINE_NAME_TO_ARCHITECTURE_NAME` 添加一条映射，让 `_class_name` 能解析到 `basic/worldplay/`
  - **示例**：
    - 加一行：`"WorldPlayHunyuan15Pipeline": "worldplay"`
  - **注意**：
    - `_class_name` 来自 `model_index.json`；如果你不想改模型文件，可以使用 `FastVideoArgs.override_pipeline_cls_name` 强制覆盖 `_class_name`（见下文 “参数/CLI”）

---

### 3) 增加 WorldPlay 推理所需的 batch 字段（pose/viewmats/Ks/action）

- **修改文件**：`fastvideo/pipelines/pipeline_batch_info.py`
  - **ForwardBatch 增加字段（建议）**：
    - `pose: str | dict | None = None`
    - `viewmats: torch.Tensor | None = None`  
      - shape 建议对齐 HY：`[B, L, 4, 4]`（L=latent_len）
    - `Ks: torch.Tensor | None = None`  
      - shape：`[B, L, 3, 3]`
    - `action_labels: torch.Tensor | None = None`  
      - shape：`[B, L]`（与 HY action_one_label 一致）
    - `worldplay_model_type: str = "ar"`（或放 SamplingParam）
    - `worldplay_chunk_latent_frames: int = 4`（AR 默认 4；BI 可 16）
    - streaming 需要的 state（建议放 `extra` 里，避免污染通用字段）：
      - `extra["worldplay_state"] = {...}`（比如 kv cache、history latents index 等）
  - **为什么要改**：后续 `WorldPlayPoseActionStage` 和 `WorldPlayDenoisingStage` 需要拿到这些条件。

---

### 4) 增加 SamplingParam（让 CLI/VideoGenerator 能传 pose 等参数）

- **修改文件**：`fastvideo/configs/sample/base.py`
  - **新增字段（建议放 SamplingParam，因为是 “每次生成”参数）**：
    - `pose: str | None = None`（既支持 `w-31` 这种 string，也支持 json path）
    - `with_ui: bool = False`（可选：生成键盘 overlay）
    - `worldplay_model_type: str = "ar"`（`ar`/`bi`）
    - `worldplay_chunk_latent_frames: int = 4`
    - `worldplay_enable_sr: bool = True`（如果后续实现 SR pipeline）
  - **并修改**：`SamplingParam.add_cli_args(...)` 增加 `--pose`、`--with-ui`、`--worldplay-model-type`、`--worldplay-chunk-latent-frames` 等 CLI 参数

> CLI 的 generation_args 是自动从 `SamplingParam` dataclass 字段收集的（见 `fastvideo/entrypoints/cli/generate.py`），所以把参数放 SamplingParam 最省事。

---

### 5) 加载 action ckpt（safetensors）并注入 transformer

- **修改文件**：`fastvideo/fastvideo_args.py`
  - **FastVideoArgs 增加字段（建议）**：
    - `worldplay_action_ckpt: str | None = None`（action 模型 safetensors 路径）
    - （可选）`worldplay_enable_prompt_rewrite: bool = False`
  - 并在 `FastVideoArgs.add_cli_args` 增加 `--worldplay-action-ckpt`

- **修改文件**：`fastvideo/models/loader/component_loader.py`
  - **改什么**：在 `TransformerLoader.load(...)` 完成 transformer `from_pretrained` 之后：
    - 如果 `fastvideo_args.worldplay_action_ckpt` 非空：
      - 调用 transformer 上的方法：`transformer.add_action_parameters()`（需要在 transformer 实现）
      - 用 `safetensors_load_file` 读 action_ckpt，并 `transformer.load_state_dict(state_dict, strict=True)`
  - **关键点**：
    - 这是推理正确性的核心：HY-WorldPlay 的行为来自 action_ckpt 的增量权重

- **（可能需要）修改文件**：`fastvideo/models/dits/...`（具体取决于当前 HunyuanVideo15 transformer 是否已有对应接口）
  - 如果 FastVideo 当前 transformer 没有 `add_action_parameters()` 或 forward 不支持 `viewmats/Ks/action`：
    - **新增 transformer 子类**（推荐，避免破坏现有模型）：
      - 新增：`fastvideo/models/dits/hunyuanvideo15_worldplay.py`
    - **修改 registry**：
      - `fastvideo/models/registry.py` 注册新 class name（供 diffusers `_class_name` 或 override 使用）
    - **修改 loader**：
      - 确保 `TransformerLoader` 能从 config 里 resolve 到这个新类（FastVideo 已有 ModelRegistry 机制）

---

### 6) Vision encoder（SigLIP）接入（替换 Hy15ImageEncodingStage 的“全零占位”）

- **新增 stage 文件**：`fastvideo/pipelines/stages/worldplay_vision_encoding.py`
  - **做什么**：对齐 `hyvideo/pipelines/worldplay_video_pipeline.py::_prepare_vision_states`
  - **输出**：`batch.image_embeds = [vision_states]`，shape `[B, 729, 1152]`
  - **预处理**：实现 `resize_and_center_crop` 逻辑（可直接复用/移植到 `fastvideo/models/vision_utils.py`）

- **修改 pipeline 文件**：`fastvideo/pipelines/basic/worldplay/worldplay_hy15_pipeline.py`
  - 把原来 HunYuan15 pipeline 里的 `Hy15ImageEncodingStage(image_encoder=None, ...)` 换成你新实现的 `WorldPlayVisionEncodingStage(image_encoder=self.get_module("image_encoder"), image_processor=self.get_module("image_processor"))`

- **修改 loader**（如果当前模型目录没有 image_encoder / image_processor diffusers config）
  - `fastvideo/models/loader/component_loader.py` 已支持 `image_encoder/image_processor` 作为 module_type
  - 你需要保证 pipeline 的 `_required_config_modules` 包含 `"image_encoder"` 和 `"image_processor"`，并且你的模型目录能提供这些组件权重路径（或你提供 override path）

---

### 7) Pose → viewmats/Ks/action 的生成逻辑

- **新增 util**：`fastvideo/utils/worldplay_pose.py`
  - **内容**：从 `hyvideo/generate.py` 移植/改写：
    - pose string parser（`w-31`, `right-1`, `up-4` 等）
    - json pose 读取（`assets/pose/*.json`）
    - `pose_to_input(...) -> (viewmats, Ks, action_labels)`

- **新增 stage**：`fastvideo/pipelines/stages/worldplay_pose_action.py`
  - **输入**：`batch.pose`（来自 SamplingParam）、`batch.num_frames`
  - **输出**：写入 `batch.viewmats / batch.Ks / batch.action_labels`
  - **校验**：对齐 HY 的约束：
    - `(num_frames - 1) % 4 == 0`
    - `latent_len = (num_frames - 1)//4 + 1`
    - pose json 的 key 数要等于 latent_len

---

### 8) WorldPlayDenoisingStage（核心：chunk rollout + memory）

- **新增 stage**：`fastvideo/pipelines/stages/worldplay_denoising.py`
  - **关键差异（相对 DenoisingStage）**：
    - 不再是简单 timestep loop；而是：
      - 以 chunk（latent chunk）为单位 rollout
      - 每个 chunk 内走 num_inference_steps 的 scheduler step
      - chunk>0 时执行 memory reconstitution：选择历史帧 index，把历史 latents/cond/viewmats/Ks/action 拼进去
    - 需要维护 kv-cache（参考 HY 的 `init_kv_cache` + `ar_txt_inference/ar_vision_inference`）
  - **需要调用 transformer.forward 的额外 kwargs**：
    - `viewmats=batch.viewmats`, `Ks=batch.Ks`, `action=batch.action_labels`
    - 以及 text/vision/byT5 的 states（FastVideo 里通常通过 `prompt_embeds` 与 `image_embeds` 表达）
  - **memory selection 算法**：
    - 第一版可直接把 `HY-WorldPlay-main/hyvideo/utils/retrieval_context.py` 的 `select_aligned_memory_frames` 移植到 `fastvideo/utils/worldplay_memory.py`

---

### 9) Streaming 接口（reset/step）

- **参考文件**：`fastvideo/pipelines/stages/matrixgame_denoising.py`
  - 已有 streaming pattern：缓存上下文 + kv_cache + `streaming_step` 只更新 action 条件并继续 denoise

- **新增/修改**：
  - `WorldPlayDenoisingStage` 增加：
    - `streaming_reset(forward_batch, fastvideo_args)`：初始化 latents/state/memory
    - `streaming_step(keyboard_action, mouse_action)`：
      - keyboard/mouse → 更新 pose 状态机（生成下一段 viewmats/Ks/action）
      - rollout 下一 chunk
  - **注意**：
    - WorldPlay 的 action 是 “pose→relative motion→label”；所以 keyboard/mouse 应该先驱动 pose 更新，再生成 action label（保持与权重训练分布一致）

---

## 参数与模型文件（落地建议）

### 推荐的运行方式（不改模型目录）

利用 FastVideo 的 override 参数：

- `--override-pipeline-cls-name WorldPlayHunyuan15Pipeline`
- `--worldplay-action-ckpt /path/to/ar_distilled_action_model/model.safetensors`

这样无需改 `model_index.json`，但代码必须注册好 `WorldPlayHunyuan15Pipeline`。

---

## 验收清单（Inference）

### Offline

- `fastvideo generate`：
  - 输入：`--image-path` + `--prompt` + `--pose w-31` + `--num-frames 125`
  - 输出：生成 mp4，且不报错

### Streaming

- `streaming_reset` 后连续 `streaming_step` N 次：
  - 显存不持续增长（无泄漏）
  - 输出视频连贯（chunk 拼接合理）


