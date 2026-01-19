# Synthetic Generator (Circle + WASD + Camera->Color)

目标：用 **不改动** `hyw/HY-WorldPlay-main/` 的前提下，生成一批“圆形小世界”的视频数据，并输出与 HY-WorldPlay 训练集读取代码一致的 `pose.json` / `action.json` 结构，保存到 `hyw/data/`。

## 1) HY-WorldPlay 训练期期望的数据格式（关键点）

HY-WorldPlay 的 `trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py` 在训练时会：

- 从一个 `manifest.json`（本 repo 里叫 `json_path`）逐条读样本；
- 每条样本至少需要：
  - `latent_path`: 一个 `.pt` 文件路径（里面含 `latent/prompt_embeds/vision_states/byt5...` 等）
  - `pose_path`: 一个 `.json` 路径（逐帧相机内外参）
  - 如果要用动作监督，还需要 `action_path`: 一个 `.json` 路径（逐帧 move/view action）

其中 **pose/action JSON 必须是“按原始帧索引”连续编号的 dict**（key 为 `"0"`, `"1"`, ...）。训练代码会按下面规则取帧：

- pose：latent 的第 `i` 帧取 `pose_keys[0]` 或 `pose_keys[4*(i-1)+4]`
- action：latent 的第 `i` 帧取 `action_keys[4*(i-1)+4]`

含义：**每 4 个原始视频帧对应 1 个 latent 帧**（因此 pose/action JSON 最安全的做法是存全量每帧的记录）。

本生成器当前只做 step1/step2：生成 **原始视频 + pose/action 标注**，并额外写出 `manifest_raw_*.json`（用于你后续做 preprocess→latent→训练时转换为真正训练 manifest）。

## 2) 生成器输出（落盘到 `hyw/data/`）

默认输出到：

- `hyw/data/sythcircle_v0/{train|val|test}/sample_00000/`
  - `video.mp4`
  - `pose.json`
  - `action.json`
- `hyw/data/sythcircle_v0/manifest_raw_{split}.json`

### `pose.json` 格式

每一帧一个条目：

- key: `"0"`, `"1"`, ...
- value: `{"intrinsic": [[...]], "w2c": [[...]], "K": [[...]], "extrinsic": [[...]]}`

训练代码会使用 `intrinsic` + `w2c`；`K/extrinsic` 是兼容字段（多余不影响）。

### `action.json` 格式

每一帧一个条目：

- key: `"0"`, `"1"`, ...
- value: `{"move_action": "W|A|S|D|", "view_action": "LR|LL|LU|LD|"}`  

注意：
- 训练里会用 `"W" in move_action` 的方式解析，所以这里用字符串更贴合原实现。
- **本生成器的真实运动方向可以是任意角度**（连续方向向量），但 `move_action` 仍会用 WASD 字符串做一个“粗粒度标签”（可能出现对角线如 `"WD"`）。

## 3) 如何运行

### 环境

在你的环境里执行（按你要求先激活 conda）：

```bash
conda activate alexfv
```

本仓库 `pyproject.toml` 已包含 `imageio/imageio-ffmpeg/pillow/numpy` 等依赖；如缺包你可以自行 `pip install ...`。

### 生成数据

从仓库根目录执行：

```bash
conda activate alexfv
python -m hyw.sythgenerator.generate_circle_dataset --split train --num_samples 8 --num_frames 125 --fps 12 --width 256 --height 256
```

### 生成 3D 场景（sythball：地面+球体，真实相机运动/转动渲染）

sythball 会生成一个简单的 3D 场景（无限平面 + 一个球体），并且：
- **WASD**：相机在世界坐标系的 XZ 平面平移
- **view_action**：相机 yaw/pitch 旋转

运行示例：

```bash
conda activate alexfv
python -m hyw.sythgenerator sythball --split train --num_samples 32 --num_frames 125 --fps 25 --width 256 --height 256
```

#### 简单/调试模式（相机直线移动、朝向不变、动作固定）

为了更容易做 overfit/debug，你可以把动作序列固定成“持续向一个方向移动”，并且禁用视角旋转：

```bash
conda activate alexfv
python -m hyw.sythgenerator sythball \
  --out_root ~/alex/FastVideo/hyw/data/sythball_simple_v1_13f \
  --split train --num_samples 32 --num_frames 13 --fps 25 --width 256 --height 256 --seed 0 \
  --fixed_move_action W --fixed_view_action ""
```

说明：
- `--fixed_move_action`: 固定 `move_action`（例如 `W/A/S/D/WA/...`），对所有 `t>0` 生效（`t=0` 仍为 `""`，更贴合 latent 对齐逻辑）。
- `--fixed_view_action`: 固定 `view_action`（例如 `""/LR/LL/LU/LD`）；用 `""` 表示相机朝向不变。

或者直接调用模块：

```bash
conda activate alexfv
python -m hyw.sythgenerator.generate_ball_dataset --split train --num_samples 8 --num_frames 125 --fps 12 --width 256 --height 256
```

指定输出目录：

```bash
conda activate alexfv
python -m hyw.sythgenerator.generate_circle_dataset \
  --out_root ~/alex/FastVideo/hyw/data/sythcircle_v1_125f \
  --split train --num_samples 16 --num_frames 125 --fps 12 --width 256 --height 256 --seed 36 \
  --macro_period 8 --move_dir_jitter_deg 8.0 --circle_radius_px 32
```

### 参数说明（与“任意角度方向/每N帧换大方向/边缘不越界”相关）

- `--macro_period`: 每 N 帧重新采样一次“移动大方向”（连续角度）。
- `--move_dir_jitter_deg`: 每帧在大方向周围做小幅角度抖动（更自然）。
- `--circle_radius_px`: 用来计算“屏幕边缘”边界，确保圆不会移动出画面。

## 4) 下一步（你要做 step3 训练时会用到）

你后续需要把 `video.mp4 + text` 预处理成 HY-WorldPlay 训练用的 `latent_path`（`.pt`），然后把 `manifest_raw_*.json` 转换为训练 manifest（每条至少含 `latent_path/pose_path/action_path`）。

本生成器已经把 `pose.json/action.json` 的字段名按 dataset 的读取逻辑对齐了，因此你只需要补齐 `latent_path` 这条链路即可进入训练。


