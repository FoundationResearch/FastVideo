## HY-WorldPlay：AR（chunk-wise causal）训练流程中文梳理（自上而下）

本文对照你当前使用的官方代码路径，解释：
- **AR 训练到底训练什么**、输入是什么、attention 怎么做；
- **不同长度视频**在训练/生成时怎么处理；
- **window_frames / memory_frames** 的含义与一个 step 里到底喂给模型多长的序列；
- **memory 选取**到底是按 *video frame*、*latent* 还是 *chunk* 作为单位（结论：主要按 **latent index / chunk 对齐**）。

> 你现在跑的入口通常是 `trainer/training/ar_hunyuan_w_mem_training_pipeline.py`，但真正训练实现基本都在 `trainer/training/ar_hunyuan_mem_training_pipeline.py`（wrapper 调用实现）。

---

### 0. 背景基础：什么是 video latent？怎么从帧数换算成 latent_T？

#### 0.1 VAE 时间压缩比与 latent_T 的公式
训练/推理都在 **latent 空间**做扩散（DiT 预测噪声/残差）。视频先经 VAE 编码为 `(B, C, T_latent, H_latent, W_latent)`。

在 trainer 侧，VAE 的时间维压缩比用 `temporal_compression_ratio`，编码得到的 latent 帧数是：

\[
T_{latent} = \left\lfloor \frac{F - 1}{r} \right\rfloor + 1 \quad\text{其中 } r = temporal\_compression\_ratio
\]

代码里就是（注意它直接 **截断**到这个长度）：

```65:77:hyw/HY-WorldPlay-main/trainer/models/vaes/common.py
def encode(self, x: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, num_frames, height, width = x.shape
    latent_num_frames = (num_frames -
                         1) // self.temporal_compression_ratio + 1
    ...
    latents = self._encode(x)[:, :, :latent_num_frames]
    return DiagonalGaussianDistribution(latents)
```

同理，解码回视频帧时：

\[
F \approx (T_{latent}-1)\cdot r + 1
\]

#### 0.2 这里 r 通常是多少？与你的 “13f/1chunk” 为何对应？
在 HY-WorldPlay / HunyuanVideo-1.5 这套里，时间压缩通常是 **4**（你的脚本/经验也符合）。

如果 `r=4`，那么：
- `F=13` → `T_latent=(13-1)//4+1=4`
- 反过来 `T_latent=4` → `F=(4-1)*4+1=13`

这就是你之前常说的 **“13 帧 = 1 个 chunk（4 个 latent）”** 的由来。

---

### 1. AR 训练的“大框架”：模型、chunk、attention 的关键约定

#### 1.1 chunk 的定义（单位是 latent，不是 video frame）
官方把 **1 个 chunk**固定为 **4 个 latent 帧**（不是 4 个视频帧）。

这个约定会同时出现在：
- 推理端：用 `chunk_latent_frames=4` 切分 `latent_frames`
- attention mask：用 `latent_seq_length * 4` 作为 1 chunk 的 token 数

推理端根据 latent_T 计算 chunk 数：

```1807:1815:hyw/HY-WorldPlay-main/hyvideo/pipelines/worldplay_video_pipeline.py
latent_frames = latents.shape[2]
...
self.chunk_num = latent_frames // chunk_latent_frames
self.chunk_latent_frames = chunk_latent_frames
...
if model_type == "ar":
    latents = self.ar_rollout(...)
```

#### 1.2 AR 的 attention：chunk 内双向，chunk 间因果（chunk-wise causal）
AR transformer 在 `torch_causal` 分支会构造一种 **chunk-wise causal mask**：
- chunk 内：全 1（双向 / full attention）
- chunk 间：第 i 个 chunk 只能 attend 到 `0..i` 的 chunk（只看历史 chunk）

关键实现（`chunk_seq_length = latent_seq_length * 4` 明确了 chunk=4 个 latent）：

```198:246:hyw/HY-WorldPlay-main/trainer/models/hyvideo/models/transformers/modules/attention.py
elif attn_mode == "torch_causal":
    ...
    latent_seq_length = int(attn_param["thw"][-1]) * int(attn_param["thw"][-2])
    chunk_seq_length = latent_seq_length * 4
    chunk_num = (vision_seq_length) // chunk_seq_length
    ...
    for i in range(chunk_num):
        ...
        for j in range(i + 1):
            ...
            # full attention within chunk i for j == i, causal for j < i
            causal_mask[start_i:end_i, start_j:end_j] = 1
```

> 所以：如果一次 forward 里真的包含很多 chunk，最后一个 chunk 理论上能看见全部历史 chunk。  
> 但实际训练不会一次喂 100 个 chunk（下面第 2 节会讲 dataset 如何把长视频采样成短序列）。

---

### 2. 数据集怎么把“任意长度视频”变成一个 step 的训练样本？

这一块核心在 `trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`（名字里 “w_mem” 就是 with memory）。

#### 2.1 先明确：dataset 内的时间轴是 latent 轴
dataset 读 `latent.pt` 后得到：
- `latent` 的时间长度用 `latent.shape[1]` 表示（代码里叫 `latent_length`）

它会检查 `latent_length >= window_frames` 才能训练，否则跳过样本，并且把 latent 长度裁到 **4 的倍数**（chunk 对齐）：

```484:492:hyw/HY-WorldPlay-main/trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py
latent_length = latent.shape[1]
if latent_length < self.window_frames:
    ...
else:
    max_frames = int(self.shared_state["max_frames"]) // 4 * 4
    max_length = min(max_frames, latent_length // 4 * 4)
latent = latent[:, :max_length, ...]
```

这里 `shared_state["max_frames"]` 实际也是用来限制 **latent_T** 的（它和 `latent_length` 比较，并且被整除到 4 的倍数）。

#### 2.2 `window_frames` 是什么？
`window_frames` 是你训练命令里 `--window_frames` 的值，dataset 存成 `self.window_frames`：

- **单位是 latent 帧（latent index）**，不是原始 video frame。
- 它定义了 “in-window” 训练时，直接取序列前 `window_frames` 个 latent 做训练。

```366:374:hyw/HY-WorldPlay-main/trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py
self.window_frames = window_frames
self.memory_frames = 20
```

#### 2.3 `memory_frames` 是什么？
`memory_frames` 在这份 dataset 里是**写死的 20**：
- 单位同样是 **latent index 的数量/上限**（最终会与 context 去重合并，实际数量可能略有变化）
- 用在 “outside-window / memory training” 的历史帧选择里

同上代码块。

#### 2.4 一个 step 实际会产生两种样本形态（in-window vs outside-window）
dataset 每次会按概率选一种形态：

**A) outside-window（概率 0.8）**：从窗口之后的某个 chunk 开始学“下一段”，但不会把整条长视频喂进去。它做的是：
- `pred_latent_size = 4`（一次预测 1 个 chunk）
- 采样一个 chunk-aligned 的 `current_frame_idx`（步长 4）
- 用 `select_aligned_memory_frames(...)` 从历史里挑 memory
- 最后把 **当前 chunk 的 4 个 latent**接到序列尾部

下面用 **ASCII** 画一下两种模式。注意单位全都是 **latent index**，并且 `pred_latent_size=4` 表示 **1 chunk = 4 个 latent**：

**B) in-window（概率 0.2）**：直接取开头一段连续 latent（长度 = `window_frames`）：

```text
原视频 latent 时间轴（latent_T 很长）:
  [0 1 2 3][4 5 6 7][8 9 10 11][12 13 14 15][16 17 18 19] ...

in-window 取法（window_frames=16 举例）:
  取前 16 个 latent（=4 chunks）
  -> 输入给模型的是:
     [0 1 2 3][4 5 6 7][8 9 10 11][12 13 14 15]

  这里要特别注意：
  - pred_latent_size=4 只用于 out-window（表示“当前要预测的 1 个 chunk”）
  - in-window 分支里 pred_latent_size = window_frames
  - 所以当 window_frames=16 时，模型会对这 16 个 latent 都输出预测（shape 与 latents 相同），loss 也默认覆盖全部 16（不像 out-window 会 mask 只监督最后 4 个）。
```

**A) out-window / memory（概率 0.8）**：从窗口之后选一个 “当前 chunk 起点” `current_frame_idx`（chunk 对齐），再从历史里挑一些 memory latent，最后把当前 chunk 拼到末尾：

```text
原视频 latent 时间轴（chunk 对齐，每 4 个 latent 一块）:
  [0 1 2 3][4 5 6 7][8 9 10 11] ... [t t+1 t+2 t+3] ... [T-4 T-3 T-2 T-1]
                              ^ current_frame_idx = t (t 是 4 的倍数，且 t >= window_frames)

out-window 最终喂给模型的序列（“被打包/重排”后的短序列）:
  [0 1 2 3] + [若干历史 memory chunks / context（不一定连续）] + [t t+1 t+2 t+3]

也就是说：
  - 模型不会看到全部历史 0..t-1
  - 而是看到：第一 chunk + 若干挑出来的历史片段 + 当前 chunk
  - 训练监督一般只落在最后这个“当前 chunk”（见后面 i2v_mask 逻辑）

补充两个容易漏掉但很关键的细节：
  1) out-window 的 memory 选择同时包含“近邻时间上下文” temporal_context_size：
     - 代码里调用 `select_aligned_memory_frames(..., temporal_context_size=12, ...)`
     - 这会强制把 `current_frame_idx` 之前最近的 12 个 latent（也就是最近 3 个 chunk）作为 context 加入集合；
     - 再在更早的历史里（按 chunk 对齐）基于 FOV overlap 挑少量 chunk 来补齐 memory。
```

```624:666:hyw/HY-WorldPlay-main/trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py
if select_prob < 0.8:
    select_window_out_flag = 1
    pred_latent_size = 4
    ...
    current_frame_idx = self.rng.randrange(start_idx, max_start + pred_latent_size, pred_latent_size)
    ...
    selected_history_frame_id = select_aligned_memory_frames(..., memory_frames=self.memory_frames, pred_latent_size=pred_latent_size, ...)
    selected_history_frame_id.extend(range(current_frame_idx, current_frame_idx + pred_latent_size))
    latent = latent[:, selected_history_frame_id]
```

```673:678:hyw/HY-WorldPlay-main/trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py
else:
    pred_latent_size = self.window_frames
    latent = latent[:, :pred_latent_size, ...]
```

#### 2.5 选出来的 memory 会“重新 apply 位置编码（positional encoding）”吗？
这里有两个“位置/编码”概念要分开看：**(1) 时空 RoPE（t/h/w）** 和 **(2) 相机 pose 的 ProPE（camera rope）**。

**(1) 时空 RoPE：会重新计算，但只基于“打包后的新序列形状”，不会保留原视频的绝对时间索引。**

原因很直接：dataset 把 `latent` / `w2c_list` / `intrinsic_list` / `action_for_pe` 都按 `selected_history_frame_id` 做了子集 + 重排：

```640:671:hyw/HY-WorldPlay-main/trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py
selected_history_frame_id = select_aligned_memory_frames(...)
selected_history_frame_id.extend(range(current_frame_idx, current_frame_idx + pred_latent_size))
latent = latent[:, selected_history_frame_id]
reset_w2c_list = w2c_list[selected_history_frame_id]
w2c_list = reset_w2c_list
reset_intrinsic_list = intrinsic_list[selected_history_frame_id]
intrinsic_list = reset_intrinsic_list
reset_action_for_pe = action_for_pe[selected_history_frame_id]
action_for_pe = reset_action_for_pe
```

而 transformer 侧计算 RoPE 的方式是：从 **当前输入张量的形状**算出 `(tt, th, tw)`，再生成 `get_rotary_pos_embed((tt, th, tw))`。它完全不知道 “这些帧原本在长视频里对应哪个绝对时间 idx”：

```783:792:hyw/HY-WorldPlay-main/trainer/models/hyvideo/models/transformers/ar_action_hunyuanvideo_1_5_transformer.py
bs, _, ot, oh, ow = x.shape
tt, th, tw = (
    ot // self.patch_size[0],
    oh // self.patch_size[1],
    ow // self.patch_size[2],
)
self.attn_param['thw'] = [tt, th, tw]
if freqs_cos is None and freqs_sin is None:
    freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))
```

所以：**被选出来的 memory 在时空 RoPE 上，会被当成一个“新的短视频序列”的第 0..tt-1 帧来编码**（按打包后的顺序）。这属于一个很重要的细节：它没有显式地传入原始时间戳/绝对帧号。

**(2) 相机 pose 的 ProPE：会按你选择后的 `viewmats/Ks` 重新编码（这部分保留了“每帧真实相机位姿”信息）。**

AR transformer 的注意力里会调用 `prope_qkv(viewmats=..., Ks=...)`，把相机位姿注入到 q/k/v（相当于一种 camera rope）：

```174:181:hyw/HY-WorldPlay-main/trainer/models/hyvideo/models/transformers/ar_action_hunyuanvideo_1_5_transformer.py
# 添加连续的camera pose，通过prope
img_q_prope, img_k_prope, img_v_prope, apply_fn_o = prope_qkv(
    img_q.permute(0, 2, 1, 3),
    img_k.permute(0, 2, 1, 3),
    img_v.permute(0, 2, 1, 3),
    viewmats=viewmats,
    Ks=Ks,
)
```

因为 dataset 在 out-window 分支里同步重排了 `w2c_list`（训练里传给 transformer 的 `viewmats`）和 `intrinsic_list`（传给 transformer 的 `Ks`），所以 **memory 帧对应的相机位姿编码仍然是正确的**；只是“纯时间位置”的 RoPE 不再代表原视频的绝对时间顺序。

> 这就是官方如何 “handle 不同长度视频” 的关键：  
> **长视频不会导致一次 attention 覆盖 100 个 chunk**，因为 dataset 会把它采样成 “窗口长度” 或 “memory+当前 chunk” 这种短序列。

---

### 3. memory 是怎么选的？选取单位到底是什么？

#### 3.1 结论先说：选的是 latent index（并且 chunk 对齐）
memory 选取的索引是 **latent index**（因此也天然是 **chunk-aligned** 的）。

证据来自两处：

1) `select_aligned_memory_frames` 里 `historical_clip_indices = range(4, ..., 4)` 明确按 4 步长扫历史（chunk 对齐），并且每次加入历史也是加 `start_idx..start_idx+3`（一个 chunk 的 4 个 latent）：

```213:292:hyw/HY-WorldPlay-main/trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py
historical_clip_indices = list(range(4, current_frame_idx - temporal_context_size, 4))
memory_frames_indices = [0,1,2,3]  # add the first chunk as context
...
for start_idx, _ in candidate_distances:
    if len(memory_frames_indices) >= memory_frames:
        break
    if start_idx not in memory_frames_indices:
        memory_frames_indices.extend(range(start_idx, start_idx + 4))
...
return sorted(list(selected_frames_set))
```

2) dataset 构造 `w2c_list` 的循环是 `for i in range(latent.shape[1])`，即按 latent 索引建的。它把 latent index 映射回原视频帧 key 时，用的是 `4*(i-1)+4`（说明 latent index ↔ 原视频帧是按 stride=4 对齐的）：

```518:520:hyw/HY-WorldPlay-main/trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py
for i in range(latent.shape[1]):
    t_key = pose_keys[0] if i == 0 else pose_keys[4 * (i - 1) + 4]
```

因此：
- **选取单位不是“video frame 逐帧”**；
- 也不是直接用 “chunk id” 存储；
- 而是用 **latent index**，并且通过步长 4 的规则保证 **chunk 对齐**。

#### 3.2 它按什么准则选 memory？（高层总结）
它的核心思想是：对当前要预测的 “query clip”（当前 chunk 的若干 latent），在历史里找一些 “视野相似” 的片段作为 memory。

从实现可以总结为：
- query：`[current_frame_idx, current_frame_idx+pred_latent_size)`（pred_latent_size=4）
- 候选历史：从 `hist_idx=4` 开始每隔 4（chunk 对齐）取一个起点
- 评分：用 `calculate_fov_overlap_similarity(...)` 计算 FOV overlap（越相似距离越小），对 query 内多个 latent 平均
- 选择：按距离从小到大选若干个历史 chunk（每次加 4 个 latent）
- 额外规则：始终包含 `[0,1,2,3]`（第一 chunk 当作 context），并与 `context_frames_indices` 做并集去重

---

### 4. 训练时监督落在哪里？（与 memory/window 的关系）

训练 pipeline 在 loss 里有一段关键逻辑：如果是 outside-window 形态并且启用了 causal，则只对最后一个 chunk 计算 loss（避免历史 memory 部分被当成训练目标）：

```764:766:hyw/HY-WorldPlay-main/trainer/training/ar_hunyuan_mem_training_pipeline.py
if training_batch.select_window_out_flag == 1 and self.causal:
    i2v_mask[:,:,:-4,...] = 0 # only compute the last chunk for outside window training
```

这对应你直觉里的“AR 训练”：历史 memory 主要用来作为条件，监督主要落在当前 chunk。

---

### 4.5 噪声/时间步（timestep）在时间维度上的分配方式（in-window vs out-window）

这里回答一个很容易困惑但非常关键的问题：**当 in-window 输入里包含多个 chunk 时，每个 chunk 的噪声时间步一样还是不一样？**

结论（对照训练代码）：
- **in-window**：timestep 是按 **chunk（4 个 latent）** 为单位分配的：**同一个 chunk 内 4 个 latent 的 timestep 相同**；不同 chunk 的 timestep 通常不同。
- **out-window（memory）**：在上面的基础上，代码会把“历史部分（除最后一个 chunk 外）”的 timestep 强行改成**很大的噪声**（随机落在 500～985），而最后一个 chunk（当前要学的 chunk）保留原本采样出来的 timestep。

对应实现都在 `_prepare_ar_dit_inputs`（注意 `chunk_latent_num=4`）：

```575:623:hyw/HY-WorldPlay-main/trainer/training/ar_hunyuan_mem_training_pipeline.py
# add a parameter: chunk_latent_num means number of latent in one chunk
chunk_latent_num = 4
first_chunk_num = 4
u = compute_density_for_timestep_sampling(
    weighting_scheme=self.training_args.weighting_scheme,
    batch_size=batch_size * ((latent_t - first_chunk_num) // chunk_latent_num + 1),
    ...
)
u = u.reshape(batch_size, -1)
...
# 关键：把每个采样到的 u 重复 4 次 -> 一个 chunk 的 4 个 latent 共用一个 timestep
u = u.unsqueeze(-1).repeat_interleave(chunk_latent_num, dim=-1).reshape(batch_size, -1).reshape(-1)
indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
indices = (self.noise_scheduler.config.num_train_timesteps - self.timestep_transform(indices, self.train_time_shift)).long()

# out-window（memory）额外处理：把除最后一个 chunk 外的 chunk timestep 改成高噪声
if training_batch.select_window_out_flag == 1:
    for i in range(0, indices.shape[0] - 4, 4):
        rand_val = torch.randint(500, 985, (1, ), device=latents.device)
        indices[i:i + 4] = rand_val

timesteps = self.noise_scheduler.timesteps[indices].to(device=self.device)
...
noisy_model_input = (1.0 - sigmas) * training_batch.latents + sigmas * noise
```

用一个直观例子说明（假设 in-window 输入 `window_frames=16`，即 4 个 chunk）：
- chunk0（latent 0..3）共享 timestep \(t_0\)
- chunk1（latent 4..7）共享 timestep \(t_1\)
- chunk2（latent 8..11）共享 timestep \(t_2\)
- chunk3（latent 12..15）共享 timestep \(t_3\)

而 out-window 时，代码会把 chunk0..chunk2 的 timestep 改成高噪声（500～985），只保留 chunk3（最后 4 个 latent）的原采样 timestep 用于“真正学习/监督”的那一段（并且 loss 也会被 mask 到最后 4 个 latent）。

### 5. 推理/生成如何处理不同长度视频？

推理端关键是：
- 根据 `latents.shape[2]`（latent_T）和 `chunk_latent_frames` 算 `chunk_num`
- `model_type=="ar"` 走 `ar_rollout`：按 chunk 逐段生成/细化

```1807:1815:hyw/HY-WorldPlay-main/hyvideo/pipelines/worldplay_video_pipeline.py
latent_frames = latents.shape[2]
self.chunk_num = latent_frames // chunk_latent_frames
...
if model_type == "ar":
    latents = self.ar_rollout(...)
```

实际使用上想要 “长度对齐/不丢尾巴”，最稳的做法是：
- 让 `latent_T` 是 **4 的倍数**（否则 `latent_frames // chunk_latent_frames` 会把尾巴丢掉）
- 进而让原视频帧数满足 \(F = 4\cdot(latent\_T-1)+1\)（例如 `latent_T=8` 对应 `F=29`）。

---

### 6. 用具体例子把 window/memory 讲清楚

假设 VAE 时间压缩比 `r=4`，chunk_latent_frames=4。

#### 例子 A：一个很长视频（假设有 100 个 chunk）
- 100 个 chunk → `latent_T = 100*4 = 400`

如果你设 `window_frames=16`：
- **in-window（20%）**：dataset 直接取 `latent[0:16]`  
  - 这一步模型看到的长度 = 16 latent = 4 chunk
- **outside-window（80%）**：dataset 会在 `{16,20,24,...,396}` 里抽一个 `current_frame_idx`，然后：
  - 从历史里挑 `memory_frames=20` 个 latent（+ 一些 context 去重合并）
  - 再拼上当前 chunk 的 4 个 latent
  - 这一步模型看到的长度 ≈ 24 latent = 6 chunk（数量级是常数，不随 100 chunk 线性增长）

所以训练不会把 100 个 chunk 全塞进一次 attention，自然也就不会 “注意力长度爆炸到不可训练”。

#### 例子 B：你当前的短视频（1chunk / 2chunk）
- 1chunk：`latent_T=4`（13 帧）
- 2chunk：`latent_T=8`（29 帧）

这时 outside-window 采样几乎会因为 “不够长”而 fallback 成 in-window（代码里 `max_start < start_idx` 会回退）。
因此你会看到训练基本就是：
- 1chunk：一次喂 4 个 latent
- 2chunk：一次喂 8 个 latent
memory_frames=20 在这种短序列下基本用不上（这是预期行为）。

