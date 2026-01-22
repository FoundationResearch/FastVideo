### Training transformer（HY-WorldPlay / trainer 侧）架构理解（关键点）

这次 `compare_forward_train_vs_hyvideo.py` 在 trainer transformer 侧报：

- `expected input ... to have 65 channels, but got 64`

它明确揭示了 **training transformer 的图像输入 latent 并不是简单的 `(x_t || cond)` 拼接**，而是遵循 `PatchEmbed(concat_condition=True)` 的固定通道布局。

### 输入张量：`hidden_states` 的真实通道定义

trainer 侧 `PatchEmbed` 定义在：
- `HY-WorldPlay-main/trainer/models/hyvideo/models/transformers/modules/embed_layers.py`

其中逻辑是（concat 模式）：
- 若 `is_reshape_temporal_channels=False`：`in_chans = in_chans * 2 + 1`
- 若 `is_reshape_temporal_channels=True`：`in_chans = in_chans + in_chans//2 + 1`

对我们现在的 i2v 训练（`concat_condition=True` 且不 reshape temporal）而言：
- `C = VAE latent channels`（本 repo 预计算出来的 `latent.pt` 是 `(B, 32, T, H, W)`，所以 C=32）
- 因此 transformer 的 patch embed 实际期望输入通道数：
  - `2*C + 1 = 65`

也就是说 training transformer 侧的 `hidden_states` 应当是：

- **noisy video latents**：`x_t`，shape `(B, C, T, H, W)`（这里 C=32）
- **conditioning latents**：`cond_latents`，shape `(B, C, T, H, W)`（通常由 `image_cond` repeat 到 T）
- **conditioning mask channel**：`cond_mask`，shape `(B, 1, T, H, W)`

拼接后：
- `hidden_states = cat([x_t, cond_latents, cond_mask], dim=1)` → `(B, 2*C+1, T, H, W)`

我们之前 compare 脚本只做了 `cat([x_t, cond_latents])` 得到 64 通道，所以触发了该错误。

### 重要推论：trainer forward 并不会“帮你补齐 mask 通道”

在 `ARHunyuanVideo_1_5_DiffusionTransformer.forward()`（trainer 侧）里，开头直接：
- `img = x = hidden_states`
- 然后 `img = self.img_in(img)`

因此 **trainer transformer 假设上游已经把 concat 输入（含 mask channel）准备好**；这也是为什么训练 pipeline 能跑，而我们对齐 compare 时会踩坑。

### 条件注入（action / pose）在 transformer 内的位置

从 trainer forward 的顺序看：
- `img_in(img)`：先把图像 latent token 化（Conv3d patch embed）
- `vec = time_in(t)`：把时间步嵌入成 modulation vector
- `vec = vec + action_in(action)`：**action 以 additive 的方式进入 modulation vector**
- `viewmats/Ks`：随后参与后续 block 内部的 attention / prope 等机制（具体在各 block 实现里使用）

因此：**action/pose 在 training transformer 里确实是“真正参与 forward” 的条件**，不是只在数据里存在但没用。

### 对齐诊断脚本的影响

为了能严格比较 “trainer transformer” vs “hyvideo transformer”的 forward 输出：
- compare 脚本必须构造完全一致的 `hidden_states`，尤其要补上 `cond_mask` 这 1 个通道
- 并且确保两边模型/输入 dtype 一致（否则会出现 conv bias dtype mismatch）

### 分布式/并行状态依赖（容易踩坑）

trainer 侧 transformer 的 forward 会直接调用 HY-WorldPlay 的并行工具（例如 `get_sp_world_size()` / `get_sp_parallel_rank()`），这些依赖全局的 sequence-parallel group（`_SP`）。

因此：
- 在正常训练（`torchrun`）里，这些 group 会在启动阶段初始化
- 但在单进程脚本里直接 import 并 forward，会触发：
  - `AssertionError: sequence model parallel group is not initialized`

所以做“单脚本 forward 对齐诊断”时，需要在脚本里显式初始化一个单进程配置（world_size=1, tp=1, sp=1）。

### 额外发现：trainer vs hyvideo forward 不是“微小差异”

我们用同一份权重、同一份输入（包含 `x_t || cond_latents || cond_mask`、pose、action、prompt/vision states）直接比较：
- trainer-side transformer forward 输出 vs hyvideo-side `forward_bi` 输出

得到的差异量级（示例）：
- `mean_abs ~ 1.0`（latent 空间）

这说明：**train 预览和 eval 推理如果分别走 trainer/hyvideo 两套 forward，很可能是“实质不同的函数”**，足以导致：
- 训练时看起来“某些 step 已经 overfit”
- 但 eval 用 hyvideo pipeline 生成时仍然更像 base

因此需要一个更强的对齐验证：直接用 trainer transformer 跑完整的 50-step denoising（同 scheduler/shift），看生成结果是否与训练预览一致。

