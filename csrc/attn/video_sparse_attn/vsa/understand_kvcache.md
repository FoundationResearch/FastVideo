### 目标场景（Causal + VSA + KV cache）

- 时间维使用 VSA 的 tile 大小 `VSA_TILE_SIZE[0] = 4`，并且 **生成时每个 block 恰好是 4 帧**（`num_frame_per_block = 4`），不做时间方向的 padding。
- attention 始终是「**前缀注意力**」：第 2 个 block 的 query 只能看到第 1、2 个 block 的 K/V（共 8 帧），以此类推。
- 我们希望 **KV cache 直接存下游 kernel 需要的一维 padded 格式**，而不是在每次前向时重新 `tile → pad` 整个前缀。

在这个设定下，可以把「KV cache + VSA」理解成：

> **KV cache = 把每个 4‑frame block 做完 VSA 的 tile+pad 之后得到的一维序列，简单地按时间顺序拼接起来。**  
> 只要时间维没有 padding，`tile(4 帧) ++ tile(4 帧) = tile(8 帧)` 在 1D 序列上是成立的。

---

### 1. 为什么 `tile(4) ++ tile(4) = tile(8)` 可以成立？

回顾 `understanded_padding.md` 里的 5×5, 2×2 toy：

- `get_tile_partition_indices` 的本质是：  
  按 `(t_tile, h_tile, w_tile)` 扫描，再在每个 tile 内做局部 flatten。
- 对时间维来说，如果 `VSA_TILE_SIZE[0] = 4`，那么：
  - `T=4` 时：只有一个 `t_tile=0`，tile 覆盖 frame 0..3；
  - `T=8` 时：有两个 `t_tile=0,1`，分别覆盖 frame 0..3 和 frame 4..7。


