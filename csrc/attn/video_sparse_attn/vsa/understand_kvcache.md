### 目标场景（Causal + VSA + KV cache）

- 时间维使用 VSA 的 tile 大小 `VSA_TILE_SIZE[0] = 4`，并且**生成时每个 block 恰好是 4 帧**（`num_frame_per_block = 4`），不做时间方向的 padding。
- attention 始终是「**前缀注意力**」：第 2 个 block 的 query 只能看到第 1、2 个 block 的 K/V（共 8 帧），以此类推。
- 我们希望 **KV cache 直接存下游 kernel 需要的一维 padded 格式**，而不是在每次前向时重新 `tile → pad` 整个前缀。

在这个设定下，可以把「KV cache + VSA」理解成：

> **KV cache = 把每个 4‑frame block 做完 VSA 的 tile+pad 之后得到的一维序列，简单地按时间顺序拼接起来。**  
> 只要时间维没有 padding，`tile(4 帧) ++ tile(4 帧) = tile(8 帧)` 在 1D 序列上是成立的。

---

### 1. 为什么 `tile(4) ++ tile(4) = tile(8)` 可以成立？

回顾 `understanded_padding.md` 里的 5×5, 2×2 toy（这里只拿时间维来类比）：

- `get_tile_partition_indices` 的本质是：  
  按 `(t_tile, h_tile, w_tile)` 扫描，再在每个 tile 内做局部 flatten。
- 对时间维来说，如果 `VSA_TILE_SIZE[0] = 4`，那么：
  - `T=4` 时：只有一个 `t_tile=0`，tile 覆盖 frame 0..3；
  - `T=8` 时：有两个 `t_tile=0,1`，分别覆盖 frame 0..3 和 frame 4..7。

关键点在于：

- 时间维不 padding，只是把 frame 分成连续的 4‑frame block。
- 每个 block 内部的 flatten 顺序，和整个 8 帧一起算 tile 时「对应 block 那一截」的顺序是一致的。

因此：

- 先对前 4 帧做一次 `tile(T=4)`，得到一段一维序列 `tile_0`；
- 再对后 4 帧做一次 `tile(T=4)`，得到一段一维序列 `tile_1`；
- 直接在一维上做 `tile_0 ++ tile_1`，等价于对整段 8 帧做一次 `tile(T=8)` 得到的结果。

这就是「按 block 做 tile+pad 再简单拼接」在时间维成立的原因。

---

### 2. KV cache 要长什么样？

下面讨论的是**单层 self‑attention 的 KV cache 设计**，对应 `CausalWanSelfAttention_VSA` / `CausalWanTransformerBlock_VSA`。

记：

- patch 之后的 3D 形状为  
  \[
  (T', H', W') = (\text{post\_patch\_num\_frames}, \text{post\_patch\_height}, \text{post\_patch\_width})
  \]
- 每次生成 4 帧（block） ⇒ `T_block = 4 / p_t` 个 patch‑frame。

**约定：**

- KV cache 里不要存「原始顺序的 K/V」，而是存**已经 tile+pad 之后的一维形式**：
  - `k_cache`: 形状大致为 `[B, L_tiled_total, num_heads, head_dim]`
  - `v_cache`: 同上
- 这里的 `L_tiled_total` 可以看成：
  \[
  L_{\text{tiled\_total}} 
  = n_{\text{blocks}} \times L_{\text{tiled\_per\_block}}
  \]
  其中 `L_{\text{tiled\_per\_block}}` 是**单个 4‑frame block** 做完 `tile+pad` 后的一维长度。

为了知道「一共有多少帧」「目前 tile 了多少 token」，KV cache 里还需要保存一些轻量的**元信息**，比如：

- `num_cached_blocks`: 已经缓存了多少个 4‑frame block；
- `dit_seq_shape_block`: 单个 block 的 `(T_block', H', W')`（patch 之后）；
- （可选）`attn_metadata`: 可以只存「当前 prefix T_total 对应的 metadata」，或者每次根据 `num_cached_blocks` 现算。

一个可能的数据结构（伪代码）：

```python
kv_cache = {
    "k": k_cache,                # [B, L_tiled_total, num_heads, head_dim]
    "v": v_cache,                # [B, L_tiled_total, num_heads, head_dim]
    "num_cached_blocks": int,    # 目前缓存了多少个 4‑frame block
    "dit_seq_shape_block": (T_block, H_prime, W_prime),
}
```

> 注意：不需要在 cache 里再存原始顺序的 K/V，也不需要存老 block 的 `gate_compress`。  
> `gate_compress` 只和「当前这一步的 Q」有关，用来压缩当前 query 对应的注意力权重。

---

### 3. 单 step 前向：如何更新 KV cache？

假设当前 step 要处理「第 `b` 个 4‑frame block」，  
输入给这一层的是**当前 block 的** `q, k, v`（已经在时间上切好，只包含这 4 帧）。

#### 3.1 当前 block 内部：先算好自己的 tile+pad

1. 先把当前 block 的 K/V 还原到 3D patch 网格上的顺序（与非 causal VSA 相同）；
2. 用**block 级别**的 `dit_seq_shape_block = (T_block', H', W')` 去算出：
   - `num_tiles_block`
   - `tile_partition_indices_block`
   - `variable_block_sizes_block`
   - `non_pad_index_block`
3. 调 `VideoSparseAttentionImpl.preprocess_qkv`（或者直接 `tile(...)`）得到：
   - `k_tiled_block`: `[B, L_tiled_per_block, num_heads, head_dim]`
   - `v_tiled_block`: 同上

这一部分和「非 causal 的 `WanTransformerBlock_VSA`」完全一样，只是现在只对「当前 4 帧」做。

#### 3.2 把新 block append 进 KV cache

有了 `k_tiled_block` / `v_tiled_block` 之后，更新 cache：

```python
if kv_cache is None:  # 第一个 block
    k_cache = k_tiled_block
    v_cache = v_tiled_block
    num_cached_blocks = 1
else:
    k_cache = torch.cat([kv_cache["k"], k_tiled_block], dim=1)
    v_cache = torch.cat([kv_cache["v"], v_tiled_block], dim=1)
    num_cached_blocks = kv_cache["num_cached_blocks"] + 1

new_kv_cache = {
    "k": k_cache,
    "v": v_cache,
    "num_cached_blocks": num_cached_blocks,
    "dit_seq_shape_block": kv_cache["dit_seq_shape_block"] if kv_cache is not None else dit_seq_shape_block,
}
```

这里利用的就是一开始的结论：  
「对每个 block 单独 tile 后在一维上拼接」等价于「对整个 prefix 一次性 tile」。

#### 3.3 本 step 的 Q 如何用 KV cache 做注意力？

这一点有两种可选设计，先在文档里把 trade‑off 摆清：

- **方案 A：只对当前 block 的 Q 做 VSA；K/V 用 cache 里的长序列**
  - Q：只包含当前 4 帧 ⇒ 长度为 `L_q_block = T_block' * H' * W'`；
  - K/V：使用 `k_cache` / `v_cache`（已经是「全 prefix tile 完」的一维序列）。
  - 这要求 VSA kernel 支持「`len(Q) < len(K) == len(V)`」的情况，  
    并且我们要能把「当前 block 的 Q」对应到全局 tile 轴上的某一段 index。

- **方案 B：把 prefix 里所有 Q 都重新算一遍，只输出最后 4 帧对应的部分**
  - 每一步都重新对**全部 prefix**做一次 `q → tile(q)`；
  - 优点：不需要处理 `len(Q) < len(K)` 的 corner case，接口最接近现有实现；
  - 缺点：时间复杂度退回到 \(O(T^2)\) 级别，只是避免了「老 K/V 反复 tile+pad」。

在没有完全确认 kernel 行为（尤其是 `video_sparse_attn` 的 Q/K 长度约束）之前，  
实现上可以先落地**方案 B（安全版）**：

1. 用 `num_cached_blocks` 推出当前 prefix 的总帧数 `T_total = num_cached_blocks * T_block`；
2. 用 `(T_total, H', W')` 调 `VideoSparseAttentionMetadataBuilder.build(...)` 得到全局 metadata；
3. 对「当前 prefix 全部 Q」做一次 tile；
4. 调 kernel 得到输出，再只取「最后 4 帧」对应的那一段，作为本 step 的结果。

---

### 4. 和现有 Causal KV cache 的关系

当前的 `CausalWanSelfAttention`（Flash / flex attention 版本）里，KV cache 大致是：

- `kv_cache["k"]`: `[B, L_max, num_heads, head_dim]`，按「时间展开的一维 token 序列」存；
- 辅助标量：
  - `global_end_index`: 全局已经写到的 token 下标；
  - `local_end_index`: 本地缓存窗口写到的下标（配合 `local_attn_size` 做 sliding window）；
  - 还有 `sink_size` 等实现 sliding window 的细节。

而在 VSA 版本里：

- 我们仍然希望「**causal 语义完全一致**」：Q 只能看到 prefix 的 K/V；
- 只是把「一维 token 轴」换成了「VSA tile 之后的一维轴」；
- sliding window（`local_attn_size`）可以先不做，或者在 tile 轴上再引入一个「只看最近若干 tile」的截断版本。

可以理解为：

> 旧版本：在「原始 token 轴」上做 KV cache + sliding window；  
> VSA 版本：在「tile 之后的 token 轴」上做 KV cache；  
> causal 性由「生成顺序 + prefix 追加」来保证，而不是显式 mask。

---

### 5. 面向 `CausalWanSelfAttention_VSA` 的接口约定

结合上面的设计，可以给出一个比较清晰的接口约束（对应当前 `causal_wanvideo.py` 里的 skeleton）：

```python
class CausalWanSelfAttention_VSA(nn.Module):
    def forward(
        self,
        q: torch.Tensor,          # [B, L_curr, num_heads, head_dim]，当前 block 的 query
        k: torch.Tensor,          # [B, L_curr, num_heads, head_dim]，当前 block 的 key（原始顺序）
        v: torch.Tensor,          # [B, L_curr, num_heads, head_dim]，当前 block 的 value（原始顺序）
        freqs_cis: tuple[Tensor, Tensor],
        block_mask: BlockMask | None,
        kv_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        ...
```

具体实现时：

- **训练阶段（不启用 KV cache）**：
  - 和非 causal VSA 一样，一次性对整个 `(T', H', W')` 做 tile+pad，然后走 `DistributedAttention_VSA`。
  - `block_mask` 可以选择用 / 不用（如果已经由 KV cache 语义保证因果性，可以省略 mask）。
- **推理 / self‑forcing 阶段（启用 KV cache）**：
  - 遵循第 3 节的流程：当前 block tile → append KV cache → 全 prefix 的 Q 做一次 VSA（先从方案 B 开始）。

这样一来：

- 非 causal 的 `WanTransformerBlock_VSA` 仍然完全复用现有逻辑；
- causal 版本只是在「Q/K/V 的来源」和「KV cache 的存取」上做了改动，
  底层 VSA kernel 与 metadata 复用现有实现即可。


