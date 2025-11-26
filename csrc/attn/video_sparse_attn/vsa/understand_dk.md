**是，你盯的这个点是对的：公式上 dK 是「所有 Q 行的和」，但在你这套设计里，pad 的 Q 行对 dK 的贡献是严格为 0，所以 kernel 里不用再额外给 Q 做 mask。**

### 1. 从公式看 dK 确实是「对所有 Q 行求和」

在块级形式里（忽略缩放常数）：

\[
dK_{\text{block}} \;+=\; dS_{\text{block}}^\top \, Q_{\text{block}}
\]

拆成行就是：

\[
dK \;=\; \sum_{i\in\text{所有 Q 行}} dS_i^\top Q_i
\]

所以如果你**真的把 pad 行当作「正常 Q 行」去参与这个 sum**，理论上会污染 dK。

---

### 2. 但在你现在的 pipeline 里，pad 行满足「dS=0」，所以贡献为 0

关键是链路：

\[
dK \leftarrow dS \leftarrow dP \leftarrow dO
\]

只要 pad 行的 \(dO_i = 0\)，那么这一行的 \(dP_i, dS_i\) 都是 0，进一步它对 dK 的贡献就是：

\[
dK \text{(来自该 pad 行)} = dS_i^\top Q_i = 0
\]

而在你当前的设计里，**pad 行的 \(dO\) 确实是 0**：

- 前向：  
  - `tile` 把真实 token 填到 `x_padded[:, non_pad_index]`，pad 位置永远是 0；  
  - `untile` 只从 `non_pad_index` 把真实位置抽出去给上层网络用。
- 反向：  
  - 上层网络的梯度只作用在 `untile` 的输出（真实位置）上；  
  - 通过 `untile` 的 backward，只有 `non_pad_index` 那几行会把梯度 scatter 回 `x_padded`；  
  - **padded 的行在计算图里从头到尾都没有被上层使用，所以它们的 `dO` 必然是 0。**

于是，在 SM90 backward 里：

- `og_smem`（dO 子块）在 pad 行是全 0；
- 这些行对应的 `dp_block_t / ds_block_t` 全是 0；
- 进而对 `dV`、`dK`、`dQ` 的增量全是 0。

所以虽然循环是 “KV‑centric” 并且数学上是「对所有 Q 行求和」，**但 pad 行那一项就是严格的 0，不会影响 dK 的数值**。

---

### 3. 对比 K/V 的 padding 为什么必须在 kernel 里管

- 对 K/V：softmax 的归一化轴在 K 这边，如果不在 kernel 里用 `lower_fill/right_fill` 把 pad 列设为 \(-\infty\)，这些列会参与归一化，改变**真实列的概率分布和梯度**，所以必须 mask。
- 对 Q：softmax 是「每一行自己算一遍」，pad 行和真实行互不竞争；pad 行的输出和梯度又在 tile/untile 这一级就被屏蔽掉了 → **它对其它 Q 行、以及对 dK/dV 的最终值不会有任何影响**。

---

**所以，回答你的问题：**

- 数学上 dK 确实是所有 Q 行的和；
- 但在当前实现中，pad Q 行的 `dO` 一定是 0 → `dS` 一定是 0 → 它们对 dK 的贡献是 0；
- 因此 **不需要在 kernel 内再额外对 Q 做 padding mask**，只需要保证外部的 tile/untile/grad 路径正确让 pad 行的梯度为 0。