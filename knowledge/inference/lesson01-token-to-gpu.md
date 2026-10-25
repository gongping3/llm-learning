# 第 1 课：从一个 token 到 GPU —— LLM 推理完整过程

> **2026-10-20 重讲版**：学习者说明尚未学习本课内容，本版从零组织。10-19 的实验数据保留在“7. 实验验证”，仅作参考证据，不代表已经掌握。
>
> 与旧版的差异：旧版把“logits `(1,S,V)` 只取最后一行”表述为“其余位置在 Decode 阶段被丢弃”。本版区分两件事：**主干仍要处理全部 S 个 token（为了产生所有位置的 K/V）；只有 `lm_head` 的投影可以只算最后位置。**

## 1. 一句话定义

LLM 推理是：把 token id 序列送进模型做前向传播，得到每个位置对下一个 token 的分数（logits），生成时只取最后一个位置采样出下一个 token，再拼回序列，循环往复。

**大模型本质上是一个被反复调用的“下一个 token 预测器”。**

## 2. 核心原理

### 2.1 文本 → token id → 向量

Tokenizer 把文本切成 token，每个 token 对应词表中的一个整数 id。

```text
ids: (B, S)          dtype = int64
X = Embedding(ids)   # (B, S, d)
```

- B：batch 大小；S：序列长度；d：模型维度；V：词表大小。
- id 只是查表索引；Embedding 是一张 `(V, d)` 的矩阵，每个 id 取出一行向量。
- 真实模型 d 常见为 4096/8192；本课用 d=64 便于观察。
- 看到张量先问三件事：**shape、dtype、device**。

### 2.2 一次前向传播的 shape 链

参数：B=1、S=5、d=64、H=4、dh=16、V=1000。

| 环节 | 计算 | shape | 说明 |
|---|---|---|---|
| 输入 ids | — | (1, 5) | int64 |
| embedding | `X = emb(ids)` | (1, 5, 64) | 每个 token 一个向量 |
| Q / K / V | `X @ Wq`、`X @ Wk`、`X @ Wv` | 各 (1, 5, 64) | 三个不同的线性投影 |
| 多头切分 | `view(1,5,4,16).transpose(1,2)` | q/k/v 各 (1, 4, 5, 16) | H 个头并行 |
| attention 分数 | `q @ k.transpose(-2,-1) / sqrt(16)` | (1, 4, 5, 5) | 每个头一张 5×5 的相关性矩阵 |
| mask + softmax | 遮住未来位置，再按最后一维归一化 | (1, 4, 5, 5) | 每行权重和为 1 |
| 加权求和 | `attn @ v` | (1, 4, 5, 16) | 每个位置聚合历史信息 |
| 多头合并 | `transpose(1,2).reshape(1,5,64)` | (1, 5, 64) | 拼回模型维度 |
| 输出投影 + 残差 + Norm | `X + out @ Wo`，再 RMSNorm | (1, 5, 64) | shape 不变 |
| FFN + 残差 | `d → 4d → d` | (1, 5, 64) | 逐位置的前馈网络 |
| lm_head | `h @ lm_head.T` | (1, 5, 1000) | 每个位置一套词表分数 |
| 取最后位置 | `logits[:, -1, :]` | (1, 1000) | 只用来预测下一个 token |

主干压缩成一行：

```text
ids (1,5) → X (1,5,64) → Q/K/V (1,5,64) → heads (1,4,5,16)
→ scores (1,4,5,5) → context (1,4,5,16) → merge (1,5,64)
→ block (1,5,64) → logits (1,5,1000) → last (1,1000)
```

除了 attention 内部的 `(1,4,5,5)`，主干上大部分张量的 shape 都是 `(B,S,d)`。Transformer 的“层”主要是在这个固定形状上反复变换数值。

### 2.3 Q / K / V

```text
Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

三个 W 都是 `(d,d)` 的可学习矩阵。

- Q：这个位置想找什么。
- K：这个位置能提供什么线索。
- V：这个位置真正要传递的内容。
- 三者来自同一个 X，但投影矩阵不同，所以角色不同。

### 2.4 Attention：分数、mask、softmax

```text
scores = q @ k.transpose(-2, -1) / sqrt(dh)   # (B,H,S,S)
attn   = softmax(mask(scores), dim=-1)        # (B,H,S,S)
context = attn @ v                            # (B,H,S,dh)
```

- 点积衡量 q 与每个 k 的匹配程度。
- 除以 `sqrt(dh)` 控制方差，避免 softmax 饱和。
- causal mask：第 i 个位置只能看 j ≤ i；把 j > i 填 `-inf`，softmax 后权重为 0。
- `torch.triu(..., diagonal=1)` 遮的是严格上三角（未来位置），对角线（自己）保留。
- softmax 沿最后一维（key 维）做，每个 query 的权重和为 1。

### 2.5 多头合并与 Transformer block

```text
context.transpose(1, 2).reshape(B, S, d)   # (B,S,d)
```

- 多头把一个 d 维空间切成 H 个 dh 维子空间并行做 attention，再拼回来用 Wo 混合。
- 合并时因为 transpose 后非连续，通常要用 `reshape`（呼应实验 1/2）。
- 输出投影 `@ Wo` + 残差 + RMSNorm，shape 保持 `(B,S,d)`。
- FFN：`d → 4d → d`，逐位置作用，不跨位置交流。
- 跨位置的信息交流全部发生在 attention 里。

### 2.6 logits 与采样

```text
logits = h @ lm_head.T   # (B,S,V)
```

- logits 是未归一化分数，不是概率，也不是“一个词”。
- softmax 后得到词表上的概率分布；argmax 是贪心，按概率采样可加 temperature/top-k/top-p。
- 生成时只需要最后一个位置：`logits[:, -1, :]` → `(B,V)`。
- 关键区分：
  - **主干必须处理全部 S 个 token**，因为每个位置的 K/V 后面都要用。
  - **`lm_head` 的 `d→V` 投影可以只对最后一个位置做**；玩具实验对全部位置做，前 S-1 行是计算浪费，但便于观察 `(1,S,V)`。
- 训练时不同：每个位置都要预测下一个 token，所以 `(B,S,V)` 全部参与 loss。

### 2.7 Prefill 与 Decode

**Prefill**：prompt 的 S 个 token 一次并行过模型；矩阵乘维度大，GPU 利用率高，compute bound；产出第一个新 token 的 logits，并产生所有位置的 K/V。

**Decode**：每次只新增 1 个 token，自回归。没有 KV Cache 时，每步把”prompt + 已生成”整个序列重新过模型，只为拿最后一个位置的 logits；第 n 步（生成第 n 个新 token）要重算 S+n−1 个 token（S 为 prompt 长度）——序列长度每步 +1，整个序列都要重算。

两个后果：

1. **计算浪费**：生成 T 个 token，线性层工作量约 O(T²)；若每步连 attention 也重算，整体到 O(T³)。
2. **瓶颈转移**：Decode 每生成 1 个 token 都要把全部权重读一遍，属于 memory bound。

具体估算：7B 参数、fp16 ≈ 14 GB 权重；1 TB/s 带宽下，光读权重约 14 ms/token，理论上限约 70 token/s，实际更低。

有 KV Cache 后：

- 新 token 只过一次主干；
- k/v 追加进 cache；
- q 只和缓存的 K/V 做 attention；
- 旧 token 的投影和 FFN 不再重算；
- 线性层工作量降为 O(T)，attention 仍需 O(T²)（每步对历史 K/V 做点积）。

### 2.8 为什么这直接导向 KV Cache

K/V 只依赖已经出现的 token；生成第 n+1 个 token 时，前 n 个 token 的 K/V 不变。所以可以缓存，不必重算。细节（存多少、怎么管理）在第 2 课和 L2.3。

## 3. 最小代码示例

```python
import math
import torch

B, S, d, H, V = 1, 5, 64, 4, 1000
dh = d // H
ids = torch.tensor([[12, 57, 231, 8, 99]])          # (1,5)

X = torch.nn.Embedding(V, d)(ids)                    # (1,5,64)
Wq, Wk, Wv = (torch.randn(d, d) for _ in range(3))
q = (X @ Wq).view(B, S, H, dh).transpose(1, 2)       # (1,4,5,16)
k = (X @ Wk).view(B, S, H, dh).transpose(1, 2)
v = (X @ Wv).view(B, S, H, dh).transpose(1, 2)

scores = q @ k.transpose(-2, -1) / math.sqrt(dh)     # (1,4,5,5)
mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
attn = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
out = (attn @ v).transpose(1, 2).reshape(B, S, d)    # (1,5,64)

lm_head = torch.randn(V, d)
logits = out @ lm_head.T                             # (1,5,1000)
next_id = logits[:, -1].argmax(dim=-1)               # (1,)
```

> 实验代码见 `experiments/attention/exp03_token_to_logits.py`（2026-10-25 已在 `/home/gongping/vllm-env` 的 torch 2.11.0+cu130 环境运行验证，见 7.2）。

## 4. 我的理解

（2026-10-25 学习者不看讲义回答，原话整理。已通过 7.3 检验，判定"真正理解"。）

```text
ids (1,5) int64
→ X (1,5,64)                Embedding 查表
→ Q/K/V 各 (1,5,64)         X @ Wq/Wk/Wv，三个投影
→ q/k/v 各 (1,4,5,16)       view(1,5,4,16).transpose(1,2) 多头切分
→ scores (1,4,5,5)          q @ kᵀ / √16   ★ attention 特有
→ attn (1,4,5,5)            causal mask + softmax
→ context (1,4,5,16)        attn @ v
→ out (1,5,64)              transpose(1,2).reshape 多头合并；+ Wo/残差/Norm/FFN 后 shape 不变
→ logits (1,5,1000)         h @ lm_head.T
→ last (1,1000) → 采样 → 下一个 token id（标量）
```

attention 特有的 shape 是 scores 的 (1,4,5,5)，即 (B,H,S,S)：主干上几乎所有张量都是 (B,S,d) 的"每位置一个向量"形态，只有 attention 造出一张 S×S 的位置对位置相关性矩阵（query 行 × key 列），跨位置的信息交流只发生在这里；FFN、投影都是逐位置的。

生成时只取最后一个位置的 logits：位置 i 的 logits 预测的是第 i+1 个 token，只有最后位置能给出"当前序列之后的新 token"。但前面的 hidden state 大有用处——最后一个位置要 attend 到全部历史位置，它的 hidden state 是所有历史 K/V 加权求和的结果。所以**主干必须算完全部 S 个位置（为了产生所有位置的 K/V），能省的只是对前面位置做 lm_head 的 d→V 投影**。训练时相反：(B,S,V) 全部参与 loss。

（一处口误已纠正：无 cache 时 Decode 第 n 步重算的是 S+n−1 个 token，不是 n 个。）

## 5. 易错点

1. 以为模型“直接输出一个词”：实际输出整个词表的 logits，再采样。
2. 以为“只取最后位置”等于“主干只算最后一个 token”：主干要算全部 S 个位置以产生 K/V；能省的是 lm_head 的投影。
3. 把 causal mask 的对角线也遮掉：`diagonal=1` 遮的是未来，自己看自己是允许的。
4. 把 attention scores 当概率：softmax 之后才是权重。
5. 以为 permute 后的张量可以随意 view：非连续时改形状会失败，合并多头要用 reshape 或先 contiguous。
6. 把 Prefill 和 Decode 的瓶颈说反：Prefill 更偏 compute bound，Decode 更偏 memory bound。
7. 以为 Decode 重算是实现缺陷：没有 cache 时，attention 需要历史 K/V，只能重算。

## 6. 与其他知识的联系

- stride / view / reshape：多头切分与合并（实验 1/2）。
- KV Cache：本课 2.7/2.8 节，第 2 课展开。
- Continuous Batching：多个请求的 prefill/decode 混合调度（阶段 3）。
- PagedAttention：KV Cache 的块式显存管理（阶段 5）。
- RoPE：位置信息注入（L1.2/L1.3）。
- Sampler：temperature/top-k/top-p（L7.3）。

## 7. 实验验证

### 7.1 2026-10-19 历史记录（仅作参考）

`experiments/attention/exp03_token_to_logits.py`（torch 2.14.1+cpu，V=1000, d=64, H=4）：

| 环节 | shape | dtype | device |
|---|---|---|---|
| 输入 ids | (1, 5) | int64 | cpu |
| embedding | (1, 5, 64) | float32 | cpu |
| Q / K / V | (1, 5, 64) | float32 | cpu |
| 多头切分后 q | (1, 4, 5, 16) | float32 | cpu |
| scores | (1, 4, 5, 5) | float32 | cpu |
| 多头合并 | (1, 5, 64) | float32 | cpu |
| logits | (1, 5, 1000) | float32 | cpu |
| 最后位置 logits | (1000,) | float32 | cpu |

Prefill 采样得到 token 350；随后 Decode 逐步生成 30 → 892 → 895 → 548 → 892 → 892。

结论：Decode 每步都要把更长的整个序列重算一遍；累计计算量至少平方增长，这就是 KV Cache 的动机。（原表述"第 n 步重算 n 个 token"不严谨：应为 S+n−1 个，S 为 prompt 长度，见 2.7。）

### 7.2 2026-10-25 本次实验（已完成，判定：通过）

环境：`/home/gongping/vllm-env`（uv venv，torch 2.11.0+cu130，CPU 运行）。2026-10-20 记录的环境阻塞已解除。

学习者先写下预测，再运行 `experiments/attention/exp03_token_to_logits.py` 对照。预测与实际 **100% 吻合**：

- Prefill 链路 11 个 shape 全部正确：ids (1,5) → embedding (1,5,64) → Q/K/V (1,5,64) → q (1,4,5,16) → scores (1,4,5,5) → 合并 (1,5,64) → block (1,5,64) → logits (1,5,1000) → last (1000,)；
- Decode 每步序列长度 6→11、logits (1,S,1000)、每步重算 S 个 token，全部正确；
- 固定种子下采样值也全部预测正确（350 → 30 → 892 → 895 → 548 → 892 → 892），说明"每步整条序列重算"的机制理解到位。

一处小问题：结论句直接引用了实验代码第 85 行的旧表述（"第 n 步重算 n 个 token，累计平方增长"），该表述已随 2.7 修正，实验代码也已同步修正（见 7.4）。

### 7.3 检验问题（2026-10-25 已回答，判定：通过）

**第 1 题：ids → 下一个 token 的张量链。**
回答：11 个张量、shape 全对（完整链路见第 4 节）；指出 attention 特有的是 scores 的 (B,H,S,S)，理由正确——主干上几乎都是 (B,S,d) 逐位置形态，只有 attention 造出 S×S 跨位置相关性矩阵。
判定：✅ 真正理解（不只是报出 shape，还解释了为什么只有 attention 有 S×S 形态）。

**第 2 题：为什么只取最后位置 logits？前面的 hidden state 还有用吗？**
回答：位置 i 的 logits 预测第 i+1 个 token，前面 S−1 个位置的 logits 预测的都是已存在的 token，采样用不上；前面的 hidden state 有用——最后一个位置要 attend 所有历史位置，其 hidden state 是全部历史 K/V 的加权和。主干必须算完全部 S 个位置（产生所有位置的 K/V），能省的只是 lm_head 的 d→V 投影。训练时相反：(B,S,V) 全部参与 loss。
判定：✅ 抓住了 2026-10-20 重讲版的核心区分（主干全算 / lm_head 可省），未复现旧版"其余位置被丢弃"的错误表述。

**第 3 题：无 KV Cache 时 Decode 为什么重算？有 cache 后什么消失、什么保留？**
回答：无 cache 时，新 token 的 attention 需要全部历史 K/V，而 K/V 没有存下来，只能把整条序列重新过一遍模型；线性层工作量累计 O(T²)，连 attention 也重算则 O(T³)。有 cache 后消失的：旧 token 的 Embedding、Q/K/V 投影、旧 token 之间的 attention 计算、FFN——全部不重算，线性层降为 O(T)；仍要做的：新 token 自己过一次主干、Q 与全部缓存 K 点积、attn 与缓存 V 加权（attention 累计仍是 O(T²)，但只是点积）、新 token 的 FFN 与 lm_head、把新 K/V 追加进 cache。
判定：✅ 机制与复杂度全对。一处口误：序列长度应为 S+n−1（S=prompt 长度），说成了 n；讲义 2.7 原表述同样不严谨，已一并修正。

### 7.4 结论复述检验（2026-10-25，判定：通过，但经历 4 轮纠正）

问题：无 cache 时 Decode 生成第 n 个 token 究竟重算了什么？累计计算量什么量级？

| 轮次 | 回答 | 问题 |
|---|---|---|
| 1 | "重算了前 n-1 个 token 的 Q 与全部缓存 K 点积、attn 与缓存 V 加权" | 把"有 cache 后仍要做的事"安到了无 cache 场景；无 cache 时 K/V 根本没存，何来"缓存 K" |
| 2 | "要算 Embedding、Q/K/V 投影、attention、FFN、lm_head；有 cache 已经算过的 attention 不需要再算" | 缺"谁的"量词（对旧 token 重算才是无 cache 的区别）；有 cache 时新 token 对历史 K/V 的 attention 仍要算 |
| 3 | "无 cache 对旧 token 全重算；有 cache 只对新 token 算 attention" | 无 cache 漏了新 token（是整条序列都重算）；有 cache 时新 token 不只算 attention，要完整过 Embedding→QKV→FFN→lm_head |
| 4 | 无 cache = 整条序列所有 token 的 Embedding、Q/K/V、attention、FFN、lm_head 全部重算；有 cache = 只算新 token 的 Embedding、Q/K/V、FFN、lm_head + 新 token 的 Q 与缓存 K/V 的点积、加权；省掉旧 token 的一切计算 | ✅ 通过 |

追加确认（新 token 的 K/V 是"现算"还是"存进 cache"？）：先答"现算"、后改答"存进 cache"——两个都对，完整顺序是**先现算、再追加进 cache**。

**教训**：无 cache 与有 cache 的对比必须带量词——"谁的 Embedding/QKV/attention"。不带量词的层名列表在两种场景都成立，等于没说。

## 8. 掌握状态

**已掌握**（2026-10-25：7.3 三问独立通过 + 7.2 shape 实验预测 100% 吻合 + 无 cache/cache 对比经纠正后完整复述。复述过程见 7.4，相关错误已记入 `reviews/mistakes.md`。）

## 9. 官方资料

- PyTorch Embedding：https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- Attention Is All You Need（3.1 / 3.2 / 3.2.1 / 3.2.2 节）：https://arxiv.org/abs/1706.03762
- HuggingFace LLM Course：https://huggingface.co/learn/llm-course/chapter1/1

## 10. 待解决问题

1. batch > 1 时每个 shape 如何加 B 维？padding 和 attention mask 怎么处理？
2. RoPE 具体如何把位置信息注入 Q/K？
3. GQA/MQA 如何改变 K/V 的头数与 KV Cache 大小？
4. temperature / top-k / top-p 的采样细节。
5. 真实推理中“只对最后位置算 lm_head”在 vLLM 里由哪个模块实现？

## 变更记录

- 2026-10-19：创建；完成实验 1～3，记录 shape 表与 Prefill/Decode 观察。
- 2026-10-20：学习者说明尚未学习本课，按”从零讲解”重写第 1～7 节；细化”主干处理全部 token / lm_head 只算最后位置”的区别；回答 causal mask `diagonal=1` 的问题；掌握状态改为”学习中”，10-19 实验数据保留为历史参考。
- 2026-10-25：学习者不看讲义回答 7.3 三问，判定通过（真正理解）；第 4 节”我的理解”填入学习者原话；修正 2.7 与 7.1 中”第 n 步重算 n 个 token”的不严谨表述（应为 S+n−1）；掌握状态升为”待验证”。
- 2026-10-25：7.2 shape 实验完成（预测与实际 100% 吻合，`/home/gongping/vllm-env` torch 2.11.0+cu130）；新增 7.4 结论复述记录（4 轮纠正后通过）；同步修正实验代码第 85 行的旧结论表述；掌握状态升为”已掌握”。
