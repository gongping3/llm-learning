# 第 1 课：从一个 token 到 GPU —— LLM 推理完整过程

## 1. 一句话定义
LLM 推理就是：**把 token id 序列送进模型做一次前向传播，得到 logits，只取最后一个位置采样出下一个 token，再把这个 token 拼回序列，循环往复。**

## 2. 核心原理：完整链路与 shape 变化

```
token ids (1, S)
   ↓ nn.Embedding
hidden states (1, S, d)
   ↓ @ Wq / Wk / Wv
Q, K, V          (1, S, d)
   ↓ view → (1, S, H, dh) → transpose(1, 2)
q, k, v          (1, H, S, dh)
   ↓ scores = q @ k.transpose(-2, -1) / sqrt(dh)
scores           (1, H, S, S)
   ↓ causal mask + softmax
attn             (1, H, S, S)
   ↓ attn @ v → transpose(1,2).reshape(1, S, d)
attn out         (1, S, d)
   ↓ @ Wo → 残差 → RMSNorm → FFN → 残差
block 输出        (1, S, d)
   ↓ @ lm_head.T
logits           (1, S, V)
   ↓ 取最后一个位置 → (V,) → argmax / 采样
下一个 token id
```

（本实验参数：V=1000, d=64, H=4, dh=16, S=5）

## 3. 实验验证

2026-10-19，`experiments/attention/exp03_token_to_logits.py`（torch 2.14.1+cpu）：

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

## 4. Prefill 与 Decode

- **Prefill**：整个 prompt 一次进入模型，一次性算出所有位置的 logits，**并行度高、计算密集（compute bound）**。
- **Decode**：每次只新增 1 个 token，但（无 KV Cache 时）必须把整个序列重新算一遍；**每步只取最后一个位置的 logits，其余全部丢弃**，属于**访存密集（memory bound）**——每生成一个 token 都要把全部权重读一遍。
- 实验证实：第 n 步重算 n 个 token，累计计算量随长度**平方增长**。

## 5. 关键结论

1. **logits 的 shape 是 `(1, S, V)`，但推理时我们只需要 `logits[0, -1]`** —— 其余 S-1 个位置的计算在 Decode 阶段全被浪费了。这是理解 vLLM 所有优化的第一块基石。
2. 多头切分用的是 `view` + `transpose`；合并回来时因为非连续，必须用 `reshape`（呼应实验 1）。
3. `k.transpose(-2, -1)` 只交换最后两维：`(1, H, S, dh) → (1, H, dh, S)`，所以 `q @ k^T` 得到 `(1, H, S, S)`。

## 6. 与其他知识的联系
stride/contiguous（实验 1）、KV Cache（第 2 课）、Continuous Batching（阶段 3）、vLLM Scheduler 的 token budget（阶段 6）

## 7. 易错点
- 以为模型"直接输出一个词"——实际输出的是整段 logits，取最后位置再采样。
- 混淆 Prefill 与 Decode 的瓶颈性质（compute bound vs memory bound）。
- 以为 `view(1, S, H, dh)` 之后可以直接 `reshape` 回 `(1, S, d)` —— 中间 transpose 过，非连续。

## 8. 掌握状态
待验证（已能看懂实验输出，待独立复述链路与 shape 推导）

## 9. 官方资料
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- https://arxiv.org/abs/1706.03762 （3.2 Scaled Dot-Product Attention）
- https://huggingface.co/learn/llm-course/chapter1/1

## 10. 待解决问题
1. 为什么 Decode 阶段不能只算新 token，而要重算整个序列？（第 2 课：因为 Attention 需要历史 K/V）
2. causal mask 为什么要 `diagonal=1`（遮住自己之后的位置）？
3. 真实模型（Qwen/Llama）里 batch>1 时 shape 如何变化？

## 记录
- 2026-10-19：第 1 课完成实验 3 验证，链路与 shape 表来自真实运行输出。
