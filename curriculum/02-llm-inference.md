# 阶段 2：理解 LLM 推理

> 时间：2～3 周
> 从"Transformer 学习者"进入"推理工程师"的关键阶段。核心问题：**为什么 LLM 推理这么慢？**

## 1. 必须理解的内容

### Prefill
整个 Prompt（如 "What is the capital of China?" 分词后的所有 token）一次进入 Transformer。

### Decode
逐 token 生成：`Prefill → Decode → Decode → ...`，每步只产生一个新 token。

### KV Cache（进入 vLLM 的第一道核心门槛）
必须彻底搞懂：

```text
K / V 为什么需要保存？
保存在哪里？保存多少？
什么时候读取？什么时候新增？
为什么占 GPU 显存？
为什么不能每次重新计算？（A B C D 已缓存，只需计算新增 token E）
```

### Memory Bandwidth
```text
FLOPS / Memory Bandwidth
Compute Bound / Memory Bound
```

## 2. 必须完成的项目（Project 2）

手写 KV Cache：

```text
request → KV Cache → decode
```

测试对比 `without KV Cache` vs `with KV Cache` 的正确性与性能。

## 3. 推荐资料

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（vLLM 原始论文，**至少读 3 遍**：
  第一遍问"为什么需要 PagedAttention"；
  第二遍问"KV Cache 到底浪费在哪里"；
  第三遍问"PagedAttention 到底怎么解决"）

## 4. 完成标准

- [ ] 能完整解释 Prefill/Decode 两阶段及各自瓶颈特征（compute bound vs memory bound）
- [ ] 能解释 KV Cache 的必要性与显存开销计算方法
- [ ] Project 2 完成并记录实验结果
