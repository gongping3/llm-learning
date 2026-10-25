# 课程总索引（LESSON_INDEX）

> 把 9 个阶段拆成可逐课勾选的课程。共 **45 课**（阶段 9 以 PR 计，不设固定课数）。
> 说明：本拆分是**依据学习计划文档的知识点密度做的估算**，非文档原文；实际节奏由掌握情况决定，一课可能跨 2～3 次学习。
> 状态标记：`[ ]` 未开始 · `[~]` 进行中/部分完成 · `[x]` 已通过检验

## 阶段 0：Python / PyTorch / GPU 环境（4 课，1～2 周）

| 课号 | 课程名 | 关键产出 | 状态 |
|---|---|---|---|
| L0.1 | Python 特性：yield / async-await / dataclass / context manager | 能读懂 vLLM 里的异步与生成器代码 | [ ] |
| L0.2 | Tensor 基础：shape / dtype / device / stride | 实验 1（exp01） | [~] |
| L0.3 | view / reshape / transpose / permute / contiguous | 实验 1、实验 2（exp01、exp02） | [~] |
| L0.4 | GPU 与环境：线程/内存层次概念、nvidia-smi、跑通 vLLM Quickstart | 环境验收记录 | [ ] |

## 阶段 1：Transformer 基础（6 课，2～3 周）

| 课号 | 课程名 | 关键产出 | 状态 |
|---|---|---|---|
| L1.1 | 从一个 token 到 GPU：LLM 推理完整过程 | 实验 3（exp03） | [x] |
| L1.2 | Tokenizer 与 Embedding | shape 打印实验 | [ ] |
| L1.3 | QKV 与 Scaled Dot-Product Attention（含 causal mask） | 手写 attention | [ ] |
| L1.4 | Multi-Head Attention 与形状变换 | 多头切分/合并实验 | [ ] |
| L1.5 | Residual / RMSNorm / FFN / Transformer Block | 拼出完整 block | [ ] |
| L1.6 | Decoder-only 与 GPT/LLaMA/Qwen 差异 | **Project 1 收尾** | [ ] |

## 阶段 2：LLM 推理原理（5 课，2～3 周）

| 课号 | 课程名 | 关键产出 | 状态 |
|---|---|---|---|
| L2.1 | Prefill 与 Decode 两阶段 | 瓶颈性质对比 | [ ] |
| L2.2 | 为什么推理慢：FLOPS / 带宽 / compute vs memory bound |  Roofline 估算 | [ ] |
| L2.3 | KV Cache 原理：为什么存、存多少、何时读何时写 | 显存开销手算 | [ ] |
| L2.4 | **Project 2**：手写 KV Cache（with vs without） | 正确性与耗时对比 | [ ] |
| L2.5 | PagedAttention 论文精读（第 1 遍：为什么需要它） | 论文笔记 | [ ] |

## 阶段 3：Continuous Batching（5 课，3～4 周）

| 课号 | 课程名 | 关键产出 | 状态 |
|---|---|---|---|
| L3.1 | Batching 与 Dynamic Batching | 对比实验 | [ ] |
| L3.2 | Continuous Batching 核心机制 | 调度时序推演 | [ ] |
| L3.3 | Prefix Cache 与 Chunked Prefill | 机制说明 | [ ] |
| L3.4 | Speculative Decoding | 机制说明 | [ ] |
| L3.5 | **Project 3**：实现 Continuous Batching | 可运行模拟器 | [ ] |

## 阶段 4：CUDA / Triton（6 课，3～5 周）

| 课号 | 课程名 | 关键产出 | 状态 |
|---|---|---|---|
| L4.1 | CUDA 线程层次：thread / block / grid / warp | 概念图 + 小实验 | [ ] |
| L4.2 | 内存层次、合并访问、occupancy | 分析练习 | [ ] |
| L4.3 | CUDA stream / event / graph | 概念说明 | [ ] |
| L4.4 | Triton 入门：vector addition | 跑通教程 | [ ] |
| L4.5 | Triton matmul 与 softmax | 跑通教程 | [ ] |
| L4.6 | Fused attention 与 kernel 性能 | 性能对比 | [ ] |

## 阶段 5：PagedAttention + vLLM 源码（6 课，4～6 周）

| 课号 | 课程名 | 关键产出 | 状态 |
|---|---|---|---|
| L5.1 | PagedAttention 核心思想 | 论文第 2 遍 | [ ] |
| L5.2 | **Project 4**：实现 Paged KV Cache | 可运行模拟器 | [ ] |
| L5.3 | vLLM V1 架构总览（进程与组件） | 架构图 | [ ] |
| L5.4 | 源码入口：entrypoints → OpenAI API server | 调用链笔记 | [ ] |
| L5.5 | EngineCore 与 Scheduler 初探 | 调用链笔记 | [ ] |
| L5.6 | KV Cache 源码：KVCacheManager / Block / BlockPool | 分配流程笔记 | [ ] |

## 阶段 6：深入 Scheduler（4 课，3～4 周）

| 课号 | 课程名 | 关键产出 | 状态 |
|---|---|---|---|
| L6.1 | Scheduler 决策过程与 token budget | 逐步推演 | [ ] |
| L6.2 | Chunked Prefill 与 prefill/decode 混合 | 机制说明 | [ ] |
| L6.3 | Request lifecycle 与状态机 | 状态图 | [ ] |
| L6.4 | 调度实验：模拟调度、对比策略 | 实验报告 | [ ] |

## 阶段 7：Model Architecture（4 课，3～4 周）

| 课号 | 课程名 | 关键产出 | 状态 |
|---|---|---|---|
| L7.1 | Qwen / Llama 在 vLLM 中的实现 | 模块层级笔记 | [ ] |
| L7.2 | Attention backend | 后端对比 | [ ] |
| L7.3 | Sampler 与 logits 处理 | 采样流程笔记 | [ ] |
| L7.4 | **Milestone**：为 vLLM 添加一个简单模型 | 可运行模型 | [ ] |

## 阶段 8：性能优化（5 课，4～6 周）

| 课号 | 课程名 | 关键产出 | 状态 |
|---|---|---|---|
| L8.1 | TP / PP / DP / EP | 对比笔记 | [ ] |
| L8.2 | 量化：FP16 / BF16 / FP8 / INT8 / INT4 | 精度-性能权衡 | [ ] |
| L8.3 | CUDA Graph / FlashAttention / FlashInfer | 机制说明 | [ ] |
| L8.4 | 性能分析工具：nsys / ncu / torch.profiler | 工具上手 | [ ] |
| L8.5 | 性能实验：定位真实瓶颈 | 分析报告 | [ ] |

## 阶段 9：成为 Contributor（长期）

以 PR 计：Level 1（文档/测试）→ Level 2（Python/Scheduler/API/Model）→ Level 3（KV Cache/Worker）→ Level 4（Triton/CUDA/Attention/量化）→ Level 5（分布式/调度/内存/内核优化）。

---

## 使用规则

1. 每完成一课，把 `[ ]` 改成 `[x]`，并同步更新 `LEARNING_STATE.md` 与 `CURRENT_SESSION.md`。
2. **只有能独立解释原理、分析代码并通过实验/问题验证，才能打 `[x]`**——看讲解不算。
3. 每课的实验代码放 `experiments/` 对应子目录，笔记放 `knowledge/` 对应子目录。
4. 4 个必做项目的要求见 `projects/`。

## 变更记录

- 2026-10-19：创建 45 课索引；L0.2、L0.3、L1.1 标记为进行中（第 1 课已讲解，实验 1～3 已跑通）。
- 2026-10-25：L1.1 标记为 [x]（7.3 三问 + 7.2 实验 + 7.4 复述全部通过，知识点升"已掌握"）。
