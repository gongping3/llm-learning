# 总体学习路线（ROADMAP）

> 学习背景：具备 C/C++、Rust、Linux、网络和系统开发经验（Suricata、K8s、GPU 环境均有接触），Transformer/LLM 推理处于入门阶段。
> 节奏：按 2 小时/天设计，总周期约 6～9 个月。**不追求速度，优先补足前置知识。**

## 一、路线总览

```
阶段 0  Python / PyTorch / Linux GPU 环境          1～2 周
        ↓
阶段 1  Transformer 基础                           2～3 周
        ↓
阶段 2  LLM 推理原理（Prefill/Decode/KV Cache）    2～3 周
        ↓
阶段 3  KV Cache / Continuous Batching             3～4 周
        ↓
阶段 4  CUDA / GPU / Attention Kernel              3～5 周
        ↓
阶段 5  PagedAttention + vLLM 核心思想 + V1 源码    4～6 周
        ↓
阶段 6  深入 Scheduler / KV Cache / Worker          3～4 周
        ↓
阶段 7  Model Architecture（Qwen/Llama/DeepSeek）  3～4 周
        ↓
阶段 8  Kernel / Quantization / Parallelism / 性能 4～6 周
        ↓
阶段 9  真正成为 vLLM Contributor                   长期
```

课程大纲详见 `curriculum/` 对应文件。

## 二、最终目标

> 看到 vLLM 一个模块，能够知道它为什么存在、数据从哪里来、经过什么模块、最后为什么这样设计，并且能够修改代码、写测试、做性能验证。

## 三、必须完成的四个项目

| 项目 | 内容 | 所属阶段 |
|---|---|---|
| Project 1 | 手写 Attention（Q/K/V/softmax/attention） | 阶段 1 |
| Project 2 | 手写 KV Cache，测试 with vs without KV Cache | 阶段 2 |
| Project 3 | 实现 Continuous Batching（队列/调度/batch/完成退出） | 阶段 3 |
| Project 4 | 实现 Paged KV Cache（块式显存管理模拟） | 阶段 5 |

## 四、知识树（最终应形成的结构）

```
                    vLLM
                      │
        ┌─────────────┴─────────────┐
        │                           │
     LLM理论                    系统工程
        │                           │
 Transformer                    Linux
        │                        Python
 Attention                       C++
        │                        Rust
 KV Cache                       IPC
        │                       Network
        │                       Process
        │                           │
        └──────────┬────────────────┘
                   │
               vLLM Core
                   │
       ┌───────────┼───────────┐
       │           │           │
   Scheduler    KV Cache    Model Runner
       │           │           │
       │           │           ├── PyTorch
       │           │           ├── Triton
       │           │           └── CUDA
       │           │           │
       └───────────┼───────────┘
                   │
             Distributed
                   │
       ┌───────────┼───────────┐
       │           │           │
      TP          PP          DP
                   │
                   ↓
              Performance
                   │
          ┌────────┼────────┐
          │        │        │
        CUDA    Kernel   Memory
```

## 五、核心原则

1. **不要"从头读 vLLM"**：按 `Transformer → Attention → KV Cache → Continuous Batching → PagedAttention → vLLM Architecture → Engine → Scheduler → KV Cache Manager → Worker → Model Runner → Attention Backend → CUDA/Triton` 的顺序，**每理解一个概念，再去源码里面找它**。
2. **不默认已掌握前置知识**：每次先检查，再前进。
3. **以官方资料为主**：官方文档、原始论文、官方源码，见 `sources/references.md`。
4. **源码学习固定版本**：进入源码阶段后，固定 vLLM 的 Git 分支或 commit 并记录版本号。
5. **不虚构进度**：只记录真实完成的学习和实验。

## 六、主教材清单

见 `sources/references.md`（共 13 项，按学习顺序排列）。

## 变更记录

- 2026-10-19：初始化 ROADMAP，依据《大模型推理学习计划》建立 9 阶段路线。
