# 阶段 5：PagedAttention + vLLM 核心思想 + V1 源码架构

> 时间：4～6 周
> 整个路线最关键的阶段：正式开始源码学习。
> ⚠️ 进入源码前：固定 vLLM 的 Git 分支或 commit，并记录版本号。

## 1. PagedAttention 核心思想

类似操作系统虚拟内存/分页的思想管理 KV Cache。

### 必须完成的项目（Project 4）：实现 Paged KV Cache

模拟块式显存管理：

```text
GPU memory
┌────┬────┬────┬────┬────┬────┐
│ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │
└────┴────┴────┴────┴────┴────┘

Request A: B0 → B3 → B5
Request B: B1 → B4
```

做完这个再看 PagedAttention 论文会非常容易。

## 2. vLLM V1 整体架构

```
Client
  │
  ▼
API Server
  │ ZMQ
  ▼
Engine Core
  ├── Scheduler
  ├── KV Cache Manager
  └── Model Execution
        │
        ▼
   GPU Worker
        │
        ▼
    Model Runner
        │
        ▼
  PyTorch / CUDA
```

三大进程：API Server Process / Engine Core Process / GPU Worker Processes。

## 3. 源码阅读顺序（五层）

| 层 | 位置 | 目标 |
|---|---|---|
| 1 入口 | `vllm/entrypoints/llm.py`、`vllm/entrypoints/cli/` | 理解 `LLM(...)` 是 offline inference 入口 |
| 2 API Server | `vllm/entrypoints/openai/` | HTTP request → OpenAI API → Request → Engine → Scheduler |
| 3 Engine Core | `vllm/v1/engine/` | 理解 EngineCore 与 Scheduler |
| 4 KV Cache | `KVCacheManager` / `Block` / `BlockPool` / allocation | 回答：一个 request 进来后，vLLM 怎么给它分配 KV Cache？ |
| 5 GPU Worker | Worker → Model Runner → Model → Attention | 理解模型加载、输入 Tensor 准备、CUDA Graph |

**不要** `grep -R "attention" .` 从头乱读。

## 4. 推荐资料

1. [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/)
2. [PagedAttention 论文](https://arxiv.org/abs/2309.06180)（配合 Project 4 重读）
3. [vLLM GitHub](https://github.com/vllm-project/vllm)（固定 commit）

## 5. 完成标准

- [ ] Project 4 完成并通过测试
- [ ] 能画出 V1 架构图并解释每个进程/组件的职责
- [ ] 能追踪一条完整调用链：`LLM.generate()` → EngineCore → Scheduler → Worker → ModelRunner
