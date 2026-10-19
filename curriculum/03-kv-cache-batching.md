# 阶段 3：LLM Serving 核心技术（KV Cache / Continuous Batching）

> 时间：3～4 周
> 这是 vLLM 的核心领域。

## 1. 必须学习

```text
Batching / Dynamic Batching
Continuous Batching
Request Scheduling / Token Budget
Prefix Cache
Chunked Prefill
Speculative Decoding
```

## 2. 重点理解 Continuous Batching

传统 Batch 必须等最慢的请求；Continuous Batching 允许请求动态加入/退出：

```text
Step 1: A B C
Step 2: A B
Step 3: A D
Step 4: A D E
```

由此引出 **Scheduler** 的概念。

## 3. 必须完成的项目（Project 3）

自己实现：

```text
Request Queue → Scheduler → Batch → Request Finish
```

模拟请求动态加入/退出的完整生命周期。

## 4. 推荐资料

- [vLLM V1 Guide](https://docs.vllm.ai/en/latest/usage/v1_guide/)（V0 已 deprecated；V1 重构了 Scheduler、KV Cache Manager、Worker、Sampler、API Server）

## 5. 完成标准

- [ ] 能解释 static batching 与 continuous batching 的性能差异及原因
- [ ] 能解释 prefix cache / chunked prefill / speculative decoding 各自解决什么问题
- [ ] Project 3 完成并通过测试
