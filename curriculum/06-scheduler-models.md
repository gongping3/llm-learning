# 阶段 6～7：深入 Scheduler 与 Model Architecture

> 时间：各 3～4 周
> 阶段 6 适合系统开发背景：把 vLLM Scheduler 理解成**一个 Token 级别的资源调度器**，而不是简单的 request queue。

## 1. 阶段 6：真正理解 Scheduler

核心问题：

```text
Request A: prompt=10000, output=1000
Request B: prompt=100,   output=500
Request C: prompt=2000,  output=100
```

GPU 每一步到底执行什么组合、按什么预算切分？这就是 Scheduler。

必须掌握：

```text
FCFS / Priority Scheduling
Token Budget / Chunked Prefill
Prefill vs Decode
Request lifecycle / Request state
Scheduling policy
```

（vLLM V1 Scheduler 支持 FCFS 和 priority-based scheduling，通过统一 token budget 同时处理 prompt/output token。）

## 2. 阶段 7：Model Architecture

研究 Qwen / Llama / DeepSeek 在 vLLM 中如何实现：

```text
QwenForCausalLM → QwenModel → DecoderLayer → (Attention, MLP) → logits → Sampler → token
```

## 3. 强烈推荐里程碑练习

自己为 vLLM 添加一个简单模型，按官方文档：

- [Implement a Basic Model](https://docs.vllm.ai/en/latest/contributing/model/basic/)

这是非常建议完成的 milestone。

## 4. 完成标准

- [ ] 能解释 Scheduler 一个 step 的完整决策过程（含 token budget 与 chunked prefill）
- [ ] 能追踪 request 从进入到完成的全部状态转移
- [ ] 能解释 Qwen 模型在 vLLM 中的模块层级与调用链
- [ ]（里程碑）按官方文档成功添加一个简单模型
