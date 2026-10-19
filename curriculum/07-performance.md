# 阶段 8：性能优化（Kernel / Quantization / Parallelism）

> 时间：4～6 周
> 进入真正的 vLLM Developer 领域。

## 1. 需要学习

```text
Tensor / Pipeline / Data / Expert Parallelism
Quantization: FP16 / BF16 / FP8 / INT8 / INT4
CUDA Graph
FlashAttention / FlashInfer
Triton
Speculative Decoding
Prefix Caching
Disaggregated Prefill
```

## 2. 重点：性能分析（回答"到底哪里慢"而不是"GPU 不够快"）

必须学会使用：

```text
nvidia-smi / nsys / ncu / torch.profiler / perf
```

分析链路：

```text
Request → Scheduler → Model Runner → Attention → CUDA Kernel
```

## 3. 完成标准

- [ ] 能解释 TP/PP/DP 各自切分什么、代价是什么
- [ ] 能解释常见量化格式的精度/性能权衡
- [ ] 能用 nsys 或 torch.profiler 定位一次推理的耗时分布并给出结论
- [ ] 能解释 CUDA Graph 为什么能加速 Decode
