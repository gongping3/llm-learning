# 阶段 4：GPU + CUDA + Attention Kernel

> 时间：3～5 周
> 从"LLM 软件开发"进入"LLM 推理系统开发"。

## 1. 必须学习

```text
CUDA Thread Hierarchy / Warp / Warp-level primitive
Memory Coalescing / Shared Memory / Register
Occupancy
CUDA Stream / Event / Graph
```

然后学习 **Triton**：

```text
PyTorch → Triton → CUDA
```

现在大量 LLM 推理优化不直接写 CUDA 而是用 Triton。以后读 vLLM kernel 会频繁碰到 Triton kernel、CUDA kernel、custom op。

## 2. Triton 重点教程

```text
Vector Addition → Matrix Multiplication → Softmax → LayerNorm → Fused Attention
```

## 3. 推荐资料

1. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
2. [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
3. [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)

## 4. 完成标准

- [ ] 能解释 Thread/Block/Grid/Warp 层级与内存层级（Global/Shared/Register）
- [ ] 完成 Triton vector addition 和 matmul 教程并能解释每一步
- [ ] 能解释为什么 Softmax/LayerNorm 需要 fused kernel（memory bound 视角）
