# 阶段 0：Python / PyTorch / Linux GPU 环境

> 时间：1～2 周（环境部分已基本完成）
> 目标：能够独立运行 PyTorch + HuggingFace + vLLM，并能够观察 GPU 上到底发生了什么。

## 1. Python 重点

不是语法，而是 vLLM 中高频出现的语言特性：

```text
class / decorator / generator / yield
async / await / asyncio
typing / dataclass
multiprocessing / threading / queue
context manager
```

重点资料：[yield 表达式（官方参考）](https://docs.python.org/zh-cn/3.12/reference/expressions.html#yield-expressions)

## 2. PyTorch 必须掌握

```text
Tensor / dtype / device / shape / stride
view / reshape / transpose / permute
matmul / softmax / embedding
nn.Module / forward / autograd
```

验收标准：能够看懂并用 shape 解释下面这段代码的每一步：

```python
Q = x @ Wq
K = x @ Wk
V = x @ Wv
scores = Q @ K.transpose(-2, -1)
scores = scores / math.sqrt(d)
attention = torch.softmax(scores, dim=-1)
output = attention @ V
```

## 3. GPU 基础概念

```text
CPU / GPU / CUDA / CUDA kernel / SM
Thread / Warp / Block / Grid
Global Memory / Shared Memory / Register
Tensor Core
```

暂不要求自己写 CUDA。

## 4. 推荐资料

1. [PyTorch Tutorials](https://docs.pytorch.org/tutorials/) —— 重点：Tensor、CUDA、torch.nn、Performance tuning
2. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)（第一遍只看 Thread/Block/Grid/Warp/Memory hierarchy，不通读）
3. [vLLM Quickstart](https://github.com/vllm-project/vllm/blob/main/docs/getting_started/quickstart.md) —— Offline batched inference、Online serving、Attention backend

## 5. 完成标准

- [ ] 能打印推理链路上每个 Tensor 的 shape / dtype / device
- [ ] 独立运行 PyTorch + HuggingFace + vLLM
- [ ] 会用 nvidia-smi 观察 GPU 显存/利用率
- [ ] stride / view / reshape / transpose / permute 相关实验全部通过检验（更新 LEARNING_STATE.md）

## 6. 当前状态

- 环境（WSL2 + NVIDIA GPU + CUDA + Python 3.12 + uv + vLLM）：✅ 已完成
- PyTorch Tensor 基础：🔄 学习中（详见 LEARNING_STATE.md）
