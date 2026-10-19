# 阶段 1：真正搞懂 Transformer

> 时间：2～3 周
> 这是非常重要的一关。当前正处于"追问 Wq/Wk/Wv、softmax、位置编码"的状态，即本阶段起点。

## 1. 必须搞懂的内容（按此顺序）

```text
Embedding
  ↓
Positional Encoding / RoPE
  ↓
Q K V
  ↓
Self Attention
  ↓
Multi-Head Attention
  ↓
Residual
  ↓
LayerNorm / RMSNorm
  ↓
FFN
  ↓
Transformer Block
  ↓
Decoder-only Transformer
```

然后理解 GPT、LLaMA、Qwen 与原始 Transformer 的关系。

## 2. 必须完成的项目（Project 1）

不要只看。用 PyTorch 自己实现：

```text
Tokenizer → Embedding → RoPE → QKV → Multi-Head Attention → RMSNorm → FFN → Transformer Block
```

可以先不训练。输入 `"hello world"`，让数据经过完整链路，打印每一步 Tensor 的 shape / dtype / device：

```text
token → embedding → position → QKV → attention → FFN → logits
```

能做到这一点，就算真正入门。

## 3. 推荐资料（按阅读顺序）

1. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) —— 先看这个
2. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) —— 第一遍只读 3.1 / 3.2 / 3.2.1 / 3.2.2 / 3.5 节
3. [HuggingFace LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) —— 重点：Tokenizer、Transformer、Datasets、Model、Inference

## 4. 完成标准

- [ ] 能不看资料画出 Decoder-only Transformer 结构图并解释每个组件
- [ ] Project 1 代码可运行，每步 shape 可解释
- [ ] 能回答：Wq/Wk/Wv 从哪里来、softmax 在 attention 中的作用、RoPE 如何编码位置
- [ ] 相关知识笔记状态更新为"已掌握"（需通过考核导师检验）
