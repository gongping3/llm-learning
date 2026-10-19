# Project 1：手写 Attention / 迷你 Transformer

> 所属课程：L1.6（阶段 1）｜预计 2～3 次学习｜状态：未开始

## 目标

用纯 PyTorch 从零实现一个 Decoder-only 迷你模型，让 `"hello world"` 这样的输入完整走一遍：

```
token → embedding → position(RoPE) → QKV → Multi-Head Attention → RMSNorm → FFN → logits
```

**可以先不训练**（用随机权重即可），重点是打通数据流。

## 必须实现

1. 极简 tokenizer（字符级即可）
2. `nn.Embedding` 查表
3. RoPE 位置编码
4. Q / K / V 投影（`@ Wq / Wk / Wv`）
5. Multi-Head Attention（含 causal mask、`/ sqrt(dh)`、softmax）
6. RMSNorm
7. FFN（SwiGLU 或简单两层）
8. 一个完整的 Transformer Block（含残差）
9. `lm_head` 得到 logits

## 验收标准

- [ ] 代码可独立运行
- [ ] 能打印链路中**每一个** Tensor 的 `shape` / `dtype` / `device`
- [ ] 多头切分与合并的 shape 变化能逐步解释清楚
- [ ] causal mask 的作用能用矩阵图说明
- [ ] **必须能回答**：为什么 scores 要除以 `sqrt(dh)`？不除会怎样？
- [ ] **必须能回答**：mask 为什么用 `diagonal=1` 而不是 0？
- [ ] **必须能回答**：`q @ k.transpose(-2, -1)` 之后 shape 为什么是 `(1, H, S, S)`？

## 提示与禁令

- 允许参考：《Attention Is All You Need》3.2 节、PyTorch 官方文档
- 先不要看 vLLM / HF 的模型实现，做完再看

## 产出位置

- 代码：`experiments/attention/` 或本目录
- 笔记：`knowledge/transformer/`
- 完成后在 `curriculum/LESSON_INDEX.md` 把 L1.6 打勾
