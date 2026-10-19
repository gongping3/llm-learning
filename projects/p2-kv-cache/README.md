# Project 2：手写 KV Cache

> 所属课程：L2.4（阶段 2）｜预计 2～3 次学习｜状态：未开始
> 这是**进入 vLLM 的第一道核心门槛**。

## 目标

实现两条推理路径并对比：

```
without KV Cache：每步把整个序列重算一遍
with    KV Cache：缓存历史 K/V，每步只算新 token
```

## 必须实现

1. 一个最小 Decoder 层（可复用 Project 1 的组件）
2. `forward_prefill(ids)`：算完整个 prompt，返回 logits 与 K/V 缓存
3. `forward_decode(new_id, kv_cache)`：只算 1 个 token，把新 K/V 追加进缓存
4. KV Cache 的数据结构：`(layers, 2, batch, heads, seq, dh)` 之类，自己决定并说明理由
5. 正确性校验：两条路径生成的 token 序列应一致（数值误差 `max|diff| < 1e-4`）
6. 性能对比：测量生成 N 个 token 的总耗时

## 验收标准

- [ ] 两条路径输出一致（给出实际 max diff 数值）
- [ ] 给出 with / without 的耗时数据（N=64、128 各测一次）
- [ ] 手算：L 层、H 头、d 维、S 长度、batch=1、fp16 时，KV Cache 占多少**字节**？给出公式
- [ ] **必须能回答**：为什么只缓存 K/V，不缓存 Q？
- [ ] **必须能回答**：Decode 阶段为什么仍要读全部权重（与 memory bound 的关系）？
- [ ] **必须能回答**：batch 变大时，KV Cache 显存如何变化？

## 提示与禁令

- 允许参考：PagedAttention 论文第 3 节（KV cache 相关部分）
- 先不要看 vLLM 的 `KVCacheManager` 实现

## 产出位置

- 代码：`experiments/kv-cache/`
- 笔记：`knowledge/inference/kv-cache.md`
- 完成后在 `curriculum/LESSON_INDEX.md` 把 L2.4 打勾
