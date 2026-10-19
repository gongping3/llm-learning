# Project 4：实现 Paged KV Cache

> 所属课程：L5.2（阶段 5）｜预计 3～4 次学习｜状态：未开始
> **做完这个再看 PagedAttention 论文，会非常容易。**

## 目标

模拟块式显存管理：把 KV Cache 切成固定大小的 block，按需分配给请求：

```
GPU memory
┌────┬────┬────┬────┬────┬────┐
│ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │
└────┴────┴────┴────┴────┴────┘

Request A: B0 → B3 → B5      （block table: [0, 3, 5]）
Request B: B1 → B4           （block table: [1, 4]）
```

## 必须实现

1. `BlockPool`：固定大小的块池（比如 block_size=4，共 N 块）
2. `BlockTable`：每个请求一张块表，记录逻辑顺序 → 物理块号的映射
3. `allocate(seq_len)` / `free(request)`：按需分配、完成时回收
4. 写入：新 token 的 K/V 写入当前块的第 `slot` 个位置，满则申请新块
5. 读取：attention 时按 block table 从**不连续的物理块**里 gather 出 K/V
6. 显存利用率统计：对比"按最大长度预分配" vs "分页按需分配"的浪费比例
7. （进阶）prefix sharing：两个相同前缀的请求共享同一批块

## 验收标准

- [ ] 块表映射正确，能从非连续块中正确读出 K/V 并算出正确 attention
- [ ] 请求完成后块被回收，可再次分配（无泄漏）
- [ ] 给出显存浪费率的量化对比数字（比如内部碎片 + 预留浪费）
- [ ] **必须能回答**：为什么要分页？连续预分配的浪费具体来自哪几部分？
- [ ] **必须能回答**：block_size 取大取小各有什么代价？
- [ ] **必须能回答**：prefix sharing 为什么在分页结构下才容易实现？

## 提示与禁令

- 允许参考：PagedAttention 论文（第 3 遍：它到底怎么解决）
- 先不要看 `vllm/v1/core/kv_cache_manager.py`

## 产出位置

- 代码：`projects/p4-paged-kv-cache/`
- 笔记：`knowledge/vllm/paged-attention.md`
- 完成后在 `curriculum/LESSON_INDEX.md` 把 L5.2 打勾
