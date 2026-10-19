# Project 3：实现 Continuous Batching

> 所属课程：L3.5（阶段 3）｜预计 2～3 次学习｜状态：未开始

## 目标

自己实现一个最小调度器，让请求能**动态加入和退出** batch：

```
t0: A B C
t1: A B C D     ← D 新加入
t2: A C D       ← B 完成退出
t3: C D E       ← E 新加入，A 完成退出
```

## 必须实现

1. `Request` 数据结构：id、prompt tokens、已生成 tokens、状态（waiting / running / finished）
2. `RequestQueue`：等待队列
3. `Scheduler.step()`：每步决定本轮 batch 由哪些请求组成
4. 一个 `token_budget` 上限（模拟 vLLM 的统一 token budget）
5. 请求完成（遇到 EOS 或达到 max_tokens）后立即退出 batch 并释放资源
6. 主循环：跑 20 步，每步打印 batch 组成

## 验收标准

- [ ] 能打印每一步的 batch 组成，并看到请求动态进出
- [ ] 请求完成时立即释放（不留占位）
- [ ] 有 token_budget 限制时，长请求不会饿死短请求（说明你的处理策略）
- [ ] **必须能回答**：static batching 为什么必须等最慢的请求？浪费在哪里？
- [ ] **必须能回答**：Continuous Batching 里，"完成"的请求是如何被感知并移除的？
- [ ] **必须能回答**：token_budget 用尽时，剩下的请求下一步怎么处理？

## 提示与禁令

- 可以用假的"模型"（随机 logits 或简单规则生成 EOS），重点是调度逻辑不是模型
- 允许参考：vLLM V1 Guide 中关于 scheduler 的描述
- 先不要照抄 `vllm/v1/core/sched/scheduler.py`

## 产出位置

- 代码：`projects/p3-continuous-batching/` 或 `experiments/`
- 笔记：`knowledge/inference/continuous-batching.md`
- 完成后在 `curriculum/LESSON_INDEX.md` 把 L3.5 打勾
