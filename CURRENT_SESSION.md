# 本次学习任务（CURRENT_SESSION）

> 每次学习开始时更新本文件；结束时由 AI（知识管理员模式）更新状态并生成下一次任务。

## 本次课程：第 1 课 —— 从一个 token 到 GPU：LLM 推理完整过程

- **计划日期**：2026-10-19
- **状态**：进行中 —— 前置知识检查阶段（等待学习者回答）
- **预计时长**：约 2 小时（10min 回忆 → 30min 新知识 → 45min 实验 → 20min 检验 → 15min 整理）
- **所属阶段**：阶段 0 收尾 + 阶段 1 开篇

## 1. 前置知识检查（当前步骤）

不看笔记，用自己的话回答：

1. 什么是 Tensor 的 `shape`？
2. 一个二维 Tensor 的两个维度分别代表什么？
3. `stride` 为什么会影响内存访问？
4. `permute` 之后得到的 Tensor，可以直接调用 `view` 吗？为什么？

> 检查目的：第 1 课要全程用 shape/dtype/device 解释数据流，必须先确认这些基础。根据回答决定是补基础还是继续新课。

### 检查结果（2026-10-19，学习者原话 + 导师评估）

| 题 | 学习者回答 | 评估 |
|---|---|---|
| 1 shape | "张量的形状，通常是维度" | 部分正确：方向对但表述不准，shape 是**每个维度的大小**，不是"维度"（维度数是 ndim）。已当场纠正。 |
| 2 二维维度 | "3 行 4 列的矩阵" | ✅ 通过 |
| 3 stride | "代表内存布局，相邻访问快，不相邻访问慢" | 部分正确：说出了**后果**（局部性），但未给出定义（逻辑索引每增 1，存储偏移增加的元素数）。已当场补充精确定义。 |
| 4 permute 后 view | "不知道" | 如实承认 ✅。这正是已记录的薄弱点，转入实验探究（实验 1：预测→运行）。 |

**结论**：shape/二维维度基础可用；stride 定义需巩固；permute/view 通过实验补齐。第 1 课继续，实验 1 融入本课。

### 实验 1 结果（预测 vs 实际，2026-10-19）

代码：`experiments/pytorch/exp01_permute_view.py`（torch 2.14.1+cpu）

| # | 预测 | 实际 | 判定 |
|---|---|---|---|
| 1 | `x.stride()` = 4 | `(4, 1)` | ⚠️ 部分正确：只答出第 0 维，漏掉 stride 是元组 |
| 2 | `y.shape`=(4,3)、`y.stride()`=3 | `(4,3)` ✅、`(1, 4)` ❌ | 概念错误：以为 permute 后变成连续张量（stride (3,1)） |
| 3 | `y.view(4,3)` 成功 | 成功（因为形状未变，无需重解释） | 结果对、理由需修正；揭示性更强的是 `y.view(12)` **失败** |
| 4 | `x.view(4,3)` 失败 | 成功，stride `(3,1)`，与 x 共享存储 | ❌ 错误：连续 Tensor 改形状完全合法 |

**本次纠正的错误理解**：
1. stride 不是"内存布局好不好"的形容词，而是精确的偏移增量元组。
2. permute 不复制数据：`x.data_ptr() == y.data_ptr()`，只换 shape 与 stride。
3. view 的判据不是"形状变没变"，而是"能否只用 stride 重新解释"；`contiguous()` 才会真正搬数据。

**新增的已验证事实**：偏移公式 `offset = i*stride[0] + j*stride[1] + storage_offset`（`x[2][3]=11` 与 `y[3][2]=11` 均落在同一存储位置 11）。

### 检验题 + 实验 2（reshape 是否复制）

- 学习者答案：不会失败、会复制数据；证明方法 = 把结果赋给新变量，改它，再与原 Tensor 比对。
- **判定**：结论在本场景下正确 ✅，改值对比法有效 ✅；但"reshape 会复制"是**过度推广** ❌。
- 实验 2（`experiments/pytorch/exp02_reshape_copy.py`）实际输出：
  - 非连续 `y.reshape(12)` → data_ptr 不同，**复制**；改 `yr[0]=999` 后 `y[0][0]` 仍为 0
  - 连续 `x.reshape(12)` → `storage().data_ptr()` 相同，**不复制**；改 `xr[0]=111` 后 `x[0][0]` 变为 111
  - 改原始 `x[0][1]=777` 后 `y[1][0]` 变为 777（permute 共享存储）
- 判据：`reshape` 只在必要时复制。另注：torch 2.14 已无模块级 `torch.shares_memory`，用 `data_ptr()` 比较。
- 已记入 `reviews/mistakes.md` 与 `knowledge/pytorch/view-reshape.md`。

## 2.5 第 1 课主体：一个 token 从进入模型到生成出来（已完成讲解 + 实验 3）

- 实验 3：`experiments/attention/exp03_token_to_logits.py`（torch 2.14.1+cpu，V=1000, d=64, H=4）
- 链路 shape（实测）：ids (1,5) → embedding (1,5,64) → QKV (1,5,64) → 多头 (1,4,5,16)
  → scores (1,4,5,5) → 合并 (1,5,64) → block (1,5,64) → **logits (1,5,1000)** → 取最后位置 (1000,) → 采样
- 核心结论：logits 是 `(1, S, V)`，但推理只用 `logits[0, -1]`，其余位置在 Decode 阶段被丢弃。
- Prefill（compute bound，一次算完） vs Decode（memory bound，逐步生成）；实验证实第 n 步重算 n 个 token。
- 已生成笔记 `knowledge/inference/lesson01-token-to-gpu.md`。

**本节待完成**：学习者复述链路（检验题）+ 填写笔记"我的理解" → 然后进入第 2 课（KV Cache）。

## 2. 本节学习目标

学完本课后，你应该能够：

1. 回答核心问题：**"一个 token 从进入模型，到最终生成出来，中间究竟发生了什么？"**
2. 用 Tensor 的 shape / dtype / device 一路解释清楚上述过程。
3. 区分 **Prefill** 和 **Decode** 两个阶段，并解释为什么 Decode 是逐 token 进行的。
4. 说出为什么推理需要 **KV Cache**（初步概念，阶段 2 深入）。

## 3. 学习流程（按 TUTOR_RULES 标准流程）

1. ☐ 检查前置知识（当前）
2. ☐ 明确本节学习目标
3. ☐ 讲解核心概念：token → embedding → 模型前向 → logits → 采样 → 下一个 token
4. ☐ 阅读官方资料（PyTorch Tutorials / HF LLM Course 相关章节）
5. ☐ 编写并运行实验：打印推理链路上每个 Tensor 的 shape / dtype / device
6. ☐ 通过提问和实践检查理解程度
7. ☐ 总结易错点、知识关联和待解决问题
8. ☐ 更新学习状态，准备下一次任务

## 4. 实验任务

（待前置检查完成后，根据实际掌握情况设计；初步方向：用最小代码手动走一遍 token → embedding → 线性层 → softmax 的 shape 变化。）

## 5. 结束时需更新

- [ ] LEARNING_STATE.md（知识点状态、薄弱点、复习安排）
- [ ] knowledge/ 新增或修改的笔记
- [ ] reviews/mistakes.md（本次发现的错误理解）
- [ ] 生成下一次学习任务

---

## 历史会话记录

（每次学习结束后在此追加一条摘要：日期 / 学习内容 / 掌握情况 / 下次衔接点。）
