# 本次学习任务（CURRENT_SESSION）

> 每次学习开始时更新本文件；结束时由 AI（知识管理员模式）更新状态并生成下一次任务。
>
> **2026-10-20 更新**：学习者说明尚未学习第 1 课内容，本次从零讲解。10-19 的实验记录仅作参考资料，不作为“已掌握”的依据；讲解后的复述与实验检验重新开始。

## 本次课程：L1.2 —— Tokenizer 与 Embedding（阶段 1）

- **计划日期**：2026-10-26
- **状态**：未开始
- **预计时长**：约 2 小时（10min 回忆 → 30min 新知识 → 45min 实验 → 20min 检验 → 15min 整理）
- **所属阶段**：阶段 1（Transformer 基础）
- **前置**：L1.1 已掌握（2026-10-25）；开始前先做间隔复习（PyTorch 内存布局第 7 天 + 第 1 课链路第 1 天，见 review-log）

## 本课任务

1. **回忆**：10-26 复习变式题（换 H、换 V 推 shape；有 cache 后新 token 的 attention 怎么算），不重讲。
2. **Tokenizer**：BPE / 词表 / token id 的关系；为什么词表是 V；特殊 token（BOS/EOS/PAD）。
3. **Embedding**：`(V,d)` 查表；dtype/device；与 lm_head 的权重是否共享。
4. **Positional Encoding / RoPE**：为什么需要位置信息（L1.1 已留伏笔）；RoPE 如何注入 Q/K（初步，L1.3 深入）。
5. **实验**：用真实 tokenizer（如 HF 的 GPT-2）把一段文本切成 ids，打印每个 tensor 的 shape/dtype/device；对比不同 tokenizer 的切分差异。
6. **官方资料**（含 L1.1 未读部分）：The Illustrated Transformer、HF LLM Course、PyTorch Embedding 文档。
7. 参考：`curriculum/01-transformer.md` 与 `sources/references.md`。

---

## L1.1 完成记录（2026-10-25）

第 1 课《从一个 token 到 GPU：LLM 推理完整过程》已收尾：

- 7.3 三问独立通过（10-25）；7.2 shape 实验预测 100% 吻合（10-25，`/home/gongping/vllm-env`，torch 2.11.0+cu130）。
- 无 cache/cache 对比复述 4 轮纠正后通过（教训：对比必须带"谁的"量词）。
- 掌握状态："一个 token 的完整推理过程" → **已掌握**（本系统第一个已掌握知识点）。
- 详细记录：`knowledge/inference/lesson01-token-to-gpu.md`（7.2/7.4 节）、`reviews/mistakes.md`。
- 未完成项：官方资料阅读（工作流第 4 步）并入 L1.2。

## 0. 历史状态记录（2026-10-20）

- 已按学习者要求从零讲解第 1 课；完整讲义写入 `knowledge/inference/lesson01-token-to-gpu.md`（2026-10-20 重讲版）。
- 2026-10-25：学习者不看讲义回答 **7.3 三问，判定通过（真正理解）**；原话已记入讲义第 4 节与 7.3 节。一处口误（Decode 序列长度 S+n−1 说成 n 个）已纠正，讲义 2.7 的不严谨表述一并修正。掌握状态升为”待验证”。
- ~~当前等待：完成 7.2 的 shape 实验~~ → 已于 2026-10-25 完成并通过（预测 100% 吻合）。
- 环境：~~阻塞~~ → 2026-10-25 已解决：`/home/gongping/vllm-env`（uv venv，torch 2.11.0+cu130）运行 exp03 验证通过。
- 下一步：~~7.2 实验~~ → 已升”已掌握”；进入 L1.2 Tokenizer 与 Embedding。

## 1. 前置知识检查（2026-10-19 历史记录）

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

### 1.5 第 1 课主体讲解记录（2026-10-19，历史参考）

- 实验 3：`experiments/attention/exp03_token_to_logits.py`（torch 2.14.1+cpu，V=1000, d=64, H=4）
- 链路 shape（实测）：ids (1,5) → embedding (1,5,64) → QKV (1,5,64) → 多头 (1,4,5,16)
  → scores (1,4,5,5) → 合并 (1,5,64) → block (1,5,64) → **logits (1,5,1000)** → 取最后位置 (1000,) → 采样
- 核心结论（旧表述，已在 2026-10-20 重讲版中细化）：logits 是 `(1, S, V)`；生成只用最后位置。注意：主干仍要处理全部 S 个 token 以产生 K/V，能省的是 lm_head 对前面位置的投影。
- Prefill（compute bound，一次算完） vs Decode（memory bound，逐步生成）；实验证实第 n 步重算 n 个 token。
- 已生成笔记 `knowledge/inference/lesson01-token-to-gpu.md`。

**本节待完成（2026-10-20）**：回答讲义 7.3 的三个问题 + 完成 7.2 的 shape 实验 + 填写笔记"我的理解"。通过后再进入 L1.2 Tokenizer 与 Embedding。

## 2. 本节学习目标

学完本课后，你应该能够：

1. 回答核心问题：**"一个 token 从进入模型，到最终生成出来，中间究竟发生了什么？"**
2. 用 Tensor 的 shape / dtype / device 一路解释清楚上述过程。
3. 区分 **Prefill** 和 **Decode** 两个阶段，并解释为什么 Decode 是逐 token 进行的。
4. 说出为什么推理需要 **KV Cache**（初步概念，阶段 2 深入）。

## 3. 学习流程（按 TUTOR_RULES 标准流程）

1. ☑ 检查前置知识（10-19 有历史记录；本次按“从零讲解”处理）
2. ☑ 明确本节学习目标
3. ☑ 讲解核心概念：token → embedding → 模型前向 → logits → 采样 → 下一个 token（2026-10-20）
4. ☐ 阅读官方资料（未完成，并入 L1.2：PyTorch Embedding / Attention Is All You Need 3.1～3.2 / HF LLM Course）
5. ☑ 编写并运行实验：打印推理链路上每个 Tensor 的 shape / dtype / device（2026-10-25 通过）
6. ☑ 通过提问和实践检查理解程度（7.3 提问 + 7.2 实验 + 7.4 复述，2026-10-25 全部通过）
7. ☑ 总结易错点、知识关联和待解决问题（mistakes.md / 讲义 7.4 / 变更记录）
8. ☑ 更新学习状态，准备下一次任务（本次，生成 L1.2）

## 4. 实验任务

见知识笔记 7.2：

1. 先预测 ids / X / q / scores / logits 的 shape，再运行 `experiments/attention/exp03_token_to_logits.py` 对照。
2. 打印每一步的 shape、dtype、device。
3. 观察 Prefill 与 Decode 每步的序列长度和 logits shape。
4. 记录预测与实际的差异。

~~当前环境缺 torch~~ → 已解决（`/home/gongping/vllm-env`）。**本任务已于 2026-10-25 完成并通过**（详见讲义 7.2 节）。

## 5. 结束时需更新

- [x] LEARNING_STATE.md（知识点状态、薄弱点、复习安排）
- [x] knowledge/ 新增或修改的笔记（lesson01-token-to-gpu.md：7.2/7.4/8 节/变更记录）
- [x] reviews/mistakes.md（本次发现的错误理解，新增 5 条）
- [x] 生成下一次学习任务（L1.2，见本文件顶部）

---

## 历史会话记录

（每次学习结束后在此追加一条摘要：日期 / 学习内容 / 掌握情况 / 下次衔接点。）

- **2026-10-25**：完成 L1.1 收尾。7.2 shape 实验预测与实际 100% 吻合（`/home/gongping/vllm-env`，torch 2.11.0+cu130，环境阻塞解除）；无 cache/cache 对比复述 4 轮纠正后通过（错误集中在"缺量词"和两场景说混，已记 mistakes）。"一个 token 的完整推理过程"升为**已掌握**（本系统第一个）。实验代码第 85 行旧结论表述同步修正。下次：10-26 先做间隔复习，再开始 L1.2 Tokenizer 与 Embedding。
