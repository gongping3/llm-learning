# 学习状态（LEARNING_STATE）

> 每次学习结束后更新。记录真实进度，不虚构已完成的课程或实验。
> 状态定义：未学习 / 学习中（看过资料但不能独立解释）/ 待验证（能解释基本原理但实验或代码理解不完整）/ 已掌握（能独立解释、分析代码并通过检验）。

**最后更新**：2026-10-25（7.2 实验通过；第 1 课链路升为已掌握，L1.1 完成）

## 一、当前所处阶段

- **阶段 0（Python/PyTorch/GPU 环境）**：原记录为 WSL2 + NVIDIA GPU + CUDA + Python 3.12 + uv；2026-10-20 在当前机器未复现出 torch 环境，需先复核或重装。PyTorch Tensor 基础正在学习中。
- **阶段 1（Transformer）**：第 1 课已从零讲解，等待复述与实验；尚不能视为已掌握。
- **当前任务**：第 1 课《从一个 token 到 GPU：LLM 推理完整过程》（见 CURRENT_SESSION.md）。

## 二、知识点状态表

| 知识点 | 状态 | 理解程度 | 实验状态 | 关联知识 | 下次行动 | 资料来源 |
|---|---|---|---|---|---|---|
| shape / dtype / device | 学习中 | 已纠正：shape 是"每个维度的大小"（维度数是 ndim）；二维维度含义回答正确 | 无记录（已排入第 1 课实验 2） | stride、view | 第 1 课实验 2 打印链路 | PyTorch 官方文档 |
| stride / 内存布局 | 学习中 | 已知偏移公式并有实验证据，但尚未独立复述（前置检查只答出局部性） | 实验 1 已运行并验证（exp01） | view、contiguous、transpose | 复述 stride 定义 + 3D 推广 | PyTorch 官方文档 |
| view / reshape | 待验证 | 已能区分 view/reshape/contiguous，并掌握"是否复制"的判据与证据 | 实验 1、实验 2 已运行并验证 | stride、contiguous | 复述 reshape 判据（连续不复制/非连续复制） | PyTorch 官方文档 |
| transpose / permute | 待验证 | 已知 permute 不复制数据、只改 shape+stride | 实验 1 已运行并验证（exp01） | stride、contiguous、view | 4D transpose 与 `K.transpose(-2,-1)` 的关系 | PyTorch 官方文档 |
| 一个 token 的完整推理过程 | 已掌握 | 7.3 三问独立通过 + 7.2 实验预测 100% 吻合 + 无 cache/cache 对比经 4 轮纠正后完整复述（2026-10-25） | 7.2 已运行验证（exp03，torch 2.11.0+cu130） | prefill/decode、logits 采样、KV Cache 动机 | 10-26 第 1 天间隔复习 | 论文 3.2 节 |
| Prefill / Decode | 待验证 | 2026-10-25 已能完整对比无 cache/cache 两种 Decode（4 轮纠正后）；O(T²)/O(T³) 表述已修正 | 7.2 已复现（10-25，exp03） | KV cache、memory bound | 用”每 token 读多少权重”解释 Decode 的 memory bound（尚未做到） | - |
| logits 与采样 | 待验证 | 2026-10-25 已独立解释”为什么只取最后位置”与”主干全算/lm_head 可省”的区别，并对比训练时行为 | 待 7.2 实验观察 | sampler、temperature | 实验观察 logits shape 与采样 | HF LLM Course |
| Wq / Wk / Wv 的来源 | 学习中 | 已复述"X @ Wq/Wk/Wv 三个投影"（2026-10-25 第 1 题）；角色差异（Q 找什么 / K 提供什么 / V 传什么）尚未复述 | 无 | QKV、matmul、embedding | 一句话说出 Q/K/V 的角色差异 | Attention Is All You Need |
| Softmax | 学习中 | 2026-10-20 已讲解 dim=-1 与 sqrt(dh) 缩放；待复述 | 无 | attention、dim 参数 | 解释 mask 后 softmax 的行为 | PyTorch 官方文档 |
| Position Embedding / RoPE | 未学习 | 第 1 课仅提及“需要位置信息”，未展开 | 无 | - | L1.2/L1.3 | 论文 3.5 节 |
| KV Cache | 未学习 | 第 1 课只建立“为什么要存”的直觉 | 无 | K/V、显存 | 第 2 课 | PagedAttention 论文 |

> 注：2026-10-25 出现第 1 个"已掌握"知识点（一个 token 的完整推理过程）；其余仍按证据不足如实记录。

## 三、薄弱点

1. **Tensor 内存布局**：stride 定义表述不准（只说局部性）、permute 后 stride 判断错误、view 判据误记为"形状变没变"。→ 实验 1 已纠正，需复述验证（10-26 复习）。
2. 尚未建立"用 shape/dtype/device 一路解释数据流"的习惯。
3. ~~Transformer 理论链路尚未复述验证~~ → 已打通（2026-10-25：7.3 + 7.2 通过）。
4. ~~环境未就绪~~ → 已解决：`/home/gongping/vllm-env`（uv venv，torch 2.11.0+cu130）。
5. **无 cache / 有 cache 的 Decode 计算对比**：2026-10-25 复述 4 轮才通过（缺"谁的"量词、把两场景说混），需在间隔复习中重点出变式题。

## 四、学习环境

- 原记录：WSL2 + NVIDIA GPU + CUDA + Python 3.12 + uv，已安装/接触过 vLLM。
- 2026-10-20 核查：Windows Python 3.13 有 numpy 2.4.4、无 torch；WSL 默认发行版为 docker-desktop；Ubuntu-24.04 未安装 torch，也未找到 uv/venv。
- 2026-10-25 核查：`/home/gongping/vllm-env`（uv venv）可用，**torch 2.11.0+cu130**；exp03 已在此环境运行验证。后续实验统一用该环境。
- 学习节奏：约 2 小时/天。

## 五、复习安排

- 2026-10-26：PyTorch 内存布局（10-19 学的第 7 天）+ 第 1 课链路（10-25 学的第 1 天）——出变式题，不重讲。
- 2026-10-28：第 1 课链路第 3 天。之后 11-01 / 11-08 / 11-24。
- 规则：学习后第 1、3、7、14、30 天复习，先出题再重讲，记录在 `reviews/review-log.md`。

## 六、里程碑进度

- [ ] 阶段 0：独立运行 PyTorch + HuggingFace + vLLM，观察 GPU 状态
- [ ] 阶段 1：Project 1 —— 手写 Attention / mini Transformer
- [ ] 阶段 2：Project 2 —— 手写 KV Cache
- [ ] 阶段 3：Project 3 —— 实现 Continuous Batching
- [ ] 阶段 5：Project 4 —— 实现 Paged KV Cache
- [ ] 阶段 7：为 vLLM 添加一个简单模型
- [ ] 阶段 9：第一个被合并的 PR
