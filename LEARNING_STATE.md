# 学习状态（LEARNING_STATE）

> 每次学习结束后更新。记录真实进度，不虚构已完成的课程或实验。
> 状态定义：未学习 / 学习中（看过资料但不能独立解释）/ 待验证（能解释基本原理但实验或代码理解不完整）/ 已掌握（能独立解释、分析代码并通过检验）。

**最后更新**：2026-10-20（第 1 课从零重讲；等待复述与实验，环境待复核）

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
| 一个 token 的完整推理过程 | 学习中 | 2026-10-20 从零讲解；能看 shape 表，待独立复述链路与推导 | 10-19 实验 3 仅作参考；本次待重新运行 | prefill/decode、logits 采样 | 回答讲义 7.3 三问 + 完成 7.2 实验 | 论文 3.2 节 |
| Prefill / Decode | 学习中 | 已讲解两阶段与 compute/memory bound；KV Cache 只建立初步概念 | 10-19 实验 3 有观察，待本次复现 | KV cache、memory bound | 用“每 token 读多少权重”解释 Decode | - |
| logits 与采样 | 学习中 | 已讲解 logits (B,S,V)、只取最后位置；区分主干与 lm_head | 待本次实验观察 | sampler、temperature | 回答“其余位置去哪了” | HF LLM Course |
| Wq / Wk / Wv 的来源 | 学习中 | 2026-10-20 已讲解是三个可学习投影矩阵；待复述 | 无 | QKV、matmul、embedding | 复述 Q/K/V 的角色差异 | Attention Is All You Need |
| Softmax | 学习中 | 2026-10-20 已讲解 dim=-1 与 sqrt(dh) 缩放；待复述 | 无 | attention、dim 参数 | 解释 mask 后 softmax 的行为 | PyTorch 官方文档 |
| Position Embedding / RoPE | 未学习 | 第 1 课仅提及“需要位置信息”，未展开 | 无 | - | L1.2/L1.3 | 论文 3.5 节 |
| KV Cache | 未学习 | 第 1 课只建立“为什么要存”的直觉 | 无 | K/V、显存 | 第 2 课 | PagedAttention 论文 |

> 注：目前尚无任何"已掌握"知识点——所有已学内容都缺少实验或测验证据，符合规范要求的诚实记录。

## 三、薄弱点

1. **Tensor 内存布局**：stride 定义表述不准（只说局部性）、permute 后 stride 判断错误、view 判据误记为"形状变没变"。→ 实验 1 已纠正，需复述验证。
2. 尚未建立"用 shape/dtype/device 一路解释数据流"的习惯。
3. Transformer 理论链路（Embedding → QKV → Attention → ... → logits）已在 2026-10-20 讲解一遍，但尚未复述验证，不能算已打通。
4. 环境未就绪：当前机器没找到 torch，第 1 课实验尚未重新运行。

## 四、学习环境

- 原记录：WSL2 + NVIDIA GPU + CUDA + Python 3.12 + uv，已安装/接触过 vLLM。
- 2026-10-20 核查：Windows Python 3.13 有 numpy 2.4.4、无 torch；WSL 默认发行版为 docker-desktop；Ubuntu-24.04 未安装 torch，也未找到 uv/venv。原记录的环境未在当前会话复现，需先确认或重装。
- 学习节奏：约 2 小时/天。

## 五、复习安排

- 目前无已进入复习周期的知识点。
- 规则：学习后第 1、3、7、14、30 天复习，先出题再重讲，记录在 `reviews/review-log.md`。

## 六、里程碑进度

- [ ] 阶段 0：独立运行 PyTorch + HuggingFace + vLLM，观察 GPU 状态
- [ ] 阶段 1：Project 1 —— 手写 Attention / mini Transformer
- [ ] 阶段 2：Project 2 —— 手写 KV Cache
- [ ] 阶段 3：Project 3 —— 实现 Continuous Batching
- [ ] 阶段 5：Project 4 —— 实现 Paged KV Cache
- [ ] 阶段 7：为 vLLM 添加一个简单模型
- [ ] 阶段 9：第一个被合并的 PR
