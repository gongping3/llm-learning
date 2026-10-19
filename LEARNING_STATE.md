# 学习状态（LEARNING_STATE）

> 每次学习结束后更新。记录真实进度，不虚构已完成的课程或实验。
> 状态定义：未学习 / 学习中（看过资料但不能独立解释）/ 待验证（能解释基本原理但实验或代码理解不完整）/ 已掌握（能独立解释、分析代码并通过检验）。

**最后更新**：2026-10-19（系统初始化，依据《如何用AI辅导学习》与《大模型推理学习计划》两份文档中的真实记录）

## 一、当前所处阶段

- **阶段 0（Python/PyTorch/GPU 环境）**：环境部分基本完成（WSL2 + NVIDIA GPU + CUDA + Python 3.12 + uv，已接触过 vLLM）；PyTorch Tensor 基础正在学习中。
- **阶段 1（Transformer）**：即将开始，当前正处于"开始追问 Wq/Wk/Wv、softmax、位置编码"的入门状态。
- **当前任务**：第 1 课《从一个 token 到 GPU：LLM 推理完整过程》（见 CURRENT_SESSION.md）。

## 二、知识点状态表

| 知识点 | 状态 | 理解程度 | 实验状态 | 关联知识 | 下次行动 | 资料来源 |
|---|---|---|---|---|---|---|
| shape / dtype / device | 学习中 | 已纠正：shape 是"每个维度的大小"（维度数是 ndim）；二维维度含义回答正确 | 无记录（已排入第 1 课实验 2） | stride、view | 第 1 课实验 2 打印链路 | PyTorch 官方文档 |
| stride / 内存布局 | 学习中 | 已知偏移公式并有实验证据，但尚未独立复述（前置检查只答出局部性） | 实验 1 已运行并验证（exp01） | view、contiguous、transpose | 复述 stride 定义 + 3D 推广 | PyTorch 官方文档 |
| view / reshape | 学习中 | 已验证 view/reshape/contiguous 三者差异；预测 3 对 1 错 | 实验 1 已运行并验证（exp01） | stride、contiguous | 判断 reshape 是否复制的实验 | PyTorch 官方文档 |
| transpose / permute | 学习中 | 已知 permute 不复制数据、只改 shape+stride；待独立复述 | 实验 1 已运行并验证（exp01） | stride、contiguous、view | 4D transpose 与 `K.transpose(-2,-1)` 的关系 | PyTorch 官方文档 |
| Wq / Wk / Wv 的来源 | 学习中 | 正在提问阶段 | 无 | QKV、matmul、embedding | 第 1～2 课 | Attention Is All You Need |
| Softmax | 学习中 | 正在提问阶段 | 无 | attention、dim 参数 | 第 2～3 课 | PyTorch 官方文档 |
| Position Embedding / RoPE | 学习中 | 正在提问阶段 | 无 | - | 第 3～4 课 | 论文 3.5 节 |
| 一个 token 的完整推理过程 | 未学习 | - | - | prefill/decode/KV cache | 第 1 课 | - |
| Prefill / Decode | 未学习 | - | - | KV cache | 第 1～5 课 | - |
| KV Cache | 未学习 | - | - | K/V、显存 | 阶段 2 | PagedAttention 论文 |

> 注：目前尚无任何"已掌握"知识点——所有已学内容都缺少实验或测验证据，符合规范要求的诚实记录。

## 三、薄弱点

1. **Tensor 内存布局**：stride 定义表述不准（只说局部性）、permute 后 stride 判断错误、view 判据误记为"形状变没变"。→ 实验 1 已纠正，需复述验证。
2. 尚未建立"用 shape/dtype/device 一路解释数据流"的习惯。
3. 系统工程背景是优势，但 Transformer 理论链路（Embedding → QKV → Attention → ... → logits）尚未打通。

## 四、学习环境

- WSL2 + NVIDIA GPU + CUDA + Python 3.12 + uv，已安装/接触过 vLLM。
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
