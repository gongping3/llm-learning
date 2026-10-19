# view 与 reshape

## 1. 一句话定义
（待填写）

## 2. 核心原理
（待填写：与 stride / 连续性的关系，何时复制数据）

> 规则：**view 只在"仅修改 stride 就能解释出新形状"时成功；reshape 永不失败，必要时复制数据。**

## 3. 最小代码示例
见 `experiments/pytorch/exp01_permute_view.py`

## 4. 我的理解
view和reshape都能改变张量形状，但是底层语义差别很大。

view 不会重排数据，他只是用新的shape和stride重新解释一块内存

reshape 返回的是副本

## 5. 易错点
- 以为"改形状就一定会失败/一定会复制"：错的。`x.view(4, 3)`（x 连续）是成功的且共享存储。
- 把"非连续 Tensor 的 view"判据记成"形状变没变"：真正判据是**能否只用 stride 重解释**。
  反例（已验证）：`y.view(4, 3)` 成功（形状没变，无需重解释），但 `y.view(12)` 失败。
- `contiguous()` 会真正搬数据（已验证：data_ptr 变化）。

## 6. 与其他知识的联系
stride、contiguous、transpose、permute

## 7. 实验验证
2026-10-19，`experiments/pytorch/exp01_permute_view.py`（torch 2.14.1+cpu）：

| 调用 | 结果 | 是否复制 |
|---|---|---|
| `x.view(4, 3)`（x 连续） | 成功，stride (3,1) | 否，共享存储 |
| `y.view(4, 3)`（y 非连续，形状未变） | 成功，stride 仍 (1,4) | 否 |
| `y.view(12)`（y 非连续，改形状） | RuntimeError | — |
| `y.contiguous().view(12)` | 成功 | **是**（新建存储） |
| `y.reshape(12)` | 成功 | 需要时复制 |

## 8. 掌握状态
学习中（前置检查预测 3 对 1 错，已通过实验纠正，待独立复述验证）

## 9. 官方资料
- https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html
- https://docs.pytorch.org/docs/stable/generated/torch.reshape.html

## 10. 待解决问题
- ~~如何判断一次 reshape 是否发生了复制？~~ **已解决（2026-10-19，实验 2）**：
  1. 比较 `a.storage().data_ptr() == b.storage().data_ptr()`；
  2. 或**改值对比法**：修改结果 Tensor，看原 Tensor 是否同步变化（学习者提出的方法，有效）；
  3. 注意 torch 2.14 中模块级 `torch.shares_memory` 已不存在，用 `data_ptr()` 更可靠。

## 11. 补充：reshape 是否复制的判据（实验 2 已验证）
| 场景 | 是否复制 | 证据 |
|---|---|---|
| 非连续 Tensor（如 permute 结果）`.reshape(...)` | **复制** | data_ptr 不同；改结果不影响原 Tensor |
| 连续 Tensor `.reshape(...)`（形状改变但连续） | **不复制** | storage().data_ptr() 相同；改结果会同步影响原 Tensor |

## 记录
- 2026-10-19：创建骨架笔记 → 完成实验 1 验证。掌握状态"学习中"。
