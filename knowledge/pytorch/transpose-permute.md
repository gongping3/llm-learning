# transpose 与 permute

## 1. 一句话定义
（待填写 —— 用自己的话写：它改变什么、不改变什么）

## 2. 核心原理
（待填写：维度重排后 shape 与 stride 如何变化；为什么不需要复制数据）

## 3. 最小代码示例
见 `experiments/pytorch/exp01_permute_view.py`

## 4. 我的理解
permute 用于重新排列张量的维度，他不改变张量的底层实际数据，只是改变访问这些张量的视图

## 5. 易错点
- **permute 不复制数据**（前置检查时的错误预期：以为 permute 后 stride 变成连续张量的 (3,1)，实际是 (1,4)）。
- permute 后的 Tensor 通常是非连续的，`is_contiguous() == False`。

## 6. 与其他知识的联系
stride、contiguous、view、reshape

## 7. 实验验证
2026-10-19，`experiments/pytorch/exp01_permute_view.py`（torch 2.14.1+cpu）：

```
x = torch.arange(12).reshape(3, 4)     # shape (3,4), stride (4,1)
y = x.permute(1, 0)                    # shape (4,3), stride (1,4)
y.is_contiguous()                      # False
x.data_ptr() == y.data_ptr()           # True  → 共享存储，未复制
y.view(12)                             # RuntimeError（非连续，无法只靠 stride 解释）
y.contiguous().view(12)                # 成功，且确认新建了存储
```

## 8. 掌握状态
学习中（前置检查答"不知道"，已通过实验建立事实，待独立复述验证）

## 9. 官方资料
- https://docs.pytorch.org/docs/stable/generated/torch.permute.html
- https://docs.pytorch.org/docs/stable/generated/torch.transpose.html

## 10. 待解决问题
- 4D Tensor（如 `[batch, head, seq, dim]`）做 `transpose(1, 2)` 后 stride 如何变化？
  这与 Attention 中 `K.transpose(-2, -1)` 的关系是什么？（第 2 课回答）

## 记录
- 2026-10-19：创建骨架笔记 → 完成实验 1 验证。掌握状态"学习中"。
