# PyTorch Tensor Stride

## 1. 一句话定义
（待填写 —— 请用你自己的话写一句话，不要把这句话抄过来）

## 2. 核心原理
（待填写：shape、stride、storage_offset 三者关系；偏移公式）

> 提示：元素 `[i][j]` 的线性存储偏移 = `i*stride[0] + j*stride[1] + storage_offset`

## 3. 最小代码示例
见 `experiments/pytorch/exp01_permute_view.py`

## 4. 我的理解
（待填写：不看资料，用自己的话解释）

## 5. 易错点
- shape 相同，不代表 stride 相同（已验证：x=(3,4) stride (4,1)；permute 后 y=(4,3) stride (1,4)）。
- stride 是**元组**，长度等于维度数；不能只回答一个数（前置检查时只答了"4"，漏掉了 (4,1) 的第二个分量）。
- permute 改变的是 stride 元数据，不是数据本身。

## 6. 与其他知识的联系
view、reshape、transpose、permute、contiguous、Triton Kernel

## 7. 实验验证
2026-10-19，`experiments/pytorch/exp01_permute_view.py`（torch 2.14.1+cpu）：

- `x = torch.arange(12).reshape(3, 4)` → `x.stride() == (4, 1)`
- 偏移公式验证：`x[2][3] = 11`，`2*4 + 3*1 + 0 = 11` ✅
- `y = x.permute(1, 0)` → `y.stride() == (1, 4)`，`y.is_contiguous() == False`，
  但 `x.data_ptr() == y.data_ptr()` → **permute 不复制数据**
- 同一元素在 y 中的偏移：`y[3][2] = 11`，`3*1 + 2*4 = 11` ✅（同一块 storage）

结论：stride 定义"逻辑索引 → 物理偏移"的映射；改 stride 就能换"视角"而不搬数据。

## 8. 掌握状态
学习中（前置检查暴露概念不准，已纠正并有实验证据，但尚未能独立复述）

## 9. 官方资料
- https://docs.pytorch.org/tutorials/ （Tensor 相关章节）

## 10. 待解决问题
- 为什么 GPU 上 stride/连续性会直接影响 kernel 性能？（待第 4 阶段 CUDA 部分回答）
- 多维（3D/4D）情况下偏移公式如何推广？

## 记录
- 2026-10-19：创建骨架笔记 → 完成实验 1 验证，补充已验证事实。掌握状态保持"学习中"。
