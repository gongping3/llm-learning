# view 与 reshape

## 1. 一句话定义
（待填写）

## 2. 核心原理
（待填写：与 stride / 连续性 (contiguous) 的关系，何时复制数据）

## 3. 最小代码示例
（待填写）

## 4. 我的理解
（待填写）

## 5. 易错点
（已知候选，待验证：不确定 reshape 何时会复制数据 —— 这是当前明确的薄弱点）

## 6. 与其他知识的联系
stride、contiguous、transpose、permute

## 7. 实验验证
（尚无实验记录）

## 8. 掌握状态
学习中（知道用途，不确定何时复制数据）

## 9. 官方资料
- https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html
- https://docs.pytorch.org/docs/stable/generated/torch.reshape.html

## 10. 待解决问题
- 什么条件下 view 会失败而 reshape 不会？
- 如何用实验证明某次 reshape 发生了数据复制？

## 记录
- 2026-10-19：创建骨架笔记。依据学习计划文档，已学但"不确定何时会复制数据"，状态记为"学习中"。
