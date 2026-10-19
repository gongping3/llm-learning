# transpose 与 permute

## 1. 一句话定义
（待填写）

## 2. 核心原理
（待填写：维度重排后 shape 与 stride 如何变化；为什么不复制数据）

## 3. 最小代码示例
（待填写）

## 4. 我的理解
（待填写）

## 5. 易错点
（已知候选，待验证：理解维度排列，但内存布局不确定 —— 当前薄弱点）

## 6. 与其他知识的联系
stride、contiguous、view、reshape

## 7. 实验验证
（尚无实验记录。已规划实验：转置后调用 view 的行为验证）

## 8. 掌握状态
学习中（理解维度排列，内存布局不确定）

## 9. 官方资料
- https://docs.pytorch.org/docs/stable/generated/torch.permute.html
- https://docs.pytorch.org/docs/stable/generated/torch.transpose.html

## 10. 待解决问题
- permute 是否一定会复制数据？
- 转置后为什么不能直接 view？contiguous() 会发生什么？

## 记录
- 2026-10-19：创建骨架笔记。依据《如何用AI辅导学习》中的记录（理解维度排列但无法解释内存布局），状态记为"学习中"。
