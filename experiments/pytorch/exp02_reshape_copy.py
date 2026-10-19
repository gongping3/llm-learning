"""
实验 2：reshape 到底会不会复制数据？（第 1 课前置知识补齐）

实验目的：
1. 验证 reshape 永不失败（与 view 的区别）
2. 验证 reshape **只在必要时**复制数据，而不是"总是复制"
3. 掌握判断是否复制的证据：data_ptr() 比较 + 改值对比法

预测（学习者提交）：reshape(12) 不会失败，会复制数据；
证明方法：把结果赋给新变量，修改它，再与原 Tensor 对比值是否一致。
→ 判定：结论正确（非连续情形），但"reshape 总是复制"是错误推广；改值对比法有效。

实际输出（torch 2.14.1+cpu，2026-10-19）：
  y.is_contiguous(): False
  非连续 reshape：data_ptr 不同 → 复制；改 yr[0]=999 后 y[0][0] 仍为 0
  连续 reshape：storage().data_ptr() 相同 → 不复制；改 xr[0]=111 后 x[0][0] 变为 111
  permute 共享存储：改原始 x[0][1]=777 后 y[1][0] 变为 777

结论：
  reshape 永不失败；是否复制取决于原 Tensor 是否连续 —— 连续则共享，非连续则复制。
  判断依据：storage().data_ptr() 是否相同，或改值后另一方是否同步变化。
  注：torch 2.14 中模块级 torch.shares_memory 已不存在，用 data_ptr() 比较更可靠。
"""

import torch

x = torch.arange(12).reshape(3, 4)
y = x.permute(1, 0)          # 非连续
print("y.is_contiguous():", y.is_contiguous())

yr = y.reshape(12)           # 非连续 → 必须复制
print("\n== 非连续 Tensor 上的 reshape（你的预测场景）==")
print("结果 shape:", tuple(yr.shape), "stride:", yr.stride())
print("是否新建存储（data_ptr 不同）:", y.data_ptr() != yr.data_ptr())
print("yr[:5]:", yr[:5].tolist())
yr[0] = 999
print("把 yr[0] 改成 999 后 —— y[0][0]:", y[0][0].item(), "（未被影响 → 确实复制了）")

xr = x.reshape(12)           # 连续 → 不复制，共享存储
print("\n== 连续 Tensor 上的 reshape（反例：不复制）==")
print("storage 地址是否相同:", x.storage().data_ptr() == xr.storage().data_ptr())
xr[0] = 111
print("把 xr[0] 改成 111 后 —— x[0][0]:", x[0][0].item(), "（同步变化 → 共享）")

print("\n== permute 结果与原 Tensor 共享存储的证据 ==")
x[0][1] = 777
print("把原始 x[0][1] 改成 777 后 —— y[1][0]:", y[1][0].item(), "（同步变化）")
