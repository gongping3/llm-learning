"""
实验 1：permute 之后发生了什么？（第 1 课前置知识补齐）

实验目的：
1. 精确理解 stride 的定义（逻辑索引每增 1，存储偏移增加的元素数）
2. 验证 permute 是否复制数据（共享 storage？stride 如何变化？）
3. 验证非连续 Tensor 能否 view，以及 contiguous() / reshape 的行为差异

预测（学习者提交，2026-10-19）：
1. x.stride()            -> 4
2. y.shape / y.stride()  -> (4, 3) / 3
3. y.view(4, 3)          -> 成功
4. x.view(4, 3)          -> 失败

实际输出（torch 2.14.1+cpu，2026-10-19）：
  x.shape=(3,4)  x.stride()=(4,1)
  y.shape=(4,3)  y.stride()=(1,4)  y.is_contiguous()=False
  x.data_ptr() == y.data_ptr() -> True（同一块存储）
  y.view(4,3)   -> 成功，stride 仍为 (1,4)
  y.view(12)    -> 失败RuntimeError: view size is not compatible with input tensor's size and stride
  y.contiguous().view(12) -> 成功，且 contiguous() 新建了存储（发生复制）
  y.reshape(12) -> 成功
  x.view(4,3)   -> 成功，stride=(3,1)，与 x 共享存储
  偏移公式验证：x[2][3]=11，2*4+3*1=11；y[3][2]=11，3*1+2*4=11（同一元素同一位置）

结论（CONCLUSION）：
1. stride 是"逻辑索引每增 1，存储偏移增加多少个元素"，是一个与维度数等长的元组。
2. permute 不复制数据：只交换 shape 与 stride，x 与 y 共享同一块 storage。
3. view 只在"仅靠修改 stride 就能解释出新形状"时成功；非连续 Tensor 改形状会失败，
   此时用 reshape（必要时会复制）。contiguous() 的作用就是真正搬运数据使其连续。
"""

import torch

x = torch.arange(12).reshape(3, 4)
print("== 原始 tensor ==")
print("x.shape:      ", x.shape)
print("x.stride():   ", x.stride())
print("x 内容:\n", x)

y = x.permute(1, 0)  # 交换维度 0 和 1
print("\n== permute(1, 0) 之后 ==")
print("y.shape:      ", y.shape)
print("y.stride():   ", y.stride())
print("y 内容:\n", y)
print("y 是否连续 (is_contiguous):", y.is_contiguous())
print("x 与 y 共享底层存储 (data_ptr 相同):", x.data_ptr() == y.data_ptr())

print("\n== view 行为对比 ==")
# 3. 对 permute 结果 view（目标形状与原形状相同）
try:
    z = y.view(4, 3)
    print("y.view(4, 3) -> 成功；结果 stride:", z.stride())
except RuntimeError as e:
    print("y.view(4, 3) -> 失败:", e)

# 3'. 揭示性更强的一问：把非连续 Tensor 拉平成 1D
try:
    z2 = y.view(12)
    print("y.view(12) -> 成功:", z2)
except RuntimeError as e:
    print("y.view(12) -> 失败:", e)

# contiguous() 之后再 view
zc = y.contiguous().view(12)
print("y.contiguous().view(12) -> 成功:", zc.tolist())
print("y.contiguous() 是否新建了存储:", y.data_ptr() != y.contiguous().data_ptr())

# reshape 永不失败（必要时复制）
zr = y.reshape(12)
print("y.reshape(12) -> 成功:", zr.tolist(), "stride:", zr.stride())

# 4. 连续 tensor 改形状
w = x.view(4, 3)
print("\nx.view(4, 3) -> 成功；shape:", w.shape, "stride:", w.stride())
print("x 与 w 共享存储:", x.data_ptr() == w.data_ptr())

# 偏移公式验证：x[i][j] 的存储偏移 = i*stride0 + j*stride1
print("\n== 偏移公式验证 ==")
print("x.storage_offset:", x.storage_offset())
i, j = 2, 3
print(f"x[{i}][{j}] =", x[i][j].item(),
      " 手算偏移 =", i * x.stride()[0] + j * x.stride()[1],
      " 实际线性索引 =", (i * x.stride()[0] + j * x.stride()[1]) + x.storage_offset())
print("整块 storage:", x.storage().tolist())

# 同一个元素在 y 中的偏移
print(f"y[{j}][{i}] =", y[j][i].item(),
      " 手算偏移 =", j * y.stride()[0] + i * y.stride()[1])
