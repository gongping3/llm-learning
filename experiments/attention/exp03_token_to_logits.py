"""
实验 3：一个 token 从进入模型到生成出来（第 1 课主体实验）

实验目的：
1. 用真实的 shape 走通完整链路：token → embedding → QKV → attention → FFN → logits → 采样
2. 观察 Prefill（一次吃整个 prompt）与 Decode（逐步生成）两个阶段
3. 为第 2 课埋下伏笔：Decode 每次都重算整个序列 → 为什么要 KV Cache

运行：python exp03_token_to_logits.py（CPU 即可；torch 2.14.1+cpu 与 2.11.0+cu130 均验证通过）
"""

import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)

V, d, H = 1000, 64, 4          # 词表大小 / 模型维度 / 注意力头数
dh = d // H                    # 每个头的维度 = 16

emb = torch.nn.Embedding(V, d)
Wq, Wk, Wv, Wo = (torch.randn(d, d) for _ in range(4))
W1, W2 = torch.randn(d, 4 * d), torch.randn(4 * d, d)
lm_head = torch.randn(V, d)

VERBOSE = True


def show(name, t):
    if VERBOSE:
        print(f"  {name:<14} shape={tuple(t.shape)}  dtype={t.dtype}  device={t.device}")


def forward(ids):
    """ids: (1, S) → logits: (1, S, V)"""
    S = ids.shape[1]
    h = emb(ids)
    show("embedding", h)

    Q, K, Vt = h @ Wq, h @ Wk, h @ Wv            # (1, S, d)
    show("Q / K / V", Q)

    q = Q.view(1, S, H, dh).transpose(1, 2)      # (1, H, S, dh)
    k = K.view(1, S, H, dh).transpose(1, 2)
    v = Vt.view(1, S, H, dh).transpose(1, 2)
    show("多头切分 q", q)

    scores = q @ k.transpose(-2, -1) / math.sqrt(dh)   # (1, H, S, S)
    show("scores", scores)

    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    attn = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
    show("mask+softmax", attn)

    out = (attn @ v).transpose(1, 2).reshape(1, S, d)  # 非连续 → 这里只能用 reshape
    show("多头合并", out)

    h2 = F.rms_norm(h + out @ Wo, (d,))          # 残差 + RMSNorm
    h3 = h2 + F.silu(h2 @ W1) @ W2               # FFN + 残差
    show("block 输出", h3)

    logits = h3 @ lm_head.T                      # (1, S, V)
    show("logits", logits)
    return logits


print("=== Prefill：整个 prompt 一次进入模型 ===")
ids = torch.tensor([[12, 57, 231, 8, 99]])
print("输入 token ids:", tuple(ids.shape))
logits = forward(ids)
last = logits[0, -1]                              # 只取最后一个位置
print("  最后一个位置 logits:", tuple(last.shape), " shape 与序列长度无关")
next_id = last.argmax()
print("  采样得到下一个 token:", next_id.item())

print("\n=== Decode：逐个生成（第 1 步打印 shape，后续只观察长度变化）===")
for step in range(1, 7):
    ids = torch.cat([ids, next_id.reshape(1, 1)], dim=1)
    logits = forward(ids)
    next_id = logits[0, -1].argmax()
    print(f"  step {step}: 序列长度={ids.shape[1]:>2}  本步重算 token 数={ids.shape[1]:>2}  "
          f"logits shape={tuple(logits.shape)}  新 token={next_id.item()}")
    VERBOSE = False

print("\n结论：Decode 每步把完整序列（prompt + 已生成）重算一遍，序列每步 +1；累计计算量至少平方增长"
      "（线性层 O(T²)，attention 也重算则整体 O(T³)）→ 这就是 KV Cache 要解决的问题（第 2 课）。")
