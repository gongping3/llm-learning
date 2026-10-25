# 错题与错误理解记录（mistakes）

> 记录每次检验中暴露的错误理解，格式：日期 / 知识点 / 我的错误答案 / 正确解释 / 纠正方式。间隔复习时优先复习本表内容。

| 日期 | 知识点 | 我的错误理解 | 正确解释 | 纠正方式 | 状态 |
|---|---|---|---|---|---|
| 2026-10-19 | stride | 把 stride 描述成"内存布局，相邻快、不相邻慢"（只说出后果，未给出定义） | stride = 逻辑索引每增 1，存储偏移增加的元素数；`offset = i*stride[0]+j*stride[1]+storage_offset` | 精确定定义 + 实验 1 偏移验证 | 已纠正，待复述 |
| 2026-10-19 | permute | 以为 permute 后得到连续张量，stride 应为 (3,1) | permute 只换 shape 与 stride，不复制数据；`(4,3)` 的 stride 是 `(1,4)`，`is_contiguous()=False` | 实验 1：`x.data_ptr()==y.data_ptr()`、偏移公式验证 | 已纠正，待复述 |
| 2026-10-19 | view | 判据记成"形状有没有变"；`x.view(4,3)` 预测为失败 | 判据是"能否只改 stride 就解释出新形状"；连续张量改形状完全合法且共享存储 | 实验 1 | 已纠正 |
| 2026-10-19 | reshape | 认为"reshape 会复制数据"（过度推广） | reshape **只在必要时**复制：非连续 → 复制；连续 → 共享 | 实验 2：data_ptr 比较 + 改值对比法反例 | 已纠正 |
| 2026-10-25 | Decode 序列长度计数 | "第 n 步时序列 = prompt + 已生成 n−1 个，共 n 个"（把 prompt 长度约掉了） | 第 n 步（生成第 n 个新 token）序列长度为 **S+n−1**（S=prompt 长度）；无 cache 时整个序列重算 | 当场指出；讲义 2.7/7.1 的原表述同步修正 | 已纠正 |
| 2026-10-25 | 无 cache / 有 cache 的 Decode 计算对比 | 把"有 cache 后仍要做的事"（新 token 的 Q 与缓存 K 点积、attn 与缓存 V 加权）说成无 cache 场景做的事；无 cache 时 K/V 根本没存，何来"缓存 K" | 无 cache：整条序列所有 token 的 Embedding/QKV/attention/FFN/lm_head 全部重算；有 cache：只算新 token 的主干 + 与缓存 K/V 的 attention | 实验 shape 反证（step 6 打印 (1,11,64)/(1,4,11,11)，11 个 token 全重算）+ 4 轮复述 | 已纠正 |
| 2026-10-25 | 无 cache 重算的范围 | "要算 Embedding、Q/K/V、attention、FFN、lm_head"——缺"谁的"量词，这句话在有 cache 场景也成立，等于没说区别 | 无 cache 的区别在于：这些层对**旧 token**也被迫全部重算；有 cache 只对新 token 算 | 引导补量词 | 已纠正 |
| 2026-10-25 | 有 cache 时新 token 的计算 | "只对新 token 算 attention" | 新 token 要产出 logits 就必须完整过 Embedding→Q/K/V→FFN→lm_head；attention 中 Q 现算、K/V 用缓存 | 因果链推演：logits→lm_head→FFN→QKV | 已纠正 |
| 2026-10-25 | 新 token 的 K/V | "现算"与"存进 cache"二选一 | 两者都对且是先后关系：**先现算**（投影）、**再追加进 cache** 供后续 token 用 | 当场确认 | 已纠正 |
| 2026-10-25 | Decode 复杂度表述 | 引用了实验代码第 85 行旧结论"第 n 步重算 n 个 token、累计平方增长" | 重算的是整条序列（每步 +1）；累计至少 O(T²)（线性层），attention 也重算则整体 O(T³) | 讲义 2.7 已修正；实验代码第 85 行同步修正 | 已纠正 |
