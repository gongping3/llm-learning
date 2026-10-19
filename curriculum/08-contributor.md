# 阶段 9：真正开始贡献 vLLM

> 时间：长期
> 最终目标：Issue → Reproduce → Understand source → Modify → Unit Test → Benchmark → PR

## 1. 贡献路径（由易到难）

### Level 1
修 Documentation / Tests / Typo / Error message

### Level 2
修改 Python / Scheduler / API / Model

### Level 3
修改 KV Cache / Scheduler / Worker

### Level 4
修改 Triton / CUDA / Attention / Quantization

### Level 5
研究 Distributed inference / Scheduling / Memory management / Kernel optimization

到 Level 5 才算真正进入 vLLM 核心开发。

## 2. 每次贡献的固定流程

```text
Issue → 复现 → 读懂相关源码 → 修改 → 单元测试 → Benchmark → PR
```

## 3. 实践建议

- 从 "good first issue" 标签的 issue 入手
- 严格跟随所固定 commit 的代码，注意 main 分支变化
- 每次贡献的学习过程整理进 knowledge/vllm/

## 4. 完成标准

- [ ] 第一个被合并的 PR（哪怕只是文档或测试）
