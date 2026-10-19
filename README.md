# llm-learning —— 大模型推理学习系统

> 最终目标：从大模型推理入门，成长为能够阅读、修改和贡献 vLLM 源码的开发者。
> 学习路线以 **vLLM V1** 为主（V0 已废弃），总周期约 6～9 个月（按 2 小时/天）。

## 一、这套系统怎么运作

核心原则：**Git 仓库中的学习文件才是长期可信的记录，AI 对话是学习过程，而不是知识存储位置。**

即使更换 AI 产品、开启新对话或模型升级，只要这个仓库还在，学习就能无缝接续。每次学习开始时，AI 导师先阅读 `ROADMAP.md`、`LEARNING_STATE.md`、`CURRENT_SESSION.md`，从上次结束的地方继续。

## 二、关键文件

| 文件 | 作用 | 更新频率 |
|---|---|---|
| `ROADMAP.md` | 未来整体学什么（9 个阶段） | 阶段变化时 |
| `LEARNING_STATE.md` | 现在学到哪里、掌握到什么程度 | 每次学习后 |
| `CURRENT_SESSION.md` | 本次具体做什么 | 每次开始、结束时 |
| `TUTOR_RULES.md` | AI 导师工作规范（长期有效） | 很少变化 |
| `knowledge/` | 已验证、值得长期保留的知识笔记 | 知识理解或验证后 |
| `curriculum/` | 各阶段的课程大纲和任务 | 按阶段更新 |
| `experiments/` | 动手实验代码 | 每次实验后 |
| `reviews/` | 复习记录、错题 | 每次复习后 |
| `sources/references.md` | 官方资料索引 | 随时补充 |

## 三、目录结构

```
llm-learning/
├── README.md                  # 本文件：学习系统总入口
├── ROADMAP.md                 # 总体学习路线（9 阶段）
├── LEARNING_STATE.md          # 当前进度与薄弱点
├── CURRENT_SESSION.md         # 本次学习任务
├── TUTOR_RULES.md             # AI 导师工作规范
│
├── curriculum/                # 阶段、课程和任务
│   ├── 00-python-pytorch.md       阶段 0
│   ├── 01-transformer.md          阶段 1
│   ├── 02-llm-inference.md        阶段 2
│   ├── 03-kv-cache-batching.md    阶段 3
│   ├── 04-cuda-triton.md          阶段 4
│   ├── 05-vllm-core.md            阶段 5
│   ├── 06-scheduler-models.md     阶段 6～7
│   ├── 07-performance.md          阶段 8
│   └── 08-contributor.md          阶段 9
│
├── knowledge/                 # 已整理的知识（统一模板，见 TEMPLATE.md）
│   ├── pytorch/
│   ├── transformer/
│   ├── inference/
│   └── vllm/
│
├── experiments/               # 动手实验
│   ├── pytorch/
│   ├── attention/
│   ├── kv-cache/
│   └── vllm/
│
├── reviews/                   # 复习、测验、错题
│   ├── mistakes.md
│   └── review-log.md
│
└── sources/
    └── references.md          # 官方资料索引
```

## 四、每日学习工作流（约 2 小时）

| 时段 | 时长 | 内容 |
|---|---|---|
| 回忆 | 10 min | 不看笔记回答 3 个问题，AI 判断哪些需要复习 |
| 新知识 | 30 min | 阅读官方资料，AI 解释难点，一次 1～2 个相互关联的概念 |
| 实验 | 45 min | 自己写代码、预测输出、运行验证；AI 辅助调试但不代替推理 |
| 检验 | 20 min | AI 通过概念解释、代码分析、变式题判断是否真正理解 |
| 整理 | 15 min | 总结笔记、记录错误、更新学习状态、生成下次任务 |

## 五、AI 导师的四种模式

- **课程导师**：按路线安排课程，先查前置再讲新知，最后给练习
- **答疑导师**：先定位你卡在哪一步，再从那里讲，用小例子和反例
- **考核导师**：先让你回答、预测输出、自己实现，再指出错误，不提前公布答案
- **知识管理员**：把有长期价值的内容整理进知识库，更新进度，生成下次衔接

详细规范见 `TUTOR_RULES.md`。

## 六、掌握状态的判定

知识点分四个状态：**未学习 → 学习中 → 待验证 → 已掌握**。

重要规则：AI 生成的解释不等于你已掌握的知识。只有你能独立解释、分析相关代码，并通过实验或问题验证之后，才标记为"已掌握"。

## 七、复习机制

学习后第 1、3、7、14、30 天安排间隔复习。复习时 AI 先出题（不先重讲旧笔记），根据回答区分"真正理解"和"仅记得结论"，并更新 `reviews/review-log.md`。

## 八、开始学习

每次开始学习时对 AI 说：

> 请先阅读 F:\llm-learning 下的 ROADMAP.md、LEARNING_STATE.md 和 CURRENT_SESSION.md，然后按 TUTOR_RULES.md 的规范继续辅导。

当前进度见 `LEARNING_STATE.md`，本次任务见 `CURRENT_SESSION.md`。
