# Trajectory-Level Alignment & Intent Drift Detection

传统对齐方法主要关注 **单步输出的正确性**（如是否回答准确、是否合规）。
 然而真实应用中，大模型往往执行 **多步推理、长链对话或多智能体协作任务**，如：

- 教学场景中的逐步知识引导
- 金融合规研究：数据 → 推理 → 建议链路
- 品牌内容生成与多角色审校
- 多 Agent 协作执行企业流程

在此类 **长时段任务** 中，即使每一步都“看似合理”，
 模型的 **整体推理轨迹** 仍可能 **逐渐偏离原始目标与约束**。

这类偏离称为：**Intent Drift（意图漂移）** —— 不是单步错误，而是 **策略随时间的偏航**。

| 场景                 | 意图漂移带来的风险                 |
| -------------------- | ---------------------------------- |
| 金融研究 → 推荐      | 逐渐越过风险/合规边界              |
| 教育辅导 → 知识讲解  | 节奏紊乱、提前“跳级”或脱离教学目标 |
| 品牌内容 → 审查链路  | 口吻/立场/承诺逐步失控             |
| 多 Agent 协作 → 调度 | 任务链失稳、无法回溯               |

**因此，对齐必须从“单步”上升到“轨迹级”。**

# IDS Minimal 做了什么？

**IDS Minimal** 提供一套 **可线上部署的“轨迹级对齐与漂移监控层”**，用于判断模型是否仍保持在既定目标轨迹内。在每一步推理中，系统会实时计算并累积三类偏移：

| Drift 类型           | 检测内容                     | 示例                           |
| -------------------- | ---------------------------- | ------------------------------ |
| **Semantic Drift**   | 当前内容是否偏离目标语义方向 | 本应讲 Q-learning 却开始讲 PPO |
| **Structural Drift** | 是否违反步骤或依赖顺序       | 未完成风险分析就直接给投资建议 |
| **Temporal Drift**   | 执行时机是否合理             | 无意义反复停留/提前终止        |

三者融合为一个 **单调累积指标**：**Intent Drift Score (IDS)**

**IDS 越高，说明轨迹越可能发生偏航**，可触发：

- 自动重规划
- 人工接管
- 回退到稳定状态

# 为什么现有对齐方法无法解决意图漂移？

主流对齐方案（SFT / RLHF / DPO / Constitutional AI）都隐含同一个假设：**模型行为在部署后是静态的。**

但在 **长链推理、多轮互动、多 Agent 协作** 中：

- 模型会根据上下文、环境反馈和历史对话 **改变策略解释方式**
- 这个变化本身 **会引入轨迹级偏离**
- 偏离不是瞬时的，而是 **逐步积累**

因此：**意图漂移不是“训练不够”，而是“动态策略系统缺少稳定性控制层”。**

要解决它，就必须：对齐行为随时间的演化轨迹，而不仅是对齐当下行为本身。

这称为：

# **Trajectory-Level Alignment**

# 为什么 IDS 是系统的“必要结构层（Default Safety Layer）”？

多步骤 LLM/多智能体系统，本质是 **动态演化系统**：

- CoT 能展开推理，但不能保证推理路径不偏离
- RLHF 能塑造偏好，但无法约束策略随时间的漂移
- DPO 能调行为，但无法判断 **“是否正在偏航”**

因此：IDS 是长链 LLM 系统的“偏航监控仪（Stability Monitor）”。

它不是增强项，而是：多步模型必须具备的稳定性基础层。

# 系统架构

```
LLM / Multi-Agent System
        │
        ▼
┌─────────────────────────────────────┐
│   IDS Layer: Trajectory Alignment   │
│  • Semantic / Structural / Temporal Drift Scoring
│  • Goal & Dependency Graph
│  • Streaming Prefix-Monotonic Drift Accumulation
└─────────────────────────────────────┘
        │
        ▼
Policy Decision: Continue / Replan / Human Override
```

**O(T) 流式计算，低开销，可直接挂接：**

- LangChain / DSPy / MCP 工作流
- 企业内容/合规/运行审计系统
- 多智能体执行框架

# 系统目标不是“让模型更聪明”

而是：

> **让模型在长链任务中保持稳定、不跑偏。**

因此具备：

| 能力                | 描述       |
| ------------------- | ---------- |
| **Controllability** | 轨迹可控   |
| **Auditability**    | 决策可溯源 |
| **Recoverability**  | 偏航可回退 |

即：模型不是变“玄”，而是变“稳”。

# 关键特性（Key Features）

| 能力                           | 价值                       |
| ------------------------------ | -------------------------- |
| **Trajectory-Level Alignment** | 防止“看似每步对，整体走偏” |
| **Intent Drift Score (IDS)**   | 提供可计算的偏航度量       |
| **流式 O(T) 更新**             | 可直接线上部署             |
| **Goal Graph**                 | 保证执行路径顺序合法       |
| **Telemetry Trace**            | 可用于风控/审计/再训练     |
| **Model-Agnostic**             | 不依赖具体模型或训练方式   |

# Repository 结构

```
ids_minimal/
├── core.py          # Intent Drift Scoring
├── goal_graph.py    # 目标依赖图
└── demo_travel.py   # 示例：旅行计划任务
```

# 声明

```
© 2025 Jianming Lai (Benjamin Daoson). All rights reserved.
This repository provides a minimal reference implementation for research and demonstration.
The full production implementation remains proprietary.
```



