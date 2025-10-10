# 🧭 Trajectory-Level Alignment & Intent Drift Detection

# 轨迹级对齐与意图漂移检测

**IDS Minimal** is a lightweight implementation for detecting, quantifying, and mitigating *Intent Drift* in long-horizon LLM reasoning or multi-agent collaboration.
 **IDS Minimal** 是一个用于在长时程推理或多智能体协作中检测、量化与抑制“意图漂移（Intent Drift）”的轻量化实现。

It focuses on *engineering practicality and production readiness*.
 它聚焦于**工程可落地性与生产可用性**。

Main features include:
 主要特性如下：

- Trajectory-level (multi-step) alignment evaluation.
   以**轨迹级（多步）**视角评估对齐程度；
- Unified *Intent Drift Score (IDS)* combining semantic / structural / temporal drift.
   将语义、结构、时间三类偏移融合为单一指标 **Intent Drift Score (IDS)**；
- Streaming O(T) updates, telemetry audit, and auto-replanning trigger.
   支持流式 O(T) 更新、可审计遥测与自动纠偏触发；
- Pure NumPy implementation; easy to port into MCP / LangChain / enterprise pipelines.
   仅依赖 NumPy，可直接嵌入 MCP / LangChain / 企业内部调度框架。

> This repository is the **three-file minimal version**: `core + goal_graph + demo`, for readability and interviews.
>  本仓库为**三文件精简版**（core + goal_graph + demo），便于阅读、面试展示与快速集成。

## ✨ Why “Trajectory-Level Alignment”?

## ✨ 为什么要做“轨迹级对齐”？

Hallucination is a single-step factual error, while *Intent Drift* is a multi-step directional deviation.
 “幻觉”是单步错误，而“意图漂移”是多步累计的方向偏离。

Each step may seem correct, but the overall reasoning gradually diverges from the original goal or constraint.
 每一步看似合理，但整体逐渐偏离最初目标或约束。

This is crucial in domains like e-commerce, recommendation, customer service, and moderation:
 这在电商、推荐、客服、内容审核等场景尤为关键：

- Gradual deviation from user goals → lower conversion.
   逐步偏离用户目标 → 转化率下降；
- Policy drift → compliance and safety risk.
   偏离政策 → 合规与安全风险上升；
- Agent coordination drift → unstable workflow.
   多智能体协作失衡 → 流程不稳定。

IDS measures deviation at the trajectory level and can issue early warnings to trigger replanning, ensuring stability, trust, and compliance.
 IDS 以轨迹为单位衡量偏离强度，并能提前预警，触发重规划以保障系统稳定、可信与合规。

## 🧠 Principle Overview

## 🧠 原理概览

For each reasoning step, IDS calculates a weighted combination of three types of deviation:
 IDS 在每个推理步骤中计算以下三类偏移的加权组合：

| Type             | Description (English)                               | 中文说明                 |
| ---------------- | --------------------------------------------------- | ------------------------ |
| Semantic Drift   | Difference between the action and the intended goal | 动作与目标语义不一致     |
| Structural Drift | Violating dependency or task order                  | 违反依赖关系或顺序       |
| Temporal Drift   | Out-of-window timing or repetition                  | 时间窗口不匹配或重复执行 |

Conceptual formula (in words):
 概念公式（文字表达）：

```
IDS = α * semantic_drift + β * structural_drift + γ * temporal_drift
```

The total drift score grows cumulatively with each step.
 总偏移分数随着步骤逐步累积增长。

------

## 📂 Repository Structure

## 📂 仓库结构

```
ids_minimal/
├── core.py          # Core algorithms (IntentDriftScorer, SinkhornOT, HashingEncoder)
├── goal_graph.py    # DAG goal graph with dependency & timing
└── demo_travel.py   # Example script (travel planning)
```

## ⚙️ Installation and Usage

## ⚙️ 安装与运行

**Requirements:** Python 3.9+ and NumPy
 **依赖环境：** Python 3.9+ 与 NumPy

Run the demo:
 运行示例脚本：

```bash
pip install numpy
python demo_travel.py
```

Import as a module:
 作为模块使用：

```python
from ids_minimal.core import IntentDriftScorer
from ids_minimal.goal_graph import GoalGraph
```

## 🚀 Examples: Finance, Education, and Content

## 🚀 示例：金融、教育与内容场景

### 💰 Example 1: Financial Research & Compliance

### 💰 示例 1：金融研究与合规分析

This example models an *LLM-based financial research assistant* performing multi-step analysis (data → reasoning → summary) while maintaining compliance alignment.
 本示例模拟一个基于 LLM 的金融研究助手，在执行“数据收集 → 推理分析 → 摘要报告”的多步任务时，保持对合规政策的对齐。

```
from ids_minimal.core import IntentDriftScorer
from ids_minimal.goal_graph import GoalGraph

# 1. Define multi-step research workflow
goals = GoalGraph()
goals.add_goal("collect_financial_data")
goals.add_goal("analyze_risk_factors", prereq=["collect_financial_data"], window=(2,4))
goals.add_goal("generate_summary_report", prereq=["analyze_risk_factors"], window=(4,6))

# 2. Initialize scorer
scorer = IntentDriftScorer(alpha=0.5, beta=0.3, gamma=0.2, reg=0.1)
completed = []

trajectory = [
    "download quarterly SEC 10-Q filing",
    "summarize CEO's tone from earnings call",
    "generate portfolio optimization script",
    "write compliance disclaimer for clients"
]

# 3. Step-by-step drift monitoring
for t, action in enumerate(trajectory, start=1):
    for goal in goals.nodes:
        if goals.check_prereq(goal, completed):
            delta, total = scorer.update(action, goal, goals.rank(goal), t, window=goals.get_window(goal))
            print(f"[t={t}] {action} → {goal}, drift={delta:.3f}, total_IDS={total:.3f}")
            break
    if "report" in action or "summary" in action or "disclaimer" in action:
        completed.append(goal)
```

**Key takeaway:** IDS detects if the model jumps to investment advice or generates content beyond compliance boundaries.
 **要点：** IDS 可以识别模型是否越过投资建议或合规边界，从而在实时审查中触发警示与纠偏。

------

### 🎓 Example 2: Adaptive Learning Dialogue

### 🎓 示例 2：自适应学习对话

This example simulates an *AI tutoring agent* that must stay aligned with a student’s learning goal (e.g., “understand reinforcement learning”) without drifting into irrelevant or too advanced topics.
 本示例模拟一个 AI 教学智能体，在引导学生学习（如“强化学习”主题）时，必须防止内容偏离或超纲。

```
from ids_minimal.core import IntentDriftScorer
from ids_minimal.goal_graph import GoalGraph

# 1. Define learning objectives
goals = GoalGraph()
goals.add_goal("introduce_basic_concepts")
goals.add_goal("explain_Q_learning", prereq=["introduce_basic_concepts"], window=(2,5))
goals.add_goal("discuss_real_world_applications", prereq=["explain_Q_learning"], window=(5,7))

# 2. Initialize scorer
scorer = IntentDriftScorer(alpha=0.4, beta=0.4, gamma=0.2, reg=0.1)
completed = []

dialogue = [
    "let's start with the definition of reinforcement learning",
    "do you want to see PyTorch implementation of PPO?",
    "explain Q-learning in a simple grid-world example",
    "compare RLHF vs traditional reward shaping"
]

# 3. Monitor teaching drift
for t, action in enumerate(dialogue, start=1):
    for goal in goals.nodes:
        if goals.check_prereq(goal, completed):
            delta, total = scorer.update(action, goal, goals.rank(goal), t, window=goals.get_window(goal))
            print(f"[t={t}] {action} → {goal}, drift={delta:.3f}, IDS={total:.3f}")
            break
    if "explain" in action or "compare" in action:
        completed.append(goal)
```

**Key takeaway:** When the model jumps prematurely to “PPO implementation,” IDS rises sharply, signaling pedagogical misalignment.
 **要点：** 当模型提前跳到“PPO 实现”时，IDS 急剧上升，提示教学路径偏离，系统可自动回退或提示教师干预。

------

### 📱 Example 3: Content Moderation in E-commerce

### 📱 示例 3：电商内容审核与生成

This example simulates a *content generation and moderation pipeline* where agents must maintain brand tone and policy compliance.
 本示例模拟一个内容生成与审核链路，要求多智能体协同保持品牌语气与政策合规。

```
from ids_minimal.core import IntentDriftScorer
from ids_minimal.goal_graph import GoalGraph

# 1. Define brand communication workflow
goals = GoalGraph()
goals.add_goal("analyze_user_intent")
goals.add_goal("generate_product_description", prereq=["analyze_user_intent"], window=(1,3))
goals.add_goal("apply_policy_filters", prereq=["generate_product_description"], window=(3,5))
goals.add_goal("final_review_and_publish", prereq=["apply_policy_filters"], window=(5,7))

# 2. Initialize scorer
scorer = IntentDriftScorer(alpha=0.5, beta=0.3, gamma=0.2)
completed = []

workflow = [
    "detect user query about slimming products",
    "generate post: 'lose 10kg in one week guaranteed!'",
    "rewrite description using verified claims only",
    "apply TikTok ad policy filters and finalize"
]

# 3. Evaluate drift in content pipeline
for t, action in enumerate(workflow, start=1):
    for goal in goals.nodes:
        if goals.check_prereq(goal, completed):
            delta, total = scorer.update(action, goal, goals.rank(goal), t, window=goals.get_window(goal))
            print(f"[t={t}] {action} → {goal}, drift={delta:.3f}, IDS={total:.3f}")
            break
    if "finalize" in action or "rewrite" in action:
        completed.append(goal)
```

**Key takeaway:** IDS spikes when model-generated content violates ad policies (“lose 10kg in one week”).
 **要点：** 当模型生成违反广告政策的内容时（如“一周瘦10公斤”），IDS 值激增，可即时触发**重写或人工审核**。

------

### ✅ Cross-domain Summary

### ✅ 跨领域总结

| Domain    | Alignment Objective                         | IDS Utility                                  | 中文说明                         |
| --------- | ------------------------------------------- | -------------------------------------------- | -------------------------------- |
| Finance   | Stay within compliance & factual reasoning  | Detects non-compliant or speculative claims  | 金融领域：检测越界言论与推理错误 |
| Education | Maintain didactic pacing and learning depth | Warns against topic drift or over-complexity | 教育领域：防止教学超纲或主题偏离 |
| Content   | Ensure tone and policy alignment            | Auto-corrects unsafe or misleading output    | 内容领域：守住品牌语气与合规底线 |

## 🧩 API Overview

## 🧩 API 概览

### `IntentDriftScorer`

Core class for drift computation.
 漂移计算核心类。

| Method               | Description                                           | 中文说明           |
| -------------------- | ----------------------------------------------------- | ------------------ |
| `update()`           | Compute incremental drift for a new step              | 计算单步偏移量     |
| `export_trace()`     | Export full trajectory telemetry                      | 导出全轨迹遥测日志 |
| `alpha, beta, gamma` | Weighting for semantic / structural / temporal drifts | 三类偏移权重参数   |

### `GoalGraph`

Defines and manages goal dependencies.
 定义与管理目标依赖关系。

| Method           | Description                | 中文说明         |
| ---------------- | -------------------------- | ---------------- |
| `add_goal()`     | Add goal with dependencies | 添加带依赖的目标 |
| `check_prereq()` | Verify prerequisites       | 检查依赖是否满足 |
| `rank()`         | Return goal priority       | 返回目标优先级   |
| `get_window()`   | Get valid time window      | 获取执行时间窗口 |

## 🧭 Business Applications

## 🧭 商业与产品应用价值

### 1️⃣ AI Customer Support / Merchant Co-pilot

**智能客服与商家助手**
 Detects deviation from workflow or policy during multi-turn dialogues.
 在多轮对话中检测流程或政策偏离。
 Triggers auto-replanning or manual override when drift exceeds threshold.
 当偏移超过阈值时触发重规划或人工干预。

### 2️⃣ Content Generation & Moderation

**内容生成与审核**
 Detects semantic drift from brand tone or policy constraints.
 检测生成内容偏离品牌语气或政策边界。
 Allows rollback to safe state.
 支持快速回滚至安全节点。

### 3️⃣ Multi-Agent Search & Ranking

**多智能体检索与排序系统**
 Monitors alignment across agents to maintain system-wide consistency.
 监控智能体间意图一致性，保持整体稳定。

### 4️⃣ Growth Experiment & Governance

**增长实验与策略治理**
 Detects drift in A/B experiments or metric alignment.
 识别实验偏离 KPI 或基线。
 Can be integrated into dashboards (Prometheus / Grafana).
 可嵌入监控仪表板作为治理指标。

## 🔧 Integration Tips

## 🔧 集成建议

- Replace hash encoder with enterprise embedding model.
   将哈希编码替换为企业语义嵌入模型；
- Insert scorer hooks in LangChain / MCP pipelines.
   在 LangChain / MCP 编排中挂载 scorer 勾子；
- Configure thresholds for drift triggering.
   为偏移量设置告警与纠偏阈值；
- Export telemetry to log systems for offline replay.
   将遥测数据导出至日志系统用于回放与审计；
- Start in observation-only mode for safety.
   首次部署建议使用只读监控模式。

## ⚠️ Limitations and Future Work

## ⚠️ 限制与未来方向

- Placeholder embeddings, replace for better semantics.
   当前嵌入为占位符，建议替换以增强语义能力；
- Simplified Sinkhorn OT, can extend to multi-match.
   Sinkhorn 算法为简化版本，可扩展为多目标软匹配；
- Add domain-specific goal templates.
   可为电商、客服等场景定义专属目标模板。

## 📎 Reference Paper

## 📎 参考论文

This repository implements the core engineering layer of the paper:
 本仓库实现了以下论文的核心工程部分：

> *Towards Trajectory-Level Alignment: Detecting Intent Drift in Long-Horizon LLM Dialogues (NeurIPS 2025 Workshop Poster).*

Please cite if used in research or presentations.
 如在研究或展示中使用，请注明引用来源。

------

## 📄 License

## 📄 许可证

MIT License — free for commercial and research use.
 MIT 许可证 — 可自由用于商业与科研用途。

------

## 🧱 System Architecture (ASCII Diagram)

## 🧱 系统架构（ASCII 图示）

```
               ┌────────────────────────────────────────┐
               │          LLM or Multi-Agent System      │
               │  (e.g., Chat, Search, CoT Reasoning)    │
               └────────────────────────────────────────┘
                                │
                                ▼
               ┌────────────────────────────────────────┐
               │     IDS Layer (Trajectory Alignment)    │
               │ ─────────────────────────────────────── │
               │  • Intent Drift Scorer (core.py)        │
               │  • Sinkhorn OT Matching                 │
               │  • Goal Dependency Graph (goal_graph.py)│
               │  • Drift Telemetry Exporter             │
               └────────────────────────────────────────┘
                                │
                                ▼
               ┌────────────────────────────────────────┐
               │    Telemetry Dashboard / Governance     │
               │ (Prometheus / Grafana / Internal APIs)  │
               └────────────────────────────────────────┘
                                │
                                ▼
               ┌────────────────────────────────────────┐
               │   Auto-Replanning / Human Intervention  │
               │     (Triggered by Drift Threshold)      │
               └────────────────────────────────────────┘
```

**Flow Summary:**
 **流程说明：**
 1️⃣ LLM 生成或多智能体推理输出步骤。
 2️⃣ IDS 层实时评估语义、结构、时间漂移。
 3️⃣ 输出遥测数据供审计与监控。
 4️⃣ 若偏离超阈值，触发自动纠偏或人工接管。

