
# IDS Minimal — Trajectory-Level Alignment & Intent Drift Detection
# IDS Minimal — 轨迹级对齐与意图漂移检测框架

**Author / 作者：** Jianming Lai (Benjamin Daoson)  
**Core Contribution / 核心贡献：** 提出并实现了首个用于 **长时程推理与多智能体系统的轨迹级对齐与意图漂移检测层**。  
此框架已在 *NeurIPS 2025 Workshop* 中展示为研究成果。

---

## 1. 🔍 Motivation — Why New Alignment is Needed?
## 1. 🔍 背景 — 为什么需要新的对齐范式？

Existing alignment methods (SFT / RLHF / DPO / Constitutional AI) assume:

> Once aligned → The model will stay aligned.

But in **long-horizon reasoning**, **multi-round dialogue**, and **multi-agent collaboration**, model behavior is **dynamic**, not static.

> **Intent Drift** = The model’s reasoning trajectory gradually deviates from the original task objective, **even when every individual step looks reasonable.**

传统对齐方法假设：

> 一旦对齐 → 行为保持不变。

但在 **长链推理、多轮对话、多智能体协作** 中，模型行为 **随上下文不断自适应更新**，导致：

> **意图漂移** = 单步合理，但整体逐渐偏离目标，最终**难以回溯与纠正**。

这会导致：

| Domain / 领域 | Failure Mode / 失效模式 |
|---|---|
| 教育智能体 | 教学节奏失衡、跳级、跑题 |
| 金融合规智能体 | 随对话推进逐渐越界风险与监管边界 |
| 电商内容生成系统 | 逐步偏离品牌语气或营销合规 |
| 多智能体协同系统 | 群体目标解释逐步分裂，系统不稳定 |

> **结论：对齐不能只发生在“行为级”，而必须发生在“轨迹级”。**

---

## 2. 🎯 What This Framework Does
## 2. 🎯 框架做什么？

At each reasoning step, we measure trajectory drift on **three dimensions**:

| Drift Type | Meaning | Example |
|-----------|---------|---------|
| **Semantic Drift** | Meaning diverges from intended goal | 讲 PPO 而非 Q-learning |
| **Structural Drift** | Wrong execution order | 未做风险分析就给投资建议 |
| **Temporal Drift** | Wrong timing / repetition | 在流程中循环或提前结束 |

These are integrated into a **single trajectory score**:
→ **IDS: Intent Drift Score**

> When IDS exceeds threshold → **Replan / Rollback / Human Override / Safety Halt**

该框架不依赖模型微调 → **可以直接加在 GPT / Claude / Qwen / DeepSeek / 多智能体系统 上。**

---

## 3. 🧩 Key Technical Contributions
## 3. 🧩 你的核心技术贡献（突出重点）

| Contribution / 贡献点 | Novelty / 创新性 | Value / 价值 |
|---|---|---|
| **提出“轨迹级对齐 (Trajectory-Level Alignment)”方法论** | 将对齐从单步提升到序列 | 解决“走着走着偏了”这一行业核心难题 |
| **定义“Intent Drift Score”统一量化偏移** | 融合语义 / 结构 / 时序稳定性 | 可作为系统级稳定性 KPI |
| **提供 O(T) 流式低开销检测** | 不依赖模型重训，不影响性能 | 可在生产系统实时运行 |
| **引入 Goal Dependency Graph 目标依赖图** | 可显式约束多步骤执行顺序 | 保证智能体链路稳定可控 |
| **支持自动回退 / 重规划 / 人工接管策略** | 实现可控自治智能体 | 迈向 Trusted AI / Safety AI |

> **这不是模型微调技术，这是大模型“操作系统稳定层”。**

---

## 4. 🧱 System Architecture
## 4. 🧱 系统架构

```

```
           ┌────────────────────────────────────┐
           │   LLM / Multi-Agent System         │
           └────────────────────────────────────┘
                           │  actions / plans
                           ▼
           ┌────────────────────────────────────┐
           │    IDS Stability Layer              │
           │  • Semantic / Structural / Temporal │
           │  • Goal Dependency Graph            │
           │  • Streaming Drift Accumulation     │
           └────────────────────────────────────┘
                           │  drift score
                           ▼
      Policy Controller: Continue / Replan / Rollback / Override
```

```

---

## 5. 📊 Experimental Results (From Full Paper)
## 5. 📊 实验结果（论文中已验证）

| Experiment Setting | Result | Interpretation |
|---|---|---|
| 长链教学对话稳定性 | IDS 预警准确率 **82%** | 显著优于 baseline 49% |
| 多智能体协作任务 | 系统失稳率下降 **> 50%** | 提升群体协调能力 |
| 金融投研推理链 | 越界建议触发率降低 **74%** | 提升合规稳定性 |
| 100k 步压力测试 | IDS 稳定，GNN 基线崩溃 | 具有长期推理韧性 |

> **实验显示：不管模型大小多强，只要是长链推理 → 必然发生意图漂移。  
加入 IDS → 才能真正稳定。**

---

## 6. 📂 Repository Structure
## 6. 📂 仓库结构

```

ids_minimal/
├── core.py          # Intent Drift Scorer (核心引擎)
├── goal_graph.py    # 任务依赖有向图
└── demo_travel.py   # 示例：多步骤规划链路

````

---

## 7. 🚀 Quick Start
## 7. 🚀 快速上手

```bash
pip install numpy
python demo_travel.py
````

---

## 8. 🛡 License & Usage Notice | 许可证与使用声明

**This repository is NOT open-source for commercial usage.**
**本仓库不允许商用，也不允许二次分发与改写。**

Released under **CC BY-NC-ND 4.0**:

| Rule | Meaning   |
| ---- | --------- |
| BY   | 必须注明作者    |
| NC   | 禁止商业使用    |
| ND   | 禁止修改与衍生发布 |

```
© 2025 Jianming Lai (Benjamin Daoson). All rights reserved.
Full production implementation (rollback, multi-agent governance,
training pipelines, safety instrumentation) is proprietary and withheld.
```

---

## 9. 🤝 Collaboration / 合作意向

If your organization works on:

* Multi-Agent Intelligent Systems
* Enterprise AI Copilot Infrastructure
* AI Safety & Governance
* Long-horizon Autonomous Agents

You can request **production partnership / closed technical briefing**:

📧 [jianming001@e.ntu.edu.sg](mailto:jianming001@e.ntu.edu.sg)
🔗 LinkedIn: [https://linkedin.com/in/benjaminrockefeller](https://linkedin.com/in/benjaminrockefeller)

```



只回答：**要 / 不要**
```
