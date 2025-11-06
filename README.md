# IDS Minimal — Trajectory-Level Alignment & Intent Drift Detection

**Author**: Jianming Lai (Benjamin Daoson)  
**Contribution**: This work introduces the first practical **trajectory-level alignment layer** for Large Language Models, enabling **intent drift detection** in long-horizon reasoning and multi-agent environments.

This repository provides a **minimal, research-oriented implementation**, demonstrating the core conceptual and algorithmic ideas behind **Intent Drift Score (IDS)**.  
Full production implementation (rollback, multi-agent arbitration, enterprise safety hooks) is **not included**.

---

## 1. Motivation

Current alignment methods such as **SFT**, **RLHF**, **DPO**, and **Constitutional AI** share a foundational assumption:

> Alignment is a **static property** of a single model policy.

However, when LLMs operate in **multi-turn dialogs**, **long reasoning chains**, or **multi-agent collaborations**, their internal goals, strategies, and reasoning paths **evolve dynamically**.

This leads to **Intent Drift**:

> A model gradually deviates from the original task objective,  
> **even if each individual step appears locally reasonable.**

This is the main failure mode behind:
- Teaching agents going off-topic or accelerating too quickly
- Compliance agents slowly approaching regulatory boundaries
- Multi-agent systems collapsing into conflict or instability
- Planning agents converging into loops or premature termination

Therefore:

> **Alignment must be lifted from step-level → trajectory-level.**

---

## 2. Approach Overview

This framework evaluates the model’s reasoning behavior over time, computing a unified **Intent Drift Score (IDS)** across three dimensions:

| Dimension | Meaning | Example Failure |
|---------|---------|----------------|
| Semantic | The meaning deviates from the intended goal | Talks about RL instead of PPO |
| Structural | Execution order misaligns | Gives recommendations before risk analysis |
| Temporal | Timing or pacing collapses | Loops or stops prematurely |

The IDS score is computed **online**, enabling:

- Continue
- Replan
- Rollback
- Human Override
- Safety Halt

No model fine-tuning is required — the layer can be applied to:
**GPT / Claude / Qwen / DeepSeek / Mixtral / Multi-Agent Systems.**

---

## 3. Key Contributions

| Contribution | Novelty | Impact |
|-------------|---------|--------|
| **Trajectory-Level Alignment** | Shifts alignment to sequences, not tokens | Directly addresses long-horizon failure modes |
| **Intent Drift Score (IDS)** | Unified, interpretable drift metric | Can be monitored as a system-level safety KPI |
| **Low-Overhead Streaming Implementation** | O(T) parallel integration | Works in real-time deployments |
| **Goal Dependency Graph** | Explicit structural constraints on reasoning | Prevents silent drift in multi-step tasks |
| **Policy Controller for Corrective Actions** | Automated stability management | Enables reliable autonomous agents |

This is not a simple training trick —  
**It acts as the *stability control layer* of the AI operating system.**

---

## 4. System Architecture

```

```
             LLM / Multi-Agent System
                         │
                         ▼
              Trajectory Monitoring Layer
              • Semantic Drift
              • Structural Drift
              • Temporal Drift
              • Goal Dependency Graph
                         │
                         ▼
          Policy Controller (Continue / Replan / Rollback)
```

```

---

## 5. Experimental Summary (Full Results in Paper)

| Scenario | Effect |
|---------|--------|
| Long-horizon teaching dialogs | 82% early drift detection accuracy |
| Multi-agent collaboration tasks | 50% reduction in system collapse rate |
| Financial advisory reasoning | 74% reduction in regulatory breach risk |
| 100k step stress tests | IDS stable — graph baselines diverged |

> **Across domains, without trajectory alignment → drift is inevitable.**

---

## 6. Repository Structure

```

ids_minimal/
├── core.py          # Intent Drift Score (core logic)
├── goal_graph.py    # Task structure dependency graph
└── demo_travel.py   # Example: multi-step planning chain

````

---

## 7. Quick Start

```bash
python demo_travel.py
````

---

## 8. License & Usage Restrictions

This repository is released under:

**CC BY-NC-ND 4.0 — Attribution · Non-Commercial · No Derivatives**

| Rule | Meaning                                    |
| ---- | ------------------------------------------ |
| BY   | Must credit the author                     |
| NC   | Commercial use prohibited                  |
| ND   | Modification and redistribution prohibited |

```
© 2025 Jianming Lai (Benjamin Daoson). All rights reserved.
Full production implementation (rollback logic, multi-agent arbitration,
training pipelines, system-level governance modules) is proprietary and withheld.
```

---

## 9. Collaboration

If your organization works on **Enterprise AI**, **Multi-Agent Copilots**, **Alignment**, or **Autonomous AI**, you may request a private briefing:

📧 Email: [jianming001@e.ntu.edu.sg](mailto:jianming001@e.ntu.edu.sg)
🔗 LinkedIn: [https://linkedin.com/in/benjaminrockefeller](https://linkedin.com/in/benjaminrockefeller)

---

<br>

---

## IDS Minimal — 轨迹级对齐与意图漂移检测框架

本项目展示了一个用于 **长时程推理与多智能体系统** 的 **轨迹级对齐层**，用于检测和控制 **意图漂移（Intent Drift）**，解决大模型在多轮推理中“走着走着偏了”的问题。

---

### 🌍 背景

传统对齐方法（SFT / RLHF / DPO）默认：

> “只要把模型训得好，它就会一直表现好。”

但现实是：

* 模型会根据上下文不断重新估计目标
* 每一步都看似合理，但整体逐渐偏离目标
* 这种偏移无法通过单步评估发现

因此：

> **对齐必须从“行为级”提升到“轨迹级”。**

---

### 🎯 核心贡献

* 提出 **轨迹级对齐（Trajectory-Level Alignment）** 概念
* 引入 **意图漂移评分 IDS**，可量化三类偏移：

  * 语义偏移
  * 结构偏移
  * 时序偏移
* 可直接加在 **任何已训练模型上**，无需重新训练
* 可触发 **继续 / 回退 / 重规划 / 人工接管 / 安全中断**

---

### 📊 实验结论（来自论文）

| 场景      | 效果提升          |
| ------- | ------------- |
| 长链教学智能体 | 早期偏移识别率 82%   |
| 多智能体任务  | 系统崩溃率下降 > 50% |
| 金融合规对话  | 越界风险降低 74%    |

---

### 🛡 许可证

本仓库 **禁止商用、禁止改写、禁止分发**。
仅用于学术研究或技术讨论。

许可证：**CC BY-NC-ND 4.0**

---

```
© 2025 赖建铭（Benjamin Daoson）版权所有。保留所有权利。
生产级实现与治理系统未公开。
```


