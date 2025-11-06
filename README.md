```markdown
# IDS Minimal — Trajectory-Level Alignment & Intent Drift Detection
**A lightweight, deployment-ready stability layer for multi-step LLM reasoning and multi-agent systems.**

Long-horizon LLM tasks (teaching dialogues, financial analysis, workflow agents, multi-agent planning) often appear correct at each individual step, yet **the overall reasoning trajectory gradually drifts away from the original goal**.  
This phenomenon is known as:

> **Intent Drift** — A gradual, trajectory-level deviation of behavior over time.

Traditional alignment (SFT / RLHF / DPO / Constitutional AI) aligns **single-step behavior**, assuming that desirable behavior **remains stable** after deployment.  
But in **multi-step and adaptive environments**, this assumption does **not** hold.

**IDS Minimal** provides a **real-time stability monitor** that detects when a model’s reasoning trajectory is drifting off-course — *before* the task fails.

---

## 🔥 Why This Matters

| Application | Risk Without IDS | What IDS Prevents |
|------------|----------------|-------------------|
| **AI Tutors** | Model “jumps topics” or teaches incorrectly | Maintains instructional pacing & topic consistency |
| **Financial / Trading Agents** | Gradual violation of risk or leverage policies | Enforces compliance & rule stability |
| **Enterprise Workflows** | Procedure steps executed out of order | Guarantees correct task sequencing |
| **Multi-Agent Collaboration** | Agents diverge from shared objectives | Preserves coordination & shared goals |

> **IDS is not about making models smarter — it is about making them stable, reliable, and controllable.**

---

## 🧭 What IDS Minimal Does

At each reasoning step, IDS computes and accumulates **trajectory drift** along three dimensions:

| Drift Type | Meaning | Example |
|-----------|---------|---------|
| **Semantic Drift** | Meaning drifts from intended task | Talking about PPO when the goal is Q-learning |
| **Structural Drift** | Steps executed out of order or missing dependencies | Making investment recommendation before doing risk analysis |
| **Temporal Drift** | Wrong timing, repetition, or premature termination | Looping or skipping key phases |

These are combined into a single **Intent Drift Score (IDS)**:

\[
IDS(\tau) = \sum_{t=1}^{T} \delta(a_t, v_t)
\]

If IDS exceeds a threshold → trigger:

- **Re-plan**
- **Human override**
- **Rollback to stable checkpoint**

---

## ✅ Key Capabilities

| Capability | Description | Value |
|-----------|-------------|-------|
| **Trajectory-Level Alignment** | Aligns *entire reasoning process*, not just individual steps | Prevents slow, undetected goal drift |
| **Streaming O(T) Drift Monitoring** | Incremental updates per step | Suitable for real-time production workloads |
| **Model-Agnostic** | Works with GPT / Claude / LLaMA / Qwen / DeepSeek / Multi-Agent systems | No retraining required |
| **Interpretable Telemetry** | Logs exactly **where** and **why** drift occurred | Supports audit, compliance, governance |

> **IDS is a *system layer*, not a training trick. It integrates into real deployments.**

---

## 🏗 Repository Structure
```

ids_minimal/
 │
 ├── core.py          # IntentDriftScorer: fusion of semantic/structural/temporal drift
 ├── goal_graph.py    # GoalGraph: defines allowed task ordering & constraints
 └── demo_travel.py   # Example: multi-step planning trajectory with drift alerts

```
### Minimal Example

```python
from core import IntentDriftScorer
from goal_graph import GoalGraph

graph = GoalGraph()
graph.add_goal("search")
graph.add_goal("evaluate", prereq=["search"])
graph.add_goal("decide", prereq=["evaluate"])

scorer = IntentDriftScorer(goal_graph=graph)

for step in trajectory:
    scorer.update(step)

print("Total Drift:", scorer.score)
print(scorer.export_trace())
```

------

## 📈 Empirical Validity (from Full Paper)

| Setting                         | Result                                                       |
| ------------------------------- | ------------------------------------------------------------ |
| **Cross-domain generalization** | IDS retains **0.79–0.85 correlation** without retraining     |
| **Multi-agent collaboration**   | Coordination failures reduced by **> 50%**                   |
| **100k-step stress tests**      | IDS maintains stability while GNN-based methods collapse     |
| **Expert human evaluation**     | IDS alarms judged correct **82%** of the time (vs. 49% baseline) |

> **Equivalent systems without IDS will drift. It’s not if — it's when.**

------

## 🛡️ Deployment Strategy

| Mode             | Usage                            | When to Use                                     |
| ---------------- | -------------------------------- | ----------------------------------------------- |
| **Monitor-only** | Log drift but do not intervene   | Observability phase                             |
| **Soft Control** | Trigger re-plan on high drift    | Production with fallback                        |
| **Hard Control** | Force rollback on drift boundary | High-stakes settings (finance, medical, policy) |

------

## ⚖️ License & Usage Notice

This repository contains a **minimal conceptual implementation** for research and demonstration.

```
© 2025 Jianming Lai (Benjamin Daoson). All rights reserved.

This code is provided for research, evaluation, and interview demonstration only.
The full production-grade version (including strategy rollback, multi-agent governance,
dataset pipelines, and deployment tooling) remains proprietary.
```

If you wish to collaborate on production deployment, please contact:
jianming001@e.ntu.edu.sg

------

## 🌍 Citation

If referencing or discussing this framework:

```
Lai, Jianming. "Trajectory-Level Alignment: Detecting Intent Drift in Long-Horizon LLM Dialogues." 2025.
```

