明白，我现在直接为你输出**最终版·中英双语 README**，可直接复制粘贴到 GitHub **原地覆盖**，无需你再做任何修改、微调或排版处理。

我会保持：

* **正式、专业、可用于面试 & 投递华为/字节/美团北斗/新加坡政府/NeurIPS Poster 展示**
* **全局结构稳定**
* **中英完全对齐，不出现内容不一致**
* **表格全部为 GitHub 可正常渲染格式，不会折行炸格式**

---

# ✅ **最终版 README.md（中英双语）**

```markdown
# IDS Minimal — Trajectory-Level Alignment & Intent Drift Detection  
# IDS Minimal — 轨迹级对齐与意图漂移检测框架

**A lightweight, deployment-ready stability layer for long-horizon LLM reasoning and multi-agent systems.**  
**一个可直接部署于长时程推理与多智能体系统的轻量级稳定性对齐组件。**

---

## 🔍 Problem Motivation | 问题背景

Most alignment methods (SFT / RLHF / DPO / Constitutional AI) assume that once aligned, the model will **remain aligned**.  
But in **multi-step reasoning** and **interactive environments**, the model’s intentions **change over time**.

> **Intent Drift** = The reasoning trajectory gradually deviates from the original goal, even though each individual step looks “reasonable”.

传统对齐方法假设模型一旦对齐便会**保持稳定**；  
但在**多步推理**与**交互场景**中，模型的目标与策略会**随时间发生偏移**，即：

> **意图漂移** = 单步看似正确，但整体逐渐偏离任务目标。

This is the key failure mode in:
- AI Tutors (内容超纲 / 脱离教学目标)
- Financial Agents (越界投资建议 / 合规风险)
- Enterprise Workflow Agents (流程执行顺序错误)
- Multi-Agent Systems (协作目标分裂 / 群体失稳)

---

## 🎯 What IDS Minimal Does | 框架核心能力

At each reasoning step, IDS evaluates **trajectory-level alignment**, not single outputs.

IDS 在每一步推理中实时评估**轨迹级对齐状态**，而非仅判断单步输出。

| Drift Type | Meaning | Example |
|-----------|---------|---------|
| **Semantic Drift** | Meaning deviates from intended task | Teaching PPO while asked to teach Q-learning |
| **Structural Drift** | Violates task dependency / sequencing | Investment recommendation before risk assessment |
| **Temporal Drift** | Too early / too late / repeated steps | Looping, skipping, or collapsing phase order |

→ These are fused into a single **Intent Drift Score (IDS)**.

> **If IDS exceeds threshold → trigger re-plan / rollback / human override.**

---

## ✅ Key Features | 框架特性

| Capability | Value | 中文说明 |
|-----------|-------|---------|
| **Trajectory-Level Alignment** | Prevents slow, undetected drift | 防止长期推理中逐步偏离 |
| **O(T) Streaming Monitoring** | Suitable for real-time systems | 流式监控，适用于在线系统 |
| **Model-Agnostic** | Works with GPT / Claude / Qwen / DeepSeek | 模型无关，无需重新训练 |
| **Auditable Telemetry** | Shows where & why drift occurs | 可审计、可解释、可追溯 |

---

## 🏛 Repository Structure | 仓库结构

```

ids_minimal/
│
├── core.py          # IntentDriftScorer (drift computation engine)
├── goal_graph.py    # GoalGraph (task dependency & sequencing)
└── demo_travel.py   # Demo example (multi-step planning)

````

---

## 🚀 Quick Start | 快速上手

### Installation | 安装
```bash
pip install numpy
````

### Run Demo | 运行示例

```bash
python demo_travel.py
```

### Minimal Usage | 最小可用示例

```python
from core import IntentDriftScorer
from goal_graph import GoalGraph

goals = GoalGraph()
goals.add_goal("search")
goals.add_goal("evaluate", prereq=["search"])
goals.add_goal("decide", prereq=["evaluate"])

scorer = IntentDriftScorer(goal_graph=goals)

trajectory = [
    "gather product info",
    "generate recommendation",
    "compare alternative suppliers"
]

for step in trajectory:
    scorer.update(step)

print("Total Drift:", scorer.score)
print(scorer.export_trace())   # For audit / visualization
```

---

## 🌍 Real-World Applications | 典型落地场景

| Domain                              | IDS Ensures                 | 中文说明         |
| ----------------------------------- | --------------------------- | ------------ |
| **AI Tutor / Education Agents**     | Stable teaching progression | 防止超纲、偏题、跳级教学 |
| **Financial / Trading Systems**     | Compliance & risk alignment | 保持风控边界，不越位   |
| **Enterprise Workflows / Copilots** | Correct step sequencing     | 保证流程有序执行     |
| **Multi-Agent Collaboration**       | Shared goal stability       | 防止智能体群体失控    |

---

## 🔧 Deployment Modes | 部署模式

| Mode         | Behavior                      | Use Case            | 中文说明           |
| ------------ | ----------------------------- | ------------------- | -------------- |
| Monitor-Only | Logs drift; no intervention   | Observability       | 观察期，只监控不干预     |
| Soft Control | Triggers re-plan on threshold | Production          | 达阈值自动纠偏        |
| Hard Control | Rollback or override          | High-stakes domains | 医疗 / 金融 / 安全场景 |

---

## 📜 Reference | 参考论文

This repository corresponds to the core engineering layer of:

```
Lai, Jianming (Benjamin Daoson).
"Towards Trajectory-Level Alignment: Detecting Intent Drift in Long-Horizon LLM Dialogues."
NeurIPS 2025 Workshop Poster.
```

---

## 📄 License | 许可证

**MIT License** — Free for academic & commercial adaptation.
**MIT 许可证** — 可自由用于科研与商业落地。

---

## 🤝 Contact | 合作交流

For production deployment, research collaboration, or enterprise alignment consulting:

**Email:** [jianming001@e.ntu.edu.sg](mailto:jianming001@e.ntu.edu.sg)
**LinkedIn:** [https://linkedin.com/in/benjaminrockefeller](https://linkedin.com/in/benjaminrockefeller)




