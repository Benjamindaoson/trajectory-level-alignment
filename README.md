# 🧭 Trajectory-Level Alignment & Intent Drift Detection

**NeurIPS 2025 Workshop on Multi-Turn Interactions in LLMs**  
*Author: [Jianming Lai (Benjamin Daoson)](https://www.linkedin.com/in/benjaminrockefeller/)*  

---

## 🌍 Overview
This repository implements **Trajectory-Level Alignment (TLA)** — a verifiable alignment framework designed to detect, quantify, and mitigate **intent drift** in long-horizon, multi-agent LLM reasoning.  
By integrating **semantic trajectory analysis**, **real-time anomaly detection**, and **alignment telemetry**, TLA enhances reasoning stability, interpretability, and trustworthiness in autonomous AI systems.

---

## 🔬 Key Contributions
- **Intent Drift Score (IDS)** — Quantifies semantic, structural, and temporal drift in LLM trajectories.  
- **Real-Time Drift Engine** — O(T) Sinkhorn Optimal Transport updates (< 3 ms/step, < 50 MB GPU) for low-latency alignment monitoring.  
- **Alignment Telemetry Layer** — Logs verifiable alignment metrics and supports auditable drift diagnostics.  
- **Multi-Agent Integration** — Compatible with LangChain and MCP for real-time re-planning and error correction.  

---

## 🧩 Architecture
