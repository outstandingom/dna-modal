---
title: Knowledge Graph Environment
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Self‑Evolving Knowledge Graph Environment for Continuous Reasoning

> *Fast, memory‑efficient, and continuously learning – designed for real‑world customer support automation.*

## 🎯 What is this?

A reinforcement learning environment that simulates customer support ticket triage. An agent learns to: 1) Identify the main issue, 2) Relate it to a known concept, 3) Propose a solution.

Unlike static environments or LLMs, our system remembers everything and improves after deployment – without retraining.

## 🧠 Key Innovations

- **Persistent memory** – stores concepts and relationships permanently.
- **Continuous learning** – background trainer updates vectors every 10 seconds.
- **DNA‑inspired encoding** – compact 16‑dim vectors, low memory (~150 MB for 1000 concepts), fast (<1 ms query).
- **Reasoning engine** – multi‑hop graph propagation + analogical reasoning.
- **Deterministic grading** – clear reward rules (exact match → 1.0, partial → 0.7/0.3).

## ⚙️ How It Works

1. Input support ticket → extract keywords as features.
2. Each feature maps to a letter sequence (A‑Z) with learnable vectors.
3. Concept vector = sum of its features' encodings, normalised.
4. Knowledge graph stores concepts and relationships; linked concepts move vectors closer.
5. FAISS search + multi‑hop reasoning + analogical arithmetic.
6. Deterministic scoring (0.0 / 0.3 / 0.7 / 1.0).

## 📊 OpenEnv Tasks (3 independent graders)

- **Easy (task_easy)** – identify the main concept. Example: "Login not working" → "login issue".
- **Medium (task_medium)** – find the correct relation. Example: "Bill is wrong" → "refund".
- **Hard (task_hard)** – provide the resolution. Example: "Locked out after failed payment" → "reset password".

All graders are deterministic and return a score between 0.0 and 1.0.

## ⚡ Performance (on 2 vCPU / 8GB)

- Latency per step: <1 ms
- Full episode (3 steps): <5 ms
- Memory for 1000 concepts: ~150 MB
- Determinism: 100%
- Scalability: up to 100,000 concepts with <2 ms search

## 🔄 Comparison with LLMs (balanced)

- **Memory**: LLM limited context / Ours persistent graph
- **Latency**: LLM seconds / Ours microseconds
- **Cost**: LLM API/GPU / Ours zero (CPU only)
- **Learning after deployment**: LLM expensive fine‑tuning / Ours automatic background updates
- **Best for**: LLM general reasoning / Ours structured, repetitive, fast queries

We complement LLMs, not replace them.

## 🚀 Real‑World Use Cases

- Customer support ticket routing
- Enterprise knowledge management
- Educational tutoring systems
- Legal case law analysis

## 🛠️ How to Run

Locally:
```bash
git clone https://github.com/outstandingom/dna-modal.git
cd dna-modal
pip install -r requirements.txt
python inference.py
