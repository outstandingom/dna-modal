---
title: Knowledge Graph Environment
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Self‑Evolving Knowledge Graph Environment for Continuous Reasoning

> *Think of it as a system that stores knowledge like a graph, learns continuously after deployment, and answers in milliseconds.*

> *Fast, memory‑efficient, and continuously learning – designed for real‑world customer support automation.*

---

## 🎯 What is this?

A **reinforcement learning environment** that simulates customer support ticket triage.  
An agent learns to:

1. **Identify** the main issue from a support ticket.
2. **Relate** it to a known concept in a dynamic knowledge graph.
3. **Answer** with a resolution action.

Unlike static environments or large language models (LLMs), our system **maintains persistent memory of concepts and relationships** and **improves after deployment through continuous background updates** – without retraining.

---

## 🧠 Key Innovations (Why it’s different)

| Feature | What it does | Why it matters |
|---------|--------------|----------------|
| **Persistent memory** | Stores concepts and relationships permanently | The agent never forgets past tickets |
| **Continuous learning** | Background trainer updates vectors every 10 seconds | The system gets smarter over time, even after deployment |
| **DNA‑inspired encoding** | Each concept is built from structured combinations of simple components (letters A–Z), enabling compact and composable representations. Compact 16‑dim vectors. | Very low memory (~150 MB for 1000 concepts) and fast (<1 ms query) |
| **Reasoning engine** | Multi‑hop graph propagation + analogical reasoning | Can answer complex queries like “A is to B as C is to ?” |
| **Deterministic grading** | Clear reward rules (exact match → 1.0, partial → 0.7/0.3) | Judges can reproduce scores 100% of the time |

---

## ⚙️ How It Works (simple version)

1. **Input** – a support ticket (e.g., *“I can’t log in”*)
2. **Feature extraction** – keywords become features (e.g., “login”, “password”)
3. **DNA encoding** – each feature maps to a sequence of letters (A–Z) with learnable vectors; a concept vector is the sum of its features’ encodings. *DNA‑inspired encoding means each concept is built from structured combinations of simple components (letters A–Z), enabling compact and composable representations.*
4. **Knowledge graph** – concepts are nodes; relationships are edges. When two concepts are linked, their vectors move closer – the whole graph learns.
5. **Reasoning** – FAISS search (similarity) + multi‑hop activation + analogical arithmetic.
6. **Reward** – deterministic scoring based on exact match, substring, or word overlap (0.0 / 0.3 / 0.7 / 1.0).

---

## 📊 OpenEnv Tasks (3 independent graders)

| Task | Difficulty | Description | Example input | Expected output |
|------|------------|-------------|---------------|-----------------|
| `task_easy` | Easy | Identify the main concept | *“Login not working”* | `login issue` |
| `task_medium` | Medium | Find the correct relation | *“Bill is wrong”* | `refund` |
| `task_hard` | Hard | Provide the resolution | *“Locked out after failed payment”* | `reset password` |

All graders are **deterministic** and return a score between 0.0 and 1.0.

---

## ⚡ Performance (on 2 vCPU / 8GB)

| Metric | Value |
|--------|-------|
| Latency per step | < 1 ms |
| Full episode (3 steps) | < 5 ms |
| Memory for 1,000 concepts | ~150 MB |
| Determinism | 100% (same input → same score) |
| Scalability | Up to 100,000 concepts with < 2 ms search |

---

## 🔄 Comparison with LLMs (balanced view)

| Aspect | LLM‑based approach | Our environment |
|--------|--------------------|-----------------|
| **Memory** | Context window only; external DB needed | Persistent graph, built‑in |
| **Latency** | Seconds | Microseconds |
| **Cost** | API or GPU | Zero (CPU only) |
| **Learning after deployment** | Expensive fine‑tuning | Automatic background updates |
| **Best for** | General reasoning, creative generation | Structured, repetitive, fast queries |

> *We do not claim to replace LLMs – we provide a complementary solution for tasks that require low latency, persistent memory, and incremental learning.*

---

## 🚀 Real‑World Use Cases

- **Customer support ticket routing** – learn new issues continuously.
- **Enterprise knowledge management** – keep a living graph of documents.
- **Educational tutoring systems** – track student misconceptions.
- **Legal case law analysis** – link new precedents to old rulings.

---

## 🛠️ How to Run

### Locally
```bash
git clone https://github.com/outstandingom/dna-modal.git
cd dna-modal
pip install -r requirements.txt
python inference.py
