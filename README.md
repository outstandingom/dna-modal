# 🧬 DNA Knowledge Graph (GraphRAG Memory Layer)

An advanced, token-efficient, dynamic "Memory Layer" that sits between your users and any Large Language Model (LLM). It completely replaces traditional Vector Database RAG by extracting semantic concepts, building relationship bonds, and injecting compressed, multi-hop reasoning into prompts.

## 🌟 What is it?
The DNA Knowledge Graph is a **Bring-Your-Own-Key (BYOK) middleware API**. 
Instead of dumping raw paragraphs of text into a vector database (standard RAG) and wasting thousands of tokens, this engine actively *learns* during the conversation. It extracts pure concepts (e.g., `Apple`, `Red`, `Fruit`), establishes mathematical and semantic bonds (`Apple -> HAS_COLOR -> Red`), and stores them in a highly compressed 128-dimensional continuous vector space.

When a user asks a question, the graph intercepts the request, traverses the mathematical bonds to pull the exact logical context needed, and seamlessly forwards the enriched prompt to the LLM (using the user's own API key).

## 🚀 Why Use This Instead of Standard RAG?

While standard RAG (Vector Databases) is useful for retrieving exact quotes from massive legal PDFs, it fails miserably at reasoning and is incredibly expensive. Here is why the DNA Graph is the future of autonomous agent memory:

| Feature | Standard RAG (Vector DB) | DNA Knowledge Graph (GraphRAG) |
| :--- | :--- | :--- |
| **Token Efficiency** | ❌ **Terrible.** Stuffs raw, 500-word paragraphs into prompts full of filler words ("the", "and"). | ✅ **Perfect.** Injects highly compressed, structured concepts. Drastically reduces token usage and cost. |
| **Reasoning** | ❌ **Dumb Search.** Matches text embeddings. Cannot connect logical dots if words aren't in the same paragraph. | ✅ **Multi-Hop Logic.** Can walk bonds (`A->B->C`) to answer complex deductive questions standard RAG fails at. |
| **Dynamic Learning** | ❌ **Static.** Requires developers to manually embed and upload new documents. | ✅ **Organic.** Learns instantly during chat. Updates physical and semantic vectors dynamically like a human brain. |
| **Offline Cost** | ❌ Requires paid embedding models to parse data. | ✅ **Zero-Token Background.** Uses a local offline `DynamicKnowledgeLoader` to map concepts for free. |

## 🏗️ Architecture & Features

### 1. Strict BYOK (Bring Your Own Key) Middleware
This engine operates strictly as a memory layer. It does not have global LLM API keys hardcoded into it, and it does not secretly consume tokens in the background to update its graph. 
- It dynamically builds isolated LLM clients per-request using the `provider` and `api_key` sent by the frontend.
- Supports **OpenAI, Google Gemini, Groq, DeepSeek, and HuggingFace**.

### 2. Multi-Tenant Memory Isolation (`session_id`)
You can host a single instance of this backend for a million users. When the frontend sends a request, it includes a unique UUID (`session_id`). The graph instantly creates a private, isolated namespace for that user (e.g., `user_123e4567:concept`). Users will perfectly resume their memory state across sessions, and memories are never cross-contaminated.

### 3. Dual-Layer Vectors
Concepts are stored using a biological DNA approach:
- **Physical Features (12-dim):** Fast, immediate heuristic routing.
- **Semantic Features (128-dim):** Deep, nuanced context understanding.

### 4. Single Orchestrator Architecture
The frontend only ever has to talk to two endpoints:
- `POST /agent`: Handles chat, intercepts memory, and forwards to the LLM.
- `GET /graph`: A single orchestrator endpoint that sweeps the brain and accurately returns every node and bond (with weight and color) for real-time 3D UI rendering.

---

## 💻 How to Use in an Existing Project

You can use this backend to supercharge the memory of *any* frontend app (React, iOS, Discord Bot).

1. Spin up this Python FastAPI server.
2. In your frontend, when a user types a message, make a POST request to `/agent`:
```javascript
const response = await fetch('https://your-dna-backend.com/agent', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "My favorite color is Blue, and I live in Kashmir.",
    session_id: "user-uuid-1234", // Isolates their memory
    provider: "gemini",           // User's chosen LLM
    api_key: "AIzaSy..."          // User's API Key
  })
});
```
3. The graph instantly extracts `Favorite Color: Blue` and `Location: Kashmir`, binds them to the user, and returns the smart LLM response. 

---

## ☁️ Deployment Guide (Testing & Production)

Because the graph saves its organic memory state to local `.pkl` files (e.g., `brain_state.pkl`), you must host it on a server with **persistent storage**. Serverless environments (like Vercel functions or AWS Lambda) will wipe the memory.

### Option 1: DigitalOcean Droplet / AWS EC2 (Recommended)
The absolute best way to host this. The IP is static and the local hard drive guarantees your memory files are never wiped.
1. Create an Ubuntu Server.
2. SSH into the server and clone this repo.
3. Install dependencies: `sudo apt install python3-pip python3-venv`
4. Setup environment: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
5. Run it forever: `nohup uvicorn knowledge_graph_env:app --host 0.0.0.0 --port 7860 > api.log 2>&1 &`

### Option 2: Render (Free Tier - No Credit Card)
If you want to test it completely for free:
1. Connect your GitHub to Render.com and create a new **Web Service**.
2. Point it to this repo.
3. Start Command: `uvicorn knowledge_graph_env:app --host 0.0.0.0 --port 10000`
4. *Warning:* Render's free tier sleeps after 15 minutes of inactivity. When it wakes up, temporary storage is wiped. However, thanks to our `DynamicKnowledgeLoader`, the base ontology will perfectly restore itself! (User-specific session memories will be lost on the free tier unless you upgrade to a paid disk).

### Option 3: Hugging Face Spaces (Free)
1. Create a new "Docker Space".
2. Upload this code.
3. Hugging Face handles the rest. (Requires setting up persistent storage in their settings to keep `.pkl` files).
