#!/usr/bin/env python3
"""
Inference script for Knowledge Graph Environment.

Runs 3 tasks (task_easy, task_medium, task_hard), grades each via the
/grade HTTP endpoint, and emits the required stdout format:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import asyncio
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Environment variables ─────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY       = HF_TOKEN
# Space URL — during hackathon eval this is the live HF Space
ENV_URL       = os.getenv("ENV_URL",
                           "https://outstandingom-knowledge-graph-env.hf.space").rstrip("/")
BENCHMARK     = "knowledge_graph_env"
MAX_STEPS     = 3
SUCCESS_THRESHOLD = 0.1

# ── Task definitions ──────────────────────────────────────────────────────────
# Each task has an initial input and a system prompt for the LLM agent.
TASK_SPECS = {
    "task_easy": {
        "input":  "I cannot login to my account. My password is not working and I keep getting locked out.",
        "system": (
            "You are a customer support AI. Identify the core issue the user is facing. "
            "Respond with a concise identification of the problem (1-2 sentences)."
        ),
    },
    "task_medium": {
        "input":  "My bill shows a double charge for my subscription this month. I need a refund for the extra payment Invoice INV-2024-891.",
        "system": (
            "You are a customer support AI specialising in billing. Identify the billing issue "
            "and what action should be taken. Respond concisely (1-2 sentences)."
        ),
    },
    "task_hard": {
        "input":  "My account is locked after multiple failed password attempts. I suspect a security breach. Please help urgently — critical issue.",
        "system": (
            "You are a security-aware customer support AI. Identify the critical security issue "
            "and recommend the immediate action. Respond concisely (1-2 sentences)."
        ),
    },
}

# ── Logging helpers ───────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    safe_action = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={safe_action} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={error if error else 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={r_str}",
        flush=True,
    )

# ── LLM helper ────────────────────────────────────────────────────────────────
def get_llm_action(client: OpenAI, system: str, user: str) -> str:
    """Ask the LLM for an action; falls back to the raw input on any error."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.3,
            max_tokens=120,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text if text else user
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", file=sys.stderr, flush=True)
        return user

# ── Grade helper ──────────────────────────────────────────────────────────────
async def grade(http: httpx.AsyncClient, task_id: str, action: str) -> float:
    """POST to /grade and return the score, or 0.0 on failure."""
    try:
        r = await http.post(
            f"{ENV_URL}/grade",
            json={"task_id": task_id, "input_text": action},
            timeout=30.0,
        )
        r.raise_for_status()
        return float(r.json().get("score", 0.0))
    except Exception as exc:
        print(f"[DEBUG] /grade failed for {task_id}: {exc}", file=sys.stderr, flush=True)
        return 0.0

# ── Single-task runner ────────────────────────────────────────────────────────
async def run_task(
    task_id: str,
    spec: dict,
    llm_client: OpenAI,
    http: httpx.AsyncClient,
) -> None:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards:   List[float] = []
    steps_done = 0
    success    = False
    score      = 0.0
    observation = spec["input"]

    for step in range(1, MAX_STEPS + 1):
        # Agent generates action from LLM
        action = get_llm_action(llm_client, spec["system"], observation)

        # Grade the action via the environment's /grade endpoint
        reward = await grade(http, task_id, action)
        reward = max(0.0001, min(0.9999, reward))

        done  = (step == MAX_STEPS)
        error = None

        rewards.append(reward)
        steps_done = step

        log_step(step=step, action=action, reward=reward, done=done, error=error)

        if done:
            break

        # For subsequent steps, feed the graded score back as context
        observation = (
            f"Previous action: {action}\n"
            f"Reward received: {reward:.4f}\n"
            f"Original issue: {spec['input']}"
        )

    score   = sum(rewards) / len(rewards) if rewards else 0.0
    score   = max(0.0, min(1.0, score))
    success = score >= SUCCESS_THRESHOLD

    log_end(success=success, steps=steps_done, score=score, rewards=rewards)

# ── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    llm_client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY if API_KEY else "no-key",
    )

    async with httpx.AsyncClient() as http:
        for task_id, spec in TASK_SPECS.items():
            await run_task(task_id, spec, llm_client, http)

if __name__ == "__main__":
    asyncio.run(main())
