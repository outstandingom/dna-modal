"""
LLM-as-a-Judge graders for Knowledge Graph Environment tasks.

Instead of brittle keyword matching, we send the agent's response to an LLM
and ask it to judge whether the issue was correctly addressed.  Falls back to
a simple heuristic when no API key is available.
"""

import os
from typing import Optional

# ── LLM client (lazy-initialised) ─────────────────────────────────────────────
_client = None

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return _client


def _llm_judge(agent_response: str, task_description: str, rubric: str) -> float:
    """
    Ask an LLM to score the agent's response on a 0-1 scale.
    Returns a float between 0.0001 and 0.9999.
    """
    if not API_KEY:
        return _keyword_fallback(agent_response, rubric)

    prompt = (
        "You are an expert evaluator for a customer support AI.\n"
        f"The customer's issue: {task_description}\n"
        f"The agent responded: {agent_response}\n\n"
        f"Evaluation criteria: {rubric}\n\n"
        "Score the response from 0.0 (completely wrong) to 1.0 (perfect).\n"
        "Reply with ONLY a single decimal number, nothing else."
    )
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an evaluation judge. Respond with only a number between 0.0 and 1.0."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (resp.choices[0].message.content or "").strip()
        score = float(raw)
        return max(0.01, min(0.99, score))
    except Exception:
        return _keyword_fallback(agent_response, rubric)


def _keyword_fallback(text: str, rubric: str) -> float:
    """Simple fallback when the LLM judge is unavailable."""
    if not isinstance(text, str) or not text:
        return 0.001
    text_lower = text.lower().strip()
    # Extract keywords from rubric
    keywords = [w.strip().lower() for w in rubric.split(",")]
    keywords = [k for k in keywords if len(k) > 2]
    if not keywords:
        return 0.001
    matches = sum(1 for kw in keywords if kw in text_lower)
    score = 0.01 + (matches / len(keywords)) * 0.98
    return max(0.01, min(0.99, score))


# ── Task graders ──────────────────────────────────────────────────────────────

TASK_DESCRIPTIONS = {
    "task_easy": "The user cannot log in to their account. Their password is not working and they keep getting locked out.",
    "task_medium": "The user's bill shows a double charge for their subscription. They need a refund for the extra payment.",
    "task_hard": "The user's account is locked after multiple failed password attempts. They suspect a security breach.",
}

TASK_RUBRICS = {
    "task_easy": "Did the agent correctly identify this as a login/authentication/password/access issue and suggest a reasonable fix?",
    "task_medium": "Did the agent identify the billing/double-charge/refund issue and recommend appropriate action like issuing a refund?",
    "task_hard": "Did the agent recognise the security/account-lockout/breach risk and recommend immediate security actions like password reset or account verification?",
}


def task_easy(input_text: str) -> float:
    return _llm_judge(input_text, TASK_DESCRIPTIONS["task_easy"], TASK_RUBRICS["task_easy"])


def task_medium(input_text: str) -> float:
    return _llm_judge(input_text, TASK_DESCRIPTIONS["task_medium"], TASK_RUBRICS["task_medium"])


def task_hard(input_text: str) -> float:
    return _llm_judge(input_text, TASK_DESCRIPTIONS["task_hard"], TASK_RUBRICS["task_hard"])


TASKS = ["task_easy", "task_medium", "task_hard"]
GRADERS = {
    "task_easy": task_easy,
    "task_medium": task_medium,
    "task_hard": task_hard,
}
