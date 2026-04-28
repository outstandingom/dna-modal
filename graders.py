"""
LLM-as-a-Judge graders for Knowledge Graph Environment tasks.

Instead of brittle keyword matching, we send the agent's response to an LLM
and ask it to judge whether the issue was correctly addressed.  Falls back to
a simple keyword heuristic when no API key is available.

IMPORTANT: All scores MUST be strictly between 0 and 1 (exclusive).
"""

import os

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


def _clamp(score: float) -> float:
    """Ensure score is strictly within (0, 1) — never 0.0 or 1.0."""
    return max(0.01, min(0.99, score))


def _llm_judge(agent_response: str, task_description: str, keywords: list) -> float:
    """
    Ask an LLM to score the agent's response on a 0-1 scale.
    Falls back to keyword matching if LLM is unavailable.
    Always returns a value strictly in (0, 1).
    """
    if not isinstance(agent_response, str) or not agent_response.strip():
        return _clamp(0.01)

    # Try LLM judge first
    if API_KEY:
        try:
            prompt = (
                "You are an expert evaluator for a customer support AI.\n"
                f"The customer's issue: {task_description}\n"
                f"The agent responded: {agent_response}\n\n"
                "Did the agent correctly identify and address the issue?\n"
                "Score the response from 0.0 (completely wrong) to 1.0 (perfect).\n"
                "Reply with ONLY a single decimal number, nothing else."
            )
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
            return _clamp(score)
        except Exception:
            pass  # Fall through to keyword fallback

    # Keyword fallback
    return _keyword_fallback(agent_response, keywords)


def _keyword_fallback(text: str, keywords: list) -> float:
    """Simple keyword matching fallback. Always returns strictly in (0, 1)."""
    if not isinstance(text, str) or not text.strip():
        return _clamp(0.01)

    text_lower = text.lower().strip()
    if not keywords:
        return _clamp(0.01)

    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    score = 0.01 + (matches / len(keywords)) * 0.98
    return _clamp(score)


# ── Task definitions ──────────────────────────────────────────────────────────

TASK_DESCRIPTIONS = {
    "task_easy": "The user cannot log in to their account. Their password is not working and they keep getting locked out.",
    "task_medium": "The user's bill shows a double charge for their subscription. They need a refund for the extra payment.",
    "task_hard": "The user's account is locked after multiple failed password attempts. They suspect a security breach.",
}

# Keywords for fallback grading — short tokens that an LLM response would
# naturally contain if it addresses the issue correctly.
TASK_KEYWORDS = {
    "task_easy":   ["login", "account", "password", "access", "sign in", "authentication", "credential", "reset"],
    "task_medium": ["bill", "payment", "charge", "invoice", "refund", "subscription", "double", "overcharge"],
    "task_hard":   ["locked", "security", "breach", "blocked", "verify", "critical", "password", "unauthorized"],
}


def task_easy(input_text: str) -> float:
    return _llm_judge(input_text, TASK_DESCRIPTIONS["task_easy"], TASK_KEYWORDS["task_easy"])


def task_medium(input_text: str) -> float:
    return _llm_judge(input_text, TASK_DESCRIPTIONS["task_medium"], TASK_KEYWORDS["task_medium"])


def task_hard(input_text: str) -> float:
    return _llm_judge(input_text, TASK_DESCRIPTIONS["task_hard"], TASK_KEYWORDS["task_hard"])


TASKS = ["task_easy", "task_medium", "task_hard"]
GRADERS = {
    "task_easy": task_easy,
    "task_medium": task_medium,
    "task_hard": task_hard,
}
