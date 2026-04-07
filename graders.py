def task_easy(input_text: str) -> float:
    if not isinstance(input_text, str) or not input_text:
        return 0.0012345
    text = input_text.lower().strip()
    keywords = ["login", "account", "password", "access", "sign in"]
    matches = sum(1 for kw in keywords if kw in text)
    score = 0.001 + (matches / len(keywords)) * 0.998
    return max(0.0001, min(0.9999, score))

def task_medium(input_text: str) -> float:
    if not isinstance(input_text, str) or not input_text:
        return 0.0023456
    text = input_text.lower().strip()
    keywords = ["bill", "payment", "charge", "invoice", "refund", "subscription"]
    matches = sum(1 for kw in keywords if kw in text)
    score = 0.001 + (matches / len(keywords)) * 0.998
    return max(0.0001, min(0.9999, score))

def task_hard(input_text: str) -> float:
    if not isinstance(input_text, str) or not input_text:
        return 0.0034567
    text = input_text.lower().strip()
    keywords = ["locked", "failed", "security", "blocked", "breach", "critical"]
    matches = sum(1 for kw in keywords if kw in text)
    score = 0.001 + (matches / len(keywords)) * 0.998
    return max(0.0001, min(0.9999, score))

TASKS = ["task_easy", "task_medium", "task_hard"]
GRADERS = {
    "task_easy": task_easy,
    "task_medium": task_medium,
    "task_hard": task_hard
}
