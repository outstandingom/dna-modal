# Knowledge Graph Environment

## Description
Customer support reasoning environment. The agent must:
1. Identify the main issue from a support ticket.
2. Find a related concept or relation.
3. Propose a solution.

## Action Space
String (e.g., "login issue", "refund", "reset password").

## Observation Space
String containing a customer support ticket description (e.g., "I'm having trouble with billing").

## Reward Function
- **Identification step**: Cosine similarity between predicted concept vector and expected concept vector (0.0–1.0).
- **Relation step**: Partial credit for correct relation (0.5) + exact match (0.5) + reasoning engine match (0.3), capped at 1.0.
- **Answer step**: Cosine similarity with expected answer vector (0.0–1.0), with bonus for exact match.

## Tasks
- **Easy**: Direct keyword matching (e.g., "login issue" → "account locked").
- **Medium**: Requires relation inference (e.g., "billing" → "refund").
- **Hard**: Multi-hop reasoning or analogy.

## Setup
1. Add your OpenAI API key as `HF_TOKEN` secret in HF Space.
2. Set `API_BASE_URL` and `MODEL_NAME` secrets.
3. Run `python inference.py` to see a baseline episode.
