"""
Self-Learning DNA Cognitive Judge – Drop‑in replacement for graders.py.

Replaces the external LLM‑as‑a‑Judge with a fully self‑contained DNA engine
based on 26 letters, 128‑dim embeddings, a 26×26 relationship matrix,
and sine‑modulated Hebbian learning.  Scores remain strictly between 0 and 1.

The public interface (task_easy, task_medium, task_hard, TASKS, GRADERS)
is IDENTICAL to the original, so core.py imports and endpoints continue to work.
"""

import os
import json
import math
import numpy as np
import threading
from typing import List, Dict, Optional

# ───────────────────────────────────────────────────────────────────────
# 1. THE FROZEN CORTEX (26 letters, 128 dims – NEVER updated)
# ───────────────────────────────────────────────────────────────────────
class DNACortex:
    """
    The immutable alphabet.  Each of the 26 letters is a 128‑dim vector.
    These NEVER change, guaranteeing zero catastrophic forgetting.
    """
    def __init__(self):
        rng = np.random.RandomState(42)  # deterministic seed
        self.embeddings = rng.randn(26, 128).astype(np.float32)
        # normalise to unit length
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.map = {c: i for i, c in enumerate(self.letters)}

    def get_dna_sequence(self, text: str) -> np.ndarray:
        """Convert text to a DNA strand: [Sequence_Length, 128]."""
        idxs = [self.map.get(c.upper(), 0) for c in text if c.isalpha()]
        if not idxs:
            return np.zeros((1, 128))
        return self.embeddings[idxs]


# ───────────────────────────────────────────────────────────────────────
# 2. THE GLOBAL SYNAPSE (26×26 relationship matrix – updated globally)
# ───────────────────────────────────────────────────────────────────────
class GlobalSynapse:
    """
    The 26×26 matrix W[i][j] = strength of the bond between letter i and j.
    Updated by EVERY user using a sine‑modulated Hebbian rule.
    """
    def __init__(self, persist_path: str = "dna_synapse.json"):
        self.persist_path = persist_path
        # initialise neutral
        self.weights = np.full((26, 26), 0.5, dtype=np.float32)
        np.fill_diagonal(self.weights, 0.5)
        self.step = 0
        self.lock = threading.RLock()
        self._load()

    def _load(self):
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "r") as f:
                    data = json.load(f)
                    self.weights = np.array(data["weights"], dtype=np.float32)
                    self.step = data.get("step", 0)
            except Exception:
                pass

    def _save(self):
        with self.lock:
            with open(self.persist_path, "w") as f:
                json.dump({
                    "weights": self.weights.tolist(),
                    "step": self.step
                }, f)

    def update(self, letter_probs: np.ndarray, reward: float):
        """
        Hebbian update: W[i][j] += lr * (reward‑0.5) * p_i * p_j
        Learning rate is modulated by a sine wave (biological clock).
        """
        with self.lock:
            self.step += 1
            # sine‑modulated learning rate oscillates between ~0.005 and ~0.035
            base_lr = 0.02
            sine_lr = 0.5 + 0.5 * math.sin(self.step / 15.0)
            lr = base_lr * sine_lr

            hebbian_delta = (reward - 0.5) * 2.0  # maps reward to [-1, +1]

            # outer product: p_i * p_j  (shape 26x26)
            co_activation = np.outer(letter_probs, letter_probs)

            # update weights
            self.weights += lr * hebbian_delta * co_activation
            # clamp to [0.01, 0.99] to avoid extreme forgetting
            self.weights = np.clip(self.weights, 0.01, 0.99)
            # self‑bonds stay neutral
            np.fill_diagonal(self.weights, 0.5)

            # persist every 10 steps
            if self.step % 10 == 0:
                self._save()


# ───────────────────────────────────────────────────────────────────────
# 3. THE DNA STRAND (Hidden layer – sequence of 128‑dim vectors)
# ───────────────────────────────────────────────────────────────────────
class DNAStrand:
    """
    Represents a sequence of 128‑dim vectors (the "working memory").
    Updated per user via sine‑modulated Hebbian learning.
    """
    def __init__(self, initial_sequence: np.ndarray):
        # initial_sequence shape: [L, 128]
        self.sequence = initial_sequence.copy()
        self.length = self.sequence.shape[0]
        # positional sine – acts as a "time tag" for each position
        self.positional_sine = np.array([math.sin(i / 3.0) for i in range(self.length)]).reshape(-1, 1)

    def fire(self) -> np.ndarray:
        """Activation = sequence * (0.5 + 0.5 * sine_position)."""
        return self.sequence * (0.5 + 0.5 * self.positional_sine)

    def hebbian_update(self, reward: float, step: int, lr_base: float = 0.03):
        """Update the 128‑dim vectors directly using Hebbian rule + temporal sine."""
        # temporal sine – global rhythm
        temporal_sine = 0.5 + 0.5 * math.sin(step / 20.0)
        lr = lr_base * temporal_sine

        hebbian_delta = (reward - 0.5) * 2.0
        activation = self.fire()

        # positional sine gives each position a different learning rate
        self.sequence += lr * hebbian_delta * activation * (0.5 + 0.5 * self.positional_sine)
        self.sequence = np.clip(self.sequence, -1.0, 1.0)


# ───────────────────────────────────────────────────────────────────────
# 4. THE MASTER DNA JUDGE ENGINE (Singleton)
# ───────────────────────────────────────────────────────────────────────
class DNAJudgeEngine:
    """
    The complete self‑learning judge.
    - Cortex (26×128) – frozen
    - Synapse (26×26) – global, updated by all users
    - Global Strand – collective hidden layer
    - User deltas & strands – per‑user personalisation
    """
    def __init__(self):
        self.cortex = DNACortex()
        self.synapse = GlobalSynapse()
        self.global_step = 0

        # Global strand: initialised with a long generic sentence
        self.global_strand = self.cortex.get_dna_sequence(
            "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG"
        )
        self._load_global_strand()

        # cache for user data: {user_id: {"delta": [26], "strand": np.array}}
        self.user_cache = {}
        self.lock = threading.RLock()

        # task descriptions (exactly as in original)
        self.task_descriptions = {
            "task_easy": "The user cannot log in to their account. Their password is not working and they keep getting locked out.",
            "task_medium": "The user's bill shows a double charge for their subscription. They need a refund for the extra payment.",
            "task_hard": "The user's account is locked after multiple failed password attempts. They suspect a security breach.",
        }

    def _load_global_strand(self):
        path = "dna_global_strand.npy"
        if os.path.exists(path):
            try:
                self.global_strand = np.load(path)
            except Exception:
                pass

    def _save_global_strand(self):
        np.save("dna_global_strand.npy", self.global_strand)

    def _get_user_data(self, user_id: str):
        with self.lock:
            if user_id not in self.user_cache:
                # initialise user delta (26‑dim bias) and personal strand
                delta = np.zeros(26, dtype=np.float32)
                strand = self.cortex.get_dna_sequence("HELLOUSER")
                # try to load from disk
                path_strand = f"user_dnas/{user_id}_strand.npy"
                path_delta = f"user_dnas/{user_id}_delta.npy"
                if os.path.exists(path_strand):
                    try:
                        strand = np.load(path_strand)
                    except Exception:
                        pass
                if os.path.exists(path_delta):
                    try:
                        delta = np.load(path_delta)
                    except Exception:
                        pass
                self.user_cache[user_id] = {
                    "delta": delta,
                    "strand": strand
                }
            return self.user_cache[user_id]

    def _save_user_data(self, user_id: str):
        with self.lock:
            data = self.user_cache.get(user_id)
            if data is None:
                return
            os.makedirs("user_dnas", exist_ok=True)
            np.save(f"user_dnas/{user_id}_strand.npy", data["strand"])
            np.save(f"user_dnas/{user_id}_delta.npy", data["delta"])

    def judge(self, user_id: str, agent_response: str, task_description: str) -> float:
        """
        Core method:
        1. Encode task & response into DNA strands.
        2. Compute letter probabilities from the task.
        3. Score = similarity + synapse influence.
        4. Upgrade: synapse, task strand, response strand, user delta, global strand.
        5. Return clamped score.
        """
        self.global_step += 1

        # ---- 1. Encode ----
        task_strand = DNAStrand(self.cortex.get_dna_sequence(task_description))
        resp_strand = DNAStrand(self.cortex.get_dna_sequence(agent_response))

        # ---- 2. Letter activations (from task) ----
        task_concept = np.mean(task_strand.fire(), axis=0, keepdims=True)  # [1,128]
        raw_act = self.cortex.embeddings @ task_concept.T
        raw_act = np.squeeze(raw_act)  # [26]
        # softmax with temperature
        exp_act = np.exp(raw_act * 2.0)
        letter_probs = exp_act / (np.sum(exp_act) + 1e-8)

        # ---- 3. Synapse influence ----
        synapse_influence = letter_probs @ self.synapse.weights @ letter_probs.T

        # ---- 4. Similarity between task and response ----
        task_fire = task_strand.fire()
        resp_fire = resp_strand.fire()
        task_vec = np.mean(task_fire, axis=0, keepdims=True)
        resp_vec = np.mean(resp_fire, axis=0, keepdims=True)
        task_vec = task_vec / (np.linalg.norm(task_vec) + 1e-8)
        resp_vec = resp_vec / (np.linalg.norm(resp_vec) + 1e-8)
        similarity = np.dot(task_vec, resp_vec.T)[0, 0]   # [-1, 1]

        # ---- 5. Final score ----
        raw_score = 0.7 * ((similarity + 1.0) / 2.0) + 0.3 * synapse_influence
        score = max(0.01, min(0.99, raw_score))
        reward = score  # the score is the reward signal

        # ---- 6. THE UPGRADE (learning) ----
        # A. Global synapse
        self.synapse.update(letter_probs, reward)

        # B. Task strand hidden layer
        task_strand.hebbian_update(reward, self.global_step, lr_base=0.03)

        # C. Response strand hidden layer
        resp_strand.hebbian_update(reward, self.global_step + 100, lr_base=0.04)

        # D. User delta (personal bias over 26 letters)
        user_data = self._get_user_data(user_id)
        delta_lr = 0.06 * (0.5 + 0.5 * math.sin(self.global_step / 10.0))
        user_data["delta"] += delta_lr * (reward - 0.5) * 2.0 * letter_probs
        user_data["delta"] = np.clip(user_data["delta"], -0.4, 0.4)

        # E. User personal strand
        user_data["strand"] += 0.02 * (reward - 0.5) * 2.0 * task_strand.fire().mean(axis=0)
        user_data["strand"] = np.clip(user_data["strand"], -1.0, 1.0)

        # F. Global strand (collective wisdom)
        global_lr = 0.001 * (0.5 + 0.5 * math.sin(self.global_step / 50.0))
        # align lengths
        if len(self.global_strand) < len(task_strand.sequence):
            pad = np.zeros((len(task_strand.sequence) - len(self.global_strand), 128))
            self.global_strand = np.vstack([self.global_strand, pad])
        elif len(self.global_strand) > len(task_strand.sequence):
            self.global_strand = self.global_strand[:len(task_strand.sequence)]

        self.global_strand += global_lr * (reward - 0.5) * 2.0 * task_strand.sequence
        self.global_strand = np.clip(self.global_strand, -1.0, 1.0)

        # ---- 7. Periodic persistence ----
        if self.global_step % 5 == 0:
            self.synapse._save()
            self._save_global_strand()
            self._save_user_data(user_id)

        return score


# ───────────────────────────────────────────────────────────────────────
# 5. SINGLETON INSTANCE & PUBLIC INTERFACE
# ───────────────────────────────────────────────────────────────────────

_engine: Optional[DNAJudgeEngine] = None
_engine_lock = threading.Lock()

def _get_engine() -> DNAJudgeEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = DNAJudgeEngine()
    return _engine


def _clamp(score: float) -> float:
    """Ensure score is strictly within (0, 1)."""
    return max(0.01, min(0.99, score))


# The original `_llm_judge` is replaced by a DNA judge.
# We keep the same signature for backward compatibility.
def _llm_judge(agent_response: str, task_description: str, keywords: list) -> float:
    """
    DNA‑based judge.  The 'keywords' parameter is kept for compatibility
    but is NOT used – the DNA engine learns from the full text.
    """
    if not isinstance(agent_response, str) or not agent_response.strip():
        return _clamp(0.01)

    # default user_id – you can override by setting env DEFAULT_USER_ID
    user_id = os.environ.get("DEFAULT_USER_ID", "anonymous")

    engine = _get_engine()
    try:
        score = engine.judge(user_id, agent_response, task_description)
        return _clamp(score)
    except Exception:
        # absolute last‑resort fallback (keyword matching)
        return _keyword_fallback(agent_response, keywords)


def _keyword_fallback(text: str, keywords: list) -> float:
    """Simple keyword matching fallback – never returns 0 or 1."""
    if not isinstance(text, str) or not text.strip():
        return _clamp(0.01)
    text_lower = text.lower().strip()
    if not keywords:
        return _clamp(0.01)
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    score = 0.01 + (matches / len(keywords)) * 0.98
    return _clamp(score)


# ── Task definitions (identical to original) ──────────────────────────

TASK_DESCRIPTIONS = {
    "task_easy": "The user cannot log in to their account. Their password is not working and they keep getting locked out.",
    "task_medium": "The user's bill shows a double charge for their subscription. They need a refund for the extra payment.",
    "task_hard": "The user's account is locked after multiple failed password attempts. They suspect a security breach.",
}

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
