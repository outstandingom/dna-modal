import os
import json
import math
import numpy as np
import threading
from typing import List, Dict, Optional

# ───────────────────────────────────────────────────────────────────────
# 1. FROZEN CORTEX
# ───────────────────────────────────────────────────────────────────────
class DNACortex:
    def __init__(self):
        rng = np.random.RandomState(42)
        self.embeddings = rng.randn(26, 128).astype(np.float32)
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.map = {c: i for i, c in enumerate(self.letters)}

    def get_dna_sequence(self, text: str) -> np.ndarray:
        idxs = [self.map.get(c.upper(), 0) for c in text if c.isalpha()]
        if not idxs:
            return np.zeros((1, 128))
        return self.embeddings[idxs]

# ───────────────────────────────────────────────────────────────────────
# 2. GLOBAL SYNAPSE (26x26)
# ───────────────────────────────────────────────────────────────────────
class GlobalSynapse:
    def __init__(self, persist_path: str = "dna_synapse.json"):
        self.persist_path = persist_path
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
                json.dump({"weights": self.weights.tolist(), "step": self.step}, f)

    def update(self, letter_probs: np.ndarray, reward: float):
        with self.lock:
            self.step += 1
            base_lr = 0.02
            sine_lr = 0.5 + 0.5 * math.sin(self.step / 15.0)
            lr = base_lr * sine_lr
            hebbian_delta = (reward - 0.5) * 2.0
            co_activation = np.outer(letter_probs, letter_probs)
            self.weights += lr * hebbian_delta * co_activation
            self.weights = np.clip(self.weights, 0.01, 0.99)
            np.fill_diagonal(self.weights, 0.5)
            if self.step % 10 == 0:
                self._save()

# ───────────────────────────────────────────────────────────────────────
# 3. DNA STRAND
# ───────────────────────────────────────────────────────────────────────
class DNAStrand:
    def __init__(self, initial_sequence: np.ndarray):
        self.sequence = initial_sequence.copy()
        self.length = self.sequence.shape[0]
        self.positional_sine = np.array([math.sin(i / 3.0) for i in range(self.length)]).reshape(-1, 1)

    def fire(self) -> np.ndarray:
        return self.sequence * (0.5 + 0.5 * self.positional_sine)

    def hebbian_update(self, reward: float, step: int, lr_base: float = 0.03):
        temporal_sine = 0.5 + 0.5 * math.sin(step / 20.0)
        lr = lr_base * temporal_sine
        hebbian_delta = (reward - 0.5) * 2.0
        activation = self.fire()
        self.sequence += lr * hebbian_delta * activation * (0.5 + 0.5 * self.positional_sine)
        self.sequence = np.clip(self.sequence, -1.0, 1.0)

# ───────────────────────────────────────────────────────────────────────
# 4. MAIN DNA JUDGE ENGINE
# ───────────────────────────────────────────────────────────────────────
class DNAJudgeEngine:
    def __init__(self):
        self.cortex = DNACortex()
        self.synapse = GlobalSynapse()
        self.global_step = 0
        self.global_strand = self.cortex.get_dna_sequence("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG")
        self._load_global_strand()
        self.user_cache = {}
        self.lock = threading.RLock()

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
                delta = np.zeros(26, dtype=np.float32)
                strand = self.cortex.get_dna_sequence("HELLOUSER")
                # load from disk if exists...
                os.makedirs("user_dnas", exist_ok=True)
                strand_path = f"user_dnas/{user_id}_strand.npy"
                delta_path = f"user_dnas/{user_id}_delta.npy"
                if os.path.exists(strand_path):
                    try: strand = np.load(strand_path)
                    except: pass
                if os.path.exists(delta_path):
                    try: delta = np.load(delta_path)
                    except: pass
                self.user_cache[user_id] = {"delta": delta, "strand": strand}
            return self.user_cache[user_id]

    def _save_user_data(self, user_id: str):
        with self.lock:
            data = self.user_cache.get(user_id)
            if data:
                os.makedirs("user_dnas", exist_ok=True)
                np.save(f"user_dnas/{user_id}_strand.npy", data["strand"])
                np.save(f"user_dnas/{user_id}_delta.npy", data["delta"])

    def judge(self, user_id: str, agent_response: str, task_description: str) -> float:
        self.global_step += 1

        task_strand = DNAStrand(self.cortex.get_dna_sequence(task_description))
        resp_strand = DNAStrand(self.cortex.get_dna_sequence(agent_response))

        # Letter probabilities from task
        task_concept = np.mean(task_strand.fire(), axis=0, keepdims=True)
        raw_act = self.cortex.embeddings @ task_concept.T
        raw_act = np.squeeze(raw_act)
        exp_act = np.exp(raw_act * 2.0)
        letter_probs = exp_act / (np.sum(exp_act) + 1e-8)

        # Synapse influence
        synapse_influence = float(letter_probs @ self.synapse.weights @ letter_probs.T)

        # Similarity
        task_vec = np.mean(task_strand.fire(), axis=0, keepdims=True)
        resp_vec = np.mean(resp_strand.fire(), axis=0, keepdims=True)
        task_vec /= (np.linalg.norm(task_vec) + 1e-8)
        resp_vec /= (np.linalg.norm(resp_vec) + 1e-8)
        similarity = float(np.dot(task_vec, resp_vec.T)[0, 0])

        # Final score
        raw_score = 0.65 * ((similarity + 1.0) / 2.0) + 0.35 * synapse_influence
        score = max(0.01, min(0.99, raw_score))
        reward = score

        # === LEARNING ===
        self.synapse.update(letter_probs, reward)
        task_strand.hebbian_update(reward, self.global_step, 0.03)
        resp_strand.hebbian_update(reward, self.global_step + 100, 0.04)

        user_data = self._get_user_data(user_id)
        delta_lr = 0.06 * (0.5 + 0.5 * math.sin(self.global_step / 10.0))
        user_data["delta"] += delta_lr * (reward - 0.5) * 2.0 * letter_probs
        user_data["delta"] = np.clip(user_data["delta"], -0.4, 0.4)

        # Global strand update (fixed padding)
        if len(self.global_strand) != len(task_strand.sequence):
            min_len = min(len(self.global_strand), len(task_strand.sequence))
            self.global_strand = self.global_strand[:min_len]

        global_lr = 0.001 * (0.5 + 0.5 * math.sin(self.global_step / 50.0))
        self.global_strand += global_lr * (reward - 0.5) * 2.0 * task_strand.sequence[:len(self.global_strand)]
        self.global_strand = np.clip(self.global_strand, -1.0, 1.0)

        if self.global_step % 5 == 0:
            self.synapse._save()
            self._save_global_strand()
            self._save_user_data(user_id)

        return score

# ───────────────────────────────────────────────────────────────────────
# Singleton + Public API (exact same as original)
# ───────────────────────────────────────────────────────────────────────
_engine = None
_engine_lock = threading.Lock()

def _get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = DNAJudgeEngine()
    return _engine

def _clamp(score: float) -> float:
    return max(0.01, min(0.99, score))

def _keyword_fallback(text: str, keywords: list) -> float:
    if not isinstance(text, str) or not text.strip() or not keywords:
        return _clamp(0.01)
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return _clamp(0.01 + (matches / len(keywords)) * 0.98)

def _llm_judge(agent_response: str, task_description: str, keywords: list) -> float:
    if not isinstance(agent_response, str) or not agent_response.strip():
        return _clamp(0.01)
    user_id = os.environ.get("DEFAULT_USER_ID", "anonymous")
    engine = _get_engine()
    try:
        return engine.judge(user_id, agent_response, task_description)
    except Exception as e:
        print(f"[DNA Judge fallback] {e}")
        return _keyword_fallback(agent_response, keywords)

# Task definitions (identical to original)
TASK_DESCRIPTIONS = { ... }  # copy from your original
TASK_KEYWORDS = { ... }

def task_easy(input_text: str) -> float:
    return _llm_judge(input_text, TASK_DESCRIPTIONS["task_easy"], TASK_KEYWORDS["task_easy"])

# ... same for task_medium, task_hard

TASKS = ["task_easy", "task_medium", "task_hard"]
GRADERS = {"task_easy": task_easy, "task_medium": task_medium, "task_hard": task_hard}
