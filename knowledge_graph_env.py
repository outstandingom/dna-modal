import os
import asyncio
import threading
import time
import random
import numpy as np
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple, Optional, Set, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ============================================================
# GRADERS — Pure Python, no heavy deps. Loaded first so
# /grade ALWAYS works even if the DNA env is still booting.
# ============================================================

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

TASKS: List[str] = ["task_easy", "task_medium", "task_hard"]
GRADERS: Dict[str, Any] = {
    "task_easy":   task_easy,
    "task_medium": task_medium,
    "task_hard":   task_hard,
}

# ============================================================
# Configuration
# ============================================================
DIMS = 16
ALPHABET = [chr(ord('A') + i) for i in range(26)]
POSITION_OFFSET = 0.1
LR_CONCEPT = 0.01
GRAD_CLIP = 1.0
MAX_CONCEPTS = 10000
BATCH_SIZE = 32
TRAIN_INTERVAL_SEC = 10

os.makedirs("./brain_data", exist_ok=True)

USE_VALIDATOR_PROXY = os.environ.get("API_BASE_URL") is not None
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

STOP_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "have", "from", "they", "been",
    "said", "each", "which", "their", "will", "other", "about", "many", "then",
    "them", "these", "some", "would", "make", "like", "into", "time", "very",
    "when", "come", "could", "than", "its", "also", "back", "after", "two",
    "how", "what", "where", "who", "why", "this", "that", "with",
}

# ============================================================
# DynamicOntology
# ============================================================
class DynamicOntology:
    def __init__(self):
        self.concept_to_features: Dict[str, List[str]] = {}
        self.feature_to_concepts: Dict[str, List[str]] = defaultdict(list)
        self._openai_client = None

    def _get_client(self):
        if self._openai_client is None:
            import openai
            self._openai_client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        return self._openai_client

    async def get_features_llm(self, concept: str) -> List[str]:
        if not API_KEY:
            return [concept]
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Extract physical and semantic features. Return a comma-separated list of up to 5 features."},
                    {"role": "user", "content": f"Features of '{concept}':"}
                ],
                temperature=0.3, max_tokens=100,
            )
            return [f.strip().lower() for f in resp.choices[0].message.content.split(",")][:5]
        except Exception:
            return [concept]

    def get_features(self, concept: str) -> List[str]:
        return self.concept_to_features.get(concept.lower(), [concept.lower()])

    def serialize(self) -> dict:
        return self.concept_to_features

    def restore(self, data: dict):
        self.concept_to_features = data
        self.feature_to_concepts = defaultdict(list)
        for concept, feats in data.items():
            for f in feats:
                self.feature_to_concepts[f].append(concept)

# ============================================================
# Feature Registry
# ============================================================
class FeatureRegistry:
    def __init__(self, ontology: DynamicOntology):
        self.ontology = ontology
        self.feature_to_id: Dict[str, int] = {}
        self.id_to_feature: Dict[int, str] = {}
        self.feature_vectors: Dict[int, np.ndarray] = {}
        self.next_id = 0

    def register(self, feature_name: str) -> int:
        name = feature_name.lower()
        if name in self.feature_to_id:
            return self.feature_to_id[name]
        fid = self.next_id
        self.next_id += 1
        self.feature_to_id[name] = fid
        self.id_to_feature[fid] = name
        self.feature_vectors[fid] = np.random.uniform(-1, 1, DIMS).astype(np.float32)
        return fid

    def update_vector(self, fid: int, delta: np.ndarray):
        self.feature_vectors[fid] += np.clip(delta, -GRAD_CLIP, GRAD_CLIP)

    def feature_to_letters(self, fid: int, length: int = 5) -> List[str]:
        vec = self.feature_vectors[fid]
        probs = np.exp(vec[:length])
        probs /= probs.sum()
        return [ALPHABET[int(np.argmax(probs)) % 26]] * length

# ============================================================
# Letter Vectors
# ============================================================
class LetterVectors:
    def __init__(self):
        self.vec = {ch: np.random.uniform(-1, 1, DIMS).astype(np.float32) for ch in ALPHABET}

    def get(self, ch: str) -> np.ndarray:
        return self.vec[ch]

    def update(self, ch: str, delta: np.ndarray):
        self.vec[ch] += np.clip(delta, -GRAD_CLIP, GRAD_CLIP)

# ============================================================
# DNA Concept
# ============================================================
class DNAConcept:
    def __init__(self, name: str, physical_features: List[int], semantic_features: List[int],
                 feature_registry: FeatureRegistry, letter_vec: LetterVectors):
        self.name = name
        self.physical_features = physical_features
        self.semantic_features = semantic_features
        self.feature_registry = feature_registry
        self.letter_vec = letter_vec
        self.vector = np.zeros(DIMS, dtype=np.float32)
        self._update_vector()

    def _encode_feature(self, fid: int, start_pos: int) -> np.ndarray:
        letters = self.feature_registry.feature_to_letters(fid, length=5)
        vec = np.zeros(DIMS, dtype=np.float32)
        for i, ch in enumerate(letters):
            base = self.letter_vec.get(ch)
            vec += np.sin(base + (start_pos + i) * POSITION_OFFSET)
        return vec

    def _update_vector(self):
        vec = np.zeros(DIMS, dtype=np.float32)
        pos = 0
        for fid in self.physical_features + self.semantic_features:
            vec += self._encode_feature(fid, pos)
            pos += 5
        norm = np.linalg.norm(vec)
        self.vector = vec / norm if norm > 0 else vec

    def move_towards(self, other: 'DNAConcept', lr: float = LR_CONCEPT):
        diff = other.vector - self.vector
        self.vector += lr * diff
        other.vector -= lr * diff
        for v in (self.vector, other.vector):
            n = np.linalg.norm(v)
            if n > 0:
                v /= n

    def cosine_similarity(self, other: 'DNAConcept') -> float:
        return float(np.dot(self.vector, other.vector) /
                     (np.linalg.norm(self.vector) * np.linalg.norm(other.vector) + 1e-8))

    def serialize(self) -> dict:
        return {"name": self.name, "physical_features": self.physical_features,
                "semantic_features": self.semantic_features, "vector": self.vector.tolist()}

    @classmethod
    def from_serialized(cls, data: dict, fr: FeatureRegistry, lv: LetterVectors):
        obj = cls(data["name"], data["physical_features"], data["semantic_features"], fr, lv)
        obj.vector = np.array(data["vector"], dtype=np.float32)
        return obj

# ============================================================
# Concept Memory — FAISS optional, falls back to numpy search
# ============================================================
class ConceptMemory:
    def __init__(self, feature_registry: FeatureRegistry, letter_vec: LetterVectors,
                 max_concepts: int = MAX_CONCEPTS):
        self.feature_registry = feature_registry
        self.letter_vec = letter_vec
        self.concepts: Dict[str, DNAConcept] = {}
        self.relationships: Dict[str, Set[str]] = defaultdict(set)
        self.max_concepts = max_concepts
        self._faiss_available = False
        self.index = None
        self.id_to_name: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
        self.next_id = 0
        # Try loading FAISS once at startup — don't block if unavailable
        try:
            import faiss  # noqa: F401
            self._faiss_available = True
        except Exception:
            self._faiss_available = False

    def _ensure_index(self):
        if not self._faiss_available:
            return
        if self.index is None:
            import faiss
            self.index = faiss.IndexFlatIP(DIMS)

    def _rebuild_index(self):
        if not self._faiss_available or not self.concepts:
            return
        import faiss
        vectors = [c.vector for c in self.concepts.values()]
        names = list(self.concepts.keys())
        vecs = np.vstack(vectors).astype(np.float32)
        self.index = faiss.IndexFlatIP(DIMS)
        self.index.add(vecs)
        self.id_to_name = {i: n for i, n in enumerate(names)}
        self.name_to_id = {n: i for i, n in enumerate(names)}
        self.next_id = len(self.concepts)

    def register(self, name: str, physical_features: List[int], semantic_features: List[int]) -> DNAConcept:
        name_low = name.lower()
        if name_low in self.concepts:
            return self.concepts[name_low]
        concept = DNAConcept(name_low, physical_features, semantic_features,
                             self.feature_registry, self.letter_vec)
        self.concepts[name_low] = concept
        if self._faiss_available:
            self._ensure_index()
            if self.index is not None:
                self.index.add(concept.vector.reshape(1, -1))
                self.id_to_name[self.next_id] = name_low
                self.name_to_id[name_low] = self.next_id
                self.next_id += 1
        self._prune()
        return concept

    def add_relationship(self, a: str, b: str):
        a_low, b_low = a.lower(), b.lower()
        if a_low in self.concepts and b_low in self.concepts:
            self.relationships[a_low].add(b_low)
            self.relationships[b_low].add(a_low)
            self.concepts[a_low].move_towards(self.concepts[b_low])

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.concepts:
            return []
        if self._faiss_available and self.index is not None and self.index.ntotal > 0:
            import faiss  # noqa: F401
            q = query_vector.reshape(1, -1).astype(np.float32)
            distances, indices = self.index.search(q, min(top_k, self.index.ntotal))
            return [(self.id_to_name[idx], float(distances[0][i]))
                    for i, idx in enumerate(indices[0]) if idx != -1 and idx in self.id_to_name]
        # Numpy fallback
        names = list(self.concepts.keys())
        vecs = np.vstack([self.concepts[n].vector for n in names])
        q = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        scores = vecs @ q
        top = np.argsort(scores)[::-1][:top_k]
        return [(names[i], float(scores[i])) for i in top]

    async def extract_and_link(self, text: str, ontology: DynamicOntology) -> List[str]:
        words = [w for w in text.lower().split() if len(w) > 3 and w not in STOP_WORDS]
        unique = list(set(words))[:15]
        for kw in unique:
            features = await ontology.get_features_llm(kw) if API_KEY else ontology.get_features(kw)
            fids = [self.feature_registry.register(f) for f in features]
            self.register(kw, fids, fids)
        for i in range(len(unique)):
            for j in range(i + 1, min(i + 4, len(unique))):
                self.add_relationship(unique[i], unique[j])
        return unique

    def _prune(self):
        if len(self.concepts) > self.max_concepts:
            to_remove = sorted(self.concepts, key=lambda n: len(self.relationships.get(n, [])))
            for name in to_remove[:len(self.concepts) - self.max_concepts]:
                self.concepts.pop(name, None)
                self.relationships.pop(name, None)
            self._rebuild_index()

# ============================================================
# Reasoning Engine
# ============================================================
class ReasoningEngine:
    def __init__(self, cm: 'ConceptMemory'):
        self.cm = cm

    def multi_hop(self, start: str, max_hops: int = 3, decay: float = 0.7) -> Dict[str, float]:
        if start not in self.cm.relationships:
            return {}
        activation: Dict[str, float] = {start: 1.0}
        for _ in range(max_hops):
            new: Dict[str, float] = {}
            for node, score in activation.items():
                for nb in self.cm.relationships.get(node, set()):
                    weight = (self.cm.concepts[node].cosine_similarity(self.cm.concepts[nb])
                              if node in self.cm.concepts and nb in self.cm.concepts else 1.0)
                    new[nb] = new.get(nb, 0) + score * weight * decay
            for k, v in new.items():
                activation[k] = activation.get(k, 0) + v
        if activation:
            mx = max(activation.values())
            activation = {k: v / mx for k, v in activation.items()}
        return activation

# ============================================================
# Background Trainer
# ============================================================
class ContinuousTrainer:
    def __init__(self, cm: ConceptMemory, interval: int = TRAIN_INTERVAL_SEC):
        self.cm = cm
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _loop(self):
        while self.running:
            time.sleep(self.interval)
            rels = [(a, b) for a, s in self.cm.relationships.items() for b in s]
            if not rels:
                continue
            batch = random.sample(rels, min(BATCH_SIZE, len(rels)))
            for a, b in batch:
                if a in self.cm.concepts and b in self.cm.concepts:
                    self.cm.concepts[a].move_towards(self.cm.concepts[b])
            self.cm._rebuild_index()

# ============================================================
# KnowledgeGraphEnv
# ============================================================
class KnowledgeGraphEnv:
    def __init__(self, start_trainer: bool = True):
        self.ontology = DynamicOntology()
        self.feature_registry = FeatureRegistry(self.ontology)
        self.letter_vec = LetterVectors()
        self.concept_memory = ConceptMemory(self.feature_registry, self.letter_vec)
        self.reasoning = ReasoningEngine(self.concept_memory)
        self._seed()
        self.trainer: Optional[ContinuousTrainer] = None
        if start_trainer:
            self.trainer = ContinuousTrainer(self.concept_memory)
            self.trainer.start()
        self.current_task: Optional[dict] = None
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False

    def _seed(self):
        seed_data = [
            ("login issue",      ["authentication", "password", "access"]),
            ("billing",          ["payment", "invoice", "refund"]),
            ("slow performance", ["latency", "response time", "optimization"]),
            ("crash",            ["bug", "error", "unstable"]),
            ("feature request",  ["enhancement", "new functionality", "suggestion"]),
            ("account locked",   ["security", "blocked", "verification"]),
        ]
        for concept, features in seed_data:
            fids = [self.feature_registry.register(f) for f in features]
            self.concept_memory.register(concept, fids, fids)
        self.concept_memory.add_relationship("login issue", "account locked")
        self.concept_memory.add_relationship("billing", "refund")
        self.concept_memory.add_relationship("slow performance", "crash")

    def reset(self) -> str:
        self.current_task = self._gen_task()
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(
                    self.concept_memory.extract_and_link(self.current_task["input"], self.ontology))
        except Exception:
            pass
        return self.current_task["input"]

    def step(self, action: str) -> Tuple[str, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode done. Call reset() first.")
        step_name = ["identify", "relate", "answer"][self.current_step]
        if step_name == "identify":
            reward = self._grade_match(action, self.current_task["expected_concept"])
            if reward >= 0.3:
                self.current_step = 1
        elif step_name == "relate":
            reward = self._grade_match(action, self.current_task["expected_relation"])
            if reward >= 0.3:
                self.current_step = 2
        else:
            reward = self._grade_match(action, self.current_task["expected_answer"])
            self.done = True
            self.current_step = 3
        self.episode_reward += reward
        return (self.current_task["input"] if not self.done else "",
                reward, self.done,
                {"step": step_name, "step_reward": reward, "total_reward": self.episode_reward})

    def state(self) -> dict:
        if not self.current_task:
            return {}
        names = ["identify", "relate", "answer"]
        return {"task": self.current_task, "step": self.current_step,
                "step_name": names[self.current_step] if self.current_step < 3 else "done"}

    def _gen_task(self) -> dict:
        concepts = list(self.concept_memory.concepts.keys())
        if not concepts:
            return {"type": "easy", "input": "My phone won't turn on",
                    "expected_concept": "hardware", "expected_relation": "battery",
                    "expected_answer": "check power"}
        base = random.choice(concepts)
        related = list(self.concept_memory.relationships.get(base, set()))
        if not related:
            similar = self.concept_memory.search(self.concept_memory.concepts[base].vector, top_k=3)
            related = [r[0] for r in similar if r[0] != base]
        rel = related[0] if related else "related issue"
        hops = self.reasoning.multi_hop(base, max_hops=2)
        answers = [c for c in hops if c not in (base, rel)]
        answer = answers[0] if answers else random.choice(concepts)
        difficulty = random.choice(["easy", "medium", "hard"])
        templates = {
            "easy":   f"I'm having trouble with {base}.",
            "medium": f"User reports {base} persists after restart.",
            "hard":   f"Critical: {base} causing system failure.",
        }
        return {"type": difficulty, "input": templates[difficulty],
                "expected_concept": base, "expected_relation": rel, "expected_answer": answer}

    def _grade_match(self, action: str, expected: str) -> float:
        a, e = action.lower().strip(), expected.lower()
        if a == e: return 0.95
        if e in a or a in e: return 0.75
        if any(w in a for w in e.split()): return 0.35
        return 0.05

    def close(self):
        if self.trainer:
            self.trainer.stop()

# ============================================================
# FastAPI — lifespan boots env in background thread so
# /health and /grade respond IMMEDIATELY on startup.
# ============================================================

_api_env: Optional[KnowledgeGraphEnv] = None
_env_ready = False


def _boot_env():
    """Run in a daemon thread so the server isn't blocked during startup."""
    global _api_env, _env_ready
    try:
        _api_env = KnowledgeGraphEnv(start_trainer=True)
        _env_ready = True
    except Exception as e:
        print(f"[WARN] KnowledgeGraphEnv boot failed: {e}", flush=True)
        _env_ready = True  # Mark ready so we don't block forever


@asynccontextmanager
async def lifespan(app: FastAPI):
    boot_thread = threading.Thread(target=_boot_env, daemon=True)
    boot_thread.start()
    yield
    if _api_env is not None:
        _api_env.close()


app = FastAPI(
    title="Knowledge Graph Environment",
    description="Self-evolving DNA-inspired knowledge graph for customer support reasoning.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic models ──────────────────────────────────────────

class ResetResponse(BaseModel):
    observation: str

class StepRequest(BaseModel):
    action: str

class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: dict

class StateResponse(BaseModel):
    state: dict

class TaskResponse(BaseModel):
    tasks: List[str]

class GradeRequest(BaseModel):
    task_id: str
    input_text: str

class GradeResponse(BaseModel):
    score: float


# ── Endpoints ────────────────────────────────────────────────

@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/tasks", response_model=TaskResponse)
async def tasks_endpoint():
    return TaskResponse(tasks=TASKS)


@app.post("/grade", response_model=GradeResponse)
async def grade_endpoint(req: GradeRequest):
    """Grade a task — uses pure-Python graders, always available instantly."""
    if req.task_id not in GRADERS:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{req.task_id}' not found. Available: {list(GRADERS.keys())}"
        )
    score = GRADERS[req.task_id](req.input_text)
    return GradeResponse(score=score)


@app.post("/reset", response_model=ResetResponse)
async def reset_endpoint():
    if _api_env is None:
        return ResetResponse(observation="Environment initializing, please retry in a moment.")
    return ResetResponse(observation=_api_env.reset())


@app.post("/step", response_model=StepResponse)
async def step_endpoint(req: StepRequest):
    if _api_env is None:
        raise HTTPException(status_code=503, detail="Environment still initializing.")
    obs, reward, done, info = _api_env.step(req.action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponse)
async def state_endpoint():
    if _api_env is None:
        raise HTTPException(status_code=503, detail="Environment still initializing.")
    return StateResponse(state=_api_env.state())
