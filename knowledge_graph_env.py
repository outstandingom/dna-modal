import os
import json
import asyncio
import threading
import time
import math
import pickle
import random
import numpy as np
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai

from graders import task_easy, task_medium, task_hard, TASKS, GRADERS
from skill_adapter import get_adapter
from projection import DimensionProjector, ProjectionCache
from dynamic_knowledge_loader import DynamicKnowledgeLoader, ConceptData

# ============================================================
# Configuration
# ============================================================
DIMS = 128
ALPHABET = [chr(ord('A') + i) for i in range(26)]
POSITION_OFFSET = 0.1

ESSENCE_DIMS   = slice(0, 4)
IDENTITY_DIMS  = slice(4, 12)
TEMPORAL_DIMS  = slice(12, 16)

class PriorityLayer(Enum):
    ATOMIC    = 0
    CLUSTER   = 1
    DOMAIN    = 2
    UNIVERSAL = 3

LR_ATOMIC    = 0.01
LR_CLUSTER   = 0.005
LR_DOMAIN    = 0.001
LR_UNIVERSAL = 0.0001
LR_LETTER      = LR_ATOMIC
LR_FEATURE_VEC = LR_CLUSTER
LR_CONCEPT     = LR_DOMAIN

GRAD_CLIP             = 1.0
MAX_CONCEPTS          = 10000
BATCH_SIZE            = 32
TRAIN_INTERVAL_SEC    = 10
PERSIST_DIR           = "./brain_data"
GLOBAL_UPDATE_FREQUENCY = 10

# ── Priority 2: Working Memory cap ──────────────────────────
WORKING_MEMORY_SIZE   = 500
# ── Priority 4: DNA Attention threshold ─────────────────────
ATTENTION_THRESHOLD   = 0.3
# ── Priority 6: Episodic Memory ─────────────────────────────
MAX_EPISODES          = 1000
MIN_EPISODE_REWARD    = 0.45
# ── Priority 11: Confidence threshold for deeper search ─────
LOW_CONFIDENCE_THRESHOLD = 0.4

os.makedirs(PERSIST_DIR, exist_ok=True)
load_dotenv()

# ============================================================
# BYOK: Multi-Provider Registry
# ============================================================
PROVIDER_REGISTRY = {
    "local_graph": {"name": "Independent Graph Mode (No LLM)", "base_url": "", "default_model": "native-12-dim-vectors", "free_tier": True, "get_key_url": ""},
    "groq":        {"name": "Groq",           "base_url": "https://api.groq.com/openai/v1",                                     "default_model": "llama-3.3-70b-versatile", "free_tier": True,  "get_key_url": "https://console.groq.com/keys"},
    "gemini":      {"name": "Google Gemini",  "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",             "default_model": "gemini-2.0-flash",        "free_tier": True,  "get_key_url": "https://aistudio.google.com/apikey"},
    "openai":      {"name": "OpenAI",         "base_url": "https://api.openai.com/v1",                                           "default_model": "gpt-4o-mini",             "free_tier": False, "get_key_url": "https://platform.openai.com/api-keys"},
    "deepseek":    {"name": "DeepSeek",       "base_url": "https://api.deepseek.com/v1",                                         "default_model": "deepseek-chat",           "free_tier": True,  "get_key_url": "https://platform.deepseek.com/api_keys"},
    "huggingface": {"name": "Hugging Face",   "base_url": "https://router.huggingface.co/v1",                                    "default_model": "Qwen/Qwen2.5-72B-Instruct","free_tier": True, "get_key_url": "https://huggingface.co/settings/tokens"},
    "custom":      {"name": "Custom / Self-Hosted", "base_url": "", "default_model": "", "free_tier": None, "get_key_url": ""},
}

STOP_WORDS: set = set()

class RelationshipColor(Enum):
    IS_A       = "IS_A"
    HAS_FEATURE= "HAS"
    GROWN_IN   = "GROWN_IN"
    LOCATION   = "LOCATION"
    RELATED_TO = "RELATED"
    CAUSES     = "CAUSES"
    PART_OF    = "PART_OF"
    OPERATOR   = "OPERATOR"
    CONDITION  = "CONDITION"

# ============================================================
# RelationshipData
# ============================================================
@dataclass
class RelationshipData:
    weight: float = 1.0
    color: str    = "RELATED"
    created_at:       float = field(default_factory=time.time)
    last_accessed:    float = field(default_factory=time.time)
    co_occurrence_count: int = 1

    def to_dict(self) -> dict:
        return {"weight": self.weight, "color": self.color,
                "created_at": self.created_at, "last_accessed": self.last_accessed,
                "co_occurrence_count": self.co_occurrence_count}

    @classmethod
    def from_dict(cls, data: dict) -> 'RelationshipData':
        return cls(weight=data.get("weight",1.0), color=data.get("color","RELATED"),
                   created_at=data.get("created_at",time.time()),
                   last_accessed=data.get("last_accessed",time.time()),
                   co_occurrence_count=data.get("co_occurrence_count",1))

# ============================================================
# ░░░░  PRIORITY 6: Episode dataclass  ░░░░
# ============================================================
@dataclass
class Episode:
    query:              str
    query_vector:       List[float]
    reasoning_path:     List[dict]
    activated_concepts: List[str]
    answer_concepts:    List[str]
    reward:             float
    created_at:         float = field(default_factory=time.time)
    use_count:          int   = 0

    def to_dict(self) -> dict:
        return {"query": self.query, "query_vector": self.query_vector,
                "reasoning_path": self.reasoning_path,
                "activated_concepts": self.activated_concepts,
                "answer_concepts": self.answer_concepts,
                "reward": self.reward, "created_at": self.created_at,
                "use_count": self.use_count}

    @classmethod
    def from_dict(cls, d: dict) -> 'Episode':
        return cls(query=d["query"], query_vector=d["query_vector"],
                   reasoning_path=d["reasoning_path"],
                   activated_concepts=d["activated_concepts"],
                   answer_concepts=d["answer_concepts"],
                   reward=d["reward"], created_at=d.get("created_at", time.time()),
                   use_count=d.get("use_count", 0))

# ============================================================
# DynamicOntology
# ============================================================
class DynamicOntology:
    def __init__(self):
        self.concept_to_features: Dict[str, List[str]] = {}
        self.feature_to_concepts: Dict[str, List[str]] = defaultdict(list)
        self.concept_domains:     Dict[str, str]        = {}
        self.llm_enabled = True

    async def add_concept(self, concept: str, context: str = ""):
        concept_low = concept.lower()
        if concept_low in self.concept_to_features:
            return
        features = self.get_features(concept_low)
        domain   = self.get_domain(concept_low)
        self.concept_to_features[concept_low] = features
        self.concept_domains[concept_low]     = domain
        for f in features:
            self.feature_to_concepts[f].append(concept_low)

    def get_features(self, concept: str, context: str = "") -> List[str]:
        concept_low = concept.lower()
        if concept_low not in self.concept_to_features:
            return [concept_low]
        return self.concept_to_features[concept_low]

    def get_domain(self, concept: str) -> str:
        return self.concept_domains.get(concept.lower(), "general")

    def serialize(self) -> dict:
        return {"concept_to_features": self.concept_to_features,
                "concept_domains": self.concept_domains}

    def restore(self, data: dict):
        self.concept_to_features = data.get("concept_to_features", {})
        self.concept_domains     = data.get("concept_domains", {})
        self.feature_to_concepts = defaultdict(list)
        for concept, feats in self.concept_to_features.items():
            for f in feats:
                self.feature_to_concepts[f].append(concept)

# ============================================================
# FeatureRegistry
# ============================================================
class FeatureRegistry:
    def __init__(self, ontology: DynamicOntology, dims: int = 128):
        self.dims = dims
        self.ontology = ontology
        self.feature_to_id:     Dict[str, int]         = {}
        self.id_to_feature:     Dict[int, str]         = {}
        self.feature_vectors:   Dict[int, np.ndarray]  = {}
        self.feature_importance:Dict[int, float]       = {}
        self.next_id = 0
        all_features: set = set()
        for feats in ontology.concept_to_features.values():
            all_features.update(feats)
        for feat in all_features:
            self.register(feat)

    def register(self, feature_name: str, importance: float = 1.0) -> int:
        name = feature_name.lower()
        if name in self.feature_to_id:
            fid = self.feature_to_id[name]
            self.feature_importance[fid] = min(10.0, self.feature_importance.get(fid, 1.0) + 0.1)
            return fid
        fid = self.next_id; self.next_id += 1
        self.feature_to_id[name]    = fid
        self.id_to_feature[fid]     = name
        self.feature_vectors[fid]   = np.random.uniform(-1, 1, self.dims).astype(np.float32)
        self.feature_importance[fid]= importance
        return fid

    def get_vector(self, fid: int) -> np.ndarray:
        return self.feature_vectors[fid]

    def get_importance(self, fid: int) -> float:
        return self.feature_importance.get(fid, 1.0)

    def update_vector(self, fid: int, delta: np.ndarray,
                      layer: PriorityLayer = PriorityLayer.CLUSTER):
        delta = np.clip(delta, -GRAD_CLIP, GRAD_CLIP)
        lr_map = {PriorityLayer.ATOMIC: LR_ATOMIC, PriorityLayer.CLUSTER: LR_CLUSTER,
                  PriorityLayer.DOMAIN: LR_DOMAIN,  PriorityLayer.UNIVERSAL: LR_UNIVERSAL}
        self.feature_vectors[fid] += delta * lr_map[layer]

    def feature_to_letters(self, fid: int, length: int = 5) -> List[str]:
        vec   = self.feature_vectors[fid]
        probs = np.exp(vec[ESSENCE_DIMS][:min(length, 4)])
        probs /= np.sum(probs) + 1e-8
        idx   = np.argmax(probs) % 26
        return [ALPHABET[idx]] * length

    def serialize(self) -> dict:
        return {"feature_to_id": self.feature_to_id,
                "id_to_feature": {str(k): v for k, v in self.id_to_feature.items()},
                "feature_vectors": {str(k): v.tolist() for k, v in self.feature_vectors.items()},
                "feature_importance": {str(k): v for k, v in self.feature_importance.items()},
                "next_id": self.next_id,
                "ontology": self.ontology.serialize()}

    def restore(self, data: dict):
        self.feature_to_id   = data["feature_to_id"]
        self.id_to_feature   = {int(k): v for k, v in data["id_to_feature"].items()}
        self.feature_vectors = {int(k): np.array(v, dtype=np.float32) for k, v in data["feature_vectors"].items()}
        self.feature_importance = {int(k): float(v) for k, v in data.get("feature_importance", {}).items()}
        self.next_id         = data["next_id"]
        self.ontology.restore(data["ontology"])

# ============================================================
# LetterVectors — FROZEN
# ============================================================
class LetterVectors:
    def __init__(self, dims: int = 128):
        self.dims = dims
        self.vec  = {ch: np.random.uniform(-1, 1, self.dims).astype(np.float32) for ch in ALPHABET}
        self.letter_importance = {ch: 1.0 for ch in ALPHABET}

    def get(self, letter: str) -> np.ndarray:
        return self.vec[letter]

    def get_importance(self, letter: str) -> float:
        return self.letter_importance.get(letter, 1.0)

    def update(self, letter: str, delta: np.ndarray, layer: PriorityLayer = PriorityLayer.ATOMIC):
        pass  # Letters are the physical laws — they never change.

    def serialize(self) -> dict:
        return {"vectors": {ch: self.vec[ch].tolist() for ch in ALPHABET},
                "importance": self.letter_importance}

    def restore(self, data: dict):
        if "vectors" in data:
            for ch, arr in data["vectors"].items():
                self.vec[ch] = np.array(arr, dtype=np.float32)
        else:
            for ch, arr in data.items():
                if ch in ALPHABET:
                    self.vec[ch] = np.array(arr, dtype=np.float32)
        if "importance" in data:
            self.letter_importance = data["importance"]

# ============================================================
# DNAConcept
# ============================================================
class DNAConcept:
    def __init__(self, name: str, physical_features: List[int], semantic_features: List[int],
                 feature_registry: 'FeatureRegistry', letter_vec: 'LetterVectors',
                 importance: float = 1.0, domain: str = "general",
                 numeric_value: Optional[float] = None, dims: int = 128):
        self.dims              = dims
        self.name              = name
        self.physical_features = physical_features
        self.semantic_features = semantic_features
        self.feature_registry  = feature_registry
        self.letter_vec        = letter_vec
        self.importance        = importance
        self.domain            = domain
        self.cluster_id: Optional[int] = None
        self.pending_deltas:   List[np.ndarray] = []
        self.numeric_value     = numeric_value
        self.vector: Optional[np.ndarray] = None
        self._update_vector()

    def _encode_feature(self, fid: int, start_pos: int) -> np.ndarray:
        letters = self.feature_registry.feature_to_letters(fid, length=5)
        vec = np.zeros(self.dims, dtype=np.float32)
        for i, ch in enumerate(letters):
            base = self.letter_vec.get(ch)
            imp  = self.letter_vec.get_importance(ch)
            vec += np.sin(base + (start_pos + i) * POSITION_OFFSET) * imp
        return vec

    def _update_vector(self):
        vec = np.zeros(self.dims, dtype=np.float32)
        pos = 0
        for fid in self.physical_features:
            vec += self._encode_feature(fid, pos); pos += 5
        for fid in self.semantic_features:
            vec += self._encode_feature(fid, pos); pos += 5
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        self.vector = vec
        self.apply_scale()

    def apply_scale(self):
        if self.vector is None: return
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = (self.vector / norm) * np.log1p(self.importance)

    def move_towards(self, other: 'DNAConcept', lr: float = LR_CONCEPT,
                     weight: float = 1.0, color: str = "RELATED"):
        if self.vector is None or other.vector is None: return
        pull       = (other.importance / (self.importance + 1e-8)) * lr * weight
        other_pull = (self.importance  / (other.importance + 1e-8)) * lr * weight
        color_mult = {"IS_A":1.2,"HAS":0.8,"GROWN_IN":0.9,"LOCATION":0.7,
                      "CAUSES":1.0,"PART_OF":0.85,"RELATED":0.5,
                      "OPERATOR":1.5,"CONDITION":1.3}.get(color, 0.5)
        pull       *= color_mult
        other_pull *= color_mult
        diff        = other.vector - self.vector
        self.vector       += pull       * diff
        other.vector      -= other_pull * diff
        self.vector  /= (np.linalg.norm(self.vector)  + 1e-8)
        other.vector /= (np.linalg.norm(other.vector) + 1e-8)
        self.apply_scale(); other.apply_scale()
        self._backpropagate_to_features(other, pull, other_pull, color)

    def _backpropagate_to_features(self, other: 'DNAConcept',
                                   pull: float, other_pull: float, color: str):
        gradient = other.vector - self.vector
        layer_map = {"IS_A": PriorityLayer.DOMAIN, "PART_OF": PriorityLayer.DOMAIN,
                     "HAS": PriorityLayer.CLUSTER, "GROWN_IN": PriorityLayer.CLUSTER,
                     "LOCATION": PriorityLayer.CLUSTER, "RELATED": PriorityLayer.ATOMIC,
                     "OPERATOR": PriorityLayer.UNIVERSAL, "CONDITION": PriorityLayer.UNIVERSAL}
        layer = layer_map.get(color, PriorityLayer.CLUSTER)
        for concept, force in [(self, pull), (other, other_pull)]:
            for fid in concept.physical_features + concept.semantic_features:
                letters = self.feature_registry.feature_to_letters(fid, length=5)
                for i, ch in enumerate(letters):
                    base = self.letter_vec.get(ch)
                    x    = base + i * POSITION_OFFSET
                    grad_sin = np.cos(x)
                    full_grad = np.zeros(self.dims)
                    full_grad[ESSENCE_DIMS]  = grad_sin[ESSENCE_DIMS]  * LR_DOMAIN
                    full_grad[IDENTITY_DIMS] = grad_sin[IDENTITY_DIMS] * LR_CLUSTER
                    full_grad[TEMPORAL_DIMS] = grad_sin[TEMPORAL_DIMS] * LR_ATOMIC
                    norm_g = np.linalg.norm(full_grad) + 1e-8
                    delta  = force * 0.5 * gradient * full_grad / norm_g
                    self.feature_registry.update_vector(fid, delta, layer)
                    self.letter_vec.update(ch, delta, PriorityLayer.ATOMIC)

    def partitioned_similarity(self, other: 'DNAConcept', partition: slice = None) -> float:
        if self.vector is None or other.vector is None: return 0.0
        v1, v2 = (self.vector[partition], other.vector[partition]) if partition else (self.vector, other.vector)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

    def cosine_similarity(self, other: 'DNAConcept') -> float:
        return self.partitioned_similarity(other)

    def strengthen_relationship(self, other_name: str, increment: float = 0.1):
        self.importance = min(10.0, self.importance + increment * 0.1)

    def serialize(self) -> dict:
        return {"name": self.name, "physical_features": self.physical_features,
                "semantic_features": self.semantic_features,
                "vector": self.vector.tolist() if self.vector is not None else None,
                "importance": self.importance, "domain": self.domain,
                "cluster_id": self.cluster_id, "numeric_value": self.numeric_value}

    @classmethod
    def from_serialized(cls, data: dict, feature_registry, letter_vec, dims: int = 128):
        obj = cls(data["name"], data["physical_features"], data["semantic_features"],
                  feature_registry, letter_vec, importance=data.get("importance", 1.0),
                  domain=data.get("domain","general"),
                  numeric_value=data.get("numeric_value"), dims=dims)
        if data.get("vector") is not None:
            obj.vector = np.array(data["vector"], dtype=np.float32)
        obj.cluster_id = data.get("cluster_id")
        return obj

# ============================================================
# SentenceDecoder
# ============================================================
class SentenceDecoder:
    def __init__(self, concept_memory: 'ConceptMemory'):
        self.concept_memory = concept_memory

    def extract_weighted_features(self, concept_name: str, top_k: int = 5) -> List[Tuple[str,float,str]]:
        if concept_name not in self.concept_memory.concepts: return []
        relationships = self.concept_memory.weighted_relationships.get(concept_name, {})
        if not relationships: return []
        sorted_rels = sorted(relationships.items(), key=lambda x: x[1].weight, reverse=True)
        return [(other, rel.weight, rel.color) for other, rel in sorted_rels[:top_k]]

    def multi_hop_traversal(self, start: str, target_color: str, max_hops: int = 3) -> List[str]:
        visited = {start}; path = [start]; current = start
        for _ in range(max_hops):
            if current not in self.concept_memory.weighted_relationships: break
            found = None
            for other, rel in self.concept_memory.weighted_relationships[current].items():
                if rel.color == target_color and other not in visited:
                    found = other; break
            if found is None: break
            path.append(found); visited.add(found); current = found
        return path

    def generate_description(self, concept_name: str) -> str:
        features = self.extract_weighted_features(concept_name, top_k=5)
        if not features: return f"{concept_name} is a concept."
        concept  = self.concept_memory.concepts.get(concept_name)
        domain   = concept.domain if concept else "general"
        is_a     = next((f[0] for f in features if f[2] == "IS_A"), None)
        has_feat = [f[0] for f in features if f[2] == "HAS"][:2]
        loc_path = self.multi_hop_traversal(concept_name, "LOCATION", max_hops=2)
        location = loc_path[-1] if len(loc_path) > 1 else None
        parts = [concept_name.capitalize()]
        if is_a:     parts.append(f"is a {is_a}")
        if has_feat: parts.append(f"with {', '.join(has_feat)}")
        if location: parts.append(f"located in {location}")
        parts.append(f"(domain: {domain})")
        return " ".join(parts) + "."

# ============================================================
# InstructionEngine
# ============================================================
class InstructionEngine:
    def __init__(self, concept_memory: 'ConceptMemory', feature_registry: 'FeatureRegistry',
                 letter_vec: 'LetterVectors'):
        self.memory = concept_memory
        self.feature_registry = feature_registry
        self.letter_vec = letter_vec
        self._ensure_operators()

    def _ensure_operators(self):
        operators = {"PLUS":10.0,"MINUS":10.0,"MULTIPLY":10.0,"DIVIDE":10.0,
                     "EQUALS":10.0,"GREATER":10.0,"LESS":10.0,
                     "IF":10.0,"THEN":10.0,"AND":10.0,"OR":10.0,"NOT":10.0}
        for op, imp in operators.items():
            op_low = op.lower()
            if op_low not in self.memory.concepts:
                physical = [self.feature_registry.register(op_low)]
                semantic = [self.feature_registry.register(op_low)]
                self.memory.register(op_low, physical, semantic, importance=imp, domain="operator")

    def get_or_create_number(self, value: float) -> DNAConcept:
        name = f"num_{value}".replace('.','_').replace('-','neg')
        if name in self.memory.concepts: return self.memory.concepts[name]
        physical = [self.feature_registry.register(str(value))]
        semantic = [self.feature_registry.register(str(value))]
        return self.memory.register(name, physical, semantic,
                                    importance=abs(value)/10+1.0,
                                    domain="number", numeric_value=value)

    def execute_arithmetic(self, operator: str, a: Union[str,float], b: Union[str,float]) -> DNAConcept:
        def _to_val(x):
            if isinstance(x,(int,float)): return float(x), self.get_or_create_number(float(x))
            c = self.memory.concepts.get(str(x).lower())
            if c is None: raise ValueError(f"Concept '{x}' not found")
            return (c.numeric_value if c.numeric_value is not None else c.importance), c
        va, ca = _to_val(a); vb, cb = _to_val(b)
        op = operator.upper()
        if   op=="PLUS":     result = va+vb
        elif op=="MINUS":    result = va-vb
        elif op=="MULTIPLY": result = va*vb
        elif op=="DIVIDE":   result = va/vb if vb!=0 else float('inf')
        else: raise ValueError(f"Unknown operator: {operator}")
        rc = self.get_or_create_number(result)
        self.memory.add_weighted_relationship(ca.name, cb.name, weight=0.9, color="OPERATOR")
        self.memory.add_weighted_relationship(operator.lower(), rc.name, weight=1.0, color="OPERATOR")
        return rc

    def evaluate_condition(self, condition_expr: Dict) -> bool:
        op    = condition_expr.get("operator","EQUALS").upper()
        left  = condition_expr.get("left");  right = condition_expr.get("right")
        def _val(x):
            if isinstance(x,(int,float)): return float(x)
            c = self.memory.concepts.get(str(x).lower())
            return (c.numeric_value if c and c.numeric_value is not None else (c.importance if c else 0))
        lv = _val(left); rv = _val(right)
        if op=="EQUALS":  return abs(lv-rv)<1e-6
        if op=="GREATER": return lv>rv
        if op=="LESS":    return lv<rv
        if op=="AND":     return bool(lv) and bool(rv)
        if op=="OR":      return bool(lv) or bool(rv)
        if op=="NOT":     return not bool(lv)
        return False

# ============================================================
# ReasoningEngine
# ============================================================
class ReasoningEngine:
    def __init__(self, concept_memory: 'ConceptMemory', feature_registry: 'FeatureRegistry',
                 letter_vec: 'LetterVectors'):
        self.concept_memory    = concept_memory
        self.decoder           = SentenceDecoder(concept_memory)
        self.instruction_engine= InstructionEngine(concept_memory, feature_registry, letter_vec)
        self.predictive_fallback = None

    def multi_hop_reasoning(self, start: str, max_hops: int = 3, decay: float = 0.7,
                            color_filter: Optional[str] = None) -> Dict[str, float]:
        if start not in self.concept_memory.weighted_relationships:
            if self.predictive_fallback is not None:
                return self.predictive_fallback.multi_hop_reasoning(start, max_hops, decay, color_filter)
            return {}
        activation = {start: 1.0}
        for _ in range(max_hops):
            new_activation: Dict[str,float] = {}
            for node, score in activation.items():
                for nb, rel in self.concept_memory.weighted_relationships.get(node, {}).items():
                    if color_filter and rel.color != color_filter: continue
                    w = rel.weight
                    if node in self.concept_memory.concepts and nb in self.concept_memory.concepts:
                        sim = self.concept_memory.concepts[node].cosine_similarity(
                            self.concept_memory.concepts[nb])
                        w  *= (0.5 + 0.5 * sim)
                    new_activation[nb] = new_activation.get(nb, 0) + score * w * decay
            for k, v in new_activation.items():
                activation[k] = activation.get(k, 0) + v
        if activation:
            mx = max(activation.values())
            activation = {k: v/mx for k, v in activation.items()}
        return activation

    def analogical_reasoning(self, a: str, b: str, c: str, partition: slice = None) -> List[str]:
        if a not in self.concept_memory.concepts or b not in self.concept_memory.concepts: return []
        va, vb = self.concept_memory.concepts[a].vector, self.concept_memory.concepts[b].vector
        direction = (vb[partition]-va[partition]) if partition else (vb-va)
        if c not in self.concept_memory.concepts: return []
        vc = self.concept_memory.concepts[c].vector
        target = vc.copy(); (target.__setitem__(partition, vc[partition]+direction)
                              if partition else target.__setitem__(slice(None), vc+direction))
        target /= (np.linalg.norm(target) + 1e-8)
        results = self.concept_memory.partitioned_search(target, top_k=5, partition=partition)
        return [r[0] for r in results if r[0] not in (a, b, c)]

    def generate_sentence(self, concept: str) -> str:
        return self.decoder.generate_description(concept)

    def calculate(self, expression: str) -> Dict:
        if not all(c in set('0123456789.+-*/() ') for c in expression):
            return {"error": "Invalid characters"}
        try:
            result  = eval(expression)
            concept = self.instruction_engine.get_or_create_number(result)
            return {"expression": expression, "result": result, "concept": concept.name}
        except Exception as e:
            return {"error": str(e)}

    def execute_instruction(self, operator: str, a, b) -> Dict:
        try:
            rc = self.instruction_engine.execute_arithmetic(operator, a, b)
            return {"operator": operator, "operands": [a, b],
                    "result": rc.numeric_value, "concept": rc.name}
        except Exception as e:
            return {"error": str(e)}

    def evaluate_rule(self, condition: Dict, action: str) -> Dict:
        result = self.instruction_engine.evaluate_condition(condition)
        return {"condition_true": result, "action_triggered": action if result else None}

# ============================================================
# ConceptMemory
# ============================================================
class ConceptMemory:
    def __init__(self, feature_registry: 'FeatureRegistry', letter_vec: 'LetterVectors',
                 max_concepts: int = MAX_CONCEPTS, dims: int = 128):
        self.dims              = dims
        self.feature_registry  = feature_registry
        self.letter_vec        = letter_vec
        self.concepts:         Dict[str, DNAConcept]              = {}
        self.weighted_relationships: Dict[str, Dict[str, RelationshipData]] = defaultdict(dict)
        self.relationships:    Dict[str, Set[str]]                = defaultdict(set)
        self.max_concepts      = max_concepts
        self.global_centroid:  Optional[np.ndarray]               = None
        self.batch_counter     = 0
        self.pending_updates:  List[Tuple[str,str,float,str]]     = []
        self.global_synapse    = np.full((26,26), 0.5, dtype=np.float32)
        np.fill_diagonal(self.global_synapse, 0.5)
        self.synapse_step      = 0
        self._load_synapse()
        self._faiss_available  = False
        self.index             = None
        self.id_to_name:       Dict[int, str]  = {}
        self.name_to_id:       Dict[str, int]  = {}
        self.next_id           = 0
        try:
            import faiss
            self._faiss_available = True
        except Exception:
            pass

    def _load_synapse(self):
        path = os.path.join(PERSIST_DIR, "global_synapse.npy")
        step_path = path.replace(".npy","_step.npy")
        if os.path.exists(path):
            try:
                self.global_synapse = np.load(path)
                if os.path.exists(step_path):
                    self.synapse_step = int(np.load(step_path))
            except Exception:
                pass

    def _save_synapse(self):
        path = os.path.join(PERSIST_DIR, "global_synapse.npy")
        np.save(path, self.global_synapse)
        np.save(path.replace(".npy","_step.npy"), np.array([self.synapse_step]))

    def update_global_centroid(self):
        if not self.concepts: return
        all_vecs = np.array([c.vector for c in self.concepts.values() if c.vector is not None])
        if len(all_vecs) > 0:
            self.global_centroid = np.mean(all_vecs, axis=0)

    def get_global_context(self) -> dict:
        if self.global_centroid is None: self.update_global_centroid()
        top = sorted(self.concepts.values(), key=lambda x: x.importance, reverse=True)[:5]
        return {"global_centroid": self.global_centroid.tolist() if self.global_centroid is not None else None,
                "anchors": [n.name for n in top], "total_concepts": len(self.concepts)}

    def _ensure_index(self):
        if not self._faiss_available: return
        if self.index is None:
            import faiss
            self.index = faiss.IndexFlatIP(self.dims)

    def _rebuild_index(self):
        if not self._faiss_available or not self.concepts: return
        import faiss
        vectors = [c.vector for c in self.concepts.values() if c.vector is not None]
        names   = [name for name, c in self.concepts.items() if c.vector is not None]
        if not vectors: return
        vecs = np.vstack(vectors).astype(np.float32)
        self.index = faiss.IndexFlatIP(self.dims)
        self.index.add(vecs)
        self.id_to_name  = {i: n for i, n in enumerate(names)}
        self.name_to_id  = {n: i for i, n in enumerate(names)}
        self.next_id     = len(self.concepts)
        self.update_global_centroid()

    def register(self, name: str, physical_features: List[int], semantic_features: List[int],
                 importance: float = 1.0, domain: str = "general",
                 numeric_value: Optional[float] = None) -> DNAConcept:
        name_low = name.lower()
        if name_low in self.concepts:
            self.concepts[name_low].importance = min(10.0, self.concepts[name_low].importance + 0.1)
            return self.concepts[name_low]
        concept = DNAConcept(name_low, physical_features, semantic_features,
                             self.feature_registry, self.letter_vec,
                             importance=importance, domain=domain,
                             numeric_value=numeric_value, dims=self.dims)
        self.concepts[name_low] = concept
        if self._faiss_available and concept.vector is not None:
            self._ensure_index()
            if self.index is not None:
                self.index.add(concept.vector.reshape(1,-1))
                self.id_to_name[self.next_id] = name_low
                self.name_to_id[name_low]     = self.next_id
                self.next_id += 1
        self._prune()
        return concept

    def _get_letter_probs(self, concept_vector: np.ndarray) -> np.ndarray:
        if concept_vector is None: return np.ones(26)/26
        if concept_vector.ndim == 1: concept_vector = concept_vector.reshape(-1,1)
        letter_matrix = np.array([self.letter_vec.get(ch) for ch in ALPHABET])
        raw = letter_matrix @ concept_vector
        raw = np.squeeze(raw)
        exp_act = np.exp(raw * 2.0)
        return exp_act / (np.sum(exp_act) + 1e-8)

    def add_weighted_relationship(self, a: str, b: str,
                                   weight: float = 1.0, color: str = "RELATED"):
        a_low, b_low = a.lower(), b.lower()
        if a_low not in self.concepts or b_low not in self.concepts: return
        for src, dst in [(a_low, b_low), (b_low, a_low)]:
            if dst not in self.weighted_relationships[src]:
                self.weighted_relationships[src][dst] = RelationshipData(weight=weight, color=color)
            else:
                rel = self.weighted_relationships[src][dst]
                rel.weight = min(1.0, rel.weight + 0.1 * weight)
                rel.co_occurrence_count += 1
                rel.last_accessed = time.time()
            self.relationships[src].add(dst)
        self.concepts[a_low].move_towards(self.concepts[b_low], lr=LR_CONCEPT,
                                           weight=weight, color=color)
        # Global Synapse (Sine-Hebbian)
        probs_a = self._get_letter_probs(self.concepts[a_low].vector)
        probs_b = self._get_letter_probs(self.concepts[b_low].vector)
        self.synapse_step += 1
        sine_lr      = 0.01 * (0.5 + 0.5 * math.sin(self.synapse_step / 15.0))
        hebbian_delta= (weight - 0.5) * 2.0
        co_act       = np.outer(probs_a, probs_b) + np.outer(probs_b, probs_a)
        self.global_synapse += sine_lr * hebbian_delta * co_act
        self.global_synapse  = np.clip(self.global_synapse, 0.01, 0.99)
        np.fill_diagonal(self.global_synapse, 0.5)
        if self.synapse_step % 10 == 0:
            self._save_synapse()

    def add_relationship(self, a: str, b: str):
        self.add_weighted_relationship(a, b, weight=1.0, color="RELATED")

    def strengthen_relationship(self, a: str, b: str, increment: float = 0.1):
        a_low, b_low = a.lower(), b.lower()
        for src, dst in [(a_low, b_low), (b_low, a_low)]:
            if src in self.weighted_relationships and dst in self.weighted_relationships[src]:
                rel = self.weighted_relationships[src][dst]
                rel.weight = min(1.0, rel.weight + increment)
                rel.co_occurrence_count += 1
                rel.last_accessed = time.time()

    def apply_decay(self, decay_rate: float = 0.001, inactive_threshold: float = 86400):
        current_time = time.time()
        for a, rels in self.weighted_relationships.items():
            to_remove = []
            for b, rel in rels.items():
                if current_time - rel.last_accessed > inactive_threshold:
                    rel.weight = max(0.05, rel.weight - decay_rate)
                    if rel.weight <= 0.06:
                        to_remove.append(b)
            for b in to_remove:
                del self.weighted_relationships[a][b]
                if b in self.relationships[a]:
                    self.relationships[a].remove(b)

    def partitioned_search(self, query_vector: np.ndarray, top_k: int = 5,
                           partition: slice = None) -> List[Tuple[str,float]]:
        if not self.concepts: return []
        q    = query_vector[partition] if partition else query_vector
        names= list(self.concepts.keys())
        vecs = np.vstack([self.concepts[n].vector[partition]
                          if partition else self.concepts[n].vector for n in names])
        q    = q / (np.linalg.norm(q) + 1e-8)
        vecs_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        scores = vecs_norm @ q
        top    = np.argsort(scores)[::-1][:top_k]
        return [(names[i], float(scores[i])) for i in top]

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str,float]]:
        return self.partitioned_search(query_vector, top_k, partition=None)

    def add_to_batch(self, a: str, b: str, weight: float, color: str):
        self.pending_updates.append((a, b, weight, color))
        self.batch_counter += 1
        if len(self.pending_updates) >= BATCH_SIZE:
            self.process_batch()

    def process_batch(self):
        if not self.pending_updates: return
        for a, b, weight, color in self.pending_updates:
            if a in self.concepts and b in self.concepts:
                self.concepts[a].move_towards(self.concepts[b], lr=LR_CONCEPT,
                                               weight=weight, color=color)
        self.pending_updates.clear()
        if self.batch_counter % GLOBAL_UPDATE_FREQUENCY == 0:
            self.update_global_centroid()
        self._rebuild_index()

    async def extract_and_link(self, text: str, ontology: 'DynamicOntology',
                                sector: str = "general") -> List[str]:
        words  = [w for w in text.lower().split() if len(w) > 1 and w not in STOP_WORDS]
        unique = list(set(words))[:15]
        for kw in unique:
            features = ontology.get_features(kw)
            domain   = ontology.get_domain(kw)
            p_fids   = [self.feature_registry.register(f) for f in features]
            s_fids   = [self.feature_registry.register(f) for f in features]
            self.register(kw, p_fids, s_fids, importance=1.0, domain=domain)
        for i in range(len(unique)):
            for j in range(i+1, min(i+4, len(unique))):
                color = self._infer_relationship_color(text, unique[i], unique[j])
                self.add_weighted_relationship(unique[i], unique[j], weight=0.8, color=color)
        return unique

    def _infer_relationship_color(self, text: str, a: str, b: str) -> str:
        t = text.lower()
        if f"{a} is {b}" in t or f"{b} is {a}" in t: return "IS_A"
        if f"{a} has {b}" in t or f"{b} has {a}" in t: return "HAS"
        if f"{a} in {b}"  in t or f"{b} in {a}"  in t: return "LOCATION"
        if "cause" in t and (a in t or b in t):          return "CAUSES"
        return "RELATED"

    def _prune(self):
        if len(self.concepts) <= self.max_concepts: return
        sorted_c = sorted(self.concepts.items(),
                          key=lambda x: (x[1].importance, len(self.weighted_relationships.get(x[0], {}))),
                          reverse=True)
        keep_threshold = int(self.max_concepts * 0.8)
        to_keep     = sorted_c[:keep_threshold]
        to_compress = sorted_c[keep_threshold:]
        if to_compress:
            super_vec = np.zeros(128); super_imp = 0
            for name, concept in to_compress:
                if concept.vector is not None:
                    sw = math.sin(concept.importance) + 0.1
                    super_vec += sw * concept.vector
                    super_imp += concept.importance
            if np.linalg.norm(super_vec) > 0:
                super_vec /= np.linalg.norm(super_vec)
                cn = f"compressed_{int(time.time())}"
                self.register(cn, [self.feature_registry.register("compressed")],
                              [self.feature_registry.register("compressed")],
                              importance=super_imp * 0.1, domain="general")
                self.concepts[cn].vector = super_vec
        self.concepts = {name: concept for name, concept in to_keep}
        keep_names = set(self.concepts.keys())
        self.weighted_relationships = defaultdict(dict, {
            k: {b: r for b, r in v.items() if b in keep_names}
            for k, v in self.weighted_relationships.items() if k in keep_names})
        self.relationships = defaultdict(set, {
            k: v.intersection(keep_names)
            for k, v in self.relationships.items() if k in keep_names})
        self._rebuild_index()

    def serialize(self) -> dict:
        return {"concepts": {n: c.serialize() for n, c in self.concepts.items()},
                "weighted_relationships": {k: {b: r.to_dict() for b, r in v.items()}
                                           for k, v in self.weighted_relationships.items()},
                "relationships": {k: list(v) for k, v in self.relationships.items()},
                "global_centroid": self.global_centroid.tolist() if self.global_centroid is not None else None,
                "global_synapse": self.global_synapse.tolist(),
                "synapse_step": self.synapse_step}

    def restore(self, data: dict):
        self.concepts = {}
        self.weighted_relationships = defaultdict(dict)
        self.relationships          = defaultdict(set)
        for name, cdata in data.get("concepts", {}).items():
            self.concepts[name] = DNAConcept.from_serialized(
                cdata, self.feature_registry, self.letter_vec, dims=self.dims)
        for k, vdict in data.get("weighted_relationships", {}).items():
            for b, rdata in vdict.items():
                self.weighted_relationships[k][b] = RelationshipData.from_dict(rdata)
                self.relationships[k].add(b)
        for k, vlist in data.get("relationships", {}).items():
            self.relationships[k].update(vlist)
        if data.get("global_centroid"):
            self.global_centroid = np.array(data["global_centroid"], dtype=np.float32)
        if data.get("global_synapse"):
            self.global_synapse = np.array(data["global_synapse"], dtype=np.float32)
            self.synapse_step   = data.get("synapse_step", 0)
        self._rebuild_index()


# ============================================================
#   ░░░░  PRIORITY 1: DNA ACTIVATION ENGINE  ░░░░
# ============================================================
class DNAActivationEngine:
    """
    Encodes a query into DNA, applies Sine oscillation,
    computes activation scores for every node (DNA Attention),
    then applies Hebbian reinforcement between co-activated nodes.
    """
    def __init__(self, concept_memory: ConceptMemory,
                 letter_vec: LetterVectors, dims: int = 128):
        self.concept_memory = concept_memory
        self.letter_vec     = letter_vec
        self.dims           = dims
        self.sine_step      = 0

    # ── Priority 1: DNA Encoder ──────────────────────────────
    def encode_query(self, query_text: str) -> np.ndarray:
        """Text → DNA vector via sine-modulated letter encoding."""
        chars = [c.upper() for c in query_text if c.upper() in ALPHABET]
        if not chars:
            return np.zeros(self.dims, dtype=np.float32)
        encoded = np.zeros(self.dims, dtype=np.float32)
        for i, ch in enumerate(chars):
            self.sine_step += 1
            sine_factor = 0.5 + 0.5 * math.sin(self.sine_step / 10.0)
            base        = self.letter_vec.get(ch)
            encoded    += base * sine_factor * math.cos(i * POSITION_OFFSET)
        norm = np.linalg.norm(encoded)
        if norm > 0:
            encoded /= norm
        return encoded.astype(np.float32)

    # ── Priority 4: DNA Attention — compute activation scores ──
    def compute_activation_scores(self, query_vector: np.ndarray,
                                  top_k: int = 500) -> Dict[str, float]:
        """
        Every node gets an activation score.
        Score = cosine_sim × hebbian_mass (log importance).
        Only top_k are returned for efficiency.
        """
        if not self.concept_memory.concepts:
            return {}
        scores: Dict[str, float] = {}
        q_norm = np.linalg.norm(query_vector) + 1e-8
        for name, concept in self.concept_memory.concepts.items():
            if concept.vector is None: continue
            sim = float(np.dot(query_vector, concept.vector) /
                        (q_norm * np.linalg.norm(concept.vector) + 1e-8))
            hebbian_mass    = math.log1p(concept.importance)
            scores[name]    = sim * hebbian_mass
        # Normalise to [0, 1]
        if scores:
            lo = min(scores.values()); hi = max(scores.values())
            rng = hi - lo + 1e-8
            scores = {k: (v - lo) / rng for k, v in scores.items()}
        # Return top_k
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k])

    # ── Hebbian Activation: co-activated nodes strengthen each other ──
    def hebbian_activate(self, activated: Dict[str, float]) -> Dict[str, float]:
        result = dict(activated)
        for node_a, score_a in list(activated.items()):
            for node_b, rel in self.concept_memory.weighted_relationships.get(node_a, {}).items():
                if node_b in activated:
                    boost = score_a * activated[node_b] * rel.weight * 0.1
                    result[node_b] = min(1.0, result.get(node_b, 0) + boost)
        return result


# ============================================================
#   ░░░░  PRIORITY 2: WORKING MEMORY  ░░░░
# ============================================================
class WorkingMemory:
    """
    Temporary, session-scoped memory.
    Loads the top-N activated nodes from the graph,
    runs reasoning over that subgraph, then is deleted.
    """
    def __init__(self, max_nodes: int = WORKING_MEMORY_SIZE):
        self.max_nodes      = max_nodes
        self.nodes:         Dict[str, float] = {}
        self.reasoning_path: List[dict]      = []
        self.session_id     = ""
        self.created_at     = time.time()

    def load(self, activation_scores: Dict[str, float],
             concept_memory: ConceptMemory):
        """Select top-N nodes + their high-weight neighbours."""
        sorted_nodes = sorted(activation_scores.items(), key=lambda x: x[1], reverse=True)
        self.nodes   = dict(sorted_nodes[:self.max_nodes])
        # Pull in neighbours of top-50 high-activation nodes
        for node in list(self.nodes.keys())[:50]:
            for nb, rel in concept_memory.weighted_relationships.get(node, {}).items():
                if nb not in self.nodes and len(self.nodes) < self.max_nodes:
                    self.nodes[nb] = activation_scores.get(nb, 0) * rel.weight * 0.5

    def add_to_path(self, step: str, node: str, score: float, color: str = ""):
        self.reasoning_path.append({"step": step, "node": node,
                                    "score": round(score, 4), "color": color,
                                    "t": round(time.time() - self.created_at, 4)})

    def clear(self):
        self.nodes.clear()
        self.reasoning_path.clear()


# ============================================================
#   ░░░░  PRIORITY 6: EPISODIC MEMORY  ░░░░
# ============================================================
class EpisodicMemory:
    """Store successful reasoning episodes and reuse them for similar queries."""
    def __init__(self, max_episodes: int = MAX_EPISODES,
                 min_reward: float = MIN_EPISODE_REWARD):
        self.episodes:     List[Episode] = []
        self.max_episodes  = max_episodes
        self.min_reward    = min_reward

    def store(self, query: str, query_vector: np.ndarray,
              working_memory: WorkingMemory, answer_concepts: List[str],
              reward: float):
        if reward < self.min_reward: return
        ep = Episode(query=query, query_vector=query_vector.tolist(),
                     reasoning_path=list(working_memory.reasoning_path),
                     activated_concepts=list(working_memory.nodes.keys())[:30],
                     answer_concepts=answer_concepts, reward=reward)
        self.episodes.append(ep)
        if len(self.episodes) > self.max_episodes:
            self.episodes.sort(key=lambda e: e.reward, reverse=True)
            self.episodes = self.episodes[:self.max_episodes]

    def retrieve_similar(self, query_vector: np.ndarray,
                         top_k: int = 3) -> List[Episode]:
        if not self.episodes: return []
        q = np.array(query_vector, dtype=np.float32)
        scored = []
        for ep in self.episodes:
            ev = np.array(ep.query_vector, dtype=np.float32)
            sim = float(np.dot(q, ev) / (np.linalg.norm(q)*np.linalg.norm(ev) + 1e-8))
            scored.append((ep, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        result = [ep for ep, sim in scored[:top_k] if sim > 0.55]
        for ep in result:
            ep.use_count += 1
        return result

    def serialize(self) -> List[dict]:
        return [e.to_dict() for e in self.episodes]

    def restore(self, data: List[dict]):
        self.episodes = [Episode.from_dict(d) for d in data]


# ============================================================
#   ░░░░  PRIORITY 11: CONFIDENCE ESTIMATOR  ░░░░
# ============================================================
class ConfidenceEstimator:
    """Every answer includes confidence, depth, nodes activated, reward estimate, evidence."""
    def estimate(self, activation_scores: Dict[str, float],
                 reasoning_path: List[dict],
                 working_memory: WorkingMemory) -> dict:
        if not activation_scores:
            return {"confidence": 0.0, "reasoning_depth": 0,
                    "activated_nodes": 0, "reward_estimate": 0.0, "evidence_count": 0}
        top_scores = sorted(activation_scores.values(), reverse=True)[:10]
        confidence = float(np.mean(top_scores))
        evidence   = len([s for s in activation_scores.values() if s > 0.5])
        depth      = len(reasoning_path)
        reward_est = min(1.0, confidence*0.65 + (evidence / max(len(activation_scores),1))*0.35)
        return {"confidence":     round(confidence, 4),
                "reasoning_depth":depth,
                "activated_nodes":len(activation_scores),
                "reward_estimate":round(reward_est, 4),
                "evidence_count": evidence}


# ============================================================
#   ░░░░  PRIORITY 14: BENCHMARK TRACKER  ░░░░
# ============================================================
class BenchmarkTracker:
    """Track retrieval accuracy, answer accuracy, nodes visited, time, memory, reward."""
    def __init__(self):
        self.metrics: List[dict] = []
        self._lock = threading.Lock()

    def record(self, query: str, retrieval_accurate: bool, answer_accurate: bool,
               nodes_visited: int, reasoning_time: float,
               memory_usage: int, reward: float):
        with self._lock:
            self.metrics.append({
                "query":              query[:60],
                "retrieval_accurate": int(retrieval_accurate),
                "answer_accurate":    int(answer_accurate),
                "nodes_visited":      nodes_visited,
                "reasoning_time_ms":  round(reasoning_time*1000, 2),
                "memory_usage_nodes": memory_usage,
                "reward":             round(reward, 4),
                "ts":                 time.time()})
            if len(self.metrics) > 10000:
                self.metrics = self.metrics[-5000:]

    def summary(self) -> dict:
        if not self.metrics:
            return {"total_queries": 0}
        recent = self.metrics[-100:]
        def avg(key): return round(float(np.mean([m[key] for m in recent])), 4)
        return {
            "total_queries": len(self.metrics),
            "recent_100": {
                "retrieval_accuracy":  avg("retrieval_accurate"),
                "answer_accuracy":     avg("answer_accurate"),
                "avg_nodes_visited":   avg("nodes_visited"),
                "avg_reasoning_ms":    avg("reasoning_time_ms"),
                "avg_memory_nodes":    avg("memory_usage_nodes"),
                "avg_reward":          avg("reward")}}


# ============================================================
#   ░░░░  PRIORITY 8: SELF REFLECTOR  ░░░░
# ============================================================
class SelfReflector:
    """After answering, evaluate reasoning quality and store improvements."""
    def __init__(self):
        self.history: List[dict] = []

    def reflect(self, query: str, reasoning_path: List[dict],
                activation_scores: Dict[str, float], reward: float) -> dict:
        unique_nodes = list({step["node"] for step in reasoning_path})
        improvements = []
        if reward < 0.5:
            improvements.append("Low reward — try different entry nodes next time")
        if len(reasoning_path) > 15:
            improvements.append("Long path — tighten attention threshold")
        if len(unique_nodes) < len(reasoning_path) * 0.6:
            improvements.append("Repeated nodes — graph may have tight cycles")
        if not activation_scores:
            improvements.append("No activation scores — concept may not exist in graph")
        top_node = (sorted(activation_scores.items(), key=lambda x: x[1], reverse=True)[:1]
                    or [("none", 0)])[0]
        reflection = {"query": query[:60], "path_length": len(reasoning_path),
                      "unique_nodes": len(unique_nodes),
                      "top_node": top_node[0], "top_score": round(top_node[1], 4),
                      "reward": round(reward, 4), "improvements": improvements,
                      "ts": time.time()}
        self.history.append(reflection)
        if len(self.history) > 500:
            self.history = self.history[-300:]
        return reflection


# ============================================================
# PersistenceManager  (enhanced: saves/loads episodes)
# ============================================================
class PersistenceManager:
    @staticmethod
    def save_all(concept_memory: ConceptMemory, feature_registry: FeatureRegistry,
                 letter_vec: LetterVectors, episodic_memory: 'EpisodicMemory' = None,
                 filepath: str = None):
        if filepath is None:
            filepath = os.path.join(PERSIST_DIR, "brain_state.pkl")
        state = {"concept_memory":  concept_memory.serialize(),
                 "feature_registry": feature_registry.serialize(),
                 "letter_vec":       letter_vec.serialize(),
                 "episodes":         episodic_memory.serialize() if episodic_memory else []}
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load_all(filepath: str = None):
        if filepath is None:
            filepath = os.path.join(PERSIST_DIR, "brain_state.pkl")
        ontology         = DynamicOntology()
        feature_registry = FeatureRegistry(ontology, dims=128)
        letter_vec       = LetterVectors(dims=128)
        concept_memory   = ConceptMemory(feature_registry, letter_vec, dims=128)
        episodic_memory  = EpisodicMemory()
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    state = pickle.load(f)
                concept_memory.restore(state.get("concept_memory", {}))
                feature_registry.restore(state.get("feature_registry", {}))
                letter_vec.restore(state.get("letter_vec", {}))
                episodic_memory.restore(state.get("episodes", []))
            except Exception as e:
                print(f"[WARN] Failed to load persisted state: {e}")
        return concept_memory, feature_registry, letter_vec, ontology, episodic_memory


# ============================================================
# ContinuousTrainer
# ============================================================
class ContinuousTrainer:
    def __init__(self, concept_memory: ConceptMemory, feature_registry: FeatureRegistry,
                 letter_vec: LetterVectors, interval_sec: int = TRAIN_INTERVAL_SEC):
        self.concept_memory   = concept_memory
        self.feature_registry = feature_registry
        self.letter_vec       = letter_vec
        self.interval         = interval_sec
        self.running          = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.thread  = threading.Thread(target=self._train_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=2)

    def _train_loop(self):
        while self.running:
            time.sleep(self.interval)
            if self.concept_memory.pending_updates:
                self.concept_memory.process_batch()
            self.concept_memory.apply_decay(decay_rate=0.001)
            rels = [(a, b, rel.weight, rel.color)
                    for a, rdict in self.concept_memory.weighted_relationships.items()
                    for b, rel in rdict.items()]
            if not rels: continue
            batch = random.sample(rels, min(BATCH_SIZE, len(rels)))
            for a, b, weight, color in batch:
                if a in self.concept_memory.concepts and b in self.concept_memory.concepts:
                    self.concept_memory.concepts[a].move_towards(
                        self.concept_memory.concepts[b],
                        lr=LR_CONCEPT*0.5, weight=weight, color=color)
            self.concept_memory._rebuild_index()
            self.concept_memory.update_global_centroid()


# ============================================================
# KnowledgeGraphEnv  — Enhanced with full DNA pipeline
# ============================================================
class KnowledgeGraphEnv:
    def __init__(self, start_trainer: bool = True):
        # ── Load persisted state ──────────────────────────────
        (self.concept_memory, self.feature_registry,
         self.letter_vec, self.ontology,
         self.episodic_memory) = PersistenceManager.load_all()

        self.reasoning_engine = ReasoningEngine(
            self.concept_memory, self.feature_registry, self.letter_vec)

        # ── Predictive 12-dim submodel ────────────────────────
        self.predictive_feature_registry = FeatureRegistry(self.ontology, dims=12)
        self.predictive_letter_vec       = LetterVectors(dims=12)
        self.predictive_concept_memory   = ConceptMemory(
            self.predictive_feature_registry, self.predictive_letter_vec, dims=12)
        self.predictive_reasoning = ReasoningEngine(
            self.predictive_concept_memory,
            self.predictive_feature_registry,
            self.predictive_letter_vec)
        self.reasoning_engine.predictive_fallback = self.predictive_reasoning

        # ── Dynamic Knowledge Loader ──────────────────────────
        self.knowledge_loader = DynamicKnowledgeLoader()

        self._seed_initial_concepts()

        self.trainer: Optional[ContinuousTrainer] = None
        if start_trainer:
            self.trainer = ContinuousTrainer(
                self.concept_memory, self.feature_registry, self.letter_vec)
            self.trainer.start()

        self.current_task: Optional[dict] = None
        self.current_step  = 0
        self.episode_reward= 0.0
        self.done          = False

        # ── Projection Layer ──────────────────────────────────
        self.projector        = DimensionProjector(
            persist_path=os.path.join(PERSIST_DIR, "projection.pkl"))
        self.projection_cache = ProjectionCache(max_size=10000)
        self._train_projector()

        # ── Priority 1: DNA Activation Engine ────────────────
        self.dna_activation_engine = DNAActivationEngine(
            self.concept_memory, self.letter_vec, dims=DIMS)

        # ── Priority 11: Confidence Estimator ────────────────
        self.confidence_estimator = ConfidenceEstimator()

        # ── Priority 14: Benchmark Tracker ───────────────────
        self.benchmark_tracker = BenchmarkTracker()

        # ── Priority 8: Self Reflector ────────────────────────
        self.self_reflector = SelfReflector()

    # ── Projector helpers ────────────────────────────────────────
    def _train_projector(self):
        try:
            vecs = [c.vector for c in self.concept_memory.concepts.values()
                    if hasattr(c,'vector') and c.vector is not None]
            if vecs:
                self.projector.train_from_concepts(vecs)
            else:
                self.projector._random_projection()
        except Exception as e:
            print(f"[PROJECTOR WARN] {e}")
            self.projector._random_projection()

    def project_concept(self, concept_name: str) -> np.ndarray:
        concept_name = concept_name.lower()
        cached = self.projection_cache.get(concept_name)
        if cached is not None: return cached
        concept = self.concept_memory.concepts.get(concept_name)
        if concept is None or concept.vector is None: return np.zeros(12)
        projected = self.projector.project_128_to_12(concept.vector)
        self.projection_cache.set(concept_name, projected)
        return projected

    def get_concept_vector_from_text(self, text: str) -> np.ndarray:
        try:
            features = self.ontology.get_features(text.lower())
            p_fids   = [self.feature_registry.register(f) for f in features]
            s_fids   = [self.feature_registry.register(f) for f in features]
            temp     = DNAConcept(text.lower(), p_fids, s_fids,
                                  self.feature_registry, self.letter_vec, dims=128)
            return temp.vector if temp.vector is not None else np.zeros(128)
        except Exception:
            return np.zeros(128)

    def get_skill_vector_from_text(self, text: str) -> np.ndarray:
        try:
            seqs = [self.letter_vec.get(ch) for ch in text.lower() if ch in self.letter_vec.vec]
            emb128 = np.mean(np.array(seqs), axis=0) if seqs else np.zeros(128)
            emb12  = self.projector.project_128_to_12(emb128)
            return get_adapter()(emb12, np.zeros(12))
        except Exception:
            return np.zeros(12)

    # ================================================================
    #   ░░░░  PRIORITY 1+2+4+7+8+11+12+13: FULL PIPELINE  ░░░░
    # ================================================================
    def full_pipeline_reasoning(self, query: str, session_id: str = "",
                                max_hops: int = 3) -> dict:
        """
        Complete DNA reasoning pipeline:
        Query → DNA Encode → Activate (Attention) → Working Memory
              → Multi-hop Reasoning → Confidence → Answer
              → Reward RL → Episode Storage → Self-Reflection → Clear WM
        """
        t_start = time.time()

        # ── Step 1: DNA Encoder ───────────────────────────────
        query_vec = self.dna_activation_engine.encode_query(query)

        # ── Step 2 (Priority 6): Check Episodic Memory ────────
        similar_eps = self.episodic_memory.retrieve_similar(query_vec, top_k=1)

        # ── Step 3 (Priority 4): DNA Attention — activation scores ─
        activation_scores = self.dna_activation_engine.compute_activation_scores(
            query_vec, top_k=600)

        # Episodic boost: if we have a similar past episode, boost its concepts
        if similar_eps:
            for c in similar_eps[0].activated_concepts[:15]:
                if c in activation_scores:
                    activation_scores[c] = min(1.0, activation_scores[c] + 0.2)

        # ── Step 4: Hebbian Activation ────────────────────────
        activation_scores = self.dna_activation_engine.hebbian_activate(activation_scores)

        # ── Step 5 (Priority 2): Build Working Memory ─────────
        wm = WorkingMemory(max_nodes=WORKING_MEMORY_SIZE)
        wm.session_id = session_id
        wm.load(activation_scores, self.concept_memory)

        # ── Step 6 (Priority 4): DNA Attention filter — only expand high-activation ─
        entry_nodes = [(n, s) for n, s in activation_scores.items()
                       if s > ATTENTION_THRESHOLD]
        entry_nodes.sort(key=lambda x: x[1], reverse=True)
        entry_nodes = entry_nodes[:25]

        # ── Step 7 (Priority 12): Multi-hop with path tracking ─
        reasoning_results: Dict[str, float] = {}
        for node, score in entry_nodes:
            wm.add_to_path("activate", node, score, "attention")
            hop_res = self.reasoning_engine.multi_hop_reasoning(node, max_hops=max_hops)
            # Filter: only concepts in working memory or very high-scoring
            for k, v in hop_res.items():
                if k in wm.nodes or v > 0.8:
                    weighted = v * score
                    if k not in reasoning_results or weighted > reasoning_results[k]:
                        reasoning_results[k] = weighted
                        wm.add_to_path("reason", k, weighted, "multi_hop")

        # ── Step 8 (Priority 11): Confidence Evaluation ───────
        confidence = self.confidence_estimator.estimate(
            reasoning_results, wm.reasoning_path, wm)

        # If confidence is low → search deeper (Priority 11)
        if confidence["confidence"] < LOW_CONFIDENCE_THRESHOLD:
            low_nodes = [(n, s) for n, s in activation_scores.items()
                         if 0.1 < s <= ATTENTION_THRESHOLD]
            low_nodes.sort(key=lambda x: x[1], reverse=True)
            for node, score in low_nodes[:8]:
                hop_res = self.reasoning_engine.multi_hop_reasoning(
                    node, max_hops=max_hops+1)
                for k, v in hop_res.items():
                    weighted = v * score * 0.6
                    if k not in reasoning_results or weighted > reasoning_results[k]:
                        reasoning_results[k] = weighted
            confidence = self.confidence_estimator.estimate(
                reasoning_results, wm.reasoning_path, wm)

        # ── Step 9: Rank answer concepts ──────────────────────
        top_pairs  = sorted(reasoning_results.items(), key=lambda x: x[1], reverse=True)[:12]
        answer_concepts = [c for c, _ in top_pairs]

        # ── Step 10 (Priority 12): Build explainable reasoning path ─
        explanation = self._build_explanation(query, wm.reasoning_path, top_pairs)

        t_end = time.time()
        reasoning_time = t_end - t_start

        # ── Step 11 (Priority 14): Benchmark ──────────────────
        self.benchmark_tracker.record(
            query=query,
            retrieval_accurate=len(answer_concepts) > 0,
            answer_accurate=confidence["confidence"] > 0.5,
            nodes_visited=len(wm.nodes),
            reasoning_time=reasoning_time,
            memory_usage=len(self.concept_memory.concepts),
            reward=confidence["reward_estimate"])

        # ── Step 12 (Priority 8): Self-Reflection ─────────────
        reflection = self.self_reflector.reflect(
            query, wm.reasoning_path, reasoning_results, confidence["reward_estimate"])

        # Capture WM data before clearing
        captured_path  = list(wm.reasoning_path)
        captured_nodes = dict(wm.nodes)

        # ── Step 13 (Priority 3): DNA Reinforcement ───────────
        self.reinforce_pipeline(query, query_vec, answer_concepts,
                                confidence["reward_estimate"])

        # ── Step 14 (Priority 6): Store Episode ───────────────
        wm_for_store = WorkingMemory()
        wm_for_store.reasoning_path = captured_path
        wm_for_store.nodes          = captured_nodes
        self.episodic_memory.store(
            query, query_vec, wm_for_store,
            answer_concepts, confidence["reward_estimate"])

        # ── Step 15 (Priority 2): Clear Working Memory ────────
        wm.clear()

        return {
            "query":               query,
            "answer_concepts":     answer_concepts,
            "top_activations":     {k: round(v,4) for k,v in top_pairs},
            "confidence":          confidence,
            "reasoning_path":      captured_path[:20],
            "explanation":         explanation,
            "reflection":          reflection,
            "reasoning_time_ms":   round(reasoning_time*1000, 2),
            "episode_reused":      len(similar_eps) > 0,
            "nodes_in_wm":         len(captured_nodes),
        }

    # ── Priority 12: Build Explainable Reasoning Trace ──────────
    def _build_explanation(self, query: str, path: List[dict],
                           top_concepts: List[Tuple]) -> str:
        lines = [f"Query: {query}"]
        seen: set = set()
        chain: List[str] = []
        for step in path[:15]:
            n = step.get("node","")
            if n and n not in seen:
                seen.add(n); chain.append(n)
        if chain:
            lines.append("Reasoning Chain: " + " → ".join(chain[:8]))
        if top_concepts:
            lines.append("Top Concepts: " +
                         ", ".join(f"{c} ({s:.2f})" for c, s in top_concepts[:5]))
        return "\n".join(lines)

    # ── Priority 3: DNA Reinforcement ───────────────────────────
    def reinforce_pipeline(self, query: str, query_vector: np.ndarray,
                           activated_concepts: List[str], reward: float):
        """
        Reward → DNA Memory (importance) → Relationship Hebbian Strength
                → Global Synapse update → Traversal Policy
        """
        if reward <= 0: return

        # 1. Update concept importance (DNA Memory)
        for cname in activated_concepts[:20]:
            if cname in self.concept_memory.concepts:
                c = self.concept_memory.concepts[cname]
                c.importance = min(10.0, c.importance + reward * 0.08)

        # 2. Strengthen co-activated relationships (Hebbian Strength)
        for i, ca in enumerate(activated_concepts[:10]):
            for cb in activated_concepts[i+1:min(i+5, len(activated_concepts))]:
                if (ca in self.concept_memory.concepts and
                        cb in self.concept_memory.concepts):
                    self.concept_memory.strengthen_relationship(ca, cb, reward*0.04)

        # 3. Update Global Synapse with reward signal
        self.concept_memory.synapse_step += 1
        sine_lr = 0.008 * (0.5 + 0.5*math.sin(self.concept_memory.synapse_step/15.0))
        for ca in activated_concepts[:6]:
            if ca not in self.concept_memory.concepts: continue
            pa = self.concept_memory._get_letter_probs(
                self.concept_memory.concepts[ca].vector)
            for cb in activated_concepts[:6]:
                if cb == ca or cb not in self.concept_memory.concepts: continue
                pb = self.concept_memory._get_letter_probs(
                    self.concept_memory.concepts[cb].vector)
                co = np.outer(pa, pb)
                self.concept_memory.global_synapse += sine_lr * reward * co
                self.concept_memory.global_synapse = np.clip(
                    self.concept_memory.global_synapse, 0.01, 0.99)

    # ── Existing methods ─────────────────────────────────────────
    def _seed_initial_concepts(self):
        if len(self.concept_memory.concepts) > 50: return
        seed_data = [
            ("login issue",      ["authentication","password","access"],   "Technology", 2.0),
            ("billing",          ["payment","invoice","refund"],            "Finance",    2.0),
            ("slow performance", ["latency","response time","optimization"],"Technology", 1.5),
            ("crash",            ["bug","error","unstable"],                "Technology", 1.5),
            ("feature request",  ["enhancement","new functionality"],       "General",    1.0),
            ("account locked",   ["security","blocked","verification"],     "Technology", 1.8),
        ]
        for concept, features, domain, importance in seed_data:
            pf = [self.feature_registry.register(f) for f in features]
            sf = [self.feature_registry.register(f) for f in features]
            self.concept_memory.register(concept, pf, sf, importance=importance, domain=domain)
            ppf = [self.predictive_feature_registry.register(f) for f in features]
            psf = [self.predictive_feature_registry.register(f) for f in features]
            self.predictive_concept_memory.register(concept, ppf, psf,
                                                    importance=importance, domain=domain)
        rels = [("login issue","account locked","CAUSES",0.9),
                ("billing","refund","RELATED",0.8),
                ("slow performance","crash","CAUSES",0.7),
                ("feature request","enhancement","IS_A",0.9),
                ("login issue","authentication","IS_A",0.95),
                ("account locked","security","HAS",0.85)]
        for a, b, color, w in rels:
            self.concept_memory.add_weighted_relationship(a, b, weight=w, color=color)
            self.predictive_concept_memory.add_weighted_relationship(a, b, weight=w, color=color)
        print("[KNOWLEDGE] Loading offline builtin knowledge graph...")
        n_loaded = self.knowledge_loader.load_builtin_knowledge()
        for cd in self.knowledge_loader.concepts:
            pf = [self.feature_registry.register(f) for f in cd.features]
            sf = [self.feature_registry.register(f) for f in cd.features]
            self.concept_memory.register(cd.name.lower(), pf, sf,
                                         importance=cd.importance, domain=cd.domain)
            ppf = [self.predictive_feature_registry.register(f) for f in cd.features]
            psf = [self.predictive_feature_registry.register(f) for f in cd.features]
            self.predictive_concept_memory.register(cd.name.lower(), ppf, psf,
                                                    importance=cd.importance, domain=cd.domain)
        for cd in self.knowledge_loader.concepts:
            for target, color, weight in cd.relationships:
                self.concept_memory.add_weighted_relationship(
                    cd.name.lower(), target.lower(), weight=weight, color=color)
                self.predictive_concept_memory.add_weighted_relationship(
                    cd.name.lower(), target.lower(), weight=weight, color=color)
        print(f"[KNOWLEDGE] Injected {n_loaded} offline concepts.")

    def extract_and_learn(self, text: str, namespace: str = "") -> dict:
        ns    = namespace
        words = [w for w in text.lower().split() if len(w) > 1 and w not in STOP_WORDS]
        unique= list(set(words))[:15]
        extracted = []
        anchor_name = f"{ns}identity" if ns else None
        if anchor_name and anchor_name not in self.concept_memory.concepts:
            features = self.ontology.get_features("identity")
            pf = [self.feature_registry.register(f) for f in features]
            sf = [self.feature_registry.register(f) for f in features]
            self.concept_memory.register(anchor_name, pf, sf,
                                         importance=2.0, domain="user_identity")
        for kw in unique:
            name = f"{ns}{kw}" if ns else kw
            if name not in self.concept_memory.concepts:
                features = self.ontology.get_features(kw)
                pf = [self.feature_registry.register(f) for f in features]
                sf = [self.feature_registry.register(f) for f in features]
                self.concept_memory.register(name, pf, sf, importance=1.0, domain="general")
                ppf = [self.predictive_feature_registry.register(f) for f in features]
                psf = [self.predictive_feature_registry.register(f) for f in features]
                self.predictive_concept_memory.register(name, ppf, psf,
                                                        importance=1.0, domain="general")
            extracted.append(name)
            if anchor_name and name != anchor_name:
                self.concept_memory.add_weighted_relationship(
                    anchor_name, name, weight=1.0, color="HAS")
                self.concept_memory.add_weighted_relationship(
                    name, anchor_name, weight=1.0, color="PART_OF")
        for i in range(len(extracted)):
            for j in range(i+1, min(i+4, len(extracted))):
                color = self.concept_memory._infer_relationship_color(
                    text, extracted[i], extracted[j])
                if (extracted[i] in self.concept_memory.concepts and
                        extracted[j] in self.concept_memory.concepts):
                    self.concept_memory.add_weighted_relationship(
                        extracted[i], extracted[j], weight=0.8, color=color)
                if (extracted[i] in self.predictive_concept_memory.concepts and
                        extracted[j] in self.predictive_concept_memory.concepts):
                    self.predictive_concept_memory.add_weighted_relationship(
                        extracted[i], extracted[j], weight=0.8, color=color)
        return {"extracted_concepts": extracted}

    # Grader delegates
    def task_easy(self, input_text: str) -> float:   return task_easy(input_text)
    def task_medium(self, input_text: str) -> float: return task_medium(input_text)
    def task_hard(self, input_text: str) -> float:   return task_hard(input_text)

    def generate_sentence(self, concept: str) -> str:
        return self.reasoning_engine.generate_sentence(concept)

    def get_global_context(self) -> dict:
        return self.concept_memory.get_global_context()

    def search_by_essence(self, concept: str, top_k: int = 5):
        if concept not in self.concept_memory.concepts: return []
        return self.concept_memory.partitioned_search(
            self.concept_memory.concepts[concept].vector, top_k, partition=ESSENCE_DIMS)

    def search_by_identity(self, concept: str, top_k: int = 5):
        if concept not in self.concept_memory.concepts: return []
        return self.concept_memory.partitioned_search(
            self.concept_memory.concepts[concept].vector, top_k, partition=IDENTITY_DIMS)

    def calculate(self, expression: str) -> Dict:
        return self.reasoning_engine.calculate(expression)

    def execute_instruction(self, operator: str, a, b) -> Dict:
        return self.reasoning_engine.execute_instruction(operator, a, b)

    def evaluate_rule(self, condition: Dict, action: str) -> Dict:
        return self.reasoning_engine.evaluate_rule(condition, action)

    # OpenEnv interface
    def reset(self) -> str:
        self.current_task   = self._generate_dynamic_task()
        self.current_step   = 0
        self.episode_reward = 0.0
        self.done           = False
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.concept_memory.extract_and_link(
                    self.current_task["input"], self.ontology))
            else:
                loop.run_until_complete(self.concept_memory.extract_and_link(
                    self.current_task["input"], self.ontology))
        except Exception as e:
            print(f"[DEBUG] Background extraction failed: {e}")
        return self.current_task["input"]

    def step(self, action: str) -> Tuple[str,float,bool,dict]:
        if self.done: raise RuntimeError("Episode done. Call reset() first.")
        step_type = ["identify","relate","answer"][self.current_step]
        reward    = 0.0
        if step_type == "identify":
            reward = self._grade_identification(action, self.current_task["expected_concept"])
            if reward >= 0.3: self.current_step = 1
        elif step_type == "relate":
            reward = self._grade_relation(action, self.current_task["expected_relation"])
            if reward >= 0.3: self.current_step = 2
        else:
            reward = self._grade_answer(action, self.current_task["expected_answer"])
            self.done = True; self.current_step = 3
        self.episode_reward += reward
        return (self.current_task["input"] if not self.done else ""), reward, self.done, {
            "task_type": self.current_task["type"],
            "step": step_type, "step_reward": reward,
            "total_reward": self.episode_reward}

    def state(self) -> dict:
        if not self.current_task: return {}
        return {"task": self.current_task, "step": self.current_step,
                "step_name": ["identify","relate","answer"][self.current_step]
                              if self.current_step < 3 else "done"}

    def _generate_dynamic_task(self) -> dict:
        concepts = list(self.concept_memory.concepts.keys())
        if not concepts:
            return {"type":"easy","input":"My phone won't turn on",
                    "expected_concept":"hardware failure","expected_relation":"battery issue",
                    "expected_answer":"replace battery or check power"}
        base    = random.choice(concepts)
        related = list(self.concept_memory.relationships.get(base, set()))
        if not related:
            sim     = self.concept_memory.search(self.concept_memory.concepts[base].vector, top_k=3)
            related = [r[0] for r in sim if r[0] != base]
        exp_rel = related[0] if related else "related issue"
        res     = self.reasoning_engine.multi_hop_reasoning(base, max_hops=2)
        possible= [c for c in res if c != base and c != exp_rel]
        exp_ans = possible[0] if possible else random.choice(concepts)
        tmpl    = {"easy":  [f"I'm having trouble with {base}.",f"{base} not working."],
                   "medium":[f"User reports {base} persists after restart.",
                              f"Ticket: {base} affecting workflow."],
                   "hard":  [f"Critical: {base} causing system failure.",
                              f"Customer says {base} is blocking all operations."]}
        diff = random.choice(["easy","medium","hard"])
        return {"type":diff,"input":random.choice(tmpl[diff]),
                "expected_concept":base,"expected_relation":exp_rel,"expected_answer":exp_ans}

    def _grade_identification(self, action: str, expected: str) -> float:
        a, e = action.lower().strip(), expected.lower()
        if a == e: return 0.95
        if e in a or a in e: return 0.75
        if any(w in a for w in e.split()): return 0.35
        return 0.05

    def _grade_relation(self, action: str, expected: str) -> float:
        a, e = action.lower().strip(), expected.lower()
        if a == e: return 0.95
        if e in a: return 0.75
        if self.current_task and a in self.concept_memory.relationships.get(
                self.current_task["expected_concept"], set()): return 0.55
        return 0.05

    def _grade_answer(self, action: str, expected: str) -> float:
        a, e = action.lower().strip(), expected.lower()
        if a == e: return 0.95
        if e in a: return 0.75
        if any(w in a for w in e.split()): return 0.35
        return 0.05

    def close(self):
        if self.trainer: self.trainer.stop()
        if self.concept_memory.pending_updates:
            self.concept_memory.process_batch()
        PersistenceManager.save_all(
            self.concept_memory, self.feature_registry, self.letter_vec,
            self.episodic_memory)


# ============================================================
# FastAPI
# ============================================================
_api_env:   Optional[KnowledgeGraphEnv] = None
_env_ready: bool = False


def _boot_env():
    global _api_env, _env_ready
    try:
        _api_env   = KnowledgeGraphEnv(start_trainer=True)
        _env_ready = True
    except Exception as e:
        print(f"[WARN] KnowledgeGraphEnv boot failed: {e}")
        _env_ready = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_boot_env, daemon=True).start()
    yield
    if _api_env is not None:
        _api_env.close()


app = FastAPI(
    title="Knowledge Graph Environment",
    description="DNA-inspired self-evolving knowledge graph with Activation Engine, "
                "Working Memory, Episodic Memory, Confidence, Explainable Reasoning, "
                "DNA Reinforcement, and Benchmarking.",
    version="5.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

# ── Pydantic models ──────────────────────────────────────────────
class ResetResponse(BaseModel):
    observation: str

class StepRequest(BaseModel):
    action: str

class StepResponse(BaseModel):
    observation: str; reward: float; done: bool; info: dict

class StateResponse(BaseModel):
    state: dict

class TaskResponse(BaseModel):
    tasks: List[str]

class GradeRequest(BaseModel):
    task_id: str; input_text: str

class GradeResponse(BaseModel):
    score: float

class SentenceRequest(BaseModel):
    concept: str

class SentenceResponse(BaseModel):
    sentence: str

class GlobalContextResponse(BaseModel):
    global_centroid: Optional[List[float]]; anchors: List[str]; total_concepts: int

class PartitionedSearchRequest(BaseModel):
    concept: str; top_k: int = 5

class SearchResult(BaseModel):
    name: str; score: float

class PartitionedSearchResponse(BaseModel):
    results: List[SearchResult]

class AddRelationshipRequest(BaseModel):
    concept_a: str; concept_b: str; weight: float = 1.0; color: str = "RELATED"

class AddRelationshipResponse(BaseModel):
    status: str

class CalculateRequest(BaseModel):
    expression: str

class InstructionRequest(BaseModel):
    operator: str; a: Union[str,float]; b: Union[str,float]

class RuleRequest(BaseModel):
    condition: Dict; action: str

# ── Priority 1+2+4+7+8+11+12+14 endpoint model ──────────────────
class ReasonRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    max_hops: int = 3

class ReasonResponse(BaseModel):
    query: str
    answer_concepts: List[str]
    top_activations: dict
    confidence: dict
    reasoning_path: List[dict]
    explanation: str
    reflection: dict
    reasoning_time_ms: float
    episode_reused: bool
    nodes_in_wm: int

# ── Priority 3: Reinforce endpoint model ────────────────────────
class ReinforceRequest(BaseModel):
    concepts: List[str]
    reward: float
    query: Optional[str] = ""

# ── Agent models ─────────────────────────────────────────────────
class AgentRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    provider: Optional[str]   = None
    api_key: Optional[str]    = None
    model: Optional[str]      = None
    custom_base_url: Optional[str] = None

class AgentResponse(BaseModel):
    response: str; tool_calls: list; provider_used: Optional[str] = None

class OrchestrateRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    provider: Optional[str]   = None
    api_key: Optional[str]    = None
    model: Optional[str]      = None
    custom_base_url: Optional[str] = None

class OrchestrateResponse(BaseModel):
    learned_concepts: List[str]; essence_matches: List[dict]; identity_matches: List[dict]
    predictions: dict; generated_sentence: Optional[str] = None
    skill_vector: List[float]; projection_status: str
    ai_summary: Optional[str] = None; provider_used: str
    total_concepts_in_graph: int; total_relationships: int

class KnowledgeLoadRequest(BaseModel):
    source: str = "technology"

class RouterReinforceRequest(BaseModel):
    reward: float; session_id: Optional[str] = None

# ── Helpers ──────────────────────────────────────────────────────
def _build_client(req: AgentRequest):
    if not req.provider or req.provider == "local_graph" or not req.api_key:
        return None, None, "local_graph"
    info = PROVIDER_REGISTRY.get(req.provider)
    if not info:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {req.provider}")
    base_url = req.custom_base_url if req.provider == "custom" else info["base_url"]
    if not base_url:
        raise HTTPException(status_code=400, detail="Custom provider requires a base_url.")
    model = req.model or info["default_model"]
    if req.provider == "groq" and model == "llama-3.1-70b-versatile":
        model = "llama-3.3-70b-versatile"
    return openai.OpenAI(base_url=base_url, api_key=req.api_key), model, req.provider


# ============================================================
# EXISTING ENDPOINTS (unchanged)
# ============================================================
@app.get("/ping")
async def ping(): return {"status": "ok"}

@app.get("/health")
async def health():
    if _api_env is None: return {"status": "initializing"}
    return {"status":"healthy",
            "total_concepts": len(_api_env.concept_memory.concepts),
            "total_relationships": sum(len(v) for v in _api_env.concept_memory.weighted_relationships.values())}

@app.get("/tasks", response_model=TaskResponse)
async def tasks_endpoint(): return TaskResponse(tasks=TASKS)

@app.post("/grade", response_model=GradeResponse)
async def grade_endpoint(req: GradeRequest):
    if req.task_id not in GRADERS:
        raise HTTPException(status_code=404, detail=f"Task '{req.task_id}' not found.")
    return GradeResponse(score=GRADERS[req.task_id](req.input_text))

@app.post("/reset", response_model=ResetResponse)
async def reset_endpoint():
    if _api_env is None:
        return ResetResponse(observation="Environment initializing, please retry.")
    return ResetResponse(observation=_api_env.reset())

@app.post("/step", response_model=StepResponse)
async def step_endpoint(req: StepRequest):
    if _api_env is None: raise HTTPException(status_code=503)
    obs, reward, done, info = _api_env.step(req.action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)

@app.get("/state", response_model=StateResponse)
async def state_endpoint():
    if _api_env is None: raise HTTPException(status_code=503)
    return StateResponse(state=_api_env.state())

@app.post("/sentence", response_model=SentenceResponse)
async def generate_sentence_endpoint(req: SentenceRequest):
    if _api_env is None: raise HTTPException(status_code=503)
    return SentenceResponse(sentence=_api_env.generate_sentence(req.concept))

@app.get("/global-context", response_model=GlobalContextResponse)
async def global_context_endpoint():
    if _api_env is None: raise HTTPException(status_code=503)
    ctx = _api_env.get_global_context()
    return GlobalContextResponse(global_centroid=ctx["global_centroid"],
                                 anchors=ctx["anchors"], total_concepts=ctx["total_concepts"])

@app.post("/search/essence", response_model=PartitionedSearchResponse)
async def search_essence_endpoint(req: PartitionedSearchRequest):
    if _api_env is None: raise HTTPException(status_code=503)
    return PartitionedSearchResponse(
        results=[SearchResult(name=r[0],score=r[1])
                 for r in _api_env.search_by_essence(req.concept, req.top_k)])

@app.post("/search/identity", response_model=PartitionedSearchResponse)
async def search_identity_endpoint(req: PartitionedSearchRequest):
    if _api_env is None: raise HTTPException(status_code=503)
    return PartitionedSearchResponse(
        results=[SearchResult(name=r[0],score=r[1])
                 for r in _api_env.search_by_identity(req.concept, req.top_k)])

@app.post("/relationships/add", response_model=AddRelationshipResponse)
async def add_relationship_endpoint(req: AddRelationshipRequest):
    if _api_env is None: raise HTTPException(status_code=503)
    for c in [req.concept_a, req.concept_b]:
        if c.lower() not in _api_env.concept_memory.concepts:
            pf = [_api_env.feature_registry.register(c)]
            sf = [_api_env.feature_registry.register(c)]
            _api_env.concept_memory.register(c, pf, sf)
    _api_env.concept_memory.add_weighted_relationship(
        req.concept_a, req.concept_b, weight=req.weight, color=req.color)
    return AddRelationshipResponse(status="relationship_added")

@app.get("/concepts")
async def list_concepts_endpoint(limit: int = 100):
    if _api_env is None: raise HTTPException(status_code=503)
    return {"concepts": [{"name":n,"importance":c.importance,"domain":c.domain,
                           "relationship_count":len(_api_env.concept_memory.weighted_relationships.get(n,{}))}
                         for n, c in list(_api_env.concept_memory.concepts.items())[:limit]],
            "total": len(_api_env.concept_memory.concepts)}

@app.get("/graph")
async def get_graph_endpoint(session_id: Optional[str] = None):
    if _api_env is None: raise HTTPException(status_code=503)
    ns    = f"user_{session_id[:8]}:" if session_id else ""
    nodes = []; links = []
    for name, concept in _api_env.concept_memory.concepts.items():
        if ns and not name.startswith(ns): continue
        display = name.replace(ns,"") if ns else name
        nodes.append({"id":display,"domain":getattr(concept,"domain","general"),
                      "importance":getattr(concept,"importance",1.0)})
    for source, targets in _api_env.concept_memory.weighted_relationships.items():
        if ns and not source.startswith(ns): continue
        ds = source.replace(ns,"") if ns else source
        for target, rel in targets.items():
            if ns and not target.startswith(ns): continue
            dt = target.replace(ns,"") if ns else target
            links.append({"source":ds,"target":dt,
                          "weight":getattr(rel,"weight",1.0),
                          "color":getattr(rel,"color","RELATED")})
    return {"nodes":nodes,"links":links}

@app.post("/calculate")
async def calculate_endpoint(req: CalculateRequest):
    if _api_env is None: raise HTTPException(status_code=503)
    return _api_env.calculate(req.expression)

@app.post("/instruction/execute")
async def execute_instruction_endpoint(req: InstructionRequest):
    if _api_env is None: raise HTTPException(status_code=503)
    return _api_env.execute_instruction(req.operator, req.a, req.b)

@app.post("/evaluate")
async def evaluate_rule_endpoint(req: RuleRequest):
    if _api_env is None: raise HTTPException(status_code=503)
    return _api_env.evaluate_rule(req.condition, req.action)

@app.get("/predict")
def predict_fuzzy(concept: str):
    if not _env_ready or _api_env is None: raise HTTPException(status_code=503)
    res = _api_env.predictive_reasoning.multi_hop_reasoning(concept.lower(), max_hops=2)
    return {"fuzzy_predictions": res}

@app.get("/providers")
async def list_providers():
    return {"providers": [{"id":k,"name":v["name"],"default_model":v["default_model"],
                           "free_tier":v["free_tier"],"get_key_url":v["get_key_url"],
                           "requires_base_url":k=="custom"}
                          for k, v in PROVIDER_REGISTRY.items()]}

@app.get("/projection/info")
async def projection_info_endpoint():
    if _api_env is None: raise HTTPException(status_code=503)
    return {"projection":_api_env.projector.explain(),
            "cache_size":len(_api_env.projection_cache.cache),
            "concepts_in_graph":len(_api_env.concept_memory.concepts)}

@app.post("/projection/retrain")
async def projection_retrain_endpoint():
    if _api_env is None: raise HTTPException(status_code=503)
    _api_env.projection_cache.clear()
    _api_env._train_projector()
    return {"status":"ok","message":"Projection retrained."}

@app.post("/knowledge/load")
async def load_knowledge_endpoint(req: KnowledgeLoadRequest):
    if _api_env is None: raise HTTPException(status_code=503)
    loader = _api_env.knowledge_loader
    n = loader._load_builtin(req.source)
    if n == 0 and req.source not in loader.builtin_sources:
        raise HTTPException(status_code=404,
                            detail=f"Unknown source '{req.source}'. "
                                   f"Available: {list(loader.builtin_sources.keys())}")
    added = []
    for cd in loader.concepts:
        pf=[_api_env.feature_registry.register(f) for f in cd.features]
        sf=[_api_env.feature_registry.register(f) for f in cd.features]
        _api_env.concept_memory.register(cd.name, pf, sf,
                                         importance=cd.importance, domain=cd.domain)
        added.append(cd.name)
        ppf=[_api_env.predictive_feature_registry.register(f) for f in cd.features]
        psf=[_api_env.predictive_feature_registry.register(f) for f in cd.features]
        _api_env.predictive_concept_memory.register(cd.name, ppf, psf,
                                                    importance=cd.importance, domain=cd.domain)
    _api_env._train_projector(); _api_env.projection_cache.clear()
    return {"status":"ok","source":req.source,
            "concepts_added":len(added),"concept_names":added[:20],
            "total_in_graph":len(_api_env.concept_memory.concepts)}


# ============================================================
#   ░░░░  NEW PRIORITY ENDPOINTS  ░░░░
# ============================================================

# ── Priority 1+2+4+7+8+11+12+13: Full DNA Reasoning Pipeline ───
@app.post("/reason", response_model=ReasonResponse,
          summary="🧬 Full DNA Reasoning Pipeline",
          description="DNA Encode → Activation (Attention) → Working Memory → "
                      "Multi-hop → Confidence → Explainable Answer → RL Update")
async def reason_endpoint(req: ReasonRequest):
    if not _env_ready or _api_env is None:
        raise HTTPException(status_code=503, detail="Environment not ready")
    result = _api_env.full_pipeline_reasoning(
        req.query, session_id=req.session_id or "", max_hops=req.max_hops)
    return ReasonResponse(**result)


# ── Priority 14: Benchmarking ───────────────────────────────────
@app.get("/benchmark",
         summary="📊 Benchmark Metrics",
         description="Retrieval accuracy, answer accuracy, nodes visited, "
                     "reasoning time, memory usage, reward over time.")
async def benchmark_endpoint():
    if _api_env is None: raise HTTPException(status_code=503)
    return _api_env.benchmark_tracker.summary()

@app.get("/benchmark/history")
async def benchmark_history_endpoint(limit: int = 50):
    if _api_env is None: raise HTTPException(status_code=503)
    return {"history": _api_env.benchmark_tracker.metrics[-limit:],
            "total": len(_api_env.benchmark_tracker.metrics)}


# ── Priority 6: Episodic Memory ─────────────────────────────────
@app.get("/episodes",
         summary="🧠 Episodic Memory",
         description="List stored reasoning episodes, sorted by reward.")
async def episodes_endpoint(limit: int = 20):
    if _api_env is None: raise HTTPException(status_code=503)
    eps = sorted(_api_env.episodic_memory.episodes, key=lambda e: e.reward, reverse=True)[:limit]
    return {"total": len(_api_env.episodic_memory.episodes),
            "episodes": [e.to_dict() for e in eps]}

@app.post("/episodes/search")
async def episode_search_endpoint(query: str, top_k: int = 3):
    """Find episodes similar to the given query."""
    if _api_env is None: raise HTTPException(status_code=503)
    q_vec = _api_env.dna_activation_engine.encode_query(query)
    eps   = _api_env.episodic_memory.retrieve_similar(q_vec, top_k=top_k)
    return {"query": query, "similar_episodes": [e.to_dict() for e in eps]}


# ── Priority 3: DNA Reinforcement ───────────────────────────────
@app.post("/reinforce",
          summary="⚡ DNA Reinforcement",
          description="Manually propagate a reward into DNA Memory, "
                      "Hebbian Strength, and Global Synapse.")
async def reinforce_endpoint(req: ReinforceRequest):
    if _api_env is None: raise HTTPException(status_code=503)
    q_vec = _api_env.dna_activation_engine.encode_query(req.query or "manual")
    _api_env.reinforce_pipeline(req.query or "manual", q_vec, req.concepts, req.reward)
    return {"status": "reinforced", "reward": req.reward,
            "concepts_updated": len(req.concepts)}


# ── Priority 8: Self-Reflection History ─────────────────────────
@app.get("/reflect",
         summary="🪞 Self-Reflection Log",
         description="View past self-reflection entries and suggested improvements.")
async def reflect_endpoint(limit: int = 30):
    if _api_env is None: raise HTTPException(status_code=503)
    return {"total": len(_api_env.self_reflector.history),
            "reflections": _api_env.self_reflector.history[-limit:]}


# ── Priority 4: DNA Attention scores ─────────────────────────────
@app.get("/attention",
         summary="👁 DNA Attention Scores",
         description="Return activation scores for all nodes given a query. "
                     "Only high-scoring nodes will be expanded in reasoning.")
async def attention_endpoint(query: str, top_k: int = 20):
    if _api_env is None: raise HTTPException(status_code=503)
    q_vec  = _api_env.dna_activation_engine.encode_query(query)
    scores = _api_env.dna_activation_engine.compute_activation_scores(q_vec, top_k=top_k)
    scores = _api_env.dna_activation_engine.hebbian_activate(scores)
    return {"query": query,
            "attention": sorted(scores.items(), key=lambda x: x[1], reverse=True),
            "threshold": ATTENTION_THRESHOLD}


# ── /agent endpoint (enhanced with DNA pipeline results) ─────────
@app.post("/agent", response_model=AgentResponse)
async def agent_endpoint(req: AgentRequest):
    if not _env_ready or _api_env is None:
        raise HTTPException(status_code=503, detail="Environment not ready")
    client, model_name, provider_used = _build_client(req)
    ns = f"user_{req.session_id[:8]}:" if req.session_id else ""
    tools = [
        {"type":"function","function":{
            "name":"search_graph",
            "description":"Searches the vector graph for a concept to find related concepts.",
            "parameters":{"type":"object","properties":{
                "concept":{"type":"string","description":"The concept to search for"}},
                "required":["concept"]}}},
        {"type":"function","function":{
            "name":"extract_and_learn",
            "description":"Parses text, extracts concepts, saves them to graph memory.",
            "parameters":{"type":"object","properties":{
                "text":{"type":"string","description":"Raw text to learn from"}},
                "required":["text"]}}}
    ]
    skill_vector     = _api_env.get_skill_vector_from_text(req.message)
    skill_vector_str = "[" + ", ".join(f"{x:.2f}" for x in skill_vector) + "]"
    messages = [
        {"role":"system","content":
            f"You are the autonomous controller of a Vector Knowledge Graph. "
            f"Use search_graph for questions, extract_and_learn for new knowledge. "
            f"After tool results give a helpful human-readable summary. "
            f"NEVER output XML or <function> tags. "
            f"[SKILL_VECTOR]: {skill_vector_str}"},
        {"role":"user","content":req.message}
    ]
    import json

    def independent_fallback():
        learned = _api_env.extract_and_learn(req.message, ns)
        lc      = learned.get("extracted_concepts",[])
        anchor  = f"{ns}identity" if ns else None
        connected = {}
        if anchor and anchor in _api_env.concept_memory.weighted_relationships:
            for t, rel in _api_env.concept_memory.weighted_relationships[anchor].items():
                dn = t.replace(ns,"") if ns else t
                connected[dn] = {"weight":getattr(rel,"weight",1.0),
                                 "type":getattr(rel,"color","RELATED")}
        # Use full pipeline for richer results
        pipeline_result = _api_env.full_pipeline_reasoning(req.message, req.session_id or "")
        parts = ["🧠 **DNA Graph Analysis** (Independent Mode)\n"]
        if connected:
            parts.append(f"🔗 **Your Memory** ({len(connected)} nodes): "
                         f"{', '.join(list(connected.keys())[:8])}")
        if pipeline_result["answer_concepts"]:
            parts.append("🌐 **Graph Reasoning**: " +
                         ", ".join(f'{c} ({s:.2f})'
                                   for c, s in list(pipeline_result["top_activations"].items())[:6]))
        if lc:
            parts.append(f"📚 **Learned**: "
                         f"{', '.join((c.replace(ns,'') if ns else c) for c in lc[:8])}")
        conf = pipeline_result["confidence"]
        parts.append(f"📊 **Confidence**: {conf['confidence']:.0%} | "
                     f"Depth: {conf['reasoning_depth']} | "
                     f"Evidence: {conf['evidence_count']}")
        parts.append(f"\n💡 {pipeline_result['explanation']}")
        return AgentResponse(response="\n".join(parts).strip(),
                             tool_calls=[], provider_used="local_graph")

    if client is None:
        return independent_fallback()

    try:
        response = client.chat.completions.create(
            model=model_name, messages=messages, tools=tools, tool_choice="auto")
    except Exception as e:
        fallback = independent_fallback()
        fallback.response = (f"⚠️ **LLM Error ({provider_used})**: {e}\n\n---\n\n"
                             + fallback.response)
        fallback.provider_used = f"{provider_used}_failed"
        return fallback

    response_message = response.choices[0].message
    try:
        if response_message.content:
            score = GRADERS["task_medium"](response_message.content)
    except Exception:
        pass

    tool_calls     = response_message.tool_calls
    executed_tools = []

    def _run_tool(function_name, function_args):
        if function_name == "search_graph":
            raw = function_args.get("concept","").lower()
            if raw in {"who am i","my name","name","identity","me"}: raw = "identity"
            words = [w for w in raw.split() if len(w)>1 and w not in STOP_WORDS] or [raw]
            res: dict = {}
            for w in words:
                nw = f"{ns}{w}" if ns else w
                hr = _api_env.reasoning_engine.multi_hop_reasoning(nw, max_hops=2)
                if hr and ns: hr = {k.replace(ns,"") if k.startswith(ns) else k: v for k,v in hr.items()}
                res.update(hr)
            if not res:
                for w in words:
                    qv   = _api_env.get_concept_vector_from_text(w)
                    sres = _api_env.concept_memory.partitioned_search(qv, top_k=3)
                    if ns: sres = [r for r in sres if r[0].startswith(ns)]
                    for name, score in sres:
                        if score > 0.4:
                            dn   = name.replace(ns,"") if ns else name
                            rels = _api_env.concept_memory.weighted_relationships.get(name,{})
                            res[dn] = [t.replace(ns,"") if ns else t for t in rels.keys()]
            return json.dumps(res)
        elif function_name == "extract_and_learn":
            return json.dumps(_api_env.extract_and_learn(
                function_args.get("text",""), namespace=ns))
        return "Tool not found"

    # Handle text-embedded tool calls (some LLMs)
    if not tool_calls and response_message.content:
        content = response_message.content.strip()
        parsed  = None
        try:
            candidate = json.loads(content)
            if isinstance(candidate,dict) and "name" in candidate:
                parsed = candidate
        except Exception:
            pass
        if not parsed:
            import re
            m = re.search(r'\{[^{}]*"name"\s*:\s*"(?:search_graph|extract_and_learn)"[^{}]*\}',
                          content)
            if m:
                try: parsed = json.loads(m.group())
                except Exception: pass
        if parsed:
            fn   = parsed.get("name","")
            args = parsed.get("parameters", parsed.get("arguments",{}))
            if isinstance(args,str):
                try: args = json.loads(args)
                except Exception: args = {}
            result = _run_tool(fn, args)
            executed_tools.append({"tool":fn,"result":result})
            messages.append({"role":"assistant","content":content})
            messages.append({"role":"user","content":
                f"Tool '{fn}' returned: {result}\nNow give a clear human-readable summary."})
            try:
                fr = client.chat.completions.create(model=model_name, messages=messages)
                return AgentResponse(response=fr.choices[0].message.content,
                                     tool_calls=executed_tools, provider_used=provider_used)
            except Exception as e:
                return AgentResponse(response=f"Graph updated, summary failed: {e}",
                                     tool_calls=executed_tools, provider_used=provider_used)

    # Native tool_calls
    if tool_calls:
        messages.append(response_message)
        for tc in tool_calls:
            fn   = tc.function.name
            try: args = json.loads(tc.function.arguments)
            except Exception: args = {}
            result = _run_tool(fn, args)
            executed_tools.append({"tool":fn,"result":result})
            messages.append({"tool_call_id":tc.id,"role":"tool",
                             "name":fn,"content":result})
        try:
            fr = client.chat.completions.create(model=model_name, messages=messages)
            return AgentResponse(response=fr.choices[0].message.content,
                                 tool_calls=executed_tools, provider_used=provider_used)
        except Exception as e:
            return AgentResponse(response=f"Graph updated, final response failed: {e}",
                                 tool_calls=executed_tools, provider_used=provider_used)

    return AgentResponse(response=response_message.content or "No action taken.",
                         tool_calls=[], provider_used=provider_used)


# ── /orchestrate (unchanged logic, enhanced with pipeline) ───────
@app.post("/orchestrate", response_model=OrchestrateResponse)
async def orchestrate_endpoint(req: OrchestrateRequest):
    if not _env_ready or _api_env is None:
        raise HTTPException(status_code=503, detail="Environment not ready")
    ns = f"user_{req.session_id[:8]}:" if req.session_id else ""
    learned         = _api_env.extract_and_learn(req.message, ns)
    learned_concepts= learned.get("extracted_concepts",[])
    essence_results = []; identity_results = []
    for cn in learned_concepts[:3]:
        if cn in _api_env.concept_memory.concepts:
            vec = _api_env.concept_memory.concepts[cn].vector
            for nm, sc in _api_env.concept_memory.partitioned_search(vec, 5, ESSENCE_DIMS):
                if nm != cn: essence_results.append({"concept":nm,"score":round(float(sc),4)})
            for nm, sc in _api_env.concept_memory.partitioned_search(vec, 5, IDENTITY_DIMS):
                if nm != cn: identity_results.append({"concept":nm,"score":round(float(sc),4)})
    def dedup(lst):
        seen=set(); out=[]
        for r in sorted(lst, key=lambda x: x["score"], reverse=True):
            if r["concept"] not in seen: seen.add(r["concept"]); out.append(r)
        return out[:5]
    essence_results  = dedup(essence_results)
    identity_results = dedup(identity_results)
    predictions = _api_env.predictive_reasoning.multi_hop_reasoning(
        req.message.lower(), max_hops=2)
    top_concept = (essence_results[0]["concept"] if essence_results
                   else (learned_concepts[0] if learned_concepts else None))
    generated_sentence = (_api_env.reasoning_engine.generate_sentence(top_concept)
                          if top_concept and top_concept in _api_env.concept_memory.concepts
                          else None)
    skill_vector     = _api_env.get_skill_vector_from_text(req.message)
    projection_status= _api_env.projector.explain().get("status","unknown")
    agent_req = AgentRequest(message=req.message, session_id=req.session_id,
                             provider=req.provider, api_key=req.api_key,
                             model=req.model, custom_base_url=req.custom_base_url)
    client, model_name, provider_used = _build_client(agent_req)
    ai_summary = None
    if client is not None:
        try:
            ctx = "\n".join(filter(None,[
                f"Learned: {', '.join(learned_concepts[:10])}" if learned_concepts else None,
                f"Related (essence): {', '.join(r['concept'] for r in essence_results[:5])}" if essence_results else None,
                f"Predictions: {', '.join(list(predictions.keys())[:5])}" if predictions else None,
                f"Description: {generated_sentence}" if generated_sentence else None]))
            sk  = "[" + ", ".join(f"{x:.2f}" for x in skill_vector) + "]"
            lr  = client.chat.completions.create(
                model=model_name, temperature=0.7, max_tokens=500,
                messages=[{"role":"system","content":
                    f"You are the DNA Knowledge Graph assistant. Skill vector: {sk}\n\nAnalysis:\n{ctx}"},
                          {"role":"user","content":req.message}])
            ai_summary = lr.choices[0].message.content
        except Exception as e:
            ai_summary = f"[LLM synthesis failed: {e}]"
    else:
        parts = ["🧠 **DNA Graph Analysis** (Independent Mode)\n"]
        if learned_concepts: parts.append(f"📚 **Learned**: {', '.join(learned_concepts[:8])}")
        if essence_results:  parts.append(f"🔬 **Essence**: {', '.join(r['concept'] for r in essence_results[:3])}")
        if identity_results: parts.append(f"🧬 **Identity**: {', '.join(r['concept'] for r in identity_results[:3])}")
        if predictions:      parts.append(f"🔮 **Predictions**: {', '.join(list(predictions.keys())[:5])}")
        if generated_sentence: parts.append(f"💡 **Insight**: {generated_sentence}")
        ai_summary = "\n".join(parts)
    total_rels = sum(len(v) for v in _api_env.concept_memory.weighted_relationships.values())
    return OrchestrateResponse(
        learned_concepts=learned_concepts, essence_matches=essence_results,
        identity_matches=identity_results, predictions=predictions,
        generated_sentence=generated_sentence,
        skill_vector=[round(float(x),6) for x in skill_vector],
        projection_status=projection_status, ai_summary=ai_summary,
        provider_used=provider_used,
        total_concepts_in_graph=len(_api_env.concept_memory.concepts),
        total_relationships=total_rels)


@app.get("/")
async def root():
    return {
        "name":    "Knowledge Graph Environment (DNA Architecture v5)",
        "version": "5.0.0",
        "status":  "online" if _api_env else "initializing",
        "new_endpoints": [
            "POST /reason          — Full DNA pipeline: encode→activate→wm→reason→confidence→explain",
            "GET  /attention       — DNA Attention scores for any query",
            "POST /reinforce       — Manual DNA Reinforcement (reward → DNA+Hebbian+Synapse)",
            "GET  /benchmark       — Retrieval accuracy, reasoning time, reward metrics",
            "GET  /benchmark/history — Full benchmark log",
            "GET  /episodes        — Episodic memory: successful past reasoning episodes",
            "POST /episodes/search — Find episodes similar to a query",
            "GET  /reflect         — Self-reflection log with improvement suggestions",
        ],
        "existing_endpoints": [
            "POST /agent           — BYOK LLM agent with tool calling",
            "POST /orchestrate     — Single unified pipeline API",
            "GET  /graph           — Full graph for 3D visualization",
            "GET  /concepts        — List all concepts",
            "POST /calculate       — Arithmetic via Instruction DNA",
            "POST /relationships/add — Add weighted colored relationship",
            "GET  /global-context  — Global centroid and anchor nodes",
            "GET  /predict         — 12-dim fuzzy predictions",
            "GET  /health / /ping  — Health checks",
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
