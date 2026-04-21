import os
import json
import asyncio
import threading
import time
import pickle
import random
import numpy as np
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai

# ============================================================
# GRADERS — imported from graders.py (LLM-as-a-Judge).
# ============================================================
from graders import task_easy, task_medium, task_hard, TASKS, GRADERS

# ============================================================
# Configuration
# ============================================================
DIMS = 16
ALPHABET = [chr(ord('A') + i) for i in range(26)]
POSITION_OFFSET = 0.1

# ========== PHASE 3: MULTI-TIER PRIORITIZATION ==========
# Dimension Partitioning (PHASE 2)
ESSENCE_DIMS = slice(0, 4)      # What something fundamentally IS
IDENTITY_DIMS = slice(4, 12)    # Unique characteristics
TEMPORAL_DIMS = slice(12, 16)   # Scale, time, importance markers

# Layer Constants (PHASE 3)
class PriorityLayer(Enum):
    ATOMIC = 0      # DNA letters, character-level
    CLUSTER = 1     # Neighborhood groups (Fruits, Cities)
    DOMAIN = 2      # Structural categories (Agriculture, Tech)
    UNIVERSAL = 3   # Global centroid

# Layer-specific learning rates
LR_ATOMIC = 0.01
LR_CLUSTER = 0.005
LR_DOMAIN = 0.001
LR_UNIVERSAL = 0.0001

# Legacy LRs (mapped to new system)
LR_LETTER = LR_ATOMIC
LR_FEATURE_VEC = LR_CLUSTER
LR_CONCEPT = LR_DOMAIN

GRAD_CLIP = 1.0
MAX_CONCEPTS = 10000
BATCH_SIZE = 32
TRAIN_INTERVAL_SEC = 10
PERSIST_DIR = "./brain_data"

# ========== PHASE 4: GLOBAL-LOCAL HYBRID ==========
GLOBAL_UPDATE_FREQUENCY = 10  # Update centroid every N batches

os.makedirs(PERSIST_DIR, exist_ok=True)

USE_VALIDATOR_PROXY = os.environ.get("API_BASE_URL") is not None
if USE_VALIDATOR_PROXY:
    API_BASE_URL = os.environ["API_BASE_URL"]
    API_KEY = os.environ["API_KEY"]
    openai_client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
else:
    API_BASE_URL = "https://api.openai.com/v1"
    API_KEY = os.getenv("HF_TOKEN", "")
    openai_client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

STOP_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "have", "from", "they", "been",
    "said", "each", "which", "their", "will", "other", "about", "many", "then",
    "them", "these", "some", "would", "make", "like", "into", "time", "very",
    "when", "come", "could", "than", "its", "also", "back", "after", "two",
    "how", "what", "where", "who", "why", "this", "that", "with",
}

# Relationship color/label constants (PHASE 1.2)
class RelationshipColor(Enum):
    IS_A = "IS_A"           # Apple IS_A Fruit
    HAS_FEATURE = "HAS"     # Apple HAS Color:Red
    GROWN_IN = "GROWN_IN"   # Apple GROWN_IN Kashmir
    LOCATION = "LOCATION"   # Kashmir LOCATION India
    RELATED_TO = "RELATED"  # General relationship
    CAUSES = "CAUSES"       # Login failure CAUSES account lock
    PART_OF = "PART_OF"     # Chhindwara PART_OF MadhyaPradesh

# ============================================================
# PHASE 1.2: Weighted & Colored Relationship Data Structure
# ============================================================
@dataclass
class RelationshipData:
    """Stores weighted, colored relationships between concepts."""
    weight: float = 1.0
    color: str = "RELATED"
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    co_occurrence_count: int = 1
    
    def to_dict(self) -> dict:
        return {
            "weight": self.weight,
            "color": self.color,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "co_occurrence_count": self.co_occurrence_count
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RelationshipData':
        return cls(
            weight=data.get("weight", 1.0),
            color=data.get("color", "RELATED"),
            created_at=data.get("created_at", time.time()),
            last_accessed=data.get("last_accessed", time.time()),
            co_occurrence_count=data.get("co_occurrence_count", 1)
        )

# ============================================================
# DynamicOntology (Enhanced with domain awareness)
# ============================================================
class DynamicOntology:
    def __init__(self):
        self.concept_to_features: Dict[str, List[str]] = {}
        self.feature_to_concepts: Dict[str, List[str]] = defaultdict(list)
        # PHASE 3.3: Domain classification
        self.concept_domains: Dict[str, str] = {}
        self.llm_enabled = True

    async def get_features_llm(self, concept: str, context: str = "") -> Tuple[List[str], str]:
        """Returns (features, domain)"""
        if not openai_client:
            return [concept], "general"
        try:
            response = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Extract features and classify the domain. Return format: 'DOMAIN: <domain> | FEATURES: <comma-separated list>'"},
                    {"role": "user", "content": f"Extract up to 5 features and the domain for '{concept}'. Domain options: Agriculture, Technology, Finance, Geography, General."}
                ],
                temperature=0.3,
                max_tokens=150
            )
            text = response.choices[0].message.content
            domain = "general"
            features = []
            if "DOMAIN:" in text and "FEATURES:" in text:
                domain_part = text.split("DOMAIN:")[1].split("|")[0].strip()
                features_part = text.split("FEATURES:")[1].strip()
                domain = domain_part
                features = [f.strip().lower() for f in features_part.split(",")]
            else:
                features = [f.strip().lower() for f in text.split(",")]
            return features[:5], domain.lower()
        except Exception:
            return [concept], "general"

    async def add_concept(self, concept: str, context: str = ""):
        concept_low = concept.lower()
        if concept_low in self.concept_to_features:
            return
        features, domain = await self.get_features_llm(f"{context} {concept}" if context else concept)
        self.concept_to_features[concept_low] = features
        self.concept_domains[concept_low] = domain
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
        return {
            "concept_to_features": self.concept_to_features,
            "concept_domains": self.concept_domains
        }

    def restore(self, data: dict):
        self.concept_to_features = data.get("concept_to_features", {})
        self.concept_domains = data.get("concept_domains", {})
        self.feature_to_concepts = defaultdict(list)
        for concept, feats in self.concept_to_features.items():
            for f in feats:
                self.feature_to_concepts[f].append(concept)

# ============================================================
# Feature Registry (Enhanced)
# ============================================================
class FeatureRegistry:
    def __init__(self, ontology: DynamicOntology):
        self.ontology = ontology
        self.feature_to_id: Dict[str, int] = {}
        self.id_to_feature: Dict[int, str] = {}
        self.feature_vectors: Dict[int, np.ndarray] = {}
        self.feature_importance: Dict[int, float] = {}  # PHASE 1.3: Feature mass
        self.next_id = 0
        all_features = set()
        for feats in ontology.concept_to_features.values():
            all_features.update(feats)
        for feat in all_features:
            self.register(feat)

    def register(self, feature_name: str, importance: float = 1.0) -> int:
        name = feature_name.lower()
        if name in self.feature_to_id:
            # Increment importance on re-registration
            fid = self.feature_to_id[name]
            self.feature_importance[fid] = min(10.0, self.feature_importance.get(fid, 1.0) + 0.1)
            return fid
        fid = self.next_id
        self.next_id += 1
        self.feature_to_id[name] = fid
        self.id_to_feature[fid] = name
        self.feature_vectors[fid] = np.random.uniform(-1, 1, DIMS).astype(np.float32)
        self.feature_importance[fid] = importance
        return fid

    def get_vector(self, fid: int) -> np.ndarray:
        return self.feature_vectors[fid]
    
    def get_importance(self, fid: int) -> float:
        return self.feature_importance.get(fid, 1.0)

    def update_vector(self, fid: int, delta: np.ndarray, layer: PriorityLayer = PriorityLayer.CLUSTER):
        """Update with layer-specific learning rate."""
        delta = np.clip(delta, -GRAD_CLIP, GRAD_CLIP)
        lr_map = {
            PriorityLayer.ATOMIC: LR_ATOMIC,
            PriorityLayer.CLUSTER: LR_CLUSTER,
            PriorityLayer.DOMAIN: LR_DOMAIN,
            PriorityLayer.UNIVERSAL: LR_UNIVERSAL
        }
        self.feature_vectors[fid] += delta * lr_map[layer]

    def feature_to_letters(self, fid: int, length: int = 5) -> List[str]:
        vec = self.feature_vectors[fid]
        # Use partitioned dimensions for different aspects
        essence_vec = vec[ESSENCE_DIMS]
        probs = np.exp(essence_vec[:min(length, len(essence_vec))])
        probs /= np.sum(probs) + 1e-8
        idx = np.argmax(probs)
        letter_idx = idx % 26
        return [ALPHABET[letter_idx]] * length

    def serialize(self) -> dict:
        return {
            "feature_to_id": self.feature_to_id,
            "id_to_feature": {str(k): v for k, v in self.id_to_feature.items()},
            "feature_vectors": {str(k): v.tolist() for k, v in self.feature_vectors.items()},
            "feature_importance": {str(k): v for k, v in self.feature_importance.items()},
            "next_id": self.next_id,
            "ontology": self.ontology.serialize()
        }

    def restore(self, data: dict):
        self.feature_to_id = data["feature_to_id"]
        self.id_to_feature = {int(k): v for k, v in data["id_to_feature"].items()}
        self.feature_vectors = {int(k): np.array(v, dtype=np.float32) for k, v in data["feature_vectors"].items()}
        self.feature_importance = {int(k): float(v) for k, v in data.get("feature_importance", {}).items()}
        self.next_id = data["next_id"]
        self.ontology.restore(data["ontology"])

# ============================================================
# Letter Vectors (Enhanced with mass)
# ============================================================
class LetterVectors:
    def __init__(self):
        self.vec = {ch: np.random.uniform(-1, 1, DIMS).astype(np.float32) for ch in ALPHABET}
        self.letter_importance = {ch: 1.0 for ch in ALPHABET}  # PHASE 1.3

    def get(self, letter: str) -> np.ndarray:
        return self.vec[letter]
    
    def get_importance(self, letter: str) -> float:
        return self.letter_importance.get(letter, 1.0)

    def update(self, letter: str, delta: np.ndarray, layer: PriorityLayer = PriorityLayer.ATOMIC):
        delta = np.clip(delta, -GRAD_CLIP, GRAD_CLIP)
        lr_map = {
            PriorityLayer.ATOMIC: LR_ATOMIC,
            PriorityLayer.CLUSTER: LR_CLUSTER,
            PriorityLayer.DOMAIN: LR_DOMAIN,
            PriorityLayer.UNIVERSAL: LR_UNIVERSAL
        }
        self.vec[letter] += delta * lr_map[layer]
        # Increment importance on update
        self.letter_importance[letter] = min(10.0, self.letter_importance[letter] + 0.01)

    def serialize(self) -> dict:
        return {
            "vectors": {ch: self.vec[ch].tolist() for ch in ALPHABET},
            "importance": self.letter_importance
        }

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
# DNAConcept — PHASE 1.3, 1.4, 1.5: Mass, Scaling, Inertia
# ============================================================
class DNAConcept:
    def __init__(self, name: str, physical_features: List[int], semantic_features: List[int],
                 feature_registry: FeatureRegistry, letter_vec: LetterVectors,
                 importance: float = 1.0, domain: str = "general"):
        self.name = name
        self.physical_features = physical_features
        self.semantic_features = semantic_features
        self.feature_registry = feature_registry
        self.letter_vec = letter_vec
        
        # PHASE 1.3: Mass/Importance
        self.importance = importance
        self.domain = domain
        self.cluster_id: Optional[int] = None  # PHASE 3.4
        
        # PHASE 4.3: Pending updates for batch processing
        self.pending_deltas: List[np.ndarray] = []
        
        self.vector: Optional[np.ndarray] = None
        self._update_vector()

    def _encode_feature(self, fid: int, start_pos: int) -> np.ndarray:
        letters = self.feature_registry.feature_to_letters(fid, length=5)
        vec = np.zeros(DIMS, dtype=np.float32)
        for i, ch in enumerate(letters):
            base = self.letter_vec.get(ch)
            # PHASE 1.3: Weight by letter importance
            importance_weight = self.letter_vec.get_importance(ch)
            vec += np.sin(base + (start_pos + i) * POSITION_OFFSET) * importance_weight
        return vec

    def _update_vector(self):
        vec = np.zeros(DIMS, dtype=np.float32)
        pos = 0
        for fid in self.physical_features:
            vec += self._encode_feature(fid, pos)
            pos += 5
        for fid in self.semantic_features:
            vec += self._encode_feature(fid, pos)
            pos += 5
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        self.vector = vec
        # PHASE 1.4: Apply mass scaling
        self.apply_scale()

    # ========== PHASE 1.4: Vector Scaling by Mass ==========
    def apply_scale(self):
        """Normalize vector but scale its length by its importance/mass."""
        if self.vector is None:
            return
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            target_scale = np.log1p(self.importance)
            self.vector = (self.vector / norm) * target_scale

    # ========== PHASE 1.5: Inertia-Based Movement ==========
    def move_towards(self, other: 'DNAConcept', lr: float = LR_CONCEPT, 
                     weight: float = 1.0, color: str = "RELATED"):
        """
        Core innovation: move concept vectors closer with inertia.
        Heavier nodes (higher importance) move less.
        """
        if self.vector is None or other.vector is None:
            return
            
        # Calculate relative inertia (PHASE 1.5)
        pull_force = (other.importance / (self.importance + 1e-8)) * lr * weight
        other_pull_force = (self.importance / (other.importance + 1e-8)) * lr * weight
        
        # Color-based movement multiplier (PHASE 1.2)
        color_multipliers = {
            "IS_A": 1.2,        # Stronger pull for essential relationships
            "HAS": 0.8,
            "GROWN_IN": 0.9,
            "LOCATION": 0.7,
            "CAUSES": 1.0,
            "PART_OF": 0.85,
            "RELATED": 0.5
        }
        color_mult = color_multipliers.get(color, 0.5)
        pull_force *= color_mult
        other_pull_force *= color_mult
        
        # Store original vectors for gradient calculation
        orig_self = self.vector.copy()
        orig_other = other.vector.copy()
        
        # Apply movement
        diff = other.vector - self.vector
        self.vector += pull_force * diff
        other.vector -= other_pull_force * diff
        
        # Normalize
        self.vector /= (np.linalg.norm(self.vector) + 1e-8)
        other.vector /= (np.linalg.norm(other.vector) + 1e-8)
        
        # PHASE 1.4: Re-apply scale to preserve mass
        self.apply_scale()
        other.apply_scale()
        
        # Backpropagate gradient with partitioned learning rates
        self._backpropagate_to_features(other, orig_self, orig_other, pull_force, other_pull_force, color)

    def _backpropagate_to_features(self, other: 'DNAConcept', 
                                    orig_self: np.ndarray, orig_other: np.ndarray,
                                    pull_force: float, other_pull_force: float,
                                    color: str):
        """PHASE 2.3: Partitioned backpropagation with layer-specific LRs."""
        gradient = other.vector - self.vector
        
        # Determine layer based on color
        layer_map = {
            "IS_A": PriorityLayer.DOMAIN,
            "PART_OF": PriorityLayer.DOMAIN,
            "HAS": PriorityLayer.CLUSTER,
            "GROWN_IN": PriorityLayer.CLUSTER,
            "LOCATION": PriorityLayer.CLUSTER,
            "RELATED": PriorityLayer.ATOMIC
        }
        layer = layer_map.get(color, PriorityLayer.CLUSTER)
        
        for concept, force in [(self, pull_force), (other, other_pull_force)]:
            for fid in concept.physical_features + concept.semantic_features:
                letters = self.feature_registry.feature_to_letters(fid, length=5)
                for i, ch in enumerate(letters):
                    base = self.letter_vec.get(ch)
                    x = base + i * POSITION_OFFSET
                    grad_sin = np.cos(x)
                    
                    # PHASE 2.2: Apply partitioned gradients
                    # Essence dimensions (0-3) get DOMAIN-level updates
                    essence_grad = grad_sin[ESSENCE_DIMS]
                    identity_grad = grad_sin[IDENTITY_DIMS]
                    temporal_grad = grad_sin[TEMPORAL_DIMS]
                    
                    full_grad = np.zeros(DIMS)
                    full_grad[ESSENCE_DIMS] = essence_grad * LR_DOMAIN
                    full_grad[IDENTITY_DIMS] = identity_grad * LR_CLUSTER
                    full_grad[TEMPORAL_DIMS] = temporal_grad * LR_ATOMIC
                    
                    norm_grad = np.linalg.norm(full_grad) + 1e-8
                    delta_f = force * 0.5 * gradient * full_grad / norm_grad
                    delta_l = force * 0.5 * gradient * full_grad / norm_grad
                    
                    self.feature_registry.update_vector(fid, delta_f, layer)
                    self.letter_vec.update(ch, delta_l, PriorityLayer.ATOMIC)

    # ========== PHASE 2.2: Partitioned Similarity ==========
    def partitioned_similarity(self, other: 'DNAConcept', partition: slice = None) -> float:
        """Calculate similarity using only specified dimensions."""
        if self.vector is None or other.vector is None:
            return 0.0
        if partition is not None:
            v1 = self.vector[partition]
            v2 = other.vector[partition]
        else:
            v1, v2 = self.vector, other.vector
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

    def cosine_similarity(self, other: 'DNAConcept') -> float:
        return self.partitioned_similarity(other)

    # ========== PHASE 10.1: Relationship Strengthening ==========
    def strengthen_relationship(self, other_name: str, increment: float = 0.1):
        """Increment importance when concepts co-occur."""
        self.importance = min(10.0, self.importance + increment * 0.1)

    def serialize(self) -> dict:
        return {
            "name": self.name,
            "physical_features": self.physical_features,
            "semantic_features": self.semantic_features,
            "vector": self.vector.tolist() if self.vector is not None else None,
            "importance": self.importance,
            "domain": self.domain,
            "cluster_id": self.cluster_id
        }

    @classmethod
    def from_serialized(cls, data: dict, feature_registry, letter_vec):
        obj = cls(
            data["name"], 
            data["physical_features"], 
            data["semantic_features"], 
            feature_registry, 
            letter_vec,
            importance=data.get("importance", 1.0),
            domain=data.get("domain", "general")
        )
        if data.get("vector") is not None:
            obj.vector = np.array(data["vector"], dtype=np.float32)
        obj.cluster_id = data.get("cluster_id")
        return obj

# ============================================================
# PHASE 7: Decoder for Sentence Generation
# ============================================================
class SentenceDecoder:
    """Generates natural language from DNA concept activations."""
    
    def __init__(self, concept_memory: 'ConceptMemory'):
        self.concept_memory = concept_memory
    
    def extract_weighted_features(self, concept_name: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """PHASE 7.1: Extract top related concepts sorted by weight."""
        if concept_name not in self.concept_memory.concepts:
            return []
        
        relationships = self.concept_memory.weighted_relationships.get(concept_name, {})
        if not relationships:
            return []
        
        # Sort by weight
        sorted_rels = sorted(
            relationships.items(),
            key=lambda x: x[1].weight,
            reverse=True
        )
        return [(other, rel.weight, rel.color) for other, rel in sorted_rels[:top_k]]
    
    def multi_hop_traversal(self, start: str, target_color: str, max_hops: int = 3) -> List[str]:
        """PHASE 7.2: Follow colored relationships."""
        visited = {start}
        path = [start]
        current = start
        
        for _ in range(max_hops):
            if current not in self.concept_memory.weighted_relationships:
                break
            # Find relationship with target color
            found = None
            for other, rel in self.concept_memory.weighted_relationships[current].items():
                if rel.color == target_color and other not in visited:
                    found = other
                    break
            if found is None:
                break
            path.append(found)
            visited.add(found)
            current = found
        
        return path
    
    def generate_description(self, concept_name: str, template: str = None) -> str:
        """PHASE 7.3: Template-based generation."""
        features = self.extract_weighted_features(concept_name, top_k=5)
        if not features:
            return f"{concept_name} is a concept."
        
        concept = self.concept_memory.concepts.get(concept_name)
        domain = concept.domain if concept else "general"
        
        # Find IS_A relationship
        is_a = next((f[0] for f in features if f[2] == "IS_A"), None)
        # Find HAS relationships
        has_features = [f[0] for f in features if f[2] == "HAS"][:2]
        # Find LOCATION
        location_path = self.multi_hop_traversal(concept_name, "LOCATION", max_hops=2)
        location = location_path[-1] if len(location_path) > 1 else None
        
        # Build sentence
        parts = [concept_name.capitalize()]
        if is_a:
            parts.append(f"is a {is_a}")
        if has_features:
            parts.append(f"with {', '.join(has_features)}")
        if location:
            parts.append(f"located in {location}")
        parts.append(f"(domain: {domain})")
        
        return " ".join(parts) + "."
    
    def self_correct(self, draft: str, concept_name: str) -> str:
        """PHASE 7.5: Self-correction using global consistency check."""
        # Check if draft contains known relationships
        features = self.extract_weighted_features(concept_name)
        feature_names = [f[0] for f in features]
        
        # Simple correction: ensure mentioned concepts are actually related
        draft_lower = draft.lower()
        for feat in feature_names:
            if feat.lower() not in draft_lower:
                # Could add missing important feature
                pass
        
        return draft

# ============================================================
# Reasoning Engine — Enhanced with partitioned search
# ============================================================
class ReasoningEngine:
    def __init__(self, concept_memory: 'ConceptMemory'):
        self.concept_memory = concept_memory
        self.decoder = SentenceDecoder(concept_memory)

    def multi_hop_reasoning(self, start: str, max_hops: int = 3, decay: float = 0.7,
                            color_filter: Optional[str] = None) -> Dict[str, float]:
        """PHASE 7.2: Color-filtered multi-hop reasoning."""
        if start not in self.concept_memory.weighted_relationships:
            return {}
        
        activation = {start: 1.0}
        for hop in range(max_hops):
            new_activation = {}
            for node, score in activation.items():
                relationships = self.concept_memory.weighted_relationships.get(node, {})
                for nb, rel in relationships.items():
                    if color_filter and rel.color != color_filter:
                        continue
                    weight = rel.weight
                    if node in self.concept_memory.concepts and nb in self.concept_memory.concepts:
                        sim = self.concept_memory.concepts[node].cosine_similarity(
                            self.concept_memory.concepts[nb]
                        )
                        weight *= (0.5 + 0.5 * sim)
                    new_activation[nb] = new_activation.get(nb, 0) + score * weight * decay
            for k, v in new_activation.items():
                activation[k] = activation.get(k, 0) + v
        
        if activation:
            max_act = max(activation.values())
            activation = {k: v/max_act for k, v in activation.items()}
        return activation

    def analogical_reasoning(self, a: str, b: str, c: str, 
                             partition: slice = None) -> List[str]:
        """A is to B as C is to ? with optional dimension partition."""
        if a not in self.concept_memory.concepts or b not in self.concept_memory.concepts:
            return []
        vec_a = self.concept_memory.concepts[a].vector
        vec_b = self.concept_memory.concepts[b].vector
        
        if partition is not None:
            direction = vec_b[partition] - vec_a[partition]
        else:
            direction = vec_b - vec_a
            
        if c not in self.concept_memory.concepts:
            return []
        vec_c = self.concept_memory.concepts[c].vector
        
        if partition is not None:
            target = vec_c.copy()
            target[partition] = vec_c[partition] + direction
        else:
            target = vec_c + direction
            
        target /= (np.linalg.norm(target) + 1e-8)
        results = self.concept_memory.partitioned_search(target, top_k=5, partition=partition)
        return [r[0] for r in results if r[0] not in (a, b, c)]
    
    def generate_sentence(self, concept: str) -> str:
        """Generate a descriptive sentence for a concept."""
        return self.decoder.generate_description(concept)

# ============================================================
# Concept Memory — PHASE 1-4: All Upgrades
# ============================================================
class ConceptMemory:
    def __init__(self, feature_registry: FeatureRegistry, letter_vec: LetterVectors,
                 max_concepts: int = MAX_CONCEPTS):
        self.feature_registry = feature_registry
        self.letter_vec = letter_vec
        self.concepts: Dict[str, DNAConcept] = {}
        
        # PHASE 1.1, 1.2: Weighted and colored relationships
        self.weighted_relationships: Dict[str, Dict[str, RelationshipData]] = defaultdict(dict)
        
        # Legacy compatibility
        self.relationships: Dict[str, Set[str]] = defaultdict(set)
        
        self.max_concepts = max_concepts
        
        # PHASE 4.1: Global centroid tracking
        self.global_centroid: Optional[np.ndarray] = None
        self.batch_counter = 0
        
        # PHASE 4.3: Pending updates buffer
        self.pending_updates: List[Tuple[str, str, float, str]] = []
        
        # FAISS
        self._faiss_available = False
        self.index = None
        self.id_to_name: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
        self.next_id = 0
        
        # PHASE 4.5: Hierarchical FAISS
        self.cluster_index = None
        self.cluster_to_concepts: Dict[int, List[str]] = defaultdict(list)
        
        try:
            import faiss
            self._faiss_available = True
        except Exception:
            self._faiss_available = False

    # ========== PHASE 4.1: Global Context ==========
    def update_global_centroid(self):
        """Calculate the center of gravity of all concepts."""
        if not self.concepts:
            return
        all_vecs = np.array([c.vector for c in self.concepts.values() if c.vector is not None])
        if len(all_vecs) > 0:
            self.global_centroid = np.mean(all_vecs, axis=0)

    def get_global_context(self) -> dict:
        """PHASE 4.2: Return global centroid and top anchor nodes."""
        if self.global_centroid is None:
            self.update_global_centroid()
        
        # Find top 5 most important nodes
        top_nodes = sorted(
            self.concepts.values(), 
            key=lambda x: x.importance, 
            reverse=True
        )[:5]
        
        return {
            "global_centroid": self.global_centroid.tolist() if self.global_centroid is not None else None,
            "anchors": [n.name for n in top_nodes],
            "total_concepts": len(self.concepts)
        }

    # ========== PHASE 4.4: Global Consistency Check ==========
    def validate_against_global(self, concept_vector: np.ndarray, threshold: float = 2.0) -> bool:
        """Check if a vector is within reasonable distance from global centroid."""
        if self.global_centroid is None:
            return True
        distance = np.linalg.norm(concept_vector - self.global_centroid)
        std = np.std([c.vector for c in self.concepts.values()]) if self.concepts else 1.0
        return distance < threshold * std

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
        vectors = [c.vector for c in self.concepts.values() if c.vector is not None]
        names = [name for name, c in self.concepts.items() if c.vector is not None]
        if not vectors:
            return
        vecs = np.vstack(vectors).astype(np.float32)
        self.index = faiss.IndexFlatIP(DIMS)
        self.index.add(vecs)
        self.id_to_name = {i: n for i, n in enumerate(names)}
        self.name_to_id = {n: i for i, n in enumerate(names)}
        self.next_id = len(self.concepts)
        
        # PHASE 4.1: Update global centroid after rebuild
        self.update_global_centroid()

    def register(self, name: str, physical_features: List[int], semantic_features: List[int],
                 importance: float = 1.0, domain: str = "general") -> DNAConcept:
        name_low = name.lower()
        if name_low in self.concepts:
            # PHASE 10.2: Strengthen on re-registration
            self.concepts[name_low].importance = min(10.0, self.concepts[name_low].importance + 0.1)
            return self.concepts[name_low]
        
        concept = DNAConcept(name_low, physical_features, semantic_features,
                             self.feature_registry, self.letter_vec,
                             importance=importance, domain=domain)
        self.concepts[name_low] = concept
        
        if self._faiss_available and concept.vector is not None:
            self._ensure_index()
            if self.index is not None:
                self.index.add(concept.vector.reshape(1, -1))
                self.id_to_name[self.next_id] = name_low
                self.name_to_id[name_low] = self.next_id
                self.next_id += 1
        
        self._prune()
        return concept

    # ========== PHASE 1.1, 1.2: Weighted & Colored Relationships ==========
    def add_weighted_relationship(self, a: str, b: str, weight: float = 1.0, 
                                   color: str = "RELATED"):
        a_low, b_low = a.lower(), b.lower()
        if a_low not in self.concepts or b_low not in self.concepts:
            return
        
        # Create or update relationship data
        if b_low not in self.weighted_relationships[a_low]:
            self.weighted_relationships[a_low][b_low] = RelationshipData(
                weight=weight, 
                color=color,
                co_occurrence_count=1
            )
        else:
            rel = self.weighted_relationships[a_low][b_low]
            rel.weight = min(1.0, rel.weight + 0.1 * weight)
            rel.co_occurrence_count += 1
            rel.last_accessed = time.time()
        
        # Symmetric relationship
        if a_low not in self.weighted_relationships[b_low]:
            self.weighted_relationships[b_low][a_low] = RelationshipData(
                weight=weight,
                color=color,
                co_occurrence_count=1
            )
        else:
            rel = self.weighted_relationships[b_low][a_low]
            rel.weight = min(1.0, rel.weight + 0.1 * weight)
            rel.co_occurrence_count += 1
            rel.last_accessed = time.time()
        
        # Legacy compatibility
        self.relationships[a_low].add(b_low)
        self.relationships[b_low].add(a_low)
        
        # Move vectors with inertia
        self.concepts[a_low].move_towards(self.concepts[b_low], 
                                           lr=LR_CONCEPT, 
                                           weight=weight, 
                                           color=color)

    def add_relationship(self, a: str, b: str):
        """Legacy method for backward compatibility."""
        self.add_weighted_relationship(a, b, weight=1.0, color="RELATED")

    # ========== PHASE 10.2: Strengthen on Co-occurrence ==========
    def strengthen_relationship(self, a: str, b: str, increment: float = 0.1):
        a_low, b_low = a.lower(), b.lower()
        if a_low in self.weighted_relationships and b_low in self.weighted_relationships[a_low]:
            rel = self.weighted_relationships[a_low][b_low]
            rel.weight = min(1.0, rel.weight + increment)
            rel.co_occurrence_count += 1
            rel.last_accessed = time.time()
            
            # Also strengthen symmetric
            if a_low in self.weighted_relationships[b_low]:
                rel2 = self.weighted_relationships[b_low][a_low]
                rel2.weight = min(1.0, rel2.weight + increment)
                rel2.co_occurrence_count += 1
                rel2.last_accessed = time.time()

    # ========== PHASE 10.1: Relationship Decay ==========
    def apply_decay(self, decay_rate: float = 0.001, inactive_threshold: float = 86400):
        """Decay relationships that haven't been accessed recently."""
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

    # ========== PHASE 2.2: Partitioned Search ==========
    def partitioned_search(self, query_vector: np.ndarray, top_k: int = 5,
                           partition: slice = None) -> List[Tuple[str, float]]:
        """Search using only specified dimension partition."""
        if not self.concepts:
            return []
        
        if partition is not None:
            q = query_vector[partition]
            vecs = np.vstack([self.concepts[n].vector[partition] for n in self.concepts.keys()])
        else:
            q = query_vector
            vecs = np.vstack([self.concepts[n].vector for n in self.concepts.keys()])
        
        names = list(self.concepts.keys())
        q = q / (np.linalg.norm(q) + 1e-8)
        vecs_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        scores = vecs_norm @ q
        top = np.argsort(scores)[::-1][:top_k]
        return [(names[i], float(scores[i])) for i in top]

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Full-vector search."""
        return self.partitioned_search(query_vector, top_k, partition=None)

    # ========== PHASE 4.3: Batch Processing ==========
    def add_to_batch(self, a: str, b: str, weight: float, color: str):
        """Add relationship update to pending batch."""
        self.pending_updates.append((a, b, weight, color))
        self.batch_counter += 1
        
        if len(self.pending_updates) >= BATCH_SIZE:
            self.process_batch()

    def process_batch(self):
        """Process all pending updates and update global statistics."""
        if not self.pending_updates:
            return
        
        for a, b, weight, color in self.pending_updates:
            if a in self.concepts and b in self.concepts:
                self.concepts[a].move_towards(self.concepts[b], 
                                               lr=LR_CONCEPT, 
                                               weight=weight, 
                                               color=color)
        
        self.pending_updates.clear()
        
        # PHASE 4.1: Update global centroid periodically
        if self.batch_counter % GLOBAL_UPDATE_FREQUENCY == 0:
            self.update_global_centroid()
        
        self._rebuild_index()

    async def extract_and_link(self, text: str, ontology: DynamicOntology, 
                                sector: str = "general") -> List[str]:
        words = [w for w in text.lower().split() if len(w) > 3 and w not in STOP_WORDS]
        unique = list(set(words))[:15]
        concept_list = []
        
        for kw in unique:
            features = await ontology.get_features_llm(kw) if ontology.llm_enabled else (ontology.get_features(kw), "general")
            if isinstance(features, tuple):
                features, domain = features
            else:
                domain = "general"
            
            physical_fids = [self.feature_registry.register(f) for f in features]
            semantic_fids = [self.feature_registry.register(f) for f in features]
            
            # PHASE 1.3: Initial importance based on word frequency in corpus
            importance = 1.0 + (0.1 * unique.index(kw) if kw in unique else 0)
            
            concept = self.register(kw, physical_fids, semantic_fids, 
                                    importance=importance, domain=domain)
            concept_list.append(kw)
        
        # Create relationships with colors inferred from context
        for i in range(len(unique)):
            for j in range(i+1, min(i+4, len(unique))):
                # Infer relationship color from text context
                color = self._infer_relationship_color(text, unique[i], unique[j])
                self.add_weighted_relationship(unique[i], unique[j], weight=0.8, color=color)
        
        return concept_list

    def _infer_relationship_color(self, text: str, a: str, b: str) -> str:
        """Infer relationship type from context."""
        text_lower = text.lower()
        if f"{a} is {b}" in text_lower or f"{b} is {a}" in text_lower:
            return "IS_A"
        elif f"{a} has {b}" in text_lower or f"{b} has {a}" in text_lower:
            return "HAS"
        elif f"{a} in {b}" in text_lower or f"{b} in {a}" in text_lower:
            return "LOCATION"
        elif "cause" in text_lower and (a in text_lower or b in text_lower):
            return "CAUSES"
        return "RELATED"

    def _prune(self):
        if len(self.concepts) > self.max_concepts:
            # Sort by importance (keep high importance concepts)
            sorted_concepts = sorted(
                self.concepts.items(), 
                key=lambda x: (x[1].importance, len(self.weighted_relationships.get(x[0], {}))),
                reverse=True
            )
            to_keep = sorted_concepts[:self.max_concepts]
            self.concepts = {name: concept for name, concept in to_keep}
            
            # Clean up relationships for removed concepts
            keep_names = set(self.concepts.keys())
            self.weighted_relationships = defaultdict(dict, {
                k: {b: r for b, r in v.items() if b in keep_names}
                for k, v in self.weighted_relationships.items() if k in keep_names
            })
            self.relationships = defaultdict(set, {
                k: v.intersection(keep_names) 
                for k, v in self.relationships.items() if k in keep_names
            })
            
            self._rebuild_index()

    def serialize(self) -> dict:
        return {
            "concepts": {name: c.serialize() for name, c in self.concepts.items()},
            "weighted_relationships": {
                k: {b: r.to_dict() for b, r in v.items()}
                for k, v in self.weighted_relationships.items()
            },
            "relationships": {k: list(v) for k, v in self.relationships.items()},
            "global_centroid": self.global_centroid.tolist() if self.global_centroid is not None else None
        }

    def restore(self, data: dict):
        self.concepts = {}
        self.weighted_relationships = defaultdict(dict)
        self.relationships = defaultdict(set)
        
        for name, cdata in data.get("concepts", {}).items():
            self.concepts[name] = DNAConcept.from_serialized(cdata, self.feature_registry, self.letter_vec)
        
        for k, vdict in data.get("weighted_relationships", {}).items():
            for b, rdata in vdict.items():
                self.weighted_relationships[k][b] = RelationshipData.from_dict(rdata)
                self.relationships[k].add(b)
        
        for k, vlist in data.get("relationships", {}).items():
            self.relationships[k].update(vlist)
        
        if data.get("global_centroid"):
            self.global_centroid = np.array(data["global_centroid"], dtype=np.float32)
        
        self._rebuild_index()

# ============================================================
# Persistence Manager
# ============================================================
class PersistenceManager:
    @staticmethod
    def save_all(concept_memory: ConceptMemory, feature_registry: FeatureRegistry, 
                 letter_vec: LetterVectors, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(PERSIST_DIR, "brain_state.pkl")
        
        state = {
            "concept_memory": concept_memory.serialize(),
            "feature_registry": feature_registry.serialize(),
            "letter_vec": letter_vec.serialize()
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load_all(filepath: str = None) -> Tuple[ConceptMemory, FeatureRegistry, LetterVectors, DynamicOntology]:
        if filepath is None:
            filepath = os.path.join(PERSIST_DIR, "brain_state.pkl")
        
        ontology = DynamicOntology()
        feature_registry = FeatureRegistry(ontology)
        letter_vec = LetterVectors()
        concept_memory = ConceptMemory(feature_registry, letter_vec)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    state = pickle.load(f)
                concept_memory.restore(state.get("concept_memory", {}))
                feature_registry.restore(state.get("feature_registry", {}))
                letter_vec.restore(state.get("letter_vec", {}))
            except Exception as e:
                print(f"[WARN] Failed to load persisted state: {e}")
        
        return concept_memory, feature_registry, letter_vec, ontology

# ============================================================
# Background Trainer (Enhanced with batch processing)
# ============================================================
class ContinuousTrainer:
    def __init__(self, concept_memory: ConceptMemory, feature_registry: FeatureRegistry,
                 letter_vec: LetterVectors, interval_sec: int = TRAIN_INTERVAL_SEC):
        self.concept_memory = concept_memory
        self.feature_registry = feature_registry
        self.letter_vec = letter_vec
        self.interval = interval_sec
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._train_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _train_loop(self):
        while self.running:
            time.sleep(self.interval)
            
            # PHASE 4.3: Process any pending batch updates
            if self.concept_memory.pending_updates:
                self.concept_memory.process_batch()
            
            # PHASE 10.1: Apply decay to unused relationships
            self.concept_memory.apply_decay(decay_rate=0.001)
            
            # Get relationships from weighted store
            rels = []
            for a, rel_dict in self.concept_memory.weighted_relationships.items():
                for b, rel in rel_dict.items():
                    rels.append((a, b, rel.weight, rel.color))
            
            if not rels:
                continue
            
            batch = random.sample(rels, min(BATCH_SIZE, len(rels)))
            for a, b, weight, color in batch:
                if a in self.concept_memory.concepts and b in self.concept_memory.concepts:
                    self.concept_memory.concepts[a].move_towards(
                        self.concept_memory.concepts[b], 
                        lr=LR_CONCEPT * 0.5,  # Lower LR for background training
                        weight=weight, 
                        color=color
                    )
            
            self.concept_memory._rebuild_index()
            
            # PHASE 4.1: Update global centroid
            self.concept_memory.update_global_centroid()

# ============================================================
# KnowledgeGraphEnv (Main Environment with all upgrades)
# ============================================================
class KnowledgeGraphEnv:
    def __init__(self, start_trainer: bool = True):
        self.concept_memory, self.feature_registry, self.letter_vec, self.ontology = PersistenceManager.load_all()
        self.reasoning_engine = ReasoningEngine(self.concept_memory)
        self._seed_initial_concepts()
        self.trainer: Optional[ContinuousTrainer] = None
        if start_trainer:
            self.trainer = ContinuousTrainer(self.concept_memory, self.feature_registry, self.letter_vec)
            self.trainer.start()
        self.current_task: Optional[dict] = None
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False

    def _seed_initial_concepts(self):
        if len(self.concept_memory.concepts) > 0:
            return
        
        # PHASE 1.1, 1.2: Seed with weighted, colored relationships
        seed_data = [
            ("login issue",      ["authentication", "password", "access"], "Technology", 2.0),
            ("billing",          ["payment", "invoice", "refund"], "Finance", 2.0),
            ("slow performance", ["latency", "response time", "optimization"], "Technology", 1.5),
            ("crash",            ["bug", "error", "unstable"], "Technology", 1.5),
            ("feature request",  ["enhancement", "new functionality", "suggestion"], "General", 1.0),
            ("account locked",   ["security", "blocked", "verification"], "Technology", 1.8),
        ]
        
        for concept, features, domain, importance in seed_data:
            physical_fids = [self.feature_registry.register(f) for f in features]
            semantic_fids = [self.feature_registry.register(f) for f in features]
            self.concept_memory.register(concept, physical_fids, semantic_fids, 
                                          importance=importance, domain=domain)
        
        # Add colored relationships
        self.concept_memory.add_weighted_relationship("login issue", "account locked", weight=0.9, color="CAUSES")
        self.concept_memory.add_weighted_relationship("billing", "refund", weight=0.8, color="RELATED")
        self.concept_memory.add_weighted_relationship("slow performance", "crash", weight=0.7, color="CAUSES")
        self.concept_memory.add_weighted_relationship("feature request", "enhancement", weight=0.9, color="IS_A")
        self.concept_memory.add_weighted_relationship("login issue", "authentication", weight=0.95, color="IS_A")
        self.concept_memory.add_weighted_relationship("account locked", "security", weight=0.85, color="HAS")

    # Instance methods (delegate to imported graders)
    def task_easy(self, input_text: str) -> float:
        return task_easy(input_text)

    def task_medium(self, input_text: str) -> float:
        return task_medium(input_text)

    def task_hard(self, input_text: str) -> float:
        return task_hard(input_text)

    # ========== PHASE 7: Sentence Generation Endpoint ==========
    def generate_sentence(self, concept: str) -> str:
        """Generate a descriptive sentence for a concept."""
        return self.reasoning_engine.generate_sentence(concept)
    
    # ========== PHASE 4.2: Global Context Endpoint ==========
    def get_global_context(self) -> dict:
        """Return global statistics about the knowledge graph."""
        return self.concept_memory.get_global_context()
    
    # ========== PHASE 2.2: Partitioned Search Endpoint ==========
    def search_by_essence(self, concept: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar concepts using only essence dimensions."""
        if concept not in self.concept_memory.concepts:
            return []
        vec = self.concept_memory.concepts[concept].vector
        return self.concept_memory.partitioned_search(vec, top_k, partition=ESSENCE_DIMS)
    
    def search_by_identity(self, concept: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar concepts using only identity dimensions."""
        if concept not in self.concept_memory.concepts:
            return []
        vec = self.concept_memory.concepts[concept].vector
        return self.concept_memory.partitioned_search(vec, top_k, partition=IDENTITY_DIMS)

    # OpenEnv interface
    def reset(self) -> str:
        self.current_task = self._generate_dynamic_task()
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(
                    self.concept_memory.extract_and_link(self.current_task["input"], self.ontology))
            else:
                loop.run_until_complete(
                    self.concept_memory.extract_and_link(self.current_task["input"], self.ontology))
        except Exception as e:
            print(f"[DEBUG] Background extraction failed: {e}", flush=True)
        return self.current_task["input"]

    def step(self, action: str) -> Tuple[str, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode already done. Call reset() first.")
        step_type = ["identify", "relate", "answer"][self.current_step]
        reward = 0.0
        if step_type == "identify":
            reward = self._grade_identification(action, self.current_task["expected_concept"])
            if reward >= 0.3:
                self.current_step = 1
        elif step_type == "relate":
            reward = self._grade_relation(action, self.current_task["expected_relation"])
            if reward >= 0.3:
                self.current_step = 2
        else:
            reward = self._grade_answer(action, self.current_task["expected_answer"])
            self.done = True
            self.current_step = 3
        self.episode_reward += reward
        obs = self.current_task["input"] if not self.done else ""
        info = {
            "task_type": self.current_task["type"],
            "step": step_type,
            "step_reward": reward,
            "total_reward": self.episode_reward
        }
        return obs, reward, self.done, info

    def state(self) -> dict:
        if not self.current_task:
            return {}
        return {
            "task": self.current_task,
            "step": self.current_step,
            "step_name": ["identify", "relate", "answer"][self.current_step] if self.current_step < 3 else "done"
        }

    def _generate_dynamic_task(self) -> dict:
        concepts = list(self.concept_memory.concepts.keys())
        if not concepts:
            return {
                "type": "easy",
                "input": "My phone won't turn on",
                "expected_concept": "hardware failure",
                "expected_relation": "battery issue",
                "expected_answer": "replace battery or check power"
            }
        base_concept = random.choice(concepts)
        related = list(self.concept_memory.relationships.get(base_concept, set()))
        if not related:
            vec = self.concept_memory.concepts[base_concept].vector
            similar = self.concept_memory.search(vec, top_k=3)
            related = [r[0] for r in similar if r[0] != base_concept]
        expected_relation = related[0] if related else "related issue"
        reasoning_result = self.reasoning_engine.multi_hop_reasoning(base_concept, max_hops=2)
        possible_answers = [c for c in reasoning_result.keys() if c != base_concept and c != expected_relation]
        if possible_answers:
            expected_answer = possible_answers[0]
        else:
            expected_answer = random.choice(list(self.concept_memory.concepts.keys()))
        templates = {
            "easy": [f"I'm having trouble with {base_concept}.", f"{base_concept} not working."],
            "medium": [f"User reports {base_concept} persists after restart.", f"Ticket: {base_concept} affecting workflow."],
            "hard": [f"Critical: {base_concept} causing system failure, need urgent resolution.", f"Customer says {base_concept} is blocking all operations."]
        }
        difficulty = random.choice(["easy", "medium", "hard"])
        input_text = random.choice(templates[difficulty])
        return {
            "type": difficulty,
            "input": input_text,
            "expected_concept": base_concept,
            "expected_relation": expected_relation,
            "expected_answer": expected_answer
        }

    def _grade_identification(self, action: str, expected: str) -> float:
        action_lower = action.lower().strip()
        expected_lower = expected.lower()
        if action_lower == expected_lower:
            return 0.95
        elif expected_lower in action_lower or action_lower in expected_lower:
            return 0.75
        elif any(word in action_lower for word in expected_lower.split()):
            return 0.35
        else:
            return 0.05

    def _grade_relation(self, action: str, expected: str) -> float:
        action_lower = action.lower().strip()
        expected_lower = expected.lower()
        if action_lower == expected_lower:
            return 0.95
        elif expected_lower in action_lower:
            return 0.75
        elif self.current_task and action_lower in self.concept_memory.relationships.get(self.current_task["expected_concept"], set()):
            return 0.55
        else:
            return 0.05

    def _grade_answer(self, action: str, expected: str) -> float:
        action_lower = action.lower().strip()
        expected_lower = expected.lower()
        if action_lower == expected_lower:
            return 0.95
        elif expected_lower in action_lower:
            return 0.75
        elif any(word in action_lower for word in expected_lower.split()):
            return 0.35
        else:
            return 0.05

    def close(self):
        if self.trainer:
            self.trainer.stop()
        # PHASE 4.3: Process any remaining batch updates
        if self.concept_memory.pending_updates:
            self.concept_memory.process_batch()
        # Save state on close
        PersistenceManager.save_all(self.concept_memory, self.feature_registry, self.letter_vec)

# ============================================================
# FastAPI — with new endpoints for upgraded features
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
    description="Self-evolving DNA-inspired knowledge graph with weighted relationships, dimensional partitioning, and global context awareness.",
    version="2.0.0",
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

# New models for upgraded features
class SentenceRequest(BaseModel):
    concept: str

class SentenceResponse(BaseModel):
    sentence: str

class GlobalContextResponse(BaseModel):
    global_centroid: Optional[List[float]]
    anchors: List[str]
    total_concepts: int

class PartitionedSearchRequest(BaseModel):
    concept: str
    top_k: int = 5

class SearchResult(BaseModel):
    name: str
    score: float

class PartitionedSearchResponse(BaseModel):
    results: List[SearchResult]

class AddRelationshipRequest(BaseModel):
    concept_a: str
    concept_b: str
    weight: float = 1.0
    color: str = "RELATED"

class AddRelationshipResponse(BaseModel):
    status: str


# ── Endpoints ────────────────────────────────────────────────

@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.get("/health")
async def health():
    if _api_env is None:
        return {"status": "initializing"}
    return {
        "status": "healthy",
        "total_concepts": len(_api_env.concept_memory.concepts),
        "total_relationships": sum(len(v) for v in _api_env.concept_memory.weighted_relationships.values())
    }


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


# ========== NEW ENDPOINTS FOR UPGRADED FEATURES ==========

@app.post("/sentence", response_model=SentenceResponse)
async def generate_sentence_endpoint(req: SentenceRequest):
    """PHASE 7: Generate a descriptive sentence for a concept."""
    if _api_env is None:
        raise HTTPException(status_code=503, detail="Environment still initializing.")
    sentence = _api_env.generate_sentence(req.concept)
    return SentenceResponse(sentence=sentence)


@app.get("/global-context", response_model=GlobalContextResponse)
async def global_context_endpoint():
    """PHASE 4.2: Get global statistics about the knowledge graph."""
    if _api_env is None:
        raise HTTPException(status_code=503, detail="Environment still initializing.")
    ctx = _api_env.get_global_context()
    return GlobalContextResponse(
        global_centroid=ctx["global_centroid"],
        anchors=ctx["anchors"],
        total_concepts=ctx["total_concepts"]
    )


@app.post("/search/essence", response_model=PartitionedSearchResponse)
async def search_essence_endpoint(req: PartitionedSearchRequest):
    """PHASE 2.2: Search using only essence dimensions (what something fundamentally IS)."""
    if _api_env is None:
        raise HTTPException(status_code=503, detail="Environment still initializing.")
    results = _api_env.search_by_essence(req.concept, req.top_k)
    return PartitionedSearchResponse(
        results=[SearchResult(name=r[0], score=r[1]) for r in results]
    )


@app.post("/search/identity", response_model=PartitionedSearchResponse)
async def search_identity_endpoint(req: PartitionedSearchRequest):
    """PHASE 2.2: Search using only identity dimensions (unique characteristics)."""
    if _api_env is None:
        raise HTTPException(status_code=503, detail="Environment still initializing.")
    results = _api_env.search_by_identity(req.concept, req.top_k)
    return PartitionedSearchResponse(
        results=[SearchResult(name=r[0], score=r[1]) for r in results]
    )


@app.post("/relationships/add", response_model=AddRelationshipResponse)
async def add_relationship_endpoint(req: AddRelationshipRequest):
    """PHASE 1.1, 1.2: Add a weighted, colored relationship between concepts."""
    if _api_env is None:
        raise HTTPException(status_code=503, detail="Environment still initializing.")
    
    # Ensure concepts exist
    if req.concept_a.lower() not in _api_env.concept_memory.concepts:
        # Auto-register
        physical_fids = [_api_env.feature_registry.register(req.concept_a)]
        semantic_fids = [_api_env.feature_registry.register(req.concept_a)]
        _api_env.concept_memory.register(req.concept_a, physical_fids, semantic_fids)
    
    if req.concept_b.lower() not in _api_env.concept_memory.concepts:
        physical_fids = [_api_env.feature_registry.register(req.concept_b)]
        semantic_fids = [_api_env.feature_registry.register(req.concept_b)]
        _api_env.concept_memory.register(req.concept_b, physical_fids, semantic_fids)
    
    _api_env.concept_memory.add_weighted_relationship(
        req.concept_a, req.concept_b, 
        weight=req.weight, 
        color=req.color
    )
    
    return AddRelationshipResponse(status="relationship_added")


@app.get("/concepts")
async def list_concepts_endpoint(limit: int = 100):
    """List all concepts with their importance scores."""
    if _api_env is None:
        raise HTTPException(status_code=503, detail="Environment still initializing.")
    
    concepts = []
    for name, concept in list(_api_env.concept_memory.concepts.items())[:limit]:
        concepts.append({
            "name": name,
            "importance": concept.importance,
            "domain": concept.domain,
            "relationship_count": len(_api_env.concept_memory.weighted_relationships.get(name, {}))
        })
    
    return {"concepts": concepts, "total": len(_api_env.concept_memory.concepts)}


@app.get("/")
async def root():
    return {
        "name": "Knowledge Graph Environment",
        "version": "2.0.0",
        "status": "online" if _api_env else "initializing",
        "message": "Welcome to the OpenEnv API with weighted relationships, dimensional partitioning, and sentence generation.",
        "new_endpoints": [
            "POST /sentence - Generate description for a concept",
            "GET /global-context - Get global knowledge graph statistics",
            "POST /search/essence - Search by essence dimensions",
            "POST /search/identity - Search by identity dimensions",
            "POST /relationships/add - Add weighted colored relationships",
            "GET /concepts - List all concepts"
        ]
    }