"""
Relational DNA Engine: Gives relationships their own first-class DNA representations.
Encodes subject, relation type, and object into a composite structure:
  R_ABC = bind(bind(Subject, Relation), Object)
"""

from dataclasses import dataclass, field
import datetime
from typing import Optional, Dict, Any
import numpy as np
from dna_memory.core.vector_operations import bind, normalize

REVERSIBLE_RELATION_MAP = {
    "purchased": "purchased_by",
    "owns": "owned_by",
    "created": "created_by",
    "belongs_to": "contains",
    "prefers": "preferred_by",
    "caused": "caused_by",
    "followed_by": "preceded_by"
}

@dataclass
class RelationDNA:
    """
    First-class DNA representation of a structured relationship.
    """
    id: str
    subject_id: str
    relation_type: str
    object_id: str
    vector: np.ndarray
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    confidence: float = 1.0
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "relation_type": self.relation_type,
            "object_id": self.object_id,
            "vector": self.vector.tolist(),
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "strength": self.strength,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RelationDNA':
        return cls(
            id=data["id"],
            subject_id=data["subject_id"],
            relation_type=data["relation_type"],
            object_id=data["object_id"],
            vector=np.array(data["vector"], dtype=np.float64),
            timestamp=data.get("timestamp", ""),
            confidence=float(data.get("confidence", 1.0)),
            strength=float(data.get("strength", 1.0)),
            metadata=data.get("metadata", {})
        )

class RelationEncoder:
    """
    Encodes relational triples:
      R_ABC = bind(bind(S_A, R_relation), O_B)
    """
    def __init__(self, dna_encoder):
        self.encoder = dna_encoder

    def encode_relation(
        self,
        subject_vec: np.ndarray,
        relation_type: str,
        object_vec: np.ndarray
    ) -> np.ndarray:
        r_type_vec = self.encoder.encode_text(f"RELATION:{relation_type.lower().strip()}")
        # Tripartite nested binding
        sr = bind(subject_vec, r_type_vec)
        sro = bind(sr, object_vec)
        return normalize(sro)
