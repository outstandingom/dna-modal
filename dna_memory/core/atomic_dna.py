"""
Atomic DNA Engine: Stores the smallest, reusable unit of knowledge.
An AtomicDNA representation encapsulates one identifiable feature, fact, event attribute, or entity.
"""

from dataclasses import dataclass, field
import datetime
import numpy as np
from typing import Dict, Any, Optional

@dataclass
class DNAFeatureMetadata:
    """Stores the semantic meaning, source, domain, and confidence of a feature."""
    meaning: str
    source: str = "direct_observation"
    domain: str = "general"
    feature_type: str = "property"  # "property", "attribute", "entity", "event_attribute"
    confidence: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "meaning": self.meaning,
            "source": self.source,
            "domain": self.domain,
            "feature_type": self.feature_type,
            "confidence": self.confidence,
            "extra": self.extra
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DNAFeatureMetadata':
        return cls(
            meaning=data.get("meaning", ""),
            source=data.get("source", "direct_observation"),
            domain=data.get("domain", "general"),
            feature_type=data.get("feature_type", "property"),
            confidence=float(data.get("confidence", 1.0)),
            extra=data.get("extra", {})
        )

@dataclass
class AtomicDNA:
    """
    Represents an atomic, indivisible representation in the DNA memory system.
    """
    id: str
    type: str  # "property", "entity", "attribute", "relation_role"
    value: Any
    vector: np.ndarray
    source: str = "default"
    confidence: float = 1.0
    metadata: Optional[DNAFeatureMetadata] = None
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def __post_init__(self):
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.asarray(self.vector, dtype=np.float64)
        # Normalize upon initialization
        norm = np.linalg.norm(self.vector)
        if norm > 1e-12:
            self.vector = self.vector / norm

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "value": str(self.value),
            "vector": self.vector.tolist(),
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AtomicDNA':
        meta = DNAFeatureMetadata.from_dict(data["metadata"]) if data.get("metadata") else None
        return cls(
            id=data["id"],
            type=data["type"],
            value=data["value"],
            vector=np.array(data["vector"], dtype=np.float64),
            source=data.get("source", "default"),
            confidence=float(data.get("confidence", 1.0)),
            metadata=meta,
            created_at=data.get("created_at", "")
        )
