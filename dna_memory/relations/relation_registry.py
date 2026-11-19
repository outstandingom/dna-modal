"""
Relational Registry: Stores, indexes, and manages first-class Relationship DNA units.
"""

from typing import Dict, List, Optional
import numpy as np
from dna_memory.core.dna_registry import DNARegistry
from dna_memory.relations.relation_dna import RelationDNA, RelationEncoder, REVERSIBLE_RELATION_MAP

class RelationRegistry:
    """
    Central registry for typed relationship DNA.
    Indexed by subject, object, and relation_type for O(1) multi-hop navigation.
    """
    def __init__(self, dna_registry: DNARegistry):
        self.dna_registry = dna_registry
        self.relation_encoder = RelationEncoder(dna_registry.encoder)
        self._relations: Dict[str, RelationDNA] = {}
        self._by_subject: Dict[str, List[str]] = {}
        self._by_object: Dict[str, List[str]] = {}
        self._by_type: Dict[str, List[str]] = {}
        self._next_rel_seq = 1

    def create_relation(
        self,
        subject_id: str,
        relation_type: str,
        object_id: str,
        confidence: float = 1.0,
        strength: float = 1.0,
        metadata: Optional[dict] = None,
        custom_id: Optional[str] = None
    ) -> RelationDNA:
        """Constructs and registers a first-class Relationship DNA entity."""
        rel_type = relation_type.lower().strip()
        rel_id = custom_id or f"rel_{rel_type[:4]}_{self._next_rel_seq:06d}"
        self._next_rel_seq += 1

        # Retrieve vectors for subject and object
        sub_atom = self.dna_registry.get(subject_id)
        obj_atom = self.dna_registry.get(object_id)

        sub_vec = sub_atom.vector if sub_atom else self.dna_registry.encoder.encode_text(subject_id)
        obj_vec = obj_atom.vector if obj_atom else self.dna_registry.encoder.encode_text(object_id)

        rel_vec = self.relation_encoder.encode_relation(sub_vec, rel_type, obj_vec)

        rel = RelationDNA(
            id=rel_id,
            subject_id=subject_id,
            relation_type=rel_type,
            object_id=object_id,
            vector=rel_vec,
            confidence=confidence,
            strength=strength,
            metadata=metadata or {}
        )

        self._relations[rel_id] = rel
        self._by_subject.setdefault(subject_id, []).append(rel_id)
        self._by_object.setdefault(object_id, []).append(rel_id)
        self._by_type.setdefault(rel_type, []).append(rel_id)
        return rel

    def get(self, relation_id: str) -> Optional[RelationDNA]:
        return self._relations.get(relation_id)

    def retrieve_relations(
        self,
        subject_id: Optional[str] = None,
        object_id: Optional[str] = None,
        relation_type: Optional[str] = None
    ) -> List[RelationDNA]:
        """Query relations with multi-facet filtering."""
        candidate_ids = None

        if subject_id is not None:
            s_ids = set(self._by_subject.get(subject_id, []))
            candidate_ids = s_ids if candidate_ids is None else candidate_ids & s_ids

        if object_id is not None:
            o_ids = set(self._by_object.get(object_id, []))
            candidate_ids = o_ids if candidate_ids is None else candidate_ids & o_ids

        if relation_type is not None:
            t_ids = set(self._by_type.get(relation_type.lower().strip(), []))
            candidate_ids = t_ids if candidate_ids is None else candidate_ids & t_ids

        if candidate_ids is None:
            return list(self._relations.values())

        return [self._relations[rid] for rid in candidate_ids if rid in self._relations]

    def reverse_relation(self, relation_id: str) -> Optional[RelationDNA]:
        """Creates or returns the inverse relation (e.g. purchased -> purchased_by)."""
        rel = self.get(relation_id)
        if not rel:
            return None

        inv_type = REVERSIBLE_RELATION_MAP.get(rel.relation_type, f"inverse_{rel.relation_type}")
        existing = self.retrieve_relations(
            subject_id=rel.object_id,
            relation_type=inv_type,
            object_id=rel.subject_id
        )
        if existing:
            return existing[0]

        return self.create_relation(
            subject_id=rel.object_id,
            relation_type=inv_type,
            object_id=rel.subject_id,
            confidence=rel.confidence,
            strength=rel.strength,
            metadata={"inverse_of": rel.id}
        )

    def update_relation_strength(self, relation_id: str, delta: float) -> float:
        """Strengthens or weakens a relationship based on feedback or recurrence."""
        rel = self.get(relation_id)
        if rel:
            rel.strength = max(0.01, min(10.0, rel.strength + delta))
            return rel.strength
        return 0.0

    def count(self) -> int:
        return len(self._relations)

    def serialize(self) -> dict:
        """Serializes all relations and indices."""
        return {
            "next_rel_seq": self._next_rel_seq,
            "relations": {rid: rel.to_dict() for rid, rel in self._relations.items()}
        }

    def restore(self, data: dict):
        """Restores relations and rebuilds subject/object/type inverted indices."""
        self._next_rel_seq = data.get("next_rel_seq", 1)
        self._relations.clear()
        self._by_subject.clear()
        self._by_object.clear()
        self._by_type.clear()
        for rid, rdata in data.get("relations", {}).items():
            rel = RelationDNA.from_dict(rdata)
            self._relations[rid] = rel
            self._by_subject.setdefault(rel.subject_id, []).append(rid)
            self._by_object.setdefault(rel.object_id, []).append(rid)
            self._by_type.setdefault(rel.relation_type, []).append(rid)
