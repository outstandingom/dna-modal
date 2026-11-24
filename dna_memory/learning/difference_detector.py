"""
Difference Detection Engine: Analyzes commonalities and distinct differences
between concept structures (e.g. comparing neem, mango, and guava leaf assemblies).

Prevents destructive false-merging by explicitly isolating unique components,
divergent predicates, and attribute variance.
"""

from dataclasses import dataclass, field
import datetime
from typing import Dict, List, Optional, Set, Any, Tuple
import numpy as np
from dna_memory.core.concept_identity import ConceptIdentityRegistry
from dna_memory.relations.relationship_lifecycle import RelationshipLifecycleEngine


@dataclass
class StructuralComparisonReport:
    """Detailed differential report between two or more concept structures."""
    entity_a: str
    entity_b: str
    shared_components: List[str]
    unique_to_a: List[str]
    unique_to_b: List[str]
    shared_relationships: List[Tuple[str, str]]   # (predicate, object_id)
    different_relationships: List[Dict[str, Any]] # predicate, obj_a, obj_b
    different_attributes: List[Dict[str, Any]]    # attr, val_a, val_b
    jaccard_similarity: float
    confidence: float
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def summary(self) -> str:
        lines = [
            f"=== STRUCTURAL COMPARISON: {self.entity_a} vs {self.entity_b} ===",
            f"Jaccard Structural Similarity: {self.jaccard_similarity:.3f} | Confidence: {self.confidence:.2f}",
            f"Shared Components ({len(self.shared_components)}):   {self.shared_components}",
            f"Unique to {self.entity_a} ({len(self.unique_to_a)}):  {self.unique_to_a}",
            f"Unique to {self.entity_b} ({len(self.unique_to_b)}):  {self.unique_to_b}",
            f"Shared Relationships ({len(self.shared_relationships)}): {self.shared_relationships}",
            f"Different Attributes: {len(self.different_attributes)}"
        ]
        return "\n".join(lines)


class DifferenceDetector:
    """
    Performs precise graph-level diffing between concepts:
      - Isolates common canonical bonds.
      - Highlights distinguishing characteristic features.
      - Quantifies structural overlap.
    """
    def __init__(
        self,
        relationship_engine: RelationshipLifecycleEngine,
        concept_registry: Optional[ConceptIdentityRegistry] = None
    ):
        self.relationship_engine = relationship_engine
        self.concept_registry = concept_registry

    def compare_entities(self, entity_a_id: str, entity_b_id: str) -> StructuralComparisonReport:
        """Compares two entities across relationships, components, and attributes."""
        # 1. Fetch active relationships for A and B
        rels_a = self.relationship_engine.find_relationships(subject_id=entity_a_id, only_active=True)
        rels_b = self.relationship_engine.find_relationships(subject_id=entity_b_id, only_active=True)

        pairs_a = set((r.predicate, r.object_id) for r in rels_a)
        pairs_b = set((r.predicate, r.object_id) for r in rels_b)

        objs_a = set(r.object_id for r in rels_a)
        objs_b = set(r.object_id for r in rels_b)

        shared_components = sorted(list(objs_a & objs_b))
        unique_to_a = sorted(list(objs_a - objs_b))
        unique_to_b = sorted(list(objs_b - objs_a))

        shared_rels = sorted(list(pairs_a & pairs_b))

        # Different relationships (same predicate, different object)
        preds_a = {r.predicate: r.object_id for r in rels_a}
        preds_b = {r.predicate: r.object_id for r in rels_b}

        diff_rels = []
        for p in set(preds_a.keys()) | set(preds_b.keys()):
            val_a = preds_a.get(p)
            val_b = preds_b.get(p)
            if val_a != val_b:
                diff_rels.append({
                    "predicate": p,
                    "target_in_a": val_a,
                    "target_in_b": val_b
                })

        # Attributes diff from concept registry
        attrs_a = {}
        attrs_b = {}
        if self.concept_registry:
            if self.concept_registry.exists(entity_a_id):
                attrs_a = self.concept_registry.get(entity_a_id).attributes
            if self.concept_registry.exists(entity_b_id):
                attrs_b = self.concept_registry.get(entity_b_id).attributes

        diff_attrs = []
        all_attr_keys = set(attrs_a.keys()) | set(attrs_b.keys())
        for k in all_attr_keys:
            v_a = attrs_a.get(k)
            v_b = attrs_b.get(k)
            if v_a != v_b:
                diff_attrs.append({
                    "attribute": k,
                    "value_in_a": v_a,
                    "value_in_b": v_b
                })

        # Jaccard similarity across total relational bonds
        union_pairs = pairs_a | pairs_b
        jaccard = (len(pairs_a & pairs_b) / len(union_pairs)) if union_pairs else 1.0

        confidence = 0.9 if (len(rels_a) >= 2 and len(rels_b) >= 2) else 0.65

        return StructuralComparisonReport(
            entity_a=entity_a_id,
            entity_b=entity_b_id,
            shared_components=shared_components,
            unique_to_a=unique_to_a,
            unique_to_b=unique_to_b,
            shared_relationships=shared_rels,
            different_relationships=diff_rels,
            different_attributes=diff_attrs,
            jaccard_similarity=jaccard,
            confidence=confidence
        )

    def compare_multiple_entities(self, entity_ids: List[str]) -> Dict[str, Any]:
        """
        Computes pairwise differential reports across a collection (e.g. neem, mango, guava)
        and identifies universal components shared across all members.
        """
        if len(entity_ids) < 2:
            return {"error": "At least 2 entities required for comparison."}

        # Find universal components
        universal_components: Optional[Set[str]] = None
        for eid in entity_ids:
            rels = self.relationship_engine.find_relationships(subject_id=eid, only_active=True)
            objs = set(r.object_id for r in rels)
            if universal_components is None:
                universal_components = objs
            else:
                universal_components &= objs

        pairwise_reports = []
        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                rep = self.compare_entities(entity_ids[i], entity_ids[j])
                pairwise_reports.append(rep)

        return {
            "entity_count": len(entity_ids),
            "universal_components": sorted(list(universal_components or set())),
            "pairwise_comparisons": pairwise_reports
        }
