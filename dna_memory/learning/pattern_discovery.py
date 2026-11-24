"""
Pattern Discovery Engine: Identifies recurring sub-graphs, structural motifs,
and relationship templates from multiple empirical observations.

Transforms raw episodic occurrences into generalized, reusable structural patterns
with evidence tracking and policy-governed promotion.
"""

from dataclasses import dataclass, field
import datetime
from collections import Counter
import threading
from typing import Dict, List, Optional, Set, Any, Tuple
from dna_memory.relations.relationship_lifecycle import RelationshipLifecycleEngine


@dataclass
class DiscoveredPattern:
    """A recurring relational pattern mined from instance observations."""
    pattern_id: str
    pattern_name: str
    domain: str
    recurring_predicates: List[str]
    shared_object_ids: List[str]
    frequency: int
    confidence: float
    supporting_instances: List[str]
    status: str = "proposed"  # "proposed", "approved", "rejected"
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "domain": self.domain,
            "recurring_predicates": self.recurring_predicates,
            "shared_object_ids": self.shared_object_ids,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "supporting_instances": self.supporting_instances,
            "status": self.status,
            "created_at": self.created_at
        }


class PatternDiscoveryEngine:
    """
    Mines graph relationships across collections of entities to surface
    recurring architectural schemas, common properties, and structural regularities.
    """
    def __init__(self, relationship_engine: RelationshipLifecycleEngine):
        self.relationship_engine = relationship_engine
        self._patterns: Dict[str, DiscoveredPattern] = {}
        self._lock = threading.RLock()

    def discover_patterns(
        self,
        instance_ids: List[str],
        domain: str = "general",
        min_frequency: int = 2,
        min_support_ratio: float = 0.5
    ) -> List[DiscoveredPattern]:
        """
        Analyzes relationships originating from instance_ids.
        Identifies predicates and objects that recur across at least min_support_ratio of instances.
        """
        with self._lock:
            if not instance_ids:
                return []

            total_instances = len(instance_ids)
            predicate_counts = Counter()
            target_counts = Counter()
            pair_counts = Counter()  # (predicate, object_id)

            instance_subgraphs: Dict[str, Set[Tuple[str, str]]] = {}

            for inst_id in instance_ids:
                rels = self.relationship_engine.find_relationships(subject_id=inst_id, only_active=True)
                pairs = set((r.predicate, r.object_id) for r in rels)
                instance_subgraphs[inst_id] = pairs

                for p, o in pairs:
                    predicate_counts[p] += 1
                    target_counts[o] += 1
                    pair_counts[(p, o)] += 1

            min_count = max(min_frequency, int(total_instances * min_support_ratio))
            discovered: List[DiscoveredPattern] = []

            # Find recurring relational motifs
            recurring_pairs = [pair for pair, count in pair_counts.items() if count >= min_count]
            if recurring_pairs:
                recurring_preds = sorted(list(set(p for p, o in recurring_pairs)))
                recurring_objs = sorted(list(set(o for p, o in recurring_pairs)))

                supporting = [
                    inst_id for inst_id, pairs in instance_subgraphs.items()
                    if any(pair in pairs for pair in recurring_pairs)
                ]

                pattern_id = f"pattern_{domain.lower()}_{len(self._patterns) + 1:04d}"
                confidence = min(0.95, len(supporting) / total_instances)

                pattern = DiscoveredPattern(
                    pattern_id=pattern_id,
                    pattern_name=f"Recurring {domain.title()} Schema ({len(recurring_pairs)} bonds)",
                    domain=domain,
                    recurring_predicates=recurring_preds,
                    shared_object_ids=recurring_objs,
                    frequency=len(supporting),
                    confidence=confidence,
                    supporting_instances=supporting,
                    status="approved" if confidence >= 0.8 else "proposed"
                )
                self._patterns[pattern_id] = pattern
                discovered.append(pattern)

            return discovered

    def get_pattern(self, pattern_id: str) -> Optional[DiscoveredPattern]:
        with self._lock:
            return self._patterns.get(pattern_id)

    def list_all_patterns(self) -> List[DiscoveredPattern]:
        with self._lock:
            return list(self._patterns.values())
