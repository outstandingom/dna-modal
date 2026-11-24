"""
Memory Consolidation Engine: Deduplicates observations, merges evidence streams,
consolidates redundant relational assertions, and elevates recurring patterns into canonical assemblies
WITHOUT destroying underlying historical provenance or factual evidence.
"""

from dataclasses import dataclass, field
import datetime
import threading
from typing import Dict, List, Optional, Set, Any, Tuple
from dna_memory.relations.relationship_lifecycle import RelationshipLifecycleEngine, RelationshipRecord
from dna_memory.core.concept_identity import ConceptIdentityRegistry


@dataclass
class ConsolidationAudit:
    """Record of an automated consolidation operation."""
    audit_id: str
    consolidated_at: str
    duplicate_relationships_merged: int
    evidence_items_aggregated: int
    aliases_merged: int
    actions_taken: List[str] = field(default_factory=list)


class MemoryConsolidator:
    """
    Sleep/idle-time memory consolidation worker:
      - Detects and combines duplicate relational claims across multiple turns.
      - Aggregates evidence lists while keeping independent source counts.
      - Maintains full reversibility by appending to history rather than deleting.
    """
    def __init__(
        self,
        relationship_engine: RelationshipLifecycleEngine,
        concept_registry: Optional[ConceptIdentityRegistry] = None
    ):
        self.relationship_engine = relationship_engine
        self.concept_registry = concept_registry
        self._audits: List[ConsolidationAudit] = []
        self._lock = threading.RLock()

    def consolidate_relationships(self) -> ConsolidationAudit:
        """
        Scans all relationships and consolidates duplicate claims with identical (s, p, o).
        Merges evidence items and preserves distinct source attribution.
        """
        with self._lock:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            audit_id = f"audit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Group active relationships by (subject_id, predicate, object_id)
            groups: Dict[Tuple[str, str, str], List[RelationshipRecord]] = {}
            for rel in self.relationship_engine.find_relationships(only_active=True):
                key = (rel.subject_id, rel.predicate, rel.object_id)
                groups.setdefault(key, []).append(rel)

            merged_count = 0
            ev_count = 0
            actions = []

            for key, rel_list in groups.items():
                if len(rel_list) <= 1:
                    continue

                # Keep the primary relationship (oldest recorded)
                primary = rel_list[0]
                duplicates = rel_list[1:]

                for dup in duplicates:
                    # Ingest evidence from duplicate into primary
                    for ev in dup.evidence:
                        if primary.add_evidence(ev, update_confidence=True):
                            ev_count += 1

                    # Deactivate duplicate
                    self.relationship_engine.deactivate_relationship(
                        dup.relationship_id,
                        reason=f"consolidated_into_{primary.relationship_id}"
                    )
                    merged_count += 1

                actions.append(
                    f"Consolidated {len(duplicates)} duplicate(s) of '{key[0]} --[{key[1]}]--> {key[2]}' "
                    f"into primary '{primary.relationship_id}'."
                )

            audit = ConsolidationAudit(
                audit_id=audit_id,
                consolidated_at=now,
                duplicate_relationships_merged=merged_count,
                evidence_items_aggregated=ev_count,
                aliases_merged=0,
                actions_taken=actions
            )
            self._audits.append(audit)
            return audit

    def get_latest_audit(self) -> Optional[ConsolidationAudit]:
        with self._lock:
            return self._audits[-1] if self._audits else None
