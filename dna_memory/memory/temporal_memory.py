"""
Temporal Memory Engine: Bitemporal (valid time vs. transaction time) reasoning,
historical knowledge reconstruction, and state timeline synthesis.

Enables accurate answers to temporal queries (e.g. 'Where did X work in 2025?')
without corrupting or confusing current knowledge states.
"""

from dataclasses import dataclass, field
import datetime
import threading
from typing import Dict, List, Optional, Any, Tuple
from dna_memory.relations.relationship_lifecycle import RelationshipRecord, RelationshipLifecycleEngine
from dna_memory.core.concept_identity import ConceptIdentityRegistry, ConceptIdentity


@dataclass
class TemporalEvent:
    """An event with both real-world occurrence time and memory recording time."""
    event_id: str
    event_type: str
    entity_id: str
    valid_time: str      # When the event occurred in the world
    recorded_time: str   # When the memory system ingested it
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class EntityHistoricalState:
    """Reconstructed state of an entity as of a specific point in time."""
    canonical_id: str
    as_of_time: str
    attributes: Dict[str, Any]
    active_relationships: List[Dict[str, Any]]
    superseded_relationships: List[Dict[str, Any]]
    reconstructed_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


class TemporalMemoryEngine:
    """
    Coordinates bitemporal queries across Concept Identities and Relationships.
    Separates:
      - Valid Time: When a fact was objectively true in reality.
      - Transaction Time: When the memory system recorded the fact.
    """
    def __init__(
        self,
        relationship_engine: RelationshipLifecycleEngine,
        concept_registry: Optional[ConceptIdentityRegistry] = None
    ):
        self.relationship_engine = relationship_engine
        self.concept_registry = concept_registry
        self._events: List[TemporalEvent] = []
        self._lock = threading.RLock()

    def record_event(
        self,
        event_id: str,
        event_type: str,
        entity_id: str,
        valid_time: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0
    ) -> TemporalEvent:
        """Records an event with both valid real-world time and transaction time."""
        with self._lock:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            event = TemporalEvent(
                event_id=event_id,
                event_type=event_type,
                entity_id=entity_id,
                valid_time=valid_time or now,
                recorded_time=now,
                details=dict(details or {}),
                confidence=confidence
            )
            self._events.append(event)
            return event

    def query_relationships_at(
        self,
        as_of_time: str,
        subject_id: Optional[str] = None,
        predicate: Optional[str] = None,
        object_id: Optional[str] = None
    ) -> List[RelationshipRecord]:
        """
        Retrieves relationships that were objectively valid at time `as_of_time`.
        Accounts for relationships valid_from and valid_until.
        """
        with self._lock:
            # Query all relationships (including historical/superseded)
            all_rels = self.relationship_engine.find_relationships(
                subject_id=subject_id,
                predicate=predicate,
                object_id=object_id,
                only_active=False
            )

            valid_at_t = []
            for r in all_rels:
                if r.status == "deactivated":
                    continue
                # Must have started on or before as_of_time
                if r.valid_from <= as_of_time:
                    # And either still open, or ended AFTER as_of_time
                    if r.valid_until is None or r.valid_until > as_of_time:
                        valid_at_t.append(r)

            return valid_at_t

    def reconstruct_entity_state(self, canonical_id: str, as_of_time: str) -> EntityHistoricalState:
        """
        Reconstructs the complete historical state of an entity at `as_of_time`.
        Includes attributes at that moment and all active/superseded relationships.
        """
        with self._lock:
            # 1. Attributes reconstruction from concept identity history
            attributes = {}
            if self.concept_registry and self.concept_registry.exists(canonical_id):
                concept = self.concept_registry.get(canonical_id)
                # Walk through history snapshots to find the latest version on or before as_of_time
                matching_attrs = None
                for h in concept.history:
                    if h.timestamp <= as_of_time:
                        matching_attrs = h.attributes
                if matching_attrs is not None:
                    attributes = dict(matching_attrs)
                elif concept.created_at <= as_of_time:
                    attributes = dict(concept.attributes)

            # 2. Relationships at as_of_time
            valid_rels = self.query_relationships_at(as_of_time=as_of_time, subject_id=canonical_id)
            active_list = [r.to_dict() for r in valid_rels]

            # Superseded before as_of_time
            all_rels = self.relationship_engine.find_relationships(subject_id=canonical_id, only_active=False)
            superseded_list = []
            for r in all_rels:
                if r.valid_until and r.valid_until <= as_of_time:
                    superseded_list.append(r.to_dict())

            return EntityHistoricalState(
                canonical_id=canonical_id,
                as_of_time=as_of_time,
                attributes=attributes,
                active_relationships=active_list,
                superseded_relationships=superseded_list
            )

    def get_timeline_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Synthesizes a chronological timeline of all facts, updates, and events
        connected to an entity.
        """
        with self._lock:
            timeline: List[Dict[str, Any]] = []

            # 1. Relationship lifecycle events
            rels = self.relationship_engine.find_relationships(subject_id=entity_id, only_active=False)
            for r in rels:
                timeline.append({
                    "timestamp": r.valid_from,
                    "type": "relationship_started",
                    "predicate": r.predicate,
                    "target": r.object_id,
                    "relationship_id": r.relationship_id,
                    "confidence": r.confidence,
                    "status": r.status
                })
                if r.valid_until:
                    timeline.append({
                        "timestamp": r.valid_until,
                        "type": f"relationship_{r.status}",
                        "predicate": r.predicate,
                        "target": r.object_id,
                        "relationship_id": r.relationship_id,
                        "superseded_by": r.superseded_by
                    })

            # 2. Discrete temporal events
            for ev in self._events:
                if ev.entity_id == entity_id:
                    timeline.append({
                        "timestamp": ev.valid_time,
                        "type": f"event_{ev.event_type}",
                        "event_id": ev.event_id,
                        "details": ev.details,
                        "confidence": ev.confidence
                    })

            # Sort chronologically
            timeline.sort(key=lambda x: x["timestamp"])
            return timeline
