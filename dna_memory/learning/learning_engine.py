"""
Memory Learning & Consolidation Engine:
Updates existing knowledge when new information arrives without silently overwriting historical facts.
"""

from typing import Dict, Any, List, Optional
import datetime
from dna_memory.core.dna_registry import DNARegistry
from dna_memory.relations.relation_registry import RelationRegistry
from dna_memory.assembly.recursive_composition import AssemblyEngine

class MemoryLearningEngine:
    """
    Handles event learning, memory consolidation, relationship reinforcement, and contradiction detection.
    """
    def __init__(
        self,
        dna_registry: DNARegistry,
        relation_registry: RelationRegistry,
        assembly_engine: AssemblyEngine
    ):
        self.dna_registry = dna_registry
        self.relation_registry = relation_registry
        self.assembly_engine = assembly_engine
        self._event_history: List[dict] = []

    def learn_from_event(
        self,
        event_type: str,
        subject_id: str,
        object_id: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        1. Creates an Episodic Event representation.
        2. Links event to subject and object via first-class Relation DNA.
        3. Updates or creates the Behaviour Assembly for the subject.
        4. Preserves historical observation intact.
        """
        ts = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
        attrs = attributes or {}

        # 1. Register event in atomic registry
        event_val = f"{event_type}_{len(self._event_history)+1}"
        event_atom = self.dna_registry.register(
            value=event_val,
            dna_type="event",
            source="event_stream",
            metadata=None
        )

        # 2. Create Relations:
        # Subject -> [participated_in] -> Event
        # Event -> [targets] -> Object
        rel_sub = self.relation_registry.create_relation(
            subject_id=subject_id,
            relation_type="participated_in",
            object_id=event_atom.id,
            metadata={"timestamp": ts, "event_type": event_type}
        )

        rel_obj = self.relation_registry.create_relation(
            subject_id=event_atom.id,
            relation_type=event_type,
            object_id=object_id,
            metadata={"timestamp": ts}
        )

        # Also create direct subject -> [event_type] -> object relation (or strengthen existing)
        existing_direct = self.relation_registry.retrieve_relations(
            subject_id=subject_id,
            relation_type=event_type,
            object_id=object_id
        )
        if existing_direct:
            for r in existing_direct:
                self.relation_registry.update_relation_strength(r.id, delta=0.5)
            direct_rel_id = existing_direct[0].id
        else:
            direct_rel = self.relation_registry.create_relation(
                subject_id=subject_id,
                relation_type=event_type,
                object_id=object_id,
                metadata={"first_seen": ts}
            )
            direct_rel_id = direct_rel.id

        # 3. Create or Update Customer Behaviour Assembly
        behaviour_asmb_id = f"asmb_behaviour_{subject_id}"
        existing_asmb = self.assembly_engine.get(behaviour_asmb_id)

        if existing_asmb:
            self.assembly_engine.add_component(
                assembly_id=behaviour_asmb_id,
                component_id=event_atom.id,
                weight=1.0,
                relation_id=direct_rel_id
            )
        else:
            self.assembly_engine.create_assembly(
                assembly_type="CustomerBehaviour",
                component_ids=[subject_id, event_atom.id],
                relation_ids=[direct_rel_id],
                custom_id=behaviour_asmb_id,
                metadata={"subject_id": subject_id}
            )

        event_record = {
            "event_id": event_atom.id,
            "event_type": event_type,
            "subject_id": subject_id,
            "object_id": object_id,
            "timestamp": ts,
            "attributes": attrs,
            "direct_relation_id": direct_rel_id
        }
        self._event_history.append(event_record)
        return event_record

    def get_event_history(self, subject_id: Optional[str] = None) -> List[dict]:
        if subject_id:
            return [e for e in self._event_history if e["subject_id"] == subject_id]
        return list(self._event_history)
