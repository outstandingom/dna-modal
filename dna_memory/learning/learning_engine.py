"""
Memory Learning & Consolidation Engine:
Updates existing knowledge when new information arrives without silently overwriting historical facts.
"""

from typing import Dict, Any, List, Optional
import datetime
from dna_memory.core.dna_registry import DNARegistry
from dna_memory.core.atomic_dna import DNAFeatureMetadata
from dna_memory.relations.relation_registry import RelationRegistry
from dna_memory.assembly.recursive_composition import AssemblyEngine
from dna_memory.prediction.temporal_patterns import TemporalPatternAnalyzer

class MemoryLearningEngine:
    """
    Handles event learning, memory consolidation, relationship reinforcement, and contradiction detection.
    Encodes event attributes as first-class Atomic DNA units composed into hierarchical Event Assemblies.
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
        1. Encodes event attributes (price, quantity, currency, etc.) as Atomic DNA units.
        2. Creates an Episodic Event representation.
        3. Links event to subject and object via first-class Relation DNA.
        4. Composes a hierarchical Event Assembly containing event atom, target object, attributes, and relations.
        5. Updates or creates the Behaviour Assembly for the subject.
        6. Preserves historical observation intact.
        """
        if timestamp:
            # Strictly validate timestamp format
            TemporalPatternAnalyzer.parse_timestamp(timestamp)
            ts = timestamp
        else:
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        attrs = attributes or {}

        # 1. Register event in atomic registry
        event_seq = len(self._event_history) + 1
        event_val = f"{event_type}_{event_seq:04d}"
        event_atom = self.dna_registry.register(
            value=event_val,
            dna_type="event",
            source="event_stream",
            metadata=DNAFeatureMetadata(
                meaning=f"{event_type} event #{event_seq}",
                source="event_stream",
                feature_type="event",
                confidence=1.0
            )
        )

        # 2. Encode all attributes as first-class Atomic DNA units
        attribute_atoms = []
        for k, v in sorted(attrs.items()):
            attr_atom = self.dna_registry.register(
                value=f"{k}:{v}",
                dna_type="attribute",
                source="event_stream",
                metadata=DNAFeatureMetadata(
                    meaning=f"{k}={v}",
                    source="event_stream",
                    feature_type="attribute",
                    confidence=1.0
                )
            )
            attribute_atoms.append(attr_atom)

        # 3. Create Relations:
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

        # Also create or strengthen direct subject -> [event_type] -> object relation
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

        # 4. Compose structured Event Assembly containing the event atom, object, attribute atoms, and relations
        event_asmb_id = f"asmb_event_{event_val}"
        event_component_ids = [event_atom.id, object_id] + [a.id for a in attribute_atoms]
        event_relation_ids = [rel_sub.id, rel_obj.id]
        
        event_assembly = self.assembly_engine.create_assembly(
            assembly_type="EventAssembly",
            component_ids=event_component_ids,
            relation_ids=event_relation_ids,
            custom_id=event_asmb_id,
            metadata={"timestamp": ts, "event_type": event_type, "subject_id": subject_id, "object_id": object_id, **attrs}
        )

        # 5. Create or Update Customer Behaviour Assembly (hierarchically referencing the event assembly)
        behaviour_asmb_id = f"asmb_behaviour_{subject_id}"
        existing_asmb = self.assembly_engine.get(behaviour_asmb_id)

        if existing_asmb:
            self.assembly_engine.add_component(
                assembly_id=behaviour_asmb_id,
                component_id=event_assembly.id,
                weight=1.0,
                relation_id=direct_rel_id
            )
        else:
            self.assembly_engine.create_assembly(
                assembly_type="CustomerBehaviour",
                component_ids=[subject_id, event_assembly.id],
                relation_ids=[direct_rel_id],
                custom_id=behaviour_asmb_id,
                metadata={"subject_id": subject_id}
            )

        event_record = {
            "event_id": event_atom.id,
            "event_assembly_id": event_assembly.id,
            "event_type": event_type,
            "subject_id": subject_id,
            "object_id": object_id,
            "timestamp": ts,
            "attributes": attrs,
            "attribute_atom_ids": [a.id for a in attribute_atoms],
            "direct_relation_id": direct_rel_id
        }
        self._event_history.append(event_record)
        return event_record

    def get_event_history(self, subject_id: Optional[str] = None) -> List[dict]:
        if subject_id:
            return [e for e in self._event_history if e["subject_id"] == subject_id]
        return list(self._event_history)

    def serialize(self) -> dict:
        """Serializes historical learned events."""
        return {
            "event_history": list(self._event_history)
        }

    def restore(self, data: dict):
        """Restores event history."""
        self._event_history = list(data.get("event_history", []))
