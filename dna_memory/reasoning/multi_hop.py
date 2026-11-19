"""
Reasoning Engine: Frontier-based multi-hop deductive traversal.
Processes strictly the newly activated frontier at each step to prevent cyclic loops and explosion.
"""

from typing import List, Dict, Any, Set
from dna_memory.relations.relation_registry import RelationRegistry
from dna_memory.assembly.recursive_composition import AssemblyEngine

class FrontierReasoningEngine:
    """
    Reasoning Engine that walks relational bonds to discover causal and associative chains.
    """
    def __init__(self, relation_registry: RelationRegistry, assembly_engine: AssemblyEngine):
        self.relation_registry = relation_registry
        self.assembly_engine = assembly_engine

    def multi_hop_reason(
        self,
        start_entity_id: str,
        target_relation_chain: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Deductive reasoning along an exact or fuzzy relational sequence.
        Example:
          Rahul -> [purchased] -> PurchaseEvent -> [occurred_at] -> Time/Location
        """
        current_frontier = [{
            "current_id": start_entity_id,
            "path": [start_entity_id],
            "relations_traversed": [],
            "confidence": 1.0
        }]

        for hop_idx, rel_type in enumerate(target_relation_chain):
            next_frontier = []
            rel_type_lower = rel_type.lower().strip()

            for state in current_frontier:
                curr_id = state["current_id"]
                # Outbound relations
                outbound = self.relation_registry.retrieve_relations(
                    subject_id=curr_id,
                    relation_type=rel_type_lower
                )
                for r in outbound:
                    if r.object_id not in state["path"]:  # cycle prevention
                        next_frontier.append({
                            "current_id": r.object_id,
                            "path": state["path"] + [r.object_id],
                            "relations_traversed": state["relations_traversed"] + [r.id],
                            "confidence": state["confidence"] * r.confidence * (r.strength / 1.0)
                        })

                # Inbound / inverse relations
                inbound = self.relation_registry.retrieve_relations(
                    object_id=curr_id,
                    relation_type=rel_type_lower
                )
                for r in inbound:
                    if r.subject_id not in state["path"]:  # cycle prevention
                        next_frontier.append({
                            "current_id": r.subject_id,
                            "path": state["path"] + [r.subject_id],
                            "relations_traversed": state["relations_traversed"] + [f"inv_{r.id}"],
                            "confidence": state["confidence"] * r.confidence * 0.9
                        })

            current_frontier = next_frontier
            if not current_frontier:
                break

        return current_frontier
