"""
DNA Activation Engine: Spreads activation across relational and hierarchical DNA structures.
Uses frontier-based expansion, decay factors, and cycle prevention.
"""

from typing import List, Dict, Any, Set, Tuple
import numpy as np
from dna_memory.core.dna_registry import DNARegistry
from dna_memory.relations.relation_registry import RelationRegistry
from dna_memory.assembly.recursive_composition import AssemblyEngine
from dna_memory.activation.retrieval_index import RetrievalIndex

class DNAActivationEngine:
    """
    Spreading activation engine over DNA memory representations.
    Moves from initial query to relevant assemblies, relations, and atomic facts.
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
        self.index = RetrievalIndex(dims=dna_registry.encoder.dims)

    def sync_index(self):
        """Indexes all atomic DNA and assemblies for fast seeded retrieval."""
        for atom in self.dna_registry._by_id.values():
            self.index.add(atom.id, atom.vector, {"type": "atomic", "value": atom.value})

        for asmb in self.assembly_engine._assemblies.values():
            self.index.add(asmb.id, asmb.vector, {"type": "assembly", "assembly_type": asmb.assembly_type})

    def activate_from_query(
        self,
        query_text: str,
        top_k: int = 5,
        max_hops: int = 2,
        decay: float = 0.65
    ) -> List[Dict[str, Any]]:
        """
        Executes query-driven spreading activation.
        1. Encodes query into DNA space.
        2. Retrieves initial seed nodes.
        3. Spreads activation across relational and assembly frontiers.
        4. Returns activated items ranked by total activation with explanations.
        """
        q_vec = self.dna_registry.encoder.encode_text(query_text)
        self.sync_index()

        initial_seeds = self.index.query(q_vec, top_k=top_k, min_score=0.1)
        if not initial_seeds:
            # Fallback to direct token matching
            for word in query_text.lower().split():
                atom = self.dna_registry.find_by_value(word)
                if atom:
                    initial_seeds.append((atom.id, 1.0, {"type": "atomic", "value": atom.value}))

        activation_map: Dict[str, float] = {}
        explanations: Dict[str, List[str]] = {}
        visited: Set[str] = set()

        frontier: List[Tuple[str, float]] = []
        for item_id, score, meta in initial_seeds:
            activation_map[item_id] = score
            explanations[item_id] = [f"Direct query seed similarity: {score:.3f}"]
            frontier.append((item_id, score))
            visited.add(item_id)

        # Multi-hop spreading activation across newly activated frontier
        for hop in range(max_hops):
            if not frontier:
                break
            next_frontier: List[Tuple[str, float]] = []

            for current_id, current_energy in frontier:
                spread_energy = current_energy * decay

                # 1. Expand via Relations (as subject or object)
                rels = self.relation_registry.retrieve_relations(subject_id=current_id)
                for r in rels:
                    target_id = r.object_id
                    effective_energy = spread_energy * r.strength * r.confidence
                    if target_id not in visited and effective_energy > 0.05:
                        visited.add(target_id)
                        activation_map[target_id] = activation_map.get(target_id, 0.0) + effective_energy
                        explanations.setdefault(target_id, []).append(
                            f"Hop {hop+1}: Traversed relation [{r.relation_type}] from {current_id} (energy: {effective_energy:.3f})"
                        )
                        next_frontier.append((target_id, effective_energy))

                # Reverse relations
                rev_rels = self.relation_registry.retrieve_relations(object_id=current_id)
                for r in rev_rels:
                    target_id = r.subject_id
                    effective_energy = spread_energy * r.strength * r.confidence * 0.8
                    if target_id not in visited and effective_energy > 0.05:
                        visited.add(target_id)
                        activation_map[target_id] = activation_map.get(target_id, 0.0) + effective_energy
                        explanations.setdefault(target_id, []).append(
                            f"Hop {hop+1}: Reverse relation [{r.relation_type}] from {current_id} (energy: {effective_energy:.3f})"
                        )
                        next_frontier.append((target_id, effective_energy))

                # 2. Expand via Parent Assemblies
                parent_assemblies = self.assembly_engine.find_parent_assemblies(current_id)
                for pid in parent_assemblies:
                    if pid not in visited:
                        visited.add(pid)
                        activation_map[pid] = activation_map.get(pid, 0.0) + spread_energy
                        explanations.setdefault(pid, []).append(
                            f"Hop {hop+1}: Activated parent assembly from constituent {current_id}"
                        )
                        next_frontier.append((pid, spread_energy))

            frontier = next_frontier

        # Sort and package activated memories
        ranked = sorted(activation_map.items(), key=lambda x: x[1], reverse=True)
        results = []
        for node_id, energy in ranked:
            item_type = "unknown"
            raw_data = None
            if node_id in self.assembly_engine._assemblies:
                item_type = "assembly"
                raw_data = self.assembly_engine._assemblies[node_id]
            elif self.dna_registry.get(node_id):
                item_type = "atomic"
                raw_data = self.dna_registry.get(node_id)
            elif self.relation_registry.get(node_id):
                item_type = "relation"
                raw_data = self.relation_registry.get(node_id)

            results.append({
                "id": node_id,
                "type": item_type,
                "activation_score": float(energy),
                "explanations": explanations.get(node_id, []),
                "data": raw_data
            })

        return results
