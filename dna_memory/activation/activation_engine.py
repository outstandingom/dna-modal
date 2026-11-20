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
        """Indexes atomic DNA and semantic assemblies for fast seeded retrieval."""
        self.index._ids.clear()
        self.index._vectors.clear()
        self.index._id_to_idx.clear()
        self.index._metadata.clear()
        self.index._is_clustered = False

        for atom in self.dna_registry._by_id.values():
            if str(atom.value).startswith("query_snippet:") or atom.id.isdigit():
                continue
            self.index.add(atom.id, atom.vector, {"type": "atomic", "value": atom.value})

        for asmb in self.assembly_engine._assemblies.values():
            if asmb.id.startswith("asmb_event_") or asmb.id.startswith("asmb_behaviour_"):
                continue
            asmb_name = asmb.metadata.get("name", asmb.id)
            self.index.add(asmb.id, asmb.vector, {"type": "assembly", "assembly_type": asmb.assembly_type, "name": asmb_name})

    def activate_from_query(
        self,
        query_text: str,
        top_k: int = 5,
        max_hops: int = 2,
        decay: float = 0.65
    ) -> List[Dict[str, Any]]:
        """
        Executes query-driven spreading activation using Compositional Vector Bundling (HRR)
        and semantic concept matching.
        1. Tokenizes query and bundles constituent concept vectors into a composite query vector.
        2. Discovers direct concept seeds (exact & subword matching across registry atoms and assemblies).
        3. Queries ANN index with high noise-floor rejection threshold (min_score=0.28).
        4. Spreads activation across relational and assembly frontiers.
        5. Returns activated items ranked by total activation with explanations.
        """
        self.sync_index()

        q_lower = query_text.lower().strip()
        import re
        tokens = re.findall(r'[a-zA-Z0-9_\-\.\@]+', q_lower)
        stopwords = {
            "is", "a", "an", "the", "i", "can", "with", "or", "do", "you", "to", "for",
            "in", "on", "at", "of", "and", "how", "what", "where", "who", "when", "why",
            "which", "my", "your", "we", "us", "are", "have", "has", "had", "will", "would",
            "could", "should", "tell", "give", "please", "some", "any", "does", "get"
        }
        clean_tokens = [t for t in tokens if t not in stopwords and len(t) > 1]

        seeds_dict: Dict[str, Tuple[float, dict]] = {}

        # 1. Direct Concept & Entity Matching (High precision)
        # Search atomic concept values and IDs
        for atom in list(self.dna_registry._by_id.values()):
            if str(atom.value).startswith("query_snippet:") or atom.id.isdigit():
                continue
            atom_val = str(atom.value).lower().strip()
            atom_id_clean = atom.id.lower().replace("_", " ")

            if atom_val and (atom_val in q_lower or (len(atom_val) > 2 and any(t == atom_val for t in clean_tokens))):
                seeds_dict[atom.id] = (1.0, {"type": "atomic", "value": atom.value})
            elif any(t in atom_val.split() for t in clean_tokens if len(t) > 2):
                seeds_dict[atom.id] = (0.85, {"type": "atomic", "value": atom.value})
            elif any(t in atom_id_clean.split() for t in clean_tokens if len(t) > 2):
                seeds_dict[atom.id] = (0.80, {"type": "atomic", "value": atom.value})
            elif atom.id.lower().replace("_", "") in q_lower.replace(" ", ""):
                seeds_dict[atom.id] = (0.90, {"type": "atomic", "value": atom.value})

        # Search assemblies
        for asmb in list(self.assembly_engine._assemblies.values()):
            if asmb.id.startswith("asmb_event_") or asmb.id.startswith("asmb_behaviour_"):
                continue
            asmb_name = asmb.metadata.get("name", "").lower().strip()
            asmb_id_clean = asmb.id.lower().replace("_", " ")
            asmb_name_collapsed = asmb_name.replace(" ", "")
            q_collapsed = q_lower.replace(" ", "")

            if asmb_name and (asmb_name in q_lower or (len(asmb_name_collapsed) > 3 and asmb_name_collapsed in q_collapsed)):
                seeds_dict[asmb.id] = (1.0, {"type": "assembly", "assembly_type": asmb.assembly_type})
            elif any(t in asmb_name.split() for t in clean_tokens if len(t) > 3):
                seeds_dict[asmb.id] = (0.85, {"type": "assembly", "assembly_type": asmb.assembly_type})
            elif any(t in asmb_id_clean.split() for t in clean_tokens if len(t) > 3):
                seeds_dict[asmb.id] = (0.80, {"type": "assembly", "assembly_type": asmb.assembly_type})

        # 2. Compositional Vector Query (HRR Superposition Bundle)
        token_vecs = []
        for t in clean_tokens:
            atom = self.dna_registry.find_by_value(t)
            if atom is not None:
                token_vecs.append(atom.vector)
            else:
                token_vecs.append(self.dna_registry.encoder.encode_text(t))

        if token_vecs:
            from dna_memory.core.vector_operations import bundle
            q_vec = bundle(token_vecs)
            # Threshold 0.28 rejects Gaussian noise (random 128D pairs are ~0.08)
            vector_seeds = self.index.query(q_vec, top_k=top_k, min_score=0.28)
            for item_id, score, meta in vector_seeds:
                if item_id not in seeds_dict or score > seeds_dict[item_id][0]:
                    seeds_dict[item_id] = (score, meta)

        initial_seeds = [(k, v[0], v[1]) for k, v in seeds_dict.items()]
        initial_seeds.sort(key=lambda x: x[1], reverse=True)
        initial_seeds = initial_seeds[:top_k]

        if not initial_seeds:
            return []

        activation_map: Dict[str, float] = {}
        explanations: Dict[str, List[str]] = {}
        visited: Set[str] = set()

        frontier: List[Tuple[str, float]] = []
        for item_id, score, meta in initial_seeds:
            activation_map[item_id] = score
            explanations[item_id] = [f"Direct query seed similarity: {score:.3f}"]
            frontier.append((item_id, score))
            visited.add(item_id)

        episodic_rels = {"participated_in", "inquired", "purchased", "has_event", "event_target"}

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
                    if r.relation_type in episodic_rels or r.object_id.isdigit():
                        continue
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
                    if r.relation_type in episodic_rels or r.subject_id.isdigit():
                        continue
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
                    if pid.startswith("asmb_event_") or pid.startswith("asmb_behaviour_"):
                        continue
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
