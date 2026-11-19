"""
Recursive Assembly Engine: Hierarchical composition of concepts and assemblies.
Supports arbitrary depth: Atomic -> Concepts -> Events -> Behaviour -> Customer Profile.
Authoritative representation: component_ids & relation_ids.
Cached representation: combined vector.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
import numpy as np
from dna_memory.core.dna_registry import DNARegistry
from dna_memory.core.vector_operations import bundle, normalize
from dna_memory.relations.relation_registry import RelationRegistry

@dataclass
class DNAAssembly:
    """
    Hierarchical assembly representing a structured, multi-level concept or event.
    """
    id: str
    assembly_type: str  # "Product", "PurchaseEvent", "CustomerBehaviour", "CustomerProfile"
    component_ids: List[str]  # IDs of AtomicDNA or child DNAAssembly instances
    relation_ids: List[str]   # IDs of RelationDNA instances linked to this assembly
    weights: List[float]      # Weights corresponding to component_ids
    vector: np.ndarray        # Cached combined vector for fast similarity search
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "assembly_type": self.assembly_type,
            "component_ids": self.component_ids,
            "relation_ids": self.relation_ids,
            "weights": self.weights,
            "vector": self.vector.tolist(),
            "version": self.version,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DNAAssembly':
        return cls(
            id=data["id"],
            assembly_type=data["assembly_type"],
            component_ids=data["component_ids"],
            relation_ids=data.get("relation_ids", []),
            weights=data.get("weights", [1.0] * len(data["component_ids"])),
            vector=np.array(data["vector"], dtype=np.float64),
            version=data.get("version", 1),
            metadata=data.get("metadata", {})
        )

class AssemblyEngine:
    """
    Manages recursive assembly lifecycle:
      creation, incremental modification, recomposition, parent discovery, and versioning.
    """
    def __init__(self, dna_registry: DNARegistry, relation_registry: RelationRegistry):
        self.dna_registry = dna_registry
        self.relation_registry = relation_registry
        self._assemblies: Dict[str, DNAAssembly] = {}
        # Inverted index: component_id -> Set of assembly_ids that contain it
        self._component_to_assemblies: Dict[str, Set[str]] = {}
        self._next_asmb_seq = 1

    def create_assembly(
        self,
        assembly_type: str,
        component_ids: List[str],
        relation_ids: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        metadata: Optional[dict] = None,
        custom_id: Optional[str] = None
    ) -> DNAAssembly:
        """Constructs a new hierarchical DNA assembly."""
        asmb_id = custom_id or f"asmb_{assembly_type.lower()}_{self._next_asmb_seq:06d}"
        self._next_asmb_seq += 1

        rel_ids = relation_ids or []
        w_list = weights or [1.0] * len(component_ids)

        # Compute initial cached vector
        vector = self._compute_assembly_vector(component_ids, rel_ids, w_list)

        assembly = DNAAssembly(
            id=asmb_id,
            assembly_type=assembly_type,
            component_ids=list(component_ids),
            relation_ids=list(rel_ids),
            weights=list(w_list),
            vector=vector,
            version=1,
            metadata=metadata or {}
        )

        self._assemblies[asmb_id] = assembly
        for cid in component_ids:
            self._component_to_assemblies.setdefault(cid, set()).add(asmb_id)

        return assembly

    def add_component(self, assembly_id: str, component_id: str, weight: float = 1.0, relation_id: Optional[str] = None) -> DNAAssembly:
        """Adds a new component or relation to an existing assembly without rebuilding everything."""
        assembly = self._assemblies.get(assembly_id)
        if not assembly:
            raise KeyError(f"Assembly '{assembly_id}' not found.")

        if component_id not in assembly.component_ids:
            assembly.component_ids.append(component_id)
            assembly.weights.append(weight)
            self._component_to_assemblies.setdefault(component_id, set()).add(assembly_id)

        if relation_id and relation_id not in assembly.relation_ids:
            assembly.relation_ids.append(relation_id)

        assembly.version += 1
        self.recompose(assembly_id)
        return assembly

    def remove_component(self, assembly_id: str, component_id: str) -> DNAAssembly:
        """Removes a component from an assembly and triggers recomposition."""
        assembly = self._assemblies.get(assembly_id)
        if not assembly:
            raise KeyError(f"Assembly '{assembly_id}' not found.")

        if component_id in assembly.component_ids:
            idx = assembly.component_ids.index(component_id)
            assembly.component_ids.pop(idx)
            assembly.weights.pop(idx)
            
            if component_id in self._component_to_assemblies:
                self._component_to_assemblies[component_id].discard(assembly_id)
                if not self._component_to_assemblies[component_id]:
                    del self._component_to_assemblies[component_id]

        assembly.version += 1
        self.recompose(assembly_id)
        return assembly

    def delete_assembly(self, assembly_id: str):
        """Removes an assembly completely and cleans up all inverted index references."""
        assembly = self._assemblies.pop(assembly_id, None)
        if not assembly:
            return

        for cid in assembly.component_ids:
            if cid in self._component_to_assemblies:
                self._component_to_assemblies[cid].discard(assembly_id)
                if not self._component_to_assemblies[cid]:
                    del self._component_to_assemblies[cid]

        # Also remove from any parent assemblies that contained this assembly
        parent_ids = list(self.find_parent_assemblies(assembly_id))
        for pid in parent_ids:
            if pid in self._assemblies:
                self.remove_component(pid, assembly_id)

    def recompose(self, assembly_id: str, visited: Optional[Set[str]] = None) -> np.ndarray:
        """Recalculates the cached composite vector for an assembly and invalidates upwards."""
        if visited is None:
            visited = set()
        if assembly_id in visited:
            return self._assemblies[assembly_id].vector if assembly_id in self._assemblies else np.zeros(self.dna_registry.encoder.dims)
        visited.add(assembly_id)

        assembly = self._assemblies.get(assembly_id)
        if not assembly:
            raise KeyError(f"Assembly '{assembly_id}' not found.")

        new_vec = self._compute_assembly_vector(
            assembly.component_ids,
            assembly.relation_ids,
            assembly.weights
        )
        assembly.vector = new_vec

        # Propagate recomposition to all parent assemblies recursively
        for parent_id in self.find_parent_assemblies(assembly_id):
            if parent_id != assembly_id and parent_id in self._assemblies:
                self.recompose(parent_id, visited=visited)

        return new_vec

    def update_atomic_component(self, atom_id: str):
        """Triggers recomposition across all assemblies that directly or indirectly contain atom_id."""
        parents = self.find_parent_assemblies(atom_id)
        visited = set()
        for pid in parents:
            if pid in self._assemblies:
                self.recompose(pid, visited=visited)

    def decompose(self, assembly_id: str) -> Dict[str, Any]:
        """
        Decomposes an assembly into its constituent components, sub-assemblies, and relations.
        """
        assembly = self._assemblies.get(assembly_id)
        if not assembly:
            raise KeyError(f"Assembly '{assembly_id}' not found.")

        components = []
        for cid in assembly.component_ids:
            # Check if component is an atomic DNA or a sub-assembly
            if cid in self._assemblies:
                components.append({"type": "assembly", "data": self._assemblies[cid]})
            else:
                atom = self.dna_registry.get(cid)
                if atom:
                    components.append({"type": "atomic", "data": atom})

        relations = [self.relation_registry.get(rid) for rid in assembly.relation_ids if self.relation_registry.get(rid)]

        return {
            "assembly_id": assembly_id,
            "assembly_type": assembly.assembly_type,
            "version": assembly.version,
            "components": components,
            "relations": relations
        }

    def find_parent_assemblies(self, component_id: str) -> List[str]:
        """Discovers which higher-level concepts or assemblies utilize this component directly."""
        return list(self._component_to_assemblies.get(component_id, set()))

    def find_ancestor_assemblies(self, component_id: str, visited: Optional[Set[str]] = None) -> List[str]:
        """Discovers all transitive ancestors (parents, grandparents, etc.) that utilize this component."""
        if visited is None:
            visited = set()
        ancestors = set()
        direct_parents = self._component_to_assemblies.get(component_id, set())
        for pid in direct_parents:
            if pid not in visited:
                visited.add(pid)
                ancestors.add(pid)
                ancestors.update(self.find_ancestor_assemblies(pid, visited=visited))
        return list(ancestors)

    def get(self, assembly_id: str) -> Optional[DNAAssembly]:
        return self._assemblies.get(assembly_id)

    def count(self) -> int:
        return len(self._assemblies)

    def serialize(self) -> dict:
        """Serializes all assemblies and index mappings."""
        return {
            "next_asmb_seq": self._next_asmb_seq,
            "assemblies": {aid: asmb.to_dict() for aid, asmb in self._assemblies.items()}
        }

    def restore(self, data: dict):
        """Restores assemblies and reconstructs the inverted index."""
        self._next_asmb_seq = data.get("next_asmb_seq", 1)
        self._assemblies.clear()
        self._component_to_assemblies.clear()
        for aid, asmb_data in data.get("assemblies", {}).items():
            asmb = DNAAssembly.from_dict(asmb_data)
            self._assemblies[aid] = asmb
            for cid in asmb.component_ids:
                self._component_to_assemblies.setdefault(cid, set()).add(aid)

    def _compute_assembly_vector(
        self,
        component_ids: List[str],
        relation_ids: List[str],
        weights: List[float]
    ) -> np.ndarray:
        """Internal recursive vector builder."""
        vectors = []
        norm_weights = []

        for cid, w in zip(component_ids, weights):
            v = None
            if cid in self._assemblies:
                v = self._assemblies[cid].vector
            else:
                atom = self.dna_registry.get(cid)
                if atom:
                    v = atom.vector
                else:
                    v = self.dna_registry.encoder.encode_text(cid)
            if v is not None:
                vectors.append(v)
                norm_weights.append(w)

        # Include relationship DNA vectors in the assembly embedding
        for rid in relation_ids:
            rel = self.relation_registry.get(rid)
            if rel:
                vectors.append(rel.vector)
                norm_weights.append(rel.strength * 0.5)

        if not vectors:
            return np.zeros(self.dna_registry.encoder.dims, dtype=np.float64)

        return bundle(vectors, weights=norm_weights)
