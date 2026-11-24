"""
DNA Registry: Central catalog and registry for all Atomic DNA representations.
Ensures every atomic representation has a permanent ID, indexed for fast O(1) retrieval.
"""

from typing import Dict, List, Optional
import numpy as np
from dna_memory.core.atomic_dna import AtomicDNA, DNAFeatureMetadata
from dna_memory.core.dna_encoder import DNAEncoder
from dna_memory.core.vector_operations import cosine_similarity

class DNARegistry:
    """
    Registry for Atomic DNA units.
    """
    def __init__(self, encoder: Optional[DNAEncoder] = None):
        self.encoder = encoder or DNAEncoder(dims=128)
        self._by_id: Dict[str, AtomicDNA] = {}
        self._by_value_type: Dict[tuple, str] = {}
        self._next_seq = 1

    def register(
        self,
        value: str,
        dna_type: str = "property",
        vector: Optional[np.ndarray] = None,
        source: str = "observation",
        confidence: float = 1.0,
        metadata: Optional[DNAFeatureMetadata] = None,
        custom_id: Optional[str] = None,
        allow_overwrite: bool = False
    ) -> AtomicDNA:
        """Register or retrieve an atomic DNA representation with collision safety."""
        key = (dna_type, str(value).lower().strip())
        
        # Check if already registered by value and type
        if key in self._by_value_type:
            existing_id = self._by_value_type[key]
            if not custom_id or existing_id == custom_id:
                return self._by_id[existing_id]

        atom_id = custom_id or f"dna_{dna_type[:3]}_{self._next_seq:06d}"
        if not custom_id:
            self._next_seq += 1

        # Check for ID collisions if custom_id was provided
        if custom_id and custom_id in self._by_id:
            existing_atom = self._by_id[custom_id]
            if str(existing_atom.value).strip().lower() == str(value).strip().lower() and existing_atom.type == dna_type:
                return existing_atom
            if not allow_overwrite:
                raise ValueError(
                    f"ID collision detected: Atom ID '{custom_id}' already exists for value '{existing_atom.value}' "
                    f"({existing_atom.type}). Cannot silently overwrite with '{value}' ({dna_type})."
                )
            # Safe overwrite: remove old value-type mapping
            old_key = (existing_atom.type, str(existing_atom.value).lower().strip())
            self._by_value_type.pop(old_key, None)

        if vector is None:
            vector = self.encoder.encode_concept(dna_type, str(value))

        if metadata is None:
            metadata = DNAFeatureMetadata(
                meaning=f"{dna_type}:{value}",
                source=source,
                feature_type=dna_type,
                confidence=confidence
            )

        atom = AtomicDNA(
            id=atom_id,
            type=dna_type,
            value=value,
            vector=vector,
            source=source,
            confidence=confidence,
            metadata=metadata
        )

        self._by_id[atom_id] = atom
        self._by_value_type[key] = atom_id
        return atom

    def get(self, atom_id: str) -> Optional[AtomicDNA]:
        """Lookup by ID."""
        return self._by_id.get(atom_id)

    def has(self, atom_id: str) -> bool:
        """Check if an atom exists by ID."""
        return atom_id in self._by_id

    def __contains__(self, atom_id: str) -> bool:
        return atom_id in self._by_id

    def unregister(self, atom_id: str) -> bool:
        """Removes an atom from the registry."""
        if atom_id in self._by_id:
            atom = self._by_id.pop(atom_id)
            key = (atom.type, str(atom.value).lower().strip())
            self._by_value_type.pop(key, None)
            return True
        return False

    def find_by_value(self, value: str, dna_type: str = "property") -> Optional[AtomicDNA]:
        """Lookup by (type, value)."""
        key = (dna_type, str(value).lower().strip())
        atom_id = self._by_value_type.get(key)
        if atom_id:
            return self._by_id.get(atom_id)
        return None

    def search_similar(self, query_vec: np.ndarray, top_k: int = 5, min_score: float = 0.0) -> List[tuple]:
        """Search registered atomic representations by cosine similarity."""
        results = []
        for atom in self._by_id.values():
            sim = cosine_similarity(query_vec, atom.vector)
            if sim >= min_score:
                results.append((atom, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        return len(self._by_id)

    def serialize(self) -> dict:
        return {
            "dims": self.encoder.dims,
            "next_seq": self._next_seq,
            "atoms": {k: v.to_dict() for k, v in self._by_id.items()}
        }

    def restore(self, data: dict):
        self.encoder.dims = data.get("dims", self.encoder.dims)
        self._next_seq = data.get("next_seq", 1)
        self._by_id.clear()
        self._by_value_type.clear()
        for k, atom_dict in data.get("atoms", {}).items():
            atom = AtomicDNA.from_dict(atom_dict)
            self._by_id[k] = atom
            self._by_value_type[(atom.type, str(atom.value).lower().strip())] = atom.id
