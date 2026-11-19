"""
Decomposition Engine: Recovers and navigates constituent representations.
Supports both lossless manifest lookup and vector-symbolic unbinding.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from dna_memory.core.atomic_dna import AtomicDNA
from dna_memory.core.dna_registry import DNARegistry
from dna_memory.composition.composition_engine import CompositeDNA
from dna_memory.composition.binding import unbind_and_match

class DecompositionEngine:
    """
    Decomposition Engine:
    - Lossless path: resolves exact component references via manifest.
    - Vector decoding path: tests whether constituent vectors can be recovered via unbinding.
    """
    def __init__(self, registry: DNARegistry):
        self.registry = registry

    def decompose_lossless(self, composite: CompositeDNA) -> List[AtomicDNA]:
        """Losslessly retrieves constituent AtomicDNA instances via authoritative manifest IDs."""
        components = []
        for cid in composite.manifest.component_ids:
            atom = self.registry.get(cid)
            if atom:
                components.append(atom)
        return components

    def decode_role_filler(
        self,
        composite: CompositeDNA,
        role_name: str,
        candidate_atoms: List[AtomicDNA]
    ) -> Tuple[Optional[AtomicDNA], float]:
        """
        Attempts vector-symbolic recovery of a filler for a given role from a bound composite.
        """
        encoder = self.registry.encoder
        role_vec = encoder.encode_text(f"ROLE:{role_name}")
        cand_vecs = [a.vector for a in candidate_atoms]
        cand_ids = [a.id for a in candidate_atoms]

        best_id, sim = unbind_and_match(role_vec, composite.vector, cand_vecs, cand_ids)
        if best_id:
            return self.registry.get(best_id), sim
        return None, 0.0
