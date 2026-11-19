"""
DNA Composition Engine: Combines multiple DNA representations into rich composite concepts.
Preserves authoritative component references in a CompositionManifest so you never
rely on lossy vector decoding to reconstruct memories.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from dna_memory.core.atomic_dna import AtomicDNA
from dna_memory.core.dna_registry import DNARegistry
from dna_memory.core.vector_operations import bundle, bind, normalize

@dataclass
class CompositionManifest:
    """
    Authoritative structural blueprint of a composite representation.
    Preserves exact IDs and roles.
    """
    composite_id: str
    composite_name: str
    operation: str  # "bundle", "bind_pairs", "hierarchical"
    component_ids: List[str]
    roles: Dict[str, str] = field(default_factory=dict)     # component_id -> role name
    weights: Dict[str, float] = field(default_factory=dict) # component_id -> weight
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "composite_id": self.composite_id,
            "composite_name": self.composite_name,
            "operation": self.operation,
            "component_ids": self.component_ids,
            "roles": self.roles,
            "weights": self.weights,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CompositionManifest':
        return cls(
            composite_id=data["composite_id"],
            composite_name=data["composite_name"],
            operation=data["operation"],
            component_ids=data["component_ids"],
            roles=data.get("roles", {}),
            weights=data.get("weights", {}),
            metadata=data.get("metadata", {})
        )

@dataclass
class CompositeDNA:
    """
    A richer representation constructed from constituent DNA components.
    Authoritative knowledge is in the manifest; vector is cached for fast retrieval.
    """
    manifest: CompositionManifest
    vector: np.ndarray

    @property
    def id(self) -> str:
        return self.manifest.composite_id

    @property
    def name(self) -> str:
        return self.manifest.composite_name

class CompositionEngine:
    """
    Engine responsible for constructing composite concepts from atomic or sub-composite DNA.
    Example: Coffee DNA = Bean + Aroma + Taste + Caffeine.
    """
    def __init__(self, registry: DNARegistry):
        self.registry = registry
        self._composites: Dict[str, CompositeDNA] = {}
        self._next_comp_seq = 1

    def compose(
        self,
        name: str,
        components: List[AtomicDNA],
        roles: Optional[Dict[str, str]] = None,
        weights: Optional[Dict[str, float]] = None,
        operation: str = "bundle",
        composite_id: Optional[str] = None
    ) -> CompositeDNA:
        """
        Creates a new composite DNA representation.
        1. Compiles authoritative manifest.
        2. Calculates combined vector via HRR vector algebra.
        """
        if not components:
            raise ValueError("Cannot compose an empty list of components.")

        cid = composite_id or f"comp_{name.lower().replace(' ', '_')}_{self._next_comp_seq:04d}"
        self._next_comp_seq += 1

        comp_ids = [c.id for c in components]
        r_map = roles or {}
        w_map = weights or {c.id: 1.0 for c in components}

        # Vector combination
        if operation == "bundle":
            vecs = [c.vector for c in components]
            ws = [w_map.get(c.id, 1.0) for c in components]
            composite_vec = bundle(vecs, weights=ws)
        elif operation == "bind_pairs":
            # Binds each component with its role, then bundles the role-filler pairs
            pairs = []
            encoder = self.registry.encoder
            for c in components:
                role_name = r_map.get(c.id, "component")
                role_vec = encoder.encode_text(f"ROLE:{role_name}")
                bound_pair = bind(role_vec, c.vector)
                pairs.append(bound_pair)
            composite_vec = bundle(pairs)
        else:
            raise ValueError(f"Unsupported composition operation: {operation}")

        manifest = CompositionManifest(
            composite_id=cid,
            composite_name=name,
            operation=operation,
            component_ids=comp_ids,
            roles=r_map,
            weights=w_map
        )

        composite = CompositeDNA(manifest=manifest, vector=composite_vec)
        self._composites[cid] = composite
        return composite

    def get_composite(self, composite_id: str) -> Optional[CompositeDNA]:
        return self._composites.get(composite_id)

    def count(self) -> int:
        return len(self._composites)
