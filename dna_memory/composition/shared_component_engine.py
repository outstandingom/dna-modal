"""
Shared Component Engine: Discovers, registers, and reuses shared foundational components
across multiple distinct concept instances (e.g. 100 leaf instances sharing blade,
vein, and petiole components while retaining distinct individual identities and traits).
"""

from dataclasses import dataclass, field
import datetime
import threading
from typing import Dict, List, Optional, Set, Any, Tuple
import numpy as np
from dna_memory.core.atomic_dna import AtomicDNA, DNAFeatureMetadata
from dna_memory.core.concept_identity import ConceptIdentityRegistry, ConceptIdentity
from dna_memory.relations.relationship_lifecycle import RelationshipLifecycleEngine


@dataclass
class SharedComponentDefinition:
    """Canonical specification of a reusable component template."""
    component_id: str
    component_name: str
    domain: str
    standard_features: List[str]
    reference_count: int = 0
    attached_instances: Set[str] = field(default_factory=set)
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "domain": self.domain,
            "standard_features": self.standard_features,
            "reference_count": self.reference_count,
            "attached_instances": sorted(list(self.attached_instances)),
            "created_at": self.created_at
        }


@dataclass
class InstanceCompositionResult:
    """Result of composing an instance using shared and unique components."""
    instance_id: str
    canonical_concept_id: str
    shared_component_ids: List[str]
    unique_component_ids: List[str]
    reuse_ratio: float  # len(shared) / total_components
    composition_vector: Optional[np.ndarray] = None


class SharedComponentEngine:
    """
    Coordinates structural component reuse across diverse instances:
      - Prevents duplicate instantiation of standard sub-structures.
      - Maintains reference counting and bidirectional links.
      - Preserves individual variation without destructive merging.
    """
    def __init__(
        self,
        concept_registry: ConceptIdentityRegistry,
        dna_registry,
        relationship_engine: RelationshipLifecycleEngine
    ):
        self.concept_registry = concept_registry
        self.dna_registry = dna_registry
        self.relationship_engine = relationship_engine
        self._shared_components: Dict[str, SharedComponentDefinition] = {}
        self._instance_to_shared: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()

    def register_shared_component(
        self,
        component_id: str,
        component_name: str,
        domain: str = "botany",
        standard_features: Optional[List[str]] = None
    ) -> SharedComponentDefinition:
        """Registers a canonical component meant to be shared across instances."""
        with self._lock:
            cid = component_id.strip()
            if cid in self._shared_components:
                return self._shared_components[cid]

            # Ensure concept is registered in identity registry
            if not self.concept_registry.exists(cid):
                self.concept_registry.register(
                    canonical_id=cid,
                    canonical_name=component_name,
                    domain=domain,
                    source="shared_component_engine"
                )

            # Ensure an AtomicDNA representation exists in DNA registry
            if not self.dna_registry.has(cid):
                from dna_memory.core.dna_encoder import DNAEncoder
                encoder = DNAEncoder(dims=getattr(self.dna_registry, "dims", 128))
                vec = encoder.encode_concept(concept_type="shared_component", value=cid)
                atom = AtomicDNA(
                    id=cid,
                    type="shared_component",
                    value=component_name,
                    vector=vec,
                    source="shared_component_engine",
                    metadata=DNAFeatureMetadata(meaning=f"Shared component: {component_name}", domain=domain)
                )
                self.dna_registry.register(atom)

            comp = SharedComponentDefinition(
                component_id=cid,
                component_name=component_name,
                domain=domain,
                standard_features=list(standard_features or [])
            )
            self._shared_components[cid] = comp
            return comp

    def attach_shared_component(
        self,
        instance_id: str,
        component_id: str,
        relationship_predicate: str = "has_part",
        confidence: float = 1.0,
        source: str = "shared_component_engine"
    ) -> bool:
        """
        Attaches a shared component to an individual instance via a versioned relationship.
        Increments reference count and records instance linkage.
        """
        with self._lock:
            if component_id not in self._shared_components:
                raise ValueError(f"Shared component '{component_id}' is not registered.")

            shared = self._shared_components[component_id]

            # Create relationship in lifecycle engine
            self.relationship_engine.create_relationship(
                subject_id=instance_id,
                predicate=relationship_predicate,
                object_id=component_id,
                confidence=confidence,
                source=source
            )

            # Update bookkeeping
            if instance_id not in shared.attached_instances:
                shared.attached_instances.add(instance_id)
                shared.reference_count += 1

            self._instance_to_shared.setdefault(instance_id, set()).add(component_id)
            return True

    def compose_instance_with_shared(
        self,
        instance_id: str,
        canonical_name: str,
        domain: str,
        required_shared_components: List[str],
        unique_features: Optional[Dict[str, Any]] = None,
        source: str = "direct_observation"
    ) -> InstanceCompositionResult:
        """
        Composes an individual concept instance reusing shared components for standard parts
        and registering unique atomic attributes for individual variance.
        """
        with self._lock:
            # 1. Register instance concept identity
            instance_concept = self.concept_registry.register(
                canonical_id=instance_id,
                canonical_name=canonical_name,
                domain=domain,
                attributes=unique_features or {},
                source=source
            )

            # Ensure an instance atomic representation exists
            if not self.dna_registry.has(instance_id):
                from dna_memory.core.dna_encoder import DNAEncoder
                encoder = DNAEncoder(dims=getattr(self.dna_registry, "dims", 128))
                vec = encoder.encode_concept(concept_type="instance", value=instance_id)
                atom = AtomicDNA(
                    id=instance_id,
                    type="instance",
                    value=canonical_name,
                    vector=vec,
                    source=source
                )
                self.dna_registry.register(atom)

            # 2. Attach required shared components
            attached_shared = []
            for sc_id in required_shared_components:
                if sc_id not in self._shared_components:
                    self.register_shared_component(sc_id, sc_id.replace("_", " ").title(), domain=domain)
                self.attach_shared_component(instance_id, sc_id, source=source)
                attached_shared.append(sc_id)

            # 3. Create unique atomic feature representations
            unique_cids = []
            if unique_features:
                from dna_memory.core.dna_encoder import DNAEncoder
                encoder = DNAEncoder(dims=getattr(self.dna_registry, "dims", 128))
                for attr_key, attr_val in unique_features.items():
                    u_id = f"{instance_id}_{attr_key}"
                    if not self.dna_registry.has(u_id):
                        u_vec = encoder.encode_concept(concept_type=attr_key, value=str(attr_val))
                        u_atom = AtomicDNA(
                            id=u_id,
                            type="attribute",
                            value=attr_val,
                            vector=u_vec,
                            source=source
                        )
                        self.dna_registry.register(u_atom)
                    self.relationship_engine.create_relationship(
                        subject_id=instance_id,
                        predicate=f"has_{attr_key}",
                        object_id=u_id,
                        source=source
                    )
                    unique_cids.append(u_id)

            total_comps = len(attached_shared) + len(unique_cids)
            reuse_ratio = (len(attached_shared) / total_comps) if total_comps > 0 else 0.0

            # 4. Generate composite bundled vector
            all_vectors = []
            for cid in attached_shared:
                atom = self.dna_registry.get(cid)
                if atom is not None:
                    all_vectors.append(atom.vector)
            for uid in unique_cids:
                atom = self.dna_registry.get(uid)
                if atom is not None:
                    all_vectors.append(atom.vector)

            from dna_memory.core.vector_operations import bundle
            comp_vector = bundle(all_vectors) if all_vectors else None

            return InstanceCompositionResult(
                instance_id=instance_id,
                canonical_concept_id=instance_concept.canonical_id,
                shared_component_ids=attached_shared,
                unique_component_ids=unique_cids,
                reuse_ratio=reuse_ratio,
                composition_vector=comp_vector
            )

    def get_shared_component(self, component_id: str) -> Optional[SharedComponentDefinition]:
        with self._lock:
            return self._shared_components.get(component_id)

    def get_instance_shared_components(self, instance_id: str) -> List[str]:
        with self._lock:
            return sorted(list(self._instance_to_shared.get(instance_id, set())))

    def list_all_shared(self) -> List[SharedComponentDefinition]:
        with self._lock:
            return list(self._shared_components.values())
