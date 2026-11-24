"""
DNAMemorySystem: Complete, unified facade for the Recursive Compositional DNA Memory Architecture.
Integrates atomic encoding, holographic vector composition, relational memory,
hierarchical recursive assemblies, spreading activation, learning, and prediction with unified disk persistence.

Phase 1-4 Architectural Extensions:
  - Concept Identity Registry & Alias Resolution
  - Relationship Lifecycle & Provenance Engine
  - Temporal Memory & Historical Reconstruction
  - Assembly Validation & Dependency Tracking
  - Shared Component Engine
  - Pattern Discovery & Difference Detection
  - Memory Consolidation
  - Hybrid Multi-Channel Retrieval & Contextual Reranking
  - Memory Integrity Audits & Database Persistence
"""

import os
import json
import gzip
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np

from dna_memory.core.atomic_dna import AtomicDNA, DNAFeatureMetadata
from dna_memory.core.dna_encoder import DNAEncoder
from dna_memory.core.dna_registry import DNARegistry
from dna_memory.core.concept_identity import ConceptIdentityRegistry, ConceptIdentity
from dna_memory.composition.composition_engine import CompositionEngine, CompositionManifest
from dna_memory.composition.decomposition_engine import DecompositionEngine
from dna_memory.composition.shared_component_engine import SharedComponentEngine, InstanceCompositionResult
from dna_memory.relations.relation_registry import RelationRegistry
from dna_memory.relations.relationship_lifecycle import RelationshipLifecycleEngine, RelationshipRecord
from dna_memory.assembly.recursive_composition import AssemblyEngine, DNAAssembly
from dna_memory.assembly.assembly_validator import AssemblyValidator
from dna_memory.assembly.dependency_manager import DependencyManager
from dna_memory.activation.retrieval_index import RetrievalIndex
from dna_memory.activation.activation_engine import DNAActivationEngine
from dna_memory.activation.hybrid_retrieval import HybridRetrievalEngine, SemanticEmbeddingAdapter, RetrievalCandidate
from dna_memory.activation.retrieval_reranker import RetrievalReranker, RerankedCandidate
from dna_memory.reasoning.multi_hop import FrontierReasoningEngine
from dna_memory.learning.learning_engine import MemoryLearningEngine
from dna_memory.learning.pattern_discovery import PatternDiscoveryEngine, DiscoveredPattern
from dna_memory.learning.difference_detector import DifferenceDetector, StructuralComparisonReport
from dna_memory.prediction.behaviour_prediction import BehaviourPredictionEngine
from dna_memory.prediction.temporal_patterns import TemporalPatternAnalyzer
from dna_memory.memory.universal_memory import UniversalUserMemory, AgentGroundingContext
from dna_memory.memory.temporal_memory import TemporalMemoryEngine, EntityHistoricalState
from dna_memory.memory.memory_consolidator import MemoryConsolidator, ConsolidationAudit
from dna_memory.memory.memory_integrity import MemoryIntegrityEngine, IntegrityCheckReport
from dna_memory.memory.persistence_adapter import SQLitePersistenceAdapter
from dna_memory.core.sine_brain_bridge import SineBrainBridge, BridgeSyncReport


class DNAMemorySystem:
    """
    Unified Recursive Compositional DNA Memory System with complete lifecycle management.
    """
    def __init__(self, dims: int = 128, seed: int = 42, db_path: Optional[str] = None):
        self.dims = dims
        self.seed = seed

        # Core Mathematical & Symbolic Foundation
        self.encoder = DNAEncoder(dims=dims, seed=seed)
        self.dna_registry = DNARegistry(self.encoder)
        self.concept_registry = ConceptIdentityRegistry()

        # Relations & Lifecycle
        self.relation_registry = RelationRegistry(self.dna_registry)
        self.relationship_lifecycle = RelationshipLifecycleEngine(concept_registry=self.concept_registry)

        # Composition & Assemblies
        self.composition_engine = CompositionEngine(self.dna_registry)
        self.decomposition_engine = DecompositionEngine(self.dna_registry)
        self.assembly_engine = AssemblyEngine(self.dna_registry, self.relation_registry)
        self.assembly_validator = AssemblyValidator()
        self.dependency_manager = DependencyManager()
        self.shared_components = SharedComponentEngine(
            concept_registry=self.concept_registry,
            dna_registry=self.dna_registry,
            relationship_engine=self.relationship_lifecycle
        )

        # Retrieval, Activation & Reasoning
        self.retrieval_index = RetrievalIndex(dims=dims)
        self.activation_engine = DNAActivationEngine(self.dna_registry, self.relation_registry, self.assembly_engine)
        self.hybrid_retrieval = HybridRetrievalEngine(
            concept_registry=self.concept_registry,
            relationship_engine=self.relationship_lifecycle,
            dna_registry=self.dna_registry
        )
        self.retrieval_reranker = RetrievalReranker()
        self.frontier_reasoning = FrontierReasoningEngine(self.relation_registry, self.assembly_engine)

        # Learning & Pattern Analytics
        self.learning_engine = MemoryLearningEngine(self.dna_registry, self.relation_registry, self.assembly_engine)
        self.pattern_discovery = PatternDiscoveryEngine(self.relationship_lifecycle)
        self.difference_detector = DifferenceDetector(
            relationship_engine=self.relationship_lifecycle,
            concept_registry=self.concept_registry
        )
        self.prediction_engine = BehaviourPredictionEngine(self.learning_engine, self.relation_registry, self.assembly_engine)

        # Memory Subsystems & Temporal Layer
        self.temporal_memory = TemporalMemoryEngine(
            relationship_engine=self.relationship_lifecycle,
            concept_registry=self.concept_registry
        )
        self.user_memory = UniversalUserMemory(
            dna_registry=self.dna_registry,
            relation_registry=self.relation_registry,
            assembly_engine=self.assembly_engine,
            lifecycle_engine=self.relationship_lifecycle,
            temporal_engine=self.temporal_memory,
            concept_registry=self.concept_registry
        )
        self.memory_consolidator = MemoryConsolidator(
            relationship_engine=self.relationship_lifecycle,
            concept_registry=self.concept_registry
        )
        self.integrity_engine = MemoryIntegrityEngine(
            dna_registry=self.dna_registry,
            concept_registry=self.concept_registry,
            relationship_engine=self.relationship_lifecycle,
            assembly_engine=self.assembly_engine
        )
        self.persistence = SQLitePersistenceAdapter(db_path=db_path or ":memory:")
        self.brain_bridge = SineBrainBridge(self)

    # -------------------------------------------------------------
    # Concept Identity Facade
    # -------------------------------------------------------------
    def register_canonical_concept(
        self,
        canonical_id: str,
        canonical_name: str,
        domain: str = "general",
        aliases: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        source: str = "direct_observation"
    ) -> ConceptIdentity:
        """Registers a canonical concept with guaranteed collision detection."""
        concept = self.concept_registry.register(
            canonical_id=canonical_id,
            canonical_name=canonical_name,
            domain=domain,
            aliases=aliases,
            attributes=attributes,
            source=source
        )
        # Register in persistence
        self.persistence.save_concept(concept.to_dict())
        return concept

    def resolve_concept(self, expression: str) -> Optional[ConceptIdentity]:
        """Resolves expression or alias to its canonical concept."""
        return self.concept_registry.resolve(expression)

    # -------------------------------------------------------------
    # Relationship Lifecycle Facade
    # -------------------------------------------------------------
    def record_relationship(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        confidence: float = 0.8,
        strength: float = 1.0,
        source: str = "direct_observation",
        evidence_content: Optional[str] = None,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None
    ) -> Tuple[RelationshipRecord, bool]:
        """Creates or updates a versioned relationship with calibrated evidence."""
        rel, is_new = self.relationship_lifecycle.create_relationship(
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            confidence=confidence,
            strength=strength,
            source=source,
            evidence_content=evidence_content,
            valid_from=valid_from,
            valid_until=valid_until
        )
        self.persistence.save_relationship(rel.to_dict())
        return rel, is_new

    def supersede_fact(
        self,
        old_relationship_id: str,
        new_object_id: str,
        source: str = "direct_observation",
        evidence_content: Optional[str] = None
    ) -> RelationshipRecord:
        """Atomically supersedes a past relationship with a new state."""
        new_rel = self.relationship_lifecycle.supersede_relationship(
            old_relationship_id=old_relationship_id,
            new_object_id=new_object_id,
            source=source,
            evidence_content=evidence_content
        )
        old_rel = self.relationship_lifecycle.get(old_relationship_id)
        self.persistence.save_relationship(old_rel.to_dict())
        self.persistence.save_relationship(new_rel.to_dict())
        return new_rel

    # -------------------------------------------------------------
    # Temporal State Reconstruction
    # -------------------------------------------------------------
    def query_facts_as_of(
        self,
        as_of_time: str,
        subject_id: Optional[str] = None,
        predicate: Optional[str] = None
    ) -> List[RelationshipRecord]:
        """Queries facts that were valid at an exact historical point in time."""
        return self.temporal_memory.query_relationships_at(
            as_of_time=as_of_time,
            subject_id=subject_id,
            predicate=predicate
        )

    def reconstruct_entity_state(self, canonical_id: str, as_of_time: str) -> EntityHistoricalState:
        """Reconstructs state of an entity as it existed at historical time as_of_time."""
        return self.temporal_memory.reconstruct_entity_state(canonical_id, as_of_time)

    # -------------------------------------------------------------
    # Shared Component & Composition Facade
    # -------------------------------------------------------------
    def compose_shared_instance(
        self,
        instance_id: str,
        canonical_name: str,
        domain: str,
        shared_component_ids: List[str],
        unique_features: Optional[Dict[str, Any]] = None
    ) -> InstanceCompositionResult:
        """Reuses shared canonical components while retaining individual variation."""
        return self.shared_components.compose_instance_with_shared(
            instance_id=instance_id,
            canonical_name=canonical_name,
            domain=domain,
            required_shared_components=shared_component_ids,
            unique_features=unique_features
        )

    # -------------------------------------------------------------
    # Structural Difference & Pattern Discovery
    # -------------------------------------------------------------
    def compare_structures(self, entity_a_id: str, entity_b_id: str) -> StructuralComparisonReport:
        """Extracts shared components, unique components, and attribute variance."""
        return self.difference_detector.compare_entities(entity_a_id, entity_b_id)

    def discover_patterns_across_instances(
        self,
        instance_ids: List[str],
        domain: str = "general"
    ) -> List[DiscoveredPattern]:
        """Discovers recurring relational motifs across observed instances."""
        return self.pattern_discovery.discover_patterns(instance_ids, domain=domain)

    # -------------------------------------------------------------
    # Hybrid Retrieval & Reranking
    # -------------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        as_of_time: Optional[str] = None
    ) -> List[RerankedCandidate]:
        """
        Executes multi-channel hybrid retrieval (exact, lexical, vector, graph)
        followed by multi-factor contextual reranking.
        """
        query_vec = self.encoder.encode_text(query)
        candidates = self.hybrid_retrieval.retrieve(
            query=query,
            query_vector=query_vec,
            as_of_time=as_of_time,
            top_k=top_k * 2
        )
        return self.retrieval_reranker.rerank(
            query=query,
            candidates=candidates,
            as_of_time=as_of_time
        )[:top_k]

    # -------------------------------------------------------------
    # Memory Consolidation & Integrity Audit
    # -------------------------------------------------------------
    def consolidate_memory(self) -> ConsolidationAudit:
        """Sleep-cycle memory consolidation: deduplicates claims and merges evidence."""
        return self.memory_consolidator.consolidate_relationships()

    def audit_integrity(self) -> IntegrityCheckReport:
        """Runs full consistency audit across vectors, relationships, and assemblies."""
        return self.integrity_engine.run_full_audit()

    # -------------------------------------------------------------
    # Backward-Compatible Universal User Memory Calls
    # -------------------------------------------------------------
    def remember_user_fact(self, user_id: str, relation_type: str, fact_value: str, confidence: float = 1.0) -> Any:
        return self.user_memory.remember_fact(user_id, relation_type, fact_value, confidence)

    def extract_and_remember(self, user_id: str, text: str) -> Dict[str, str]:
        return self.user_memory.extract_from_dialogue(user_id, text)

    def reconstruct_user_profile(self, user_id: str) -> Dict[str, str]:
        return self.user_memory.reconstruct_profile(user_id)

    def unbind_relation(self, subject_id: str, relation_type: str) -> Optional[Tuple[str, float]]:
        return self.user_memory.unbind_relation(subject_id, relation_type)

    def synthesize_agent_context(self, user_id: str, query_text: str, top_k: int = 5) -> AgentGroundingContext:
        activated_nodes = self.search_memory(query_text, top_k=top_k)
        return self.user_memory.synthesize_agent_grounding(user_id, query_text, activated_nodes)

    def register_concept(
        self,
        value: str,
        concept_type: str = "concept",
        source: str = "manual",
        confidence: float = 1.0,
        custom_id: Optional[str] = None
    ) -> AtomicDNA:
        return self.dna_registry.register(
            value=value,
            dna_type=concept_type,
            source=source,
            confidence=confidence,
            custom_id=custom_id
        )

    def search_memory(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.activation_engine.activate_from_query(query_text, top_k=top_k)

    def compose_product(
        self,
        name: str,
        components: List[AtomicDNA],
        custom_id: Optional[str] = None
    ) -> CompositionManifest:
        """Composes product DNA from constituent properties (e.g. Bean + Aroma + Taste + Caffeine)."""
        return self.composition_engine.compose(
            name=name,
            components=components,
            operation="bundle",
            composite_id=custom_id
        )

    def record_interaction(
        self,
        customer_id: str,
        event_type: str,
        target_id: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Learns an interaction event:
        Creates atomic representations for attributes, builds Event Assembly, links relations,
        and dynamically recomposes the customer behaviour assembly.
        """
        return self.learning_engine.learn_from_event(
            event_type=event_type,
            subject_id=customer_id,
            object_id=target_id,
            attributes=attributes,
            timestamp=timestamp
        )

    def predict_next_action(self, customer_id: str) -> Dict[str, Any]:
        """Synthesizes customer behaviour assemblies and cadences to predict next action & window."""
        return self.prediction_engine.predict_next_event(customer_id)

    def reason_path(self, start_id: str, max_hops: int = 3) -> Dict[str, Any]:
        return self.frontier_reasoning.traverse(start_id, max_hops=max_hops)

    def sync_index(self):
        self.activation_engine.sync_index()

    def save_to_disk(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "version": "2.1.0",
            "dims": self.dims,
            "seed": self.seed,
            "dna_registry": self.dna_registry.serialize(),
            "relation_registry": self.relation_registry.serialize(),
            "assembly_engine": self.assembly_engine.serialize(),
            "learning_engine": self.learning_engine.serialize(),
            "retrieval_index": self.activation_engine.index.serialize(),
            "concept_registry": self.concept_registry.to_dict(),
            "relationship_lifecycle": self.relationship_lifecycle.to_dict()
        }
        content = json.dumps(payload, indent=2)
        if filepath.endswith(".gz"):
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

    def load_from_disk(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"State file not found at '{filepath}'.")

        if filepath.endswith(".gz"):
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)

        self.dims = payload.get("dims", self.dims)
        self.seed = payload.get("seed", self.seed)

        if "dna_registry" in payload:
            self.dna_registry.restore(payload["dna_registry"])
        if "relation_registry" in payload:
            self.relation_registry.restore(payload["relation_registry"])
        if "assembly_engine" in payload:
            self.assembly_engine.restore(payload["assembly_engine"])
        if "learning_engine" in payload:
            self.learning_engine.restore(payload["learning_engine"])
        if "concept_registry" in payload:
            self.concept_registry = ConceptIdentityRegistry.from_dict(payload["concept_registry"])
        if "relationship_lifecycle" in payload:
            self.relationship_lifecycle = RelationshipLifecycleEngine.from_dict(
                payload["relationship_lifecycle"],
                concept_registry=self.concept_registry
            )
        if "retrieval_index" in payload:
            self.activation_engine.index.restore(payload["retrieval_index"])
        else:
            self.sync_index()
