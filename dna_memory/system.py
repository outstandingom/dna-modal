"""
DNAMemorySystem: Complete, unified facade for the Recursive Compositional DNA Memory Architecture.
Integrates atomic encoding, holographic vector composition, relational memory,
hierarchical recursive assemblies, spreading activation, learning, and prediction with unified disk persistence.
"""

import os
import json
import gzip
from typing import Dict, Any, List, Optional
import numpy as np

from dna_memory.core.atomic_dna import AtomicDNA, DNAFeatureMetadata
from dna_memory.core.dna_encoder import DNAEncoder
from dna_memory.core.dna_registry import DNARegistry
from dna_memory.composition.composition_engine import CompositionEngine, CompositionManifest
from dna_memory.composition.decomposition_engine import DecompositionEngine
from dna_memory.relations.relation_registry import RelationRegistry
from dna_memory.assembly.recursive_composition import AssemblyEngine, DNAAssembly
from dna_memory.activation.retrieval_index import RetrievalIndex
from dna_memory.activation.activation_engine import DNAActivationEngine
from dna_memory.reasoning.multi_hop import FrontierReasoningEngine
from dna_memory.learning.learning_engine import MemoryLearningEngine
from dna_memory.prediction.behaviour_prediction import BehaviourPredictionEngine
from dna_memory.prediction.temporal_patterns import TemporalPatternAnalyzer

class DNAMemorySystem:
    """
    Unified Recursive Compositional DNA Memory System.
    """
    def __init__(self, dims: int = 128, seed: int = 42):
        self.dims = dims
        self.seed = seed

        self.encoder = DNAEncoder(dims=dims, seed=seed)
        self.dna_registry = DNARegistry(self.encoder)
        self.relation_registry = RelationRegistry(self.dna_registry)
        self.composition_engine = CompositionEngine(self.dna_registry)
        self.decomposition_engine = DecompositionEngine(self.dna_registry)
        self.assembly_engine = AssemblyEngine(self.dna_registry, self.relation_registry)
        self.retrieval_index = RetrievalIndex(dims=dims)
        self.activation_engine = DNAActivationEngine(self.dna_registry, self.relation_registry, self.assembly_engine)
        self.frontier_reasoning = FrontierReasoningEngine(self.relation_registry, self.assembly_engine)
        self.learning_engine = MemoryLearningEngine(self.dna_registry, self.relation_registry, self.assembly_engine)
        self.prediction_engine = BehaviourPredictionEngine(self.learning_engine, self.relation_registry, self.assembly_engine)

    def register_concept(
        self,
        value: str,
        concept_type: str = "concept",
        source: str = "manual",
        confidence: float = 1.0,
        custom_id: Optional[str] = None
    ) -> AtomicDNA:
        """Registers a basic concept or entity in Atomic DNA."""
        return self.dna_registry.register(
            value=value,
            dna_type=concept_type,
            source=source,
            confidence=confidence,
            custom_id=custom_id
        )

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

    def search_memory(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Spreading activation and neural retrieval across DNA memory."""
        return self.activation_engine.activate_from_query(query_text, top_k=top_k)

    def reason_path(self, start_id: str, max_hops: int = 3) -> Dict[str, Any]:
        """Multi-hop relational frontier traversal."""
        return self.frontier_reasoning.traverse(start_id, max_hops=max_hops)

    def sync_index(self):
        """Synchronizes all atoms and assemblies to the ANN retrieval index."""
        self.activation_engine.sync_index()

    def save_to_disk(self, filepath: str):
        """Serializes and persists the complete system state to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "version": "2.0.0",
            "dims": self.dims,
            "seed": self.seed,
            "dna_registry": self.dna_registry.serialize(),
            "relation_registry": self.relation_registry.serialize(),
            "assembly_engine": self.assembly_engine.serialize(),
            "learning_engine": self.learning_engine.serialize(),
            "retrieval_index": self.activation_engine.index.serialize()
        }

        content = json.dumps(payload, indent=2)
        if filepath.endswith(".gz"):
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

    def load_from_disk(self, filepath: str):
        """Restores complete system state from disk."""
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
        if "retrieval_index" in payload:
            self.activation_engine.index.restore(payload["retrieval_index"])
        else:
            self.sync_index()
