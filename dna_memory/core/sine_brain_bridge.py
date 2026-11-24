"""
Sine Brain Bridge (System A <-> System B Adapter)

Defines explicit architectural responsibilities and provides a tested interface
between the two DNA engines in the repository:

  • System A (Compositional Relational DNA - `dna_memory`):
    - Role: Canonical long-term memory, concept identity, relationship lifecycle,
      bi-temporal validity, hierarchical assemblies, and hybrid multi-channel retrieval.
    - Mathematical representation: 128D Holographic Reduced Representations (HRR)
      using continuous circular convolution and correlation.

  • System B (Adaptive Sine-Based Cognitive Brain - `knowledge_graph_env.py`):
    - Role: Continuous oscillatory cognitive substrate, 26-letter alphabet positional
      sine encoding, Hebbian synaptic adaptation, and 12D heuristic routing.
    - Mathematical representation: Sine-phase modulated continuous embeddings.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class BridgeSyncReport:
    """Status report of synchronization between System A and System B."""
    concepts_synced_to_brain: int
    relations_synced_to_brain: int
    learned_synapses_ingested: int
    status: str = "SYNCHRONIZED"


class SineBrainBridge:
    """
    Bi-directional interface linking System A (Compositional Memory) and
    System B (Sine-Based Adaptive Cognitive Brain).
    """

    def __init__(self, memory_system: Any, brain_env: Optional[Any] = None):
        self.memory = memory_system
        self.brain_env = brain_env
        self._synced_concepts: Dict[str, np.ndarray] = {}

    def set_brain_env(self, brain_env: Any) -> None:
        """Attach or update the System B runtime environment."""
        self.brain_env = brain_env

    def sync_memory_to_brain(self, limit: int = 100) -> BridgeSyncReport:
        """
        Synchronizes canonical concepts and verified relationships from System A
        into System B's active concept map.
        """
        c_count = 0
        r_count = 0

        # Export active concepts from System A
        if hasattr(self.memory, "dna_registry"):
            for atom_id, atom in list(self.memory.dna_registry._by_id.items())[:limit]:
                self._synced_concepts[atom_id] = atom.vector
                c_count += 1
                if self.brain_env is not None and hasattr(self.brain_env, "concepts"):
                    # Populate System B concept registry if missing
                    if atom_id not in self.brain_env.concepts:
                        self.brain_env.concepts[atom_id] = atom.vector

        # Export active relationships from System A
        if hasattr(self.memory, "relationship_lifecycle"):
            active_rels = self.memory.relationship_lifecycle.query_relationships(status="ACTIVE")
            for r in active_rels[:limit]:
                r_count += 1
                if self.brain_env is not None and hasattr(self.brain_env, "relationships"):
                    self.brain_env.relationships[r.relationship_id] = {
                        "subject": r.subject_id,
                        "predicate": r.predicate,
                        "object": r.object_id,
                        "confidence": r.confidence
                    }

        return BridgeSyncReport(
            concepts_synced_to_brain=c_count,
            relations_synced_to_brain=r_count,
            learned_synapses_ingested=0,
            status="SUCCESS"
        )

    def ingest_brain_synapses_to_memory(self, min_weight: float = 0.5) -> int:
        """
        Extracts strong associative synapses learned in System B (via Hebbian updates)
        and registers them as candidate relational hypotheses in System A's lifecycle engine.
        """
        if self.brain_env is None or not hasattr(self.brain_env, "synapses"):
            return 0

        ingested = 0
        synapses = getattr(self.brain_env, "synapses", {})
        for (src, dst), weight in synapses.items():
            if weight >= min_weight:
                if hasattr(self.memory, "relationship_lifecycle"):
                    self.memory.relationship_lifecycle.create_relationship(
                        subject_id=str(src),
                        predicate="associatively_linked_to",
                        object_id=str(dst),
                        confidence=float(min(1.0, weight)),
                        source="system_b_hebbian_synapse",
                        evidence_content=f"Synaptic weight: {weight:.4f}"
                    )
                    ingested += 1

        return ingested

    def unified_context_grounding(self, user_id: str, query_text: str) -> Dict[str, Any]:
        """
        Combines System A's lossless structural facts and authoritative manifest with
        System B's continuous predictive routing.
        """
        # 1. Structural facts from System A
        user_facts = {}
        if hasattr(self.memory, "user_memory"):
            user_facts = self.memory.user_memory.reconstruct_profile(user_id)

        # 2. Heuristic prediction / routing from System B if available
        heuristic_prediction = {}
        if self.brain_env is not None and hasattr(self.brain_env, "predict_heuristic"):
            heuristic_prediction = self.brain_env.predict_heuristic(user_id, query_text)

        return {
            "user_id": user_id,
            "structural_facts": user_facts,
            "heuristic_routing": heuristic_prediction,
            "architecture_mode": "DUAL_ENGINE_UNIFIED"
        }
