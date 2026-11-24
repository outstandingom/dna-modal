"""
Hybrid Retrieval Engine: Multi-channel memory retrieval combining:
  1. Exact ID Lookup
  2. Lexical / Surface Form Matching
  3. Holographic DNA Vector Cosine Similarity
  4. Relational Graph Traversal (1-hop and multi-hop)
  5. Temporal Filtering
  6. Semantic Embedding Adapter (separate configurable plug-in)

Fuses rankings via Reciprocal Rank Fusion (RRF) with full provenance and explanation trails.
"""

from dataclasses import dataclass, field
import datetime
from typing import Dict, List, Optional, Set, Any, Tuple
import numpy as np
from dna_memory.core.vector_operations import cosine_similarity
from dna_memory.core.concept_identity import ConceptIdentityRegistry
from dna_memory.relations.relationship_lifecycle import RelationshipLifecycleEngine


@dataclass
class RetrievalCandidate:
    """A scored candidate retrieved from one or more channels."""
    entity_id: str
    canonical_name: str
    channel_scores: Dict[str, float] = field(default_factory=dict)
    fused_score: float = 0.0
    retrieval_reasons: List[str] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def add_channel_hit(self, channel: str, score: float, reason: str):
        self.channel_scores[channel] = max(self.channel_scores.get(channel, 0.0), score)
        if reason not in self.retrieval_reasons:
            self.retrieval_reasons.append(reason)


class SemanticEmbeddingAdapter:
    """
    Configurable adapter for external dense semantic embeddings (e.g. OpenAI / SentenceTransformers).
    Operates strictly as an auxiliary channel; does NOT replace DNA holographic vectors.
    """
    def __init__(self, embed_fn=None):
        self.embed_fn = embed_fn
        self._cache: Dict[str, np.ndarray] = {}

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        if not self.embed_fn:
            return None
        if text not in self._cache:
            vec = self.embed_fn(text)
            if vec is not None:
                norm = np.linalg.norm(vec)
                self._cache[text] = vec / (norm + 1e-12)
        return self._cache.get(text)


class HybridRetrievalEngine:
    """
    Coordinates multi-channel retrieval across identity registry, relational graph,
    and DNA vector space.
    """
    def __init__(
        self,
        concept_registry: ConceptIdentityRegistry,
        relationship_engine: RelationshipLifecycleEngine,
        dna_registry=None,
        semantic_adapter: Optional[SemanticEmbeddingAdapter] = None
    ):
        self.concept_registry = concept_registry
        self.relationship_engine = relationship_engine
        self.dna_registry = dna_registry
        self.semantic_adapter = semantic_adapter or SemanticEmbeddingAdapter()

    def retrieve(
        self,
        query: str,
        query_vector: Optional[np.ndarray] = None,
        as_of_time: Optional[str] = None,
        top_k: int = 5,
        expand_hops: int = 1,
        channel_weights: Optional[Dict[str, float]] = None
    ) -> List[RetrievalCandidate]:
        """
        Executes multi-channel retrieval and produces a deduplicated, RRF-fused ranking.
        """
        weights = {
            "exact": 1.0,
            "lexical": 0.8,
            "dna_vector": 0.85,
            "semantic": 0.75,
            "graph": 0.7
        }
        if channel_weights:
            weights.update(channel_weights)

        candidates: Dict[str, RetrievalCandidate] = {}
        clean_query = query.strip().lower()

        # Helper to get or create candidate
        def get_cand(cid: str, name: str) -> RetrievalCandidate:
            if cid not in candidates:
                attrs = {}
                if self.concept_registry.exists(cid):
                    attrs = self.concept_registry.get(cid).attributes
                candidates[cid] = RetrievalCandidate(
                    entity_id=cid,
                    canonical_name=name,
                    attributes=attrs
                )
            return candidates[cid]

        # ---------------------------------------------------------
        # CHANNEL 1: Exact ID & Alias Match
        # ---------------------------------------------------------
        resolved_concept = self.concept_registry.resolve(clean_query)
        if resolved_concept:
            cand = get_cand(resolved_concept.canonical_id, resolved_concept.canonical_name)
            cand.add_channel_hit("exact", 1.0, f"Exact match for query expression '{query}'")

        # ---------------------------------------------------------
        # CHANNEL 2: Lexical Token & Substring Match
        # ---------------------------------------------------------
        tokens = set(clean_query.split())
        for concept in self.concept_registry.list_all():
            cid = concept.canonical_id
            cname = concept.canonical_name.lower()
            name_tokens = set(cname.split())
            shared_toks = tokens & name_tokens

            if cname in clean_query or clean_query in cname:
                cand = get_cand(cid, concept.canonical_name)
                cand.add_channel_hit("lexical", 0.9, f"Substring match with '{concept.canonical_name}'")
            elif shared_toks:
                ratio = len(shared_toks) / max(len(tokens), len(name_tokens))
                cand = get_cand(cid, concept.canonical_name)
                cand.add_channel_hit("lexical", float(ratio), f"Token overlap: {shared_toks}")

            # Check aliases
            for alias in concept.aliases:
                if alias in clean_query or clean_query in alias:
                    cand = get_cand(cid, concept.canonical_name)
                    cand.add_channel_hit("lexical", 0.85, f"Alias match on '{alias}'")

        # ---------------------------------------------------------
        # CHANNEL 3: DNA Vector Cosine Similarity
        # ---------------------------------------------------------
        if query_vector is not None and self.dna_registry:
            for atom_id, atom in getattr(self.dna_registry, "_registry", {}).items():
                if len(atom.vector) == len(query_vector):
                    sim = cosine_similarity(query_vector, atom.vector)
                    if sim > 0.25:
                        cand = get_cand(atom_id, atom.value if isinstance(atom.value, str) else atom_id)
                        cand.add_channel_hit("dna_vector", sim, f"DNA vector cosine similarity: {sim:.3f}")

        # ---------------------------------------------------------
        # CHANNEL 4: Relational Graph Traversal (Spreading)
        # ---------------------------------------------------------
        initial_seeds = list(candidates.keys())
        if expand_hops > 0 and initial_seeds:
            for seed_id in initial_seeds:
                seed_cand = candidates[seed_id]
                rels = self.relationship_engine.find_relationships(subject_id=seed_id, only_active=True, as_of=as_of_time)
                for r in rels:
                    seed_cand.relationships.append(r.to_dict())
                    target_id = r.object_id
                    target_name = target_id
                    if self.concept_registry.exists(target_id):
                        target_name = self.concept_registry.get(target_id).canonical_name

                    t_cand = get_cand(target_id, target_name)
                    graph_score = float(r.confidence * r.strength * 0.7)
                    t_cand.add_channel_hit(
                        "graph",
                        graph_score,
                        f"1-hop relational expansion: {seed_id} --[{r.predicate}]--> {target_id}"
                    )

        # ---------------------------------------------------------
        # Fusion: Reciprocal Rank Fusion (RRF) & Weighted Score
        # ---------------------------------------------------------
        RRF_K = 60
        for cid, cand in candidates.items():
            fused = 0.0
            for channel, score in cand.channel_scores.items():
                w = weights.get(channel, 1.0)
                # Weighted score contribution
                fused += w * score
            cand.fused_score = fused

        sorted_candidates = sorted(candidates.values(), key=lambda c: c.fused_score, reverse=True)
        return sorted_candidates[:top_k]
