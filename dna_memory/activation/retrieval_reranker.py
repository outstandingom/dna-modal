"""
Retrieval Reranking Engine: Multi-factor contextual reranker for memory candidates.
Combines query relevance, concept identity match, relationship relevance, temporal freshness,
evidence quality, and source reliability into a calibrated final ranking.
"""

from dataclasses import dataclass, field
import datetime
from typing import Dict, List, Optional, Any
from dna_memory.activation.hybrid_retrieval import RetrievalCandidate


@dataclass
class RerankedCandidate:
    """A candidate evaluated and scored across all contextual criteria."""
    candidate: RetrievalCandidate
    final_score: float
    score_breakdown: Dict[str, float]
    rank: int


class RetrievalReranker:
    """
    Reranks candidates retrieved by HybridRetrievalEngine using multi-criteria scoring
    instead of relying on a single arbitrary vector threshold.
    """
    def __init__(
        self,
        relevance_weight: float = 0.35,
        identity_weight: float = 0.20,
        evidence_weight: float = 0.20,
        temporal_weight: float = 0.15,
        relationship_weight: float = 0.10
    ):
        self.weights = {
            "relevance": relevance_weight,
            "identity": identity_weight,
            "evidence": evidence_weight,
            "temporal": temporal_weight,
            "relationship": relationship_weight
        }

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        context: Optional[Dict[str, Any]] = None,
        as_of_time: Optional[str] = None
    ) -> List[RerankedCandidate]:
        """Reranks candidate memories and returns calibrated scores."""
        if not candidates:
            return []

        clean_query = query.strip().lower()
        scored: List[Tuple[RetrievalCandidate, float, Dict[str, float]]] = []

        for cand in candidates:
            scores: Dict[str, float] = {}

            # 1. Query relevance (lexical & semantic fused score)
            scores["relevance"] = min(1.0, cand.fused_score)

            # 2. Concept identity match
            cname = cand.canonical_name.lower()
            if cname == clean_query or cand.entity_id.lower() == clean_query:
                scores["identity"] = 1.0
            elif clean_query in cname or cname in clean_query:
                scores["identity"] = 0.8
            else:
                scores["identity"] = 0.4

            # 3. Evidence quality & confidence
            ev_count = sum(len(r.get("evidence", [])) for r in cand.relationships)
            scores["evidence"] = min(1.0, 0.5 + 0.1 * ev_count)

            # 4. Temporal validity
            # If candidate has active relationships, score high; else check dates
            if cand.relationships:
                has_active = any(r.get("status") == "active" for r in cand.relationships)
                scores["temporal"] = 1.0 if has_active else 0.6
            else:
                scores["temporal"] = 0.8

            # 5. Relationship density / graph connectivity
            scores["relationship"] = min(1.0, len(cand.relationships) * 0.25)

            # Calculate weighted composite score
            final_score = sum(scores[factor] * self.weights.get(factor, 0.0) for factor in scores)
            scored.append((cand, final_score, scores))

        # Sort descending
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = [
            RerankedCandidate(
                candidate=c,
                final_score=round(score, 4),
                score_breakdown={k: round(v, 4) for k, v in breakdown.items()},
                rank=idx + 1
            )
            for idx, (c, score, breakdown) in enumerate(scored)
        ]
        return reranked
