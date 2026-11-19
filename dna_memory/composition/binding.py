"""
Binding & Unbinding Operations: Encodes and decodes structured associations between representations.
Uses Circular Convolution and Circular Correlation (Involution FFT).
"""

from typing import Tuple, List, Optional
import numpy as np
from dna_memory.core.vector_operations import bind, unbind, cosine_similarity

def bind_role_filler(role_vector: np.ndarray, filler_vector: np.ndarray) -> np.ndarray:
    """
    Encodes a role-filler pair:
      Pair = Role (*) Filler
    The result is quasi-orthogonal to both Role and Filler.
    """
    return bind(role_vector, filler_vector)

def unbind_filler(role_vector: np.ndarray, bound_vector: np.ndarray) -> np.ndarray:
    """
    Recovers the filler vector from a bound representation:
      Recovered_Filler = Role^(-1) (*) Bound
    """
    return unbind(role_vector, bound_vector)

def unbind_and_match(
    role_vector: np.ndarray,
    bound_vector: np.ndarray,
    candidate_vectors: List[np.ndarray],
    candidate_ids: List[str]
) -> Tuple[Optional[str], float]:
    """
    Unbinds the representation and finds the closest matching candidate.
    Returns: (best_match_id, similarity_score)
    """
    recovered = unbind(role_vector, bound_vector)
    best_id = None
    best_sim = -1.0

    for cand_id, cand_vec in zip(candidate_ids, candidate_vectors):
        sim = cosine_similarity(recovered, cand_vec)
        if sim > best_sim:
            best_sim = sim
            best_id = cand_id

    return best_id, best_sim
