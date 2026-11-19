"""
Bundling Operations: Combines several DNA vectors into a shared superposed representation.
"""

from typing import List, Tuple
import numpy as np
from dna_memory.core.vector_operations import bundle, normalize, cosine_similarity

def bundle_representations(
    vectors: List[np.ndarray],
    weights: List[float] = None
) -> Tuple[np.ndarray, List[float]]:
    """
    Bundles representations with confidence-weighted superposition.
    Returns:
      (composite_vector, normalized_weights)
    """
    if not vectors:
        raise ValueError("Cannot bundle empty vector list.")

    if weights is None:
        weights = [1.0] * len(vectors)

    total_w = sum(weights)
    if total_w > 1e-12:
        norm_weights = [w / total_w for w in weights]
    else:
        norm_weights = [1.0 / len(weights)] * len(weights)

    composite = bundle(vectors, weights=norm_weights)
    return composite, norm_weights

def measure_bundle_preservation(composite_vector: np.ndarray, constituent_vectors: List[np.ndarray]) -> List[float]:
    """Measures the cosine similarity of the bundled representation to each of its constituents."""
    return [cosine_similarity(composite_vector, v) for v in constituent_vectors]
