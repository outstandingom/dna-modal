"""
Holographic Reduced Representations (HRR) and Vector Symbolic Architecture (VSA) operations.
Implements:
  - normalize: unit L2 normalization with numerical epsilon safety.
  - cosine_similarity: dot product between normalized representations.
  - bundle: superposition / addition of representations.
  - bind: circular convolution (via FFT) creating quasi-orthogonal associations.
  - unbind: circular correlation (approximate inverse) extracting bound features.
"""

import numpy as np

def normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalization with division-by-zero protection."""
    norm = np.linalg.norm(vec)
    if norm < eps:
        return np.zeros_like(vec, dtype=np.float64)
    return vec / norm

def cosine_similarity(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity between two vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

def bundle(vectors: list, weights: list = None) -> np.ndarray:
    """
    Superposition operation: combines multiple vectors into a shared representation.
    Preserves partial similarity to each constituent component.
    """
    if not vectors:
        raise ValueError("Cannot bundle an empty list of vectors.")
    
    vecs = [np.asarray(v, dtype=np.float64) for v in vectors]
    d = len(vecs[0])
    
    if weights is None:
        weights = [1.0] * len(vecs)
    else:
        if len(weights) != len(vecs):
            raise ValueError(f"Weights length ({len(weights)}) does not match vectors length ({len(vecs)}).")

    combined = np.zeros(d, dtype=np.float64)
    for v, w in zip(vecs, weights):
        combined += w * normalize(v)
        
    return normalize(combined)

def bind(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    HRR Binding via Circular Convolution:
      c = v1 * v2
    Computes circular convolution in O(D log D) time using Fast Fourier Transform.
    The resulting vector is quasi-orthogonal to both v1 and v2.
    """
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions must match for binding: {len(a)} vs {len(b)}")

    # FFT-based circular convolution
    fa = np.fft.fft(a)
    fb = np.fft.fft(b)
    bound = np.fft.ifft(fa * fb).real
    return normalize(bound)

def unbind(key_vector: np.ndarray, bound_vector: np.ndarray) -> np.ndarray:
    """
    HRR Unbinding via Circular Correlation:
    To unbind key_vector from bound_vector = bind(key_vector, value_vector),
    convolve the exact involution / inverse of key_vector with bound_vector.
    
    The involution k* is k with indices reversed modulo D:
      k*[i] = k[(-i) % D]
    """
    k = np.asarray(key_vector, dtype=np.float64)
    b = np.asarray(bound_vector, dtype=np.float64)
    d = len(k)
    if len(b) != d:
        raise ValueError(f"Vector dimensions must match for unbinding: {d} vs {len(b)}")

    # Compute involution k*
    k_inv = np.zeros(d, dtype=np.float64)
    k_inv[0] = k[0]
    k_inv[1:] = k[:0:-1]

    # Circular convolution of k_inv with b
    fk_inv = np.fft.fft(k_inv)
    fb = np.fft.fft(b)
    unbound = np.fft.ifft(fk_inv * fb).real
    return normalize(unbound)
