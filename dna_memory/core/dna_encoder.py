"""
DNA Encoder: Converts text, entities, properties, and values into normalized representations.
Supports variable dimensions (128, 512, 1024, 2048).
"""

import hashlib
import numpy as np
from dna_memory.core.vector_operations import normalize

class DNAEncoder:
    """
    Deterministic vector encoder for Atomic DNA features.
    Produces high-capacity pseudo-random or continuous representations for symbolic tokens.
    """
    def __init__(self, dims: int = 128, seed: int = 42):
        self.dims = dims
        self.seed = seed
        self._cache = {}

    def encode_text(self, text: str) -> np.ndarray:
        """Deterministic seed-based orthogonal encoding for a text token."""
        key = (text, self.dims)
        if key in self._cache:
            return self._cache[key].copy()

        # Generate deterministic pseudo-random Gaussian vector from SHA256
        h = hashlib.sha256(f"{self.seed}:{text}".encode("utf-8")).digest()
        seed_int = int.from_bytes(h[:8], byteorder="big") % (2**32)
        rng = np.random.RandomState(seed_int)
        
        raw_vec = rng.normal(0.0, 1.0, size=self.dims)
        vec = normalize(raw_vec)
        self._cache[key] = vec
        return vec.copy()

    def encode_numeric(self, value: float, min_val: float = 0.0, max_val: float = 1000.0) -> np.ndarray:
        """
        Encodes a continuous numerical value using circular phase progression.
        Nearby values have high cosine similarity; distant values become quasi-orthogonal.
        """
        clamped = max(min_val, min(max_val, value))
        norm_val = (clamped - min_val) / (max_val - min_val + 1e-8)
        
        vec = np.zeros(self.dims, dtype=np.float64)
        for i in range(self.dims // 2):
            freq = 2.0 ** (i / 4.0)
            angle = norm_val * 2.0 * np.pi * freq
            vec[2 * i] = np.cos(angle)
            vec[2 * i + 1] = np.sin(angle)
            
        return normalize(vec)

    def encode_concept(self, concept_type: str, value: str) -> np.ndarray:
        """
        Encodes a type-qualified concept:
        binds Type vector with Value vector to prevent cross-talk.
        e.g., Color:Red vs Person:Red.
        """
        v_type = self.encode_text(f"TYPE:{concept_type}")
        v_val = self.encode_text(f"VALUE:{value}")
        from dna_memory.core.vector_operations import bind
        return bind(v_type, v_val)
