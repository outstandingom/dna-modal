"""
Retrieval Index: Fast similarity indexing over DNA representations (atomic, assemblies, relations).
Avoids full scans by using top-k scoring and candidate partition filtering.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from dna_memory.core.vector_operations import cosine_similarity

class RetrievalIndex:
    """
    In-memory vector index with fast cosine ranking and partition pruning.
    """
    def __init__(self, dims: int = 128):
        self.dims = dims
        self._ids: List[str] = []
        self._vectors: Optional[np.ndarray] = None  # Matrix of shape (N, D)
        self._metadata: Dict[str, dict] = {}
        self._id_to_idx: Dict[str, int] = {}

    def add(self, item_id: str, vector: np.ndarray, meta: Optional[dict] = None):
        """Add or update an item in the index."""
        vec = np.asarray(vector, dtype=np.float64).reshape(1, -1)
        if item_id in self._id_to_idx:
            idx = self._id_to_idx[item_id]
            self._vectors[idx] = vec[0]
            self._metadata[item_id] = meta or {}
            return

        idx = len(self._ids)
        self._ids.append(item_id)
        self._id_to_idx[item_id] = idx
        self._metadata[item_id] = meta or {}

        if self._vectors is None:
            self._vectors = vec
        else:
            self._vectors = np.vstack([self._vectors, vec])

    def query(self, query_vec: np.ndarray, top_k: int = 5, min_score: float = 0.0) -> List[Tuple[str, float, dict]]:
        """Matrix-accelerated cosine similarity search."""
        if self._vectors is None or len(self._ids) == 0:
            return []

        q = np.asarray(query_vec, dtype=np.float64).reshape(1, -1)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            return []
        q_unit = q / q_norm

        # Matrix dot product (N,)
        sims = np.dot(self._vectors, q_unit.T).flatten()

        # Partition top_k
        k = min(top_k, len(self._ids))
        if k == 0:
            return []

        partition_indices = np.argpartition(-sims, k - 1)[:k]
        sorted_indices = partition_indices[np.argsort(-sims[partition_indices])]

        results = []
        for idx in sorted_indices:
            score = float(sims[idx])
            if score >= min_score:
                item_id = self._ids[idx]
                results.append((item_id, score, self._metadata.get(item_id, {})))

        return results

    def remove(self, item_id: str):
        if item_id not in self._id_to_idx:
            return
        idx = self._id_to_idx[item_id]
        self._ids.pop(idx)
        self._metadata.pop(item_id, None)
        self._vectors = np.delete(self._vectors, idx, axis=0)
        self._id_to_idx = {iid: i for i, iid in enumerate(self._ids)}

    def count(self) -> int:
        return len(self._ids)
