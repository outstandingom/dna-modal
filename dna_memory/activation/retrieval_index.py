"""
Retrieval Index: High-performance approximate nearest neighbor (ANN) vector indexing
using partitioned Voronoi clustering (Inverted File Index / IVF) in pure NumPy.
Prunes candidate search space by querying cluster centroids first, avoiding full linear scans.
"""

from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np

class RetrievalIndex:
    """
    Partitioned / Clustered Approximate Nearest Neighbor (ANN) Index.
    When N < 32, uses fast exact vectorized cosine search.
    When N >= 32, partitions vector space into Voronoi clusters (IVF).
    Queries probe only the top `nprobe` closest cluster partitions,
    achieving sub-linear search complexity with high recall.
    """
    def __init__(self, dims: int = 128, n_clusters: Optional[int] = None, cluster_threshold: int = 32):
        self.dims = dims
        self._target_clusters = n_clusters
        self.cluster_threshold = cluster_threshold
        
        self._ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        self._vectors: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, dict] = {}

        # Clustering partitions (IVF)
        self._centroids: Optional[np.ndarray] = None  # Shape (K, D)
        self._cluster_to_ids: Dict[int, List[str]] = {}
        self._id_to_cluster: Dict[str, int] = {}
        self._is_clustered: bool = False

    def add(self, item_id: str, vector: np.ndarray, meta: Optional[dict] = None):
        """Add or update an item in the index."""
        vec = np.asarray(vector, dtype=np.float64).flatten()
        norm = np.linalg.norm(vec)
        unit_vec = vec / (norm + 1e-12)

        if item_id not in self._id_to_idx:
            self._id_to_idx[item_id] = len(self._ids)
            self._ids.append(item_id)

        self._vectors[item_id] = unit_vec
        self._metadata[item_id] = meta or {}

        # If already clustered, assign to nearest centroid
        if self._is_clustered and self._centroids is not None:
            c_sims = np.dot(self._centroids, unit_vec)
            best_c = int(np.argmax(c_sims))
            old_c = self._id_to_cluster.get(item_id)
            if old_c is not None and old_c in self._cluster_to_ids:
                if item_id in self._cluster_to_ids[old_c]:
                    self._cluster_to_ids[old_c].remove(item_id)
            self._id_to_cluster[item_id] = best_c
            self._cluster_to_ids.setdefault(best_c, []).append(item_id)
        elif len(self._ids) >= self.cluster_threshold:
            self.rebuild_clusters()

    def rebuild_clusters(self, n_clusters: Optional[int] = None, max_iter: int = 10):
        """Builds Voronoi partitions using spherical k-means."""
        n = len(self._ids)
        if n < 4:
            self._is_clustered = False
            return

        k = n_clusters or self._target_clusters or max(4, int(np.sqrt(n)))
        k = min(k, n)

        # Collect matrix (N, D)
        mat = np.vstack([self._vectors[iid] for iid in self._ids])

        # Initialize centroids randomly with fixed seed for determinism
        rng = np.random.RandomState(42)
        init_indices = rng.choice(n, size=k, replace=False)
        centroids = mat[init_indices].copy()

        # Spherical k-means
        for _ in range(max_iter):
            # (N, K) dot product similarities
            sims = np.dot(mat, centroids.T)
            assignments = np.argmax(sims, axis=1)

            new_centroids = np.zeros_like(centroids)
            for c_idx in range(k):
                members = mat[assignments == c_idx]
                if len(members) > 0:
                    sum_vec = np.sum(members, axis=0)
                    norm = np.linalg.norm(sum_vec)
                    new_centroids[c_idx] = sum_vec / (norm + 1e-12)
                else:
                    new_centroids[c_idx] = centroids[c_idx]
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids

        self._centroids = centroids
        self._cluster_to_ids = {c_idx: [] for c_idx in range(k)}
        self._id_to_cluster = {}

        sims = np.dot(mat, self._centroids.T)
        final_assignments = np.argmax(sims, axis=1)

        for i, iid in enumerate(self._ids):
            c_idx = int(final_assignments[i])
            self._id_to_cluster[iid] = c_idx
            self._cluster_to_ids[c_idx].append(iid)

        self._is_clustered = True

    def query(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0,
        nprobe: int = 2,
        exact: bool = False
    ) -> List[Tuple[str, float, dict]]:
        """
        Similarity query.
        When exact=False and index is clustered, searches only candidate clusters (ANN).
        """
        if not self._ids:
            return []

        q = np.asarray(query_vec, dtype=np.float64).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            return []
        q_unit = q / q_norm

        # Determine candidate IDs to scan
        if exact or not self._is_clustered or self._centroids is None or len(self._ids) < self.cluster_threshold:
            candidates = self._ids
        else:
            # Query centroids (K similarities)
            c_sims = np.dot(self._centroids, q_unit)
            n_probe_actual = min(nprobe, len(self._centroids))
            top_clusters = np.argpartition(-c_sims, n_probe_actual - 1)[:n_probe_actual]

            candidates = []
            for c_idx in top_clusters:
                candidates.extend(self._cluster_to_ids.get(int(c_idx), []))

            # Fallback if empty
            if not candidates:
                candidates = self._ids

        # Evaluate candidate dot products
        scores = []
        for cid in candidates:
            sim = float(np.dot(self._vectors[cid], q_unit))
            if sim >= min_score:
                scores.append((cid, sim, self._metadata.get(cid, {})))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def remove(self, item_id: str):
        """Removes an item from the index."""
        if item_id not in self._vectors:
            return

        del self._vectors[item_id]
        self._metadata.pop(item_id, None)

        if item_id in self._id_to_cluster:
            c = self._id_to_cluster.pop(item_id)
            if c in self._cluster_to_ids and item_id in self._cluster_to_ids[c]:
                self._cluster_to_ids[c].remove(item_id)

        self._ids.remove(item_id)
        self._id_to_idx = {iid: i for i, iid in enumerate(self._ids)}

        if len(self._ids) < self.cluster_threshold:
            self._is_clustered = False
            self._centroids = None

    def count(self) -> int:
        return len(self._ids)

    def is_clustered(self) -> bool:
        return self._is_clustered

    def serialize(self) -> dict:
        """Serializes index state, vectors, and clustering assignments."""
        return {
            "dims": self.dims,
            "target_clusters": self._target_clusters,
            "cluster_threshold": self.cluster_threshold,
            "is_clustered": self._is_clustered,
            "centroids": self._centroids.tolist() if self._centroids is not None else None,
            "ids": list(self._ids),
            "vectors": {iid: vec.tolist() for iid, vec in self._vectors.items()},
            "metadata": self._metadata,
            "id_to_cluster": self._id_to_cluster,
            "cluster_to_ids": {str(k): v for k, v in self._cluster_to_ids.items()}
        }

    def restore(self, data: dict):
        """Restores index from serialized state."""
        self.dims = data.get("dims", self.dims)
        self._target_clusters = data.get("target_clusters")
        self.cluster_threshold = data.get("cluster_threshold", 32)
        self._is_clustered = data.get("is_clustered", False)

        c_raw = data.get("centroids")
        self._centroids = np.array(c_raw, dtype=np.float64) if c_raw is not None else None

        self._ids = list(data.get("ids", []))
        self._id_to_idx = {iid: i for i, iid in enumerate(self._ids)}
        self._vectors = {iid: np.array(vec, dtype=np.float64) for iid, vec in data.get("vectors", {}).items()}
        self._metadata = data.get("metadata", {})
        self._id_to_cluster = data.get("id_to_cluster", {})
        self._cluster_to_ids = {int(k): v for k, v in data.get("cluster_to_ids", {}).items()}
