# ============================================================
# PROJECTION LAYER: Bridge between 128-dim Brain and 12-dim Judge
# ============================================================

import numpy as np
from typing import Optional, List, Tuple
import pickle
import os

class DimensionProjector:
    """
    Efficiently projects between 128-dim (knowledge graph) and 12-dim (fast judge).
    Uses SVD to find optimal projection that preserves maximum information.
    
    Think of it as: JPEG compression for your brain vectors!
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path or os.path.join(PERSIST_DIR, "projection.pkl")
        self.projection_matrix: Optional[np.ndarray] = None  # 128x12
        self.reconstruction_matrix: Optional[np.ndarray] = None  # 12x128
        self.explained_variance: Optional[np.ndarray] = None
        self.is_trained = False
        self._load()
    
    def train_from_concepts(self, concept_vectors: List[np.ndarray]) -> None:
        """
        Learn the optimal projection from existing concept vectors.
        Uses SVD to find the 12 most important dimensions.
        
        Args:
            concept_vectors: List of 128-dim concept vectors
        """
        if not concept_vectors:
            print("[WARN] No concept vectors provided. Using random projection.")
            self._random_projection()
            return
        
        # Stack all vectors
        vectors = np.vstack([v.reshape(1, -1) for v in concept_vectors if v is not None])
        if len(vectors) == 0:
            print("[WARN] No valid concept vectors. Using random projection.")
            self._random_projection()
            return
        
        # Center the data
        mean_vec = np.mean(vectors, axis=0)
        centered = vectors - mean_vec
        
        # SVD: find the most important directions
        # Vt has shape (128, 128) - each row is a principal component
        U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
        
        # Take top 12 components as projection
        # Vt[:12] is 12x128, so we transpose to 128x12
        self.projection_matrix = Vt[:12].T  # Shape: (128, 12)
        
        # Store reconstruction matrix (for going back from 12 to 128)
        # Using pseudo-inverse for best reconstruction
        self.reconstruction_matrix = np.linalg.pinv(self.projection_matrix.T)  # Shape: (12, 128)
        
        # Store explained variance (how much info we keep)
        self.explained_variance = S[:12] / np.sum(S)
        self.is_trained = True
        
        # Save to disk
        self._save()
        
        print(f"[INFO] Projection trained on {len(vectors)} concepts")
        print(f"[INFO] Explained variance: {np.sum(self.explained_variance)*100:.1f}%")
    
    def _random_projection(self) -> None:
        """Fallback: Use random orthonormal projection."""
        rng = np.random.RandomState(42)
        random_matrix = rng.randn(128, 12)
        # Orthonormalize using QR decomposition
        Q, _ = np.linalg.qr(random_matrix)
        self.projection_matrix = Q  # Shape: (128, 12)
        self.reconstruction_matrix = Q.T  # Shape: (12, 128)
        self.is_trained = True
    
    def project_128_to_12(self, vector_128: np.ndarray) -> np.ndarray:
        """
        Project a 128-dim vector down to 12-dim.
        
        Args:
            vector_128: np.ndarray of shape (128,) or (batch_size, 128)
        
        Returns:
            np.ndarray of shape (12,) or (batch_size, 12)
        """
        if not self.is_trained:
            self._random_projection()
        
        # Handle batch input
        if vector_128.ndim == 1:
            vector_128 = vector_128.reshape(1, -1)
        
        # Project: (batch, 128) @ (128, 12) = (batch, 12)
        vector_12 = vector_128 @ self.projection_matrix
        
        return vector_12.squeeze()
    
    def project_12_to_128(self, vector_12: np.ndarray) -> np.ndarray:
        """
        Reconstruct approximate 128-dim vector from 12-dim.
        This is lossy - you get an approximation, not the original!
        
        Args:
            vector_12: np.ndarray of shape (12,) or (batch_size, 12)
        
        Returns:
            np.ndarray of shape (128,) or (batch_size, 128)
        """
        if not self.is_trained:
            self._random_projection()
        
        if vector_12.ndim == 1:
            vector_12 = vector_12.reshape(1, -1)
        
        # Reconstruct: (batch, 12) @ (12, 128) = (batch, 128)
        vector_128 = vector_12 @ self.reconstruction_matrix
        
        return vector_128.squeeze()
    
    def explain(self) -> dict:
        """
        Get information about how well the projection preserves information.
        """
        if not self.is_trained:
            return {"status": "untrained"}
        
        return {
            "status": "trained",
            "input_dim": 128,
            "output_dim": 12,
            "explained_variance_ratio": self.explained_variance.tolist(),
            "total_explained_variance": float(np.sum(self.explained_variance)),
            "compression_ratio": 128 / 12,
            "memory_saved_per_vector": 116  # 128 - 12
        }
    
    def _save(self):
        """Save projection to disk."""
        if self.persist_path:
            try:
                os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
                data = {
                    "projection_matrix": self.projection_matrix,
                    "reconstruction_matrix": self.reconstruction_matrix,
                    "explained_variance": self.explained_variance,
                    "is_trained": self.is_trained
                }
                with open(self.persist_path, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                print(f"[WARN] Failed to save projection: {e}")
    
    def _load(self):
        """Load projection from disk."""
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, 'rb') as f:
                    data = pickle.load(f)
                self.projection_matrix = data["projection_matrix"]
                self.reconstruction_matrix = data["reconstruction_matrix"]
                self.explained_variance = data.get("explained_variance")
                self.is_trained = data.get("is_trained", True)
                print("[INFO] Projection loaded from disk")
            except Exception as e:
                print(f"[WARN] Failed to load projection: {e}")

# ============================================================
# FAST PROJECTION CACHE (for performance)
# ============================================================

class ProjectionCache:
    """
    Caches projected vectors to avoid recomputing.
    """
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[np.ndarray]:
        return self.cache.get(key)
    
    def set(self, key: str, vector: np.ndarray):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = vector
    
    def clear(self):
        self.cache.clear()

# ============================================================
# INTEGRATION INTO KNOWLEDGE GRAPH ENV
# ============================================================

# Add this to KnowledgeGraphEnv.__init__:
"""
# Add dimension projector
self.projector = DimensionProjector()

# Learn projection from existing concepts
if len(self.concept_memory.concepts) > 0:
    concept_vectors = [c.vector for c in self.concept_memory.concepts.values() 
                      if c.vector is not None]
    if concept_vectors:
        self.projector.train_from_concepts(concept_vectors)
else:
    # Use random projection as fallback
    self.projector._random_projection()

# Add cache for performance
self.projection_cache = ProjectionCache()
"""

# Add this method to KnowledgeGraphEnv:
"""
def project_concept(self, concept_name: str) -> np.ndarray:
    '''Get 12-dim projection of a concept.'''
    concept_name = concept_name.lower()
    
    # Check cache first
    cached = self.projection_cache.get(concept_name)
    if cached is not None:
        return cached
    
    # Get concept
    concept = self.concept_memory.concepts.get(concept_name)
    if concept is None or concept.vector is None:
        return np.zeros(12)
    
    # Project
    projected = self.projector.project_128_to_12(concept.vector)
    
    # Cache
    self.projection_cache.set(concept_name, projected)
    
    return projected

def get_skill_vector_from_text(self, text: str) -> np.ndarray:
    '''Get 12-dim skill vector from text.'''
    # Get 128-dim embedding
    seq = self.predictive_letter_vec.get_dna_sequence(text)
    if len(seq) == 0:
        return np.zeros(12)
    
    # Average to 128-dim
    embedding_128 = np.mean(seq, axis=0)
    
    # Project to 12-dim
    embedding_12 = self.projector.project_128_to_12(embedding_128)
    
    # Get memory context (12-dim)
    memory_context = np.zeros(12)
    
    # Get skill vector
    skill_vector = get_adapter()(embedding_12, memory_context)
    return skill_vector
"""

# ============================================================
# FASTAPI ENDPOINT FOR PROJECTION INFO
# ============================================================

# Add this endpoint to your FastAPI app:
"""
@app.get("/projection/info")
async def projection_info_endpoint():
    '''Get information about the dimension projection layer.'''
    if _api_env is None:
        raise HTTPException(status_code=503, detail="Environment still initializing.")
    
    info = _api_env.projector.explain()
    return {
        "projection": info,
        "cache_size": len(_api_env.projection_cache.cache),
        "concepts_available": len(_api_env.concept_memory.concepts)
    }

@app.post("/projection/test")
async def test_projection_endpoint(concept: str):
    '''Test projection on a specific concept.'''
    if _api_env is None:
        raise HTTPException(status_code=503, detail="Environment still initializing.")
    
    concept = concept.lower()
    if concept not in _api_env.concept_memory.concepts:
        raise HTTPException(status_code=404, detail="Concept not found")
    
    concept_obj = _api_env.concept_memory.concepts[concept]
    vector_128 = concept_obj.vector
    
    if vector_128 is None:
        raise HTTPException(status_code=400, detail="Concept has no vector")
    
    # Project
    vector_12 = _api_env.projector.project_128_to_12(vector_128)
    vector_128_reconstructed = _api_env.projector.project_12_to_128(vector_12)
    
    # Calculate reconstruction error
    error = np.linalg.norm(vector_128 - vector_128_reconstructed)
    
    return {
        "concept": concept,
        "original_128_dim": vector_128.tolist()[:5],  # Show first 5 dims
        "projected_12_dim": vector_12.tolist(),
        "reconstructed_128_dim": vector_128_reconstructed.tolist()[:5],
        "reconstruction_error": float(error),
        "compression_ratio": "128→12 (saves 90.6% memory)"
    }
"""
