"""
skill_adapter.py — Pure NumPy fallback implementation.
Replaced torch-based MLP with a lightweight NumPy equivalent 
so the server can deploy on Render without the 800MB PyTorch dependency.
"""
import numpy as np
import os
from typing import List


class SkillAdapter:
    """
    Lightweight NumPy skill adapter that replaces the torch-based MLP.
    Uses Xavier initialization and tanh activation — same behaviour,
    zero heavy-framework dependencies.
    """
    def __init__(self, input_dim=24, hidden_dim=24, output_dim=12):
        # Xavier uniform init
        def xavier(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, (fan_out, fan_in)) * 0.5

        self.W1 = xavier(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = xavier(hidden_dim, hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = xavier(hidden_dim, output_dim)
        self.b3 = np.zeros(output_dim)

    def _relu(self, x):
        return np.maximum(0, x)

    def forward(self, user_embedding: np.ndarray, memory_context: np.ndarray) -> np.ndarray:
        x = np.concatenate([user_embedding, memory_context])
        h1 = self._relu(self.W1 @ x + self.b1)
        h2 = self._relu(self.W2 @ h1 + self.b2)
        skill = np.tanh(self.W3 @ h2 + self.b3)
        return skill

    def update_from_reward(self, user_embedding: np.ndarray, memory_context: np.ndarray, reward: float, lr: float = 0.001):
        """Simple Hebbian-style update — no autograd needed."""
        skill = self.forward(user_embedding, memory_context)
        # Reinforce or dampen weights based on reward direction
        scale = lr * (reward - 0.5) * 2
        x = np.concatenate([user_embedding, memory_context])
        self.W1 += scale * np.outer(skill[:self.W1.shape[0]], x[:self.W1.shape[1]])
        return skill

    def save(self, path="skill_adapter.npz"):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    def load(self, path="skill_adapter.npz"):
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.W3, self.b3 = data["W3"], data["b3"]


# Singleton adapter
_adapter = None

def get_adapter():
    global _adapter
    if _adapter is None:
        _adapter = SkillAdapter()
        if os.path.exists("skill_adapter.npz"):
            _adapter.load("skill_adapter.npz")
    return _adapter
