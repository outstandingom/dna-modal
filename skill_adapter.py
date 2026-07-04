import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import os
class SkillAdapter(nn.Module):
    """
    A tiny neural network that learns to transform the user input + DNA context
    into a 'skill embedding' that improves the LLM's response.
    We'll use a simple MLP with 2 hidden layers.
    """
    def __init__(self, input_dim=24, hidden_dim=24, output_dim=12):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Initialize weights small
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.zeros_(layer.bias)

    def forward(self, user_embedding: np.ndarray, memory_context: np.ndarray) -> np.ndarray:
        """
        user_embedding: 128-dim vector from user input (e.g., average of DNA strand)
        memory_context: 128-dim vector from DNA memory (e.g., centroid of relevant concepts)
        Returns: skill vector (128-dim) that will be injected into the prompt.
        """
        # Convert to torch tensors
        x = torch.tensor(np.concatenate([user_embedding, memory_context]), dtype=torch.float32).unsqueeze(0)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        skill = self.tanh(self.fc3(x))  # Range [-1, 1]
        return skill.detach().numpy().squeeze()  # for external use

    def update_from_reward(self, user_embedding: np.ndarray, memory_context: np.ndarray, reward: float, lr: float = 0.001):
        """
        Use the reward (score from DNA Judge) to update the adapter via a simple
        policy gradient-like rule. We treat the skill vector as an action and
        reinforce it if reward > 0.5.
        """
        # We'll do a quick gradient ascent on the output skill to maximize reward.
        # For simplicity, we use a Hebbian-like update: if reward is high, strengthen
        # the weights that produced this skill.
        with torch.enable_grad():
            x = torch.tensor(np.concatenate([user_embedding, memory_context]), dtype=torch.float32).requires_grad_(True)
            # Forward pass
            h1 = self.relu(self.fc1(x))
            h2 = self.relu(self.fc2(h1))
            skill = self.tanh(self.fc3(h2))
            # We want to increase skill magnitude if reward > 0.5, decrease otherwise.
            # Use the reward as a scalar multiplier for the gradient.
            loss = -reward * torch.norm(skill)  # maximize norm (or we could use a target)
            loss.backward()
            # Apply gradient to parameters with learning rate
            with torch.no_grad():
                for param in self.parameters():
                    param += lr * param.grad
                # zero gradients for next call
                self.zero_grad()
        return skill.detach().numpy().squeeze()

# Singleton adapter, to be loaded/saved
_adapter = None
def get_adapter():
    global _adapter
    if _adapter is None:
        _adapter = SkillAdapter()
        # load saved state if exists
        if os.path.exists("skill_adapter.pth"):
            _adapter.load_state_dict(torch.load("skill_adapter.pth"))
    return _adapter
