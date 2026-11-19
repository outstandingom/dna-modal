"""
Working Memory: Manages currently activated representations, dynamic focus, and active reasoning frontier.
"""

from typing import List, Dict, Any, Optional

class WorkingMemory:
    """
    Temporary scratchpad holding representations active in the current interaction turn.
    Automatically resets or decays across conversation turns.
    """
    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self._active_representations: Dict[str, Dict[str, Any]] = {}

    def retain(self, item_id: str, representation_type: str, data: Any, activation_score: float = 1.0):
        """Retains an activated representation in working memory."""
        self._active_representations[item_id] = {
            "id": item_id,
            "type": representation_type,
            "data": data,
            "score": activation_score
        }
        # If over capacity, prune lowest activation scores
        if len(self._active_representations) > self.capacity:
            sorted_items = sorted(self._active_representations.items(), key=lambda x: x[1]["score"])
            excess = len(self._active_representations) - self.capacity
            for i in range(excess):
                del self._active_representations[sorted_items[i][0]]

    def get_context_items(self) -> List[Dict[str, Any]]:
        """Returns sorted list of active context representations."""
        items = list(self._active_representations.values())
        items.sort(key=lambda x: x["score"], reverse=True)
        return items

    def clear(self):
        self._active_representations.clear()
