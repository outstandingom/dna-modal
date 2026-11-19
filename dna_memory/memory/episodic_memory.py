"""
Episodic Memory: Preserves individual events and interactions as immutable historical records.
Ensures history is never silently rewritten.
"""

from typing import List, Dict, Any, Optional
import datetime

class EpisodicMemory:
    """
    Immutable chronological ledger of all observed events.
    """
    def __init__(self):
        self._episodes: List[Dict[str, Any]] = []

    def record_episode(
        self,
        event_type: str,
        subject: str,
        details: Dict[str, Any],
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        ts = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
        episode = {
            "episode_id": f"ep_{len(self._episodes) + 1:06d}",
            "event_type": event_type,
            "subject": subject,
            "details": details,
            "timestamp": ts
        }
        self._episodes.append(episode)
        return episode

    def get_subject_episodes(self, subject: str) -> List[Dict[str, Any]]:
        return [e for e in self._episodes if e["subject"] == subject]

    def all_episodes(self) -> List[Dict[str, Any]]:
        return list(self._episodes)

    def count(self) -> int:
        return len(self._episodes)
