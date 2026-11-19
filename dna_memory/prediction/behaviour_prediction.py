"""
Behaviour Prediction Engine:
Combines temporal pattern analysis, relational preference strengths, and recursive assemblies
to estimate the next event window, likely product/service, confidence, and supporting evidence.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import datetime
from dna_memory.relations.relation_registry import RelationRegistry
from dna_memory.assembly.recursive_composition import AssemblyEngine
from dna_memory.prediction.temporal_patterns import TemporalPatternAnalyzer

if TYPE_CHECKING:
    from dna_memory.learning.learning_engine import MemoryLearningEngine

class BehaviourPredictionEngine:
    """
    Predicts next customer action, optimal product recommendation, and anticipated purchase window.
    """
    def __init__(
        self,
        learning_engine: Any,
        relation_registry: RelationRegistry,
        assembly_engine: AssemblyEngine
    ):
        self.learning_engine = learning_engine
        self.relation_registry = relation_registry
        self.assembly_engine = assembly_engine

    def predict_next_event(self, customer_id: str) -> Dict[str, Any]:
        """
        Synthesizes historical events and relational strengths into an actionable prediction.
        """
        history = self.learning_engine.get_event_history(subject_id=customer_id)
        if not history:
            return {
                "customer": customer_id,
                "predicted_event": "unknown",
                "predicted_window": "insufficient_data",
                "confidence": 0.0,
                "evidence": ["No historical events recorded for this customer."]
            }

        # 1. Product preference determination via relational strength
        obj_counts = {}
        for ev in history:
            obj = ev["object_id"]
            obj_counts[obj] = obj_counts.get(obj, 0) + 1

        top_product = max(obj_counts.items(), key=lambda x: x[1])[0]

        # 2. Temporal pattern analysis
        timestamps = [ev["timestamp"] for ev in history]
        cadence = TemporalPatternAnalyzer.analyze_cadence(timestamps)

        mean_interval = cadence.get("mean_interval_days")
        regularity = cadence.get("regularity_score", 0.5)

        # 3. Determine window and calibrate confidence
        evidence = [
            f"Observed {len(history)} past interactions/purchases",
            f"Top product preference: '{top_product}' ({obj_counts[top_product]} times)"
        ]

        if mean_interval is not None and len(history) >= 2:
            evidence.append(f"Recurring cadence: mean interval {mean_interval:.1f} days (regularity {regularity:.2f})")
            
            # Predict window based on last timestamp + mean_interval
            last_ts = TemporalPatternAnalyzer.parse_timestamp(history[-1]["timestamp"])
            now = datetime.datetime.now(datetime.timezone.utc)
            days_since_last = (now - last_ts).total_seconds() / 86400.0
            days_until_next = max(0.0, mean_interval - days_since_last)

            if days_until_next <= 1.0:
                pred_window = "next_1_to_2_days"
            elif days_until_next <= 3.0:
                pred_window = "next_1_to_3_days"
            elif days_until_next <= 7.0:
                pred_window = "next_3_to_7_days"
            else:
                pred_window = f"in_approximately_{int(days_until_next)}_days"

            # Confidence scaled by sample size and regularity score
            sample_factor = min(1.0, len(history) / 10.0)
            confidence = round(0.5 * sample_factor + 0.45 * regularity, 2)
        else:
            pred_window = "initial_onboarding_window"
            confidence = 0.50
            evidence.append("Limited historical frequency data available.")

        return {
            "customer": customer_id,
            "predicted_event": f"{top_product}_reorder",
            "predicted_target": top_product,
            "predicted_window": pred_window,
            "estimated_interval_days": mean_interval,
            "confidence": confidence,
            "evidence": evidence
        }
