"""
Temporal Pattern Engine: Analyzes recurring event intervals, calculates periodicity, variance, and drift.
"""

from typing import List, Dict, Any, Optional
import datetime
import numpy as np

class TemporalPatternAnalyzer:
    """
    Computes recurrence statistics across event timestamps.
    """
    @staticmethod
    def parse_timestamp(ts_str: str) -> datetime.datetime:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                dt = datetime.datetime.strptime(ts_str.split("+")[0].strip(), fmt)
                return dt.replace(tzinfo=datetime.timezone.utc)
            except Exception:
                continue
        return datetime.datetime.now(datetime.timezone.utc)

    @classmethod
    def calculate_intervals_days(cls, timestamps: List[str]) -> List[float]:
        if len(timestamps) < 2:
            return []

        dts = sorted([cls.parse_timestamp(t) for t in timestamps])
        intervals = []
        for i in range(1, len(dts)):
            diff_days = (dts[i] - dts[i-1]).total_seconds() / 86400.0
            intervals.append(max(0.1, diff_days))
        return intervals

    @classmethod
    def analyze_cadence(cls, timestamps: List[str]) -> Dict[str, Any]:
        intervals = cls.calculate_intervals_days(timestamps)
        if not intervals:
            return {
                "sample_count": len(timestamps),
                "mean_interval_days": None,
                "std_interval_days": None,
                "regularity_score": 0.0
            }

        mean_val = float(np.mean(intervals))
        std_val = float(np.std(intervals)) if len(intervals) > 1 else 0.0
        # Regularity score: 1.0 = perfectly periodic, 0.0 = high variance
        cv = std_val / (mean_val + 1e-6)
        regularity = float(max(0.0, min(1.0, 1.0 / (1.0 + cv))))

        return {
            "sample_count": len(timestamps),
            "mean_interval_days": round(mean_val, 2),
            "std_interval_days": round(std_val, 2),
            "regularity_score": round(regularity, 3),
            "intervals": [round(x, 2) for x in intervals]
        }
