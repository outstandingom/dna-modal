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
        """
        Strictly parses ISO-8601 timestamps and standard datetime strings.
        Raises ValueError on malformed or invalid timestamps without silent fallback.
        Preserves timezone offset and normalizes to UTC.
        """
        if not ts_str or not isinstance(ts_str, str):
            raise ValueError(f"Invalid timestamp value: {ts_str!r}. Expected non-empty string.")

        s = ts_str.strip()
        try:
            clean_s = s.replace("Z", "+00:00") if s.endswith("Z") else s
            dt = datetime.datetime.fromisoformat(clean_s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            else:
                dt = dt.astimezone(datetime.timezone.utc)
            return dt
        except Exception:
            pass

        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S.%f",
            "%d-%m-%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S"
        ):
            try:
                dt = datetime.datetime.strptime(s, fmt)
                return dt.replace(tzinfo=datetime.timezone.utc)
            except Exception:
                continue

        raise ValueError(
            f"Invalid timestamp format: '{ts_str}'. Expected valid ISO-8601 (e.g. '2026-09-01T10:00:00Z' or 'YYYY-MM-DD HH:MM:SS')."
        )

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
