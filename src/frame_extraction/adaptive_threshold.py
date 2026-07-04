"""
Adaptive Threshold Module.

Computes a robust threshold from a distribution of values using
Median + IQR (Interquartile Range) or Median + MAD.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np


class AdaptiveVideoThreshold:
    """Adaptive threshold using Median + IQR (or Median + MAD)."""

    def __init__(
        self,
        k: float = 1.5,
        method: str = "median_iqr",
    ) -> None:
        """
        Args:
            k:          Multiplier applied to the spread measure.
            method:     "median_iqr" or "median_mad".
        """
        if method not in {"median_iqr", "median_mad"}:
            raise ValueError(f"Unknown method: {method}")

        self.k = k
        self.method = method
        self.threshold: Optional[float] = None
        self.stats: Dict[str, float] = {}

    def fit(self, values: Union[List[float], np.ndarray]) -> float:
        """
        Compute and store the adaptive threshold from the given values.

        Returns:
            The computed threshold value (never negative).
        """
        values = np.asarray(values, dtype=np.float32)

        if values.size == 0:
            raise ValueError("Empty input.")

        median = float(np.median(values))
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1

        self.stats = {"median": median, "q1": q1, "q3": q3, "iqr": iqr}

        if self.method == "median_iqr":
            threshold = median + self.k * iqr
        else:
            mad = float(np.median(np.abs(values - median)))
            self.stats["mad"] = mad
            threshold = median + self.k * 1.4826 * mad

        self.threshold = max(0.0, float(threshold))
        return self.threshold

    def get_threshold(self) -> Optional[float]:
        """Return the last computed threshold, or None if ``fit()`` has not run."""
        return self.threshold
