"""
Cycle Diagnostics Module - Detect Non-Transitive Preferences

Tests for intransitivity (Condorcet cycles) in pairwise preferences.
"""

import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass
from itertools import combinations


@dataclass
class CycleDiagnostics:
    """Results of cycle diagnostic tests."""
    cycle_rate: float
    kendall_w: float
    transitivity_score: float
    n_cycles_detected: int
    n_triplets_tested: int
    confidence_interval: tuple[float, float]


class CycleDetector:
    """Detect cycles in pairwise preferences."""

    def __init__(self, confidence_level: float = 0.95):
        """
        Args:
            confidence_level: Confidence level for bootstrap CI
        """
        self.confidence_level = confidence_level

    def detect_cycles(
        self,
        pairwise_matrix: np.ndarray,
        item_ids: list[str],
        n_samples: int = 1000
    ) -> CycleDiagnostics:
        """Detect cycles in pairwise preference matrix.

        Args:
            pairwise_matrix: (n_items, n_items) matrix where [i,j] = P(i > j)
            item_ids: List of item ID strings
            n_samples: Number of triplets to sample

        Returns:
            CycleDiagnostics
        """
        n_items = len(item_ids)
        cycles = 0
        total_triplets = 0

        # Sample triplets
        triplet_indices = list(combinations(range(n_items), 3))

        if len(triplet_indices) > n_samples:
            rng = np.random.default_rng()
            sample_idx = rng.choice(
                len(triplet_indices), n_samples, replace=False
            )
            triplet_indices = [triplet_indices[i] for i in sample_idx]

        for i, j, k in triplet_indices:
            total_triplets += 1

            # Check for cycle: i > j, j > k, k > i
            prob_i_j = pairwise_matrix[i, j]
            prob_j_k = pairwise_matrix[j, k]
            prob_k_i = pairwise_matrix[k, i]

            # Cycle exists if all three comparisons favor different items
            if (prob_i_j > 0.5 and prob_j_k > 0.5 and prob_k_i > 0.5):
                cycles += 1
            # Anti-cycle: j > i, k > j, i > k
            elif (prob_i_j < 0.5 and prob_j_k < 0.5 and prob_k_i < 0.5):
                cycles += 1

        cycle_rate = cycles / total_triplets if total_triplets > 0 else 0.0

        return CycleDiagnostics(
            cycle_rate=cycle_rate,
            kendall_w=0.0,  # Computed separately
            transitivity_score=1.0 - cycle_rate,
            n_cycles_detected=cycles,
            n_triplets_tested=total_triplets,
            confidence_interval=(0.0, 0.0)  # Computed with bootstrap
        )

    def bootstrap_cycle_rate(
        self,
        pairwise_matrix: np.ndarray,
        n_items: int,
        n_bootstrap: int = 100,
        n_samples: int = 500
    ) -> tuple[float, float, float]:
        """Bootstrap confidence interval for cycle rate.

        Returns:
            (mean_cycle_rate, ci_lower, ci_upper)
        """
        rng = np.random.default_rng()
        cycle_rates = []

        for _ in range(n_bootstrap):
            # Resample items with replacement
            sampled_items = rng.choice(n_items, min(n_items, 10), replace=False)
            sampled_matrix = pairwise_matrix[np.ix_(sampled_items, sampled_items)]

            # Compute cycle rate for subsample
            triplets = list(combinations(range(len(sampled_items)), 3))
            if len(triplets) > n_samples:
                sample_idx = rng.choice(len(triplets), n_samples, replace=False)
                triplets = [triplets[i] for i in sample_idx]

            cycles = 0
            for i, j, k in triplets:
                prob_i_j = sampled_matrix[i, j]
                prob_j_k = sampled_matrix[j, k]
                prob_k_i = sampled_matrix[k, i]

                if (prob_i_j > 0.5 and prob_j_k > 0.5 and prob_k_i > 0.5):
                    cycles += 1
                elif (prob_i_j < 0.5 and prob_j_k < 0.5 and prob_k_i < 0.5):
                    cycles += 1

            cycle_rates.append(cycles / len(triplets) if triplets else 0)

        mean_rate = np.mean(cycle_rates)
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(cycle_rates, alpha / 2 * 100)
        ci_upper = np.percentile(cycle_rates, (1 - alpha / 2) * 100)

        return mean_rate, ci_lower, ci_upper


class TransitivityAnalyzer:
    """Analyze transitivity of pairwise preferences."""

    def __init__(self):
        pass

    def kendall_w(
        self,
        rankings: list[list[int]]
    ) -> float:
        """Compute Kendall's W (coefficient of concordance).

        Args:
            rankings: List of rankings, each ranking is a list of item indices in rank order

        Returns:
            Kendall's W in [0, 1]
        """
        if not rankings:
            return 0.0

        n_items = len(rankings[0])
        n_rankers = len(rankings)

        # Convert to rank matrix (1 = best, n_items = worst)
        rank_matrix = np.zeros((n_rankers, n_items))
        for r, ranking in enumerate(rankings):
            for rank, item in enumerate(ranking):
                rank_matrix[r, item] = rank + 1

        # Compute sum of ranks per item
        rank_sums = np.sum(rank_matrix, axis=0)

        # Mean of rank sums
        mean_rank_sum = np.mean(rank_sums)

        # Sum of squared deviations
        S = np.sum((rank_sums - mean_rank_sum) ** 2)

        # Kendall's W formula
        W = 12 * S / (n_rankers ** 2 * (n_items ** 3 - n_items))

        return float(W)

    def copeland_score(
        self,
        pairwise_matrix: np.ndarray,
        item_ids: list[str]
    ) -> list[tuple[str, int]]:
        """Compute Copeland scores (pairwise win count).

        Args:
            pairwise_matrix: (n_items, n_items) matrix where [i,j] = P(i > j)
            item_ids: List of item IDs

        Returns:
            List of (item_id, copeland_score) sorted by score
        """
        n_items = len(item_ids)
        scores = []

        for i in range(n_items):
            wins = 0
            for j in range(n_items):
                if i != j:
                    if pairwise_matrix[i, j] > 0.5:
                        wins += 1
                    elif pairwise_matrix[i, j] == 0.5:
                        wins += 0.5
            scores.append((item_ids[i], int(wins)))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    def transitivity_violation_rate(
        self,
        pairwise_matrix: np.ndarray,
        n_samples: int = 1000
    ) -> float:
        """Compute rate of transitive violations.

        Returns:
            Fraction of triplets with transitive violations
        """
        n_items = pairwise_matrix.shape[0]
        violations = 0
        total = 0

        triplet_indices = list(combinations(range(n_items), 3))

        if len(triplet_indices) > n_samples:
            rng = np.random.default_rng()
            sample_idx = rng.choice(len(triplet_indices), n_samples, replace=False)
            triplet_indices = [triplet_indices[i] for i in sample_idx]

        for i, j, k in triplet_indices:
            total += 1

            # True if i > j
            i_beats_j = pairwise_matrix[i, j] > 0.5
            j_beats_k = pairwise_matrix[j, k] > 0.5

            # If i > j and j > k, violation if k > i
            if i_beats_j and j_beats_k:
                if pairwise_matrix[k, i] > 0.5:
                    violations += 1

            # Also check reverse
            j_beats_i = pairwise_matrix[j, i] > 0.5
            k_beats_j = pairwise_matrix[k, j] > 0.5

            if j_beats_i and k_beats_j:
                if pairwise_matrix[i, k] > 0.5:
                    violations += 1

        return violations / total if total > 0 else 0.0


def interpret_cycle_results(diagnostics: CycleDiagnostics) -> str:
    """Generate interpretation of cycle diagnostics."""
    interpretation = []
    interpretation.append("Cycle Diagnostics Results")
    interpretation.append("=" * 50)
    interpretation.append(f"Cycles detected: {diagnostics.n_cycles_detected}")
    interpretation.append(f"Triplets tested: {diagnostics.n_triplets_tested}")
    interpretation.append(f"Cycle rate: {diagnostics.cycle_rate:.2%}")
    interpretation.append(f"Transitivity score: {diagnostics.transitivity_score:.2%}")

    if diagnostics.cycle_rate > 0.10:
        interpretation.append("\n[!] HIGH CYCLE RATE: Transitive models may be inappropriate")
        interpretation.append("Consider:")
        interpretation.append("- Mixture of prototypes models")
        interpretation.append("- Context-dependent utilities")
        interpretation.append("- Pairwise probability matrix factorization")
    elif diagnostics.cycle_rate > 0.05:
        interpretation.append("\n[~] MODERATE CYCLE RATE: Some transitivity violations")
        interpretation.append("Consider reporting both transitive and non-transitive models")
    else:
        interpretation.append("\n[OK] LOW CYCLE RATE: Transitive models appropriate")

    return "\n".join(interpretation)


def build_pairwise_matrix(
    pairs: list[tuple[str, str, str]],  # (item_A, item_B, winner)
    item_ids: list[str]
) -> np.ndarray:
    """Build pairwise probability matrix from comparison list.

    Args:
        pairs: List of (item_A, item_B, winner) tuples
        item_ids: List of all item IDs

    Returns:
        (n_items, n_items) probability matrix
    """
    n_items = len(item_ids)
    item_to_idx = {item: i for i, item in enumerate(item_ids)}

    # Count wins
    win_counts = np.zeros((n_items, n_items))
    total_counts = np.zeros((n_items, n_items))

    for item_a, item_b, winner in pairs:
        if item_a not in item_to_idx or item_b not in item_to_idx:
            continue

        i, j = item_to_idx[item_a], item_to_idx[item_b]

        if winner == "A":
            win_counts[i, j] += 1
        else:
            win_counts[j, i] += 1

        total_counts[i, j] += 1
        total_counts[j, i] += 1

    # Compute probabilities
    matrix = np.zeros((n_items, n_items))
    for i in range(n_items):
        for j in range(n_items):
            if i != j:
                total = total_counts[i, j] + total_counts[j, i]
                if total > 0:
                    matrix[i, j] = win_counts[i, j] / total

    return matrix
