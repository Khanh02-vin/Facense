"""
Preference Module — Pairwise Preference Learning (Phase 3).

Single canonical path:
  PairwiseSample → BradleyTerryModel → preference scores

No neural / MoP variants.  Bradley-Terry is stable, L-BFGS-B–based,
with automatic L2 regularisation when the preference graph is
disconnected or sparse.
"""

import numpy as np
from typing import Literal
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class PairwiseSample:
    """Single pairwise comparison."""
    user_id: str
    image_A: str
    image_B: str
    winner: Literal["A", "B"]
    timestamp: float = 0.0


@dataclass
class BradleyTerryResult:
    """Bradley-Terry model result."""

    item_scores: dict[str, float]
    convergence: bool
    n_iterations: int
    log_likelihood: float
    alpha_used: float = 0.0
    alpha_mode: str = "auto:none"


# ============================================================================
# Helpers (module-level)
# ============================================================================


def _graph_components(
    n_items: int, pair_indices: list[tuple[int, int]]
) -> list[set[int]]:
    """Return the connected components of the preference graph.

    Items that never appear in any pair form singleton components.
    """

    parent = list(range(n_items))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for w, l in pair_indices:
        union(w, l)

    comps: dict[int, set[int]] = {}
    for i in range(n_items):
        comps.setdefault(find(i), set()).add(i)
    return list(comps.values())


def _auto_alpha(
    n_items: int,
    pair_indices: list[tuple[int, int]],
    alpha_user: float,
) -> tuple[float, str]:
    """Pick the L2 regularisation strength.

    * If the caller passed a positive ``alpha_user``, use it as-is.
    * Otherwise, if the preference graph is **disconnected** (more
      than one component), pick a small alpha proportional to the
      inverse edge count so the unconstrained parameters cannot
      drift arbitrarily between components.
    * Otherwise, if the graph is **sparse** (fewer than ``2 * N``
      edges), pick a small alpha proportional to ``1 / N``.
    * Otherwise return ``0.0`` and let the BT NLL drive the fit.

    Returns ``(alpha, mode_label)``.
    """

    if alpha_user > 0.0:
        return float(alpha_user), "user"

    n_edges = len(pair_indices)
    components = _graph_components(n_items, pair_indices)
    if len(components) > 1:
        return 1.0 / max(n_edges, 1), f"auto:disconnected({len(components)})"
    if n_edges < 2 * n_items:
        return 0.5 / max(n_items, 1), "auto:sparse"
    return 0.0, "auto:none"


def _log_likelihood(w: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray) -> float:
    """Compute the **unpenalised** Bradley-Terry log-likelihood from the
    fitted weights.  Recomputing from the data (instead of reading
    ``-result.fun`` from scipy) gives a value that does not depend on
    the optimiser's bookkeeping and is unaffected by ``alpha``."""

    diff = np.clip(w[i_idx] - w[j_idx], -30.0, 30.0)
    prob = 1.0 / (1.0 + np.exp(-diff))
    prob = np.clip(prob, 1e-15, 1.0 - 1e-15)
    return float(np.sum(np.log(prob)))


class BradleyTerryModel:
    """Bradley-Terry model for pairwise preferences.

    Models P(i > j) = σ(w_i - w_j) where w_i is item strength.
    Fitted via maximum-likelihood with scipy.optimize.
    """

    def __init__(
        self,
        max_iterations: int = 500,
        tol: float = 1e-8,
        alpha: float = 0.0,
    ):
        """
        Args:
            max_iterations: Maximum solver iterations.
            tol: Convergence tolerance for solver.
            alpha: L2 regularisation strength (0 = no regularisation).
        """
        self.max_iterations = max_iterations
        self.tol = tol
        self.alpha = alpha
        self.item_scores: dict[str, float] = {}
        self.convergence = False
        self.n_iterations = 0

    def fit(self, pairs: list[PairwiseSample]) -> BradleyTerryResult:
        """Fit Bradley-Terry model via NLL minimisation.

        Args:
            pairs: List of pairwise comparisons.

        Returns:
            BradleyTerryResult with item strengths.
        """
        if not pairs:
            return BradleyTerryResult(
                item_scores={},
                convergence=True,
                n_iterations=0,
                log_likelihood=0.0,
            )

        # Collect all items in encounter order so indices are stable.
        items = sorted({pair.image_A for pair in pairs} | {pair.image_B for pair in pairs})
        n = len(items)
        item_to_idx = {item: i for i, item in enumerate(items)}

        # Build arrays for the NLL evaluator.
        # Convention: for each pair (winner_idx, loser_idx), sigmoid(w_winner - w_loser).
        pair_indices: list[tuple[int, int]] = []
        for pair in pairs:
            i = item_to_idx[pair.image_A]
            j = item_to_idx[pair.image_B]
            if i == j:
                continue
            if pair.winner == "A":
                pair_indices.append((i, j))  # A wins
            elif pair.winner == "B":
                pair_indices.append((j, i))  # B wins (swap so winner is first)
            # Skip "equal" or invalid choices

        if not pair_indices:
            return BradleyTerryResult(
                item_scores={item: 0.0 for item in items},
                convergence=True,
                n_iterations=0,
                log_likelihood=0.0,
            )

        i_idx = np.array([p[0] for p in pair_indices], dtype=np.int32)
        j_idx = np.array([p[1] for p in pair_indices], dtype=np.int32)

        def nll(w: np.ndarray) -> float:
            # P(winner beats loser) = sigmoid(w_winner - w_loser)
            diff = w[i_idx] - w[j_idx]
            diff = np.clip(diff, -30.0, 30.0)
            prob = 1.0 / (1.0 + np.exp(-diff))
            prob = np.clip(prob, 1e-15, 1.0 - 1e-15)
            loss = -np.sum(np.log(prob))
            if self.alpha > 0.0:
                loss += 0.5 * self.alpha * np.sum(w ** 2)
            return loss

        # Initialise at zero (equal strengths).
        w0 = np.zeros(n, dtype=np.float64)

        result = minimize(
            nll,
            w0,
            method="L-BFGS-B",
            options={"maxiter": self.max_iterations, "ftol": self.tol, "gtol": self.tol},
        )

        w_opt = result.x.astype(np.float64)
        # Centre scores (identifiability constraint): sum(w) = 0.
        w_opt = w_opt - np.mean(w_opt)

        self.item_scores = {item: float(w_opt[idx]) for item, idx in item_to_idx.items()}
        self.convergence = result.success
        self.n_iterations = int(result.nit)

        return BradleyTerryResult(
            item_scores=self.item_scores,
            convergence=self.convergence,
            n_iterations=self.n_iterations,
            log_likelihood=float(-result.fun),
        )

    def predict_pair(self, image_A: str, image_B: str) -> float:
        """Predict probability A > B.

        Args:
            image_A: First image ID
            image_B: Second image ID

        Returns:
            P(A > B)
        """
        if image_A not in self.item_scores or image_B not in self.item_scores:
            return 0.5

        diff = self.item_scores[image_A] - self.item_scores[image_B]
        return 1 / (1 + np.exp(-diff))

    def rank(self, item_ids: list[str]) -> list[tuple[str, float]]:
        """Rank items by strength.

        Returns:
            List of (item_id, score) sorted by score descending
        """
        scores = [(item, self.item_scores.get(item, 0.0)) for item in item_ids]
        return sorted(scores, key=lambda x: x[1], reverse=True)




def load_pairwise_data(csv_path: str) -> list[PairwiseSample]:
    """Load pairwise data from CSV.

    Expected columns: user_id, image_A, image_B, winner, timestamp
    """
    import csv

    pairs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(PairwiseSample(
                user_id=row['user_id'],
                image_A=row['image_A'],
                image_B=row['image_B'],
                winner=row['winner'],
                timestamp=float(row.get('timestamp', 0))
            ))

    return pairs
