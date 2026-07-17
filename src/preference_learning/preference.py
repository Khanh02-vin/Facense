"""
Preference Module - Pairwise Preference Learning

Models:
- Bradley-Terry Model
- Neural Reward Model
- Mixture of Prototypes
"""

import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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


class NeuralRewardModel:
    """Neural reward model for pairwise preferences."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        lr: float = 0.001,
        dropout: float = 0.1
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.dropout = dropout
        self.model = None
        self.is_fitted = False

    def _build_model(self):
        """Build PyTorch model."""
        try:
            import torch
            import torch.nn as nn

            class RewardNet(nn.Module):
                def __init__(self, embedding_dim, hidden_dim, dropout):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(embedding_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, 1)
                    )

                def forward(self, x):
                    return self.net(x)

            return RewardNet(self.embedding_dim, self.hidden_dim, self.dropout)
        except ImportError:
            raise ImportError("PyTorch required: pip install torch")

    def fit(
        self,
        embeddings_A: np.ndarray,
        embeddings_B: np.ndarray,
        winners: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> dict:
        """Train reward model with utility-pairwise loss.

        The network f(·) is trained so that:
            sigmoid(f(emb_A) - f(emb_B)) ≈ P(A beats B)

        This keeps predict(embedding) and predict_pair() consistent.

        Args:
            embeddings_A: (n_pairs, embedding_dim) embeddings of image A.
            embeddings_B: (n_pairs, embedding_dim) embeddings of image B.
            winners: (n_pairs,) 1 if A wins, 0 if B wins.
            epochs: Training epochs.
            batch_size: Batch size.
            validation_split: Fraction for validation.

        Returns:
            Training history dict.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model — maps single embedding → scalar score
        self.model = self._build_model().to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Prepare data: input is (emb_A, emb_B), label is 1 if A > B else 0
        X_A = torch.FloatTensor(embeddings_A)
        X_B = torch.FloatTensor(embeddings_B)
        y = torch.FloatTensor(winners)

        dataset = TensorDataset(X_A, X_B, y)

        n_train = int(len(dataset) * (1 - validation_split))
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train, len(dataset) - n_train]
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        history = {"train_loss": [], "val_auc": []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for emb_A, emb_B, label in train_loader:
                emb_A, emb_B, label = emb_A.to(device), emb_B.to(device), label.to(device)

                optimizer.zero_grad()
                score_A = self.model(emb_A).squeeze()
                score_B = self.model(emb_B).squeeze()
                # sigmoid(score_A - score_B) = P(A > B)
                logits = score_A - score_B
                loss = nn.BCEWithLogitsLoss()(logits, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            history["train_loss"].append(train_loss / len(train_loader))

            # Validation
            self.model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for emb_A, emb_B, label in val_loader:
                    emb_A, emb_B = emb_A.to(device), emb_B.to(device)
                    score_A = self.model(emb_A).squeeze()
                    score_B = self.model(emb_B).squeeze()
                    probs = torch.sigmoid(score_A - score_B).cpu().numpy()
                    probs = np.atleast_1d(probs)
                    val_preds.extend(probs.tolist())
                    val_labels.extend(label.numpy().tolist())

            try:
                val_auc = roc_auc_score(val_labels, val_preds)
            except Exception:
                val_auc = 0.5
            history["val_auc"].append(val_auc)

        self.is_fitted = True
        self.device = device
        return history

    def predict_pair(self, emb_A: np.ndarray, emb_B: np.ndarray) -> float:
        """Predict P(A beats B) from a pair of embeddings.

        Args:
            emb_A: embedding of image A.
            emb_B: embedding of image B.

        Returns:
            Probability in [0, 1].
        """
        if not self.is_fitted:
            return 0.5
        import torch
        self.model.eval()
        with torch.no_grad():
            ta = torch.FloatTensor(emb_A).unsqueeze(0).to(self.device)
            tb = torch.FloatTensor(emb_B).unsqueeze(0).to(self.device)
            diff = self.model(ta) - self.model(tb)
            return float(torch.sigmoid(diff).item())

    def predict(self, embedding: np.ndarray) -> float:
        """Predict reward for single embedding.

        Args:
            embedding: (embedding_dim,) image embedding

        Returns:
            Reward score
        """
        if not self.is_fitted:
            return 0.5

        import torch

        self.model.eval()
        with torch.no_grad():
            emb_t = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
            reward = self.model(emb_t).item()

        return float(reward)

    def predict_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict rewards for batch of embeddings."""
        if not self.is_fitted:
            return np.full(len(embeddings), 0.5)

        import torch
        from torch.utils.data import TensorDataset, DataLoader

        self.model.eval()
        dataset = TensorDataset(torch.FloatTensor(embeddings))
        loader = DataLoader(dataset, batch_size=32)

        rewards = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                r = self.model(batch).squeeze(-1).cpu().numpy()
                rewards.extend(r if r.ndim > 0 else [r])

        return np.array(rewards)


class MixtureOfPrototypes:
    """Preference as mixture of prototypes (multi-modal)."""

    def __init__(
        self,
        n_prototypes: int = 5,
        temperature: float = 0.1,
        max_iterations: int = 100
    ):
        self.n_prototypes = n_prototypes
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.prototypes = None
        self.weights = None

    def fit(
        self,
        positive_embeddings: np.ndarray,
        negative_embeddings: np.ndarray = None
    ) -> dict:
        """Fit mixture model.

        Args:
            positive_embeddings: (n_pos, dim) embeddings of liked images
            negative_embeddings: Optional (n_neg, dim) embeddings of disliked images

        Returns:
            Fit results dict
        """
        n_samples, dim = positive_embeddings.shape

        # Initialize prototypes from positive embeddings
        if n_samples >= self.n_prototypes:
            # K-means initialization
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_prototypes, random_state=42, n_init=10)
            kmeans.fit(positive_embeddings)
            self.prototypes = kmeans.cluster_centers_
            self.weights = np.ones(self.n_prototypes) / self.n_prototypes
        else:
            # Random initialization
            idx = np.random.choice(n_samples, self.n_prototypes, replace=True)
            self.prototypes = positive_embeddings[idx]
            self.weights = np.ones(self.n_prototypes) / self.n_prototypes

        # EM-like optimization
        for iteration in range(self.max_iterations):
            prev_prototypes = self.prototypes.copy()

            # E-step: compute responsibilities
            responsibilities = self._compute_responsibilities(positive_embeddings)

            # M-step: update prototypes and weights
            for k in range(self.n_prototypes):
                resp_k = responsibilities[:, k]
                total_resp = np.sum(resp_k) + 1e-10

                # Update prototype as weighted mean
                self.prototypes[k] = np.sum(
                    positive_embeddings * resp_k[:, np.newaxis],
                    axis=0
                ) / total_resp

                # Update weight
                self.weights[k] = total_resp / n_samples

            # Normalize prototypes
            self.prototypes = self.prototypes / (
                np.linalg.norm(self.prototypes, axis=1, keepdims=True) + 1e-10
            )

            # Check convergence
            delta = np.max(np.abs(self.prototypes - prev_prototypes))
            if delta < 1e-4:
                break

        return {
            "n_iterations": iteration + 1,
            "converged": delta < 1e-4,
            "prototype_weights": self.weights.tolist(),
            "n_prototypes": self.n_prototypes
        }

    def _compute_responsibilities(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute soft clustering responsibilities."""
        n_samples = len(embeddings)
        responsibilities = np.zeros((n_samples, self.n_prototypes))

        for k in range(self.n_prototypes):
            sim = np.dot(embeddings, self.prototypes[k])
            responsibilities[:, k] = np.exp(sim / self.temperature)

        # Normalize
        responsibilities = responsibilities / (
            responsibilities.sum(axis=1, keepdims=True) + 1e-10
        )

        return responsibilities

    def score(self, embedding: np.ndarray) -> float:
        """Score embedding against mixture.

        Returns:
            Log-likelihood under mixture model
        """
        if self.prototypes is None:
            return 0.0

        # Compute weighted sum of similarities
        similarities = np.array([
            np.dot(embedding, proto) for proto in self.prototypes
        ])

        weighted_sim = np.sum(
            self.weights * np.exp(similarities / self.temperature)
        )

        return float(np.log(weighted_sim + 1e-10))

    def score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Score batch of embeddings."""
        return np.array([self.score(emb) for emb in embeddings])


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
