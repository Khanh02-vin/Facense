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

    Models P(i > j) = sigmoid(w_i - w_j) where w_i is item strength.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tol: float = 1e-6,
        alpha: float = 0.0
    ):
        """
        Args:
            max_iterations: Maximum EM iterations
            tol: Convergence tolerance
            alpha: Regularization strength
        """
        self.max_iterations = max_iterations
        self.tol = tol
        self.alpha = alpha
        self.item_scores = {}
        self.convergence = False
        self.n_iterations = 0

    def fit(self, pairs: list[PairwiseSample]) -> BradleyTerryResult:
        """Fit Bradley-Terry model.

        Args:
            pairs: List of pairwise comparisons

        Returns:
            BradleyTerryResult with item strengths
        """
        if not pairs:
            return BradleyTerryResult(
                item_scores={},
                convergence=True,
                n_iterations=0,
                log_likelihood=0.0
            )

        # Collect all items
        items = set()
        for pair in pairs:
            items.add(pair.image_A)
            items.add(pair.image_B)

        # Initialize scores
        item_scores = {item: 0.0 for item in items}
        log_likelihoods = []

        n = len(items)

        for iteration in range(self.max_iterations):
            # E-step: compute expected counts
            exp_counts = {item: 0.0 for item in items}
            total_counts = {item: 0.0 for item in items}
            ll = 0.0

            for pair in pairs:
                i, j = pair.image_A, pair.image_B
                diff = item_scores[i] - item_scores[j]

                # sigmoid
                prob_i = 1 / (1 + np.exp(-diff))
                prob_i = np.clip(prob_i, 1e-10, 1 - 1e-10)

                # Weighted by winner
                if pair.winner == "A":
                    exp_counts[i] += prob_i
                    exp_counts[j] += (1 - prob_i)
                else:
                    exp_counts[i] += (1 - prob_i)
                    exp_counts[j] += prob_i

                total_counts[i] += 1
                total_counts[j] += 1

                # Log-likelihood
                if pair.winner == "A":
                    ll += np.log(prob_i)
                else:
                    ll += np.log(1 - prob_i)

            log_likelihoods.append(ll)

            # M-step: update scores with regularization
            prev_scores = item_scores.copy()

            for item in items:
                if total_counts[item] > 0:
                    # Bradley-Terry update with regularization toward 0
                    item_scores[item] = exp_counts[item] / total_counts[item]
                    # Add regularization
                    item_scores[item] = (1 - self.alpha) * item_scores[item]

            # Center scores (identifiability constraint)
            mean_score = np.mean(list(item_scores.values()))
            for item in items:
                item_scores[item] -= mean_score

            # Check convergence
            delta = sum(
                abs(item_scores[i] - prev_scores[i])
                for i in items
            )

            if delta < self.tol:
                self.convergence = True
                self.n_iterations = iteration + 1
                break

        self.item_scores = item_scores

        return BradleyTerryResult(
            item_scores=item_scores,
            convergence=self.convergence,
            n_iterations=self.n_iterations,
            log_likelihood=log_likelihoods[-1] if log_likelihoods else 0.0
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
        """Train reward model.

        Args:
            embeddings_A: (n_pairs, embedding_dim) embeddings of image A
            embeddings_B: (n_pairs, embedding_dim) embeddings of image B
            winners: (n_pairs,) 1 if A wins, 0 if B wins
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Fraction for validation

        Returns:
            Training history dict
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        self.model = self._build_model().to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        # Prepare data
        X_diff = embeddings_A - embeddings_B  # Difference feature
        dataset = TensorDataset(
            torch.FloatTensor(X_diff),
            torch.FloatTensor(winners)
        )

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
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                logits = self.model(X_batch).squeeze()
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            history["train_loss"].append(train_loss / len(train_loader))

            # Validation
            self.model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    logits = self.model(X_batch).squeeze()
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds.extend(probs)
                    val_labels.extend(y_batch.numpy())

            try:
                val_auc = roc_auc_score(val_labels, val_preds)
            except:
                val_auc = 0.5
            history["val_auc"].append(val_auc)

        self.is_fitted = True
        self.device = device
        return history

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
