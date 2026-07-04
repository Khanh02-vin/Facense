"""
Train Preference Model

Train Random Forest Regressor to learn user's preference.

Workflow:
1. Load features + labels
2. Train model
3. Evaluate
4. Save model
5. Explain feature importance
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TrainingResult:
    """Result from training."""
    model: any
    feature_names: List[str]
    feature_importance: Dict[str, float]
    cv_score: float
    train_score: float
    test_score: float
    n_samples: int
    n_features: int
    model_type: str

    def to_dict(self) -> dict:
        return {
            "feature_importance": self.feature_importance,
            "cv_score": float(self.cv_score),
            "train_score": float(self.train_score),
            "test_score": float(self.test_score),
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
        }


class PreferenceModelTrainer:
    """
    Train preference model from features and labels.

    Supports:
    - Random Forest Regressor
    - Gradient Boosting Regressor
    - Ridge Regression
    """

    FEATURE_NAMES = [
        # Layer 1
        'motion_energy',
        'motion_peak',
        'motion_variance',
        'blur',
        'brightness',
        'brightness_std',
        'face_visibility',
        'face_detected',
        # Layer 2
        'smile',
        'mouth_open',
        'eye_contact',
        'pupil_left',
        'pupil_right',
        'head_yaw',
        'head_pitch',
        'head_roll',
        'face_symmetry',
        'face_clarity',
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_importance = {}

    def prepare_data(
        self,
        features_list: List[dict],
        labels: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for training.

        Args:
            features_list: List of feature dicts
            labels: List of ratings (1-5)

        Returns:
            X, y arrays
        """
        X = []
        y = []

        for features, label in zip(features_list, labels):
            # Extract feature vector
            feature_vec = []
            for name in self.FEATURE_NAMES:
                feature_vec.append(features.get(name, 0.0))

            X.append(feature_vec)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        print(f"\n[DATA]")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {len(self.FEATURE_NAMES)}")
        print(f"  Labels: {np.unique(y, return_counts=True)}")

        return X, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = 'random_forest',
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> TrainingResult:
        """
        Train preference model.

        Args:
            X: Feature matrix
            y: Labels (1-5)
            model_type: 'random_forest', 'gradient_boosting', 'ridge'
            test_size: Fraction for test set
            cv_folds: Cross-validation folds

        Returns:
            TrainingResult
        """
        # Normalize
        X_scaled = self.scaler.fit_transform(X)

        # Split
        n_test = max(1, int(len(X) * test_size))
        indices = np.random.permutation(len(X))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"\n[TRAINING]")
        print(f"  Model: {model_type}")
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

        # Train model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        else:  # ridge
            self.model = Ridge(alpha=1.0)

        self.model.fit(X_train, y_train)

        # Scores
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        train_score = r2_score(y_train, train_pred)
        test_score = r2_score(y_test, test_pred)

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_scaled, y,
            cv=cv_folds,
            scoring='r2'
        )
        cv_score = np.mean(cv_scores)

        print(f"\n[RESULTS]")
        print(f"  Train R2: {train_score:.3f}")
        print(f"  Test R2: {test_score:.3f}")
        print(f"  CV R2: {cv_score:.3f} +/- {np.std(cv_scores):.3f}")
        print(f"  Test MAE: {mean_absolute_error(y_test, test_pred):.3f}")

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            importance = np.abs(self.model.coef_)

        self.feature_importance = {
            name: float(imp)
            for name, imp in zip(self.FEATURE_NAMES, importance)
        }

        # Sort by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: -abs(x[1])
        )

        print(f"\n[FEATURE IMPORTANCE]")
        for name, imp in sorted_features[:10]:
            sign = "+" if imp > 0 else "-"
            bar = "█" * int(abs(imp) * 50)
            print(f"  {sign} {name:20s}: {bar} {imp:.3f}")

        return TrainingResult(
            model=self.model,
            feature_names=self.FEATURE_NAMES,
            feature_importance=self.feature_importance,
            cv_score=cv_score,
            train_score=train_score,
            test_score=test_score,
            n_samples=len(X),
            n_features=len(self.FEATURE_NAMES),
            model_type=model_type
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict preference scores."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> Dict:
        """
        Predict with confidence.

        Returns:
            Dict with 'score', 'confidence', 'explanation'
        """
        scores = self.predict(X)

        # Confidence based on model uncertainty
        # For RF, use variance across trees
        if hasattr(self.model, 'estimators_'):
            predictions = np.array([
                est.predict(self.scaler.transform(X))
                for est in self.model.estimators_
            ])
            uncertainty = np.std(predictions, axis=0)
            confidence = 1.0 - np.clip(uncertainty / 2, 0, 1)
        else:
            confidence = np.ones(len(scores))

        # Explanation
        explanations = []
        for i in range(len(scores)):
            top_features = sorted(
                self.feature_importance.items(),
                key=lambda x: -abs(x[1])
            )[:3]

            positive = [f for f, v in top_features if v > 0]
            negative = [f for f, v in top_features if v < 0]

            if positive:
                explanations.append(f"Liked: {', '.join(positive)}")
            if negative:
                explanations.append(f"Disliked: {', '.join(negative)}")

        return {
            'score': scores,
            'confidence': confidence,
            'explanation': explanations
        }

    def save(self, path: str):
        """Save model and scaler."""
        import pickle

        data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'feature_names': self.FEATURE_NAMES
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        print(f"\n[SAVED] Model to: {path}")

    def load(self, path: str):
        """Load model and scaler."""
        import pickle

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_importance = data['feature_importance']
        self.FEATURE_NAMES = data['feature_names']

        print(f"[LOADED] Model from: {path}")


# ============================================================
# SIMULATED TRAINING
# ============================================================

def simulate_training():
    """Simulate training with sample data."""
    print("="*60)
    print("PREFERENCE MODEL TRAINING - SIMULATION")
    print("="*60)

    # Simulate features
    np.random.seed(42)
    n_samples = 100

    features_list = []
    for i in range(n_samples):
        features = {
            'motion_energy': np.random.uniform(0, 30),
            'motion_peak': np.random.uniform(0, 50),
            'motion_variance': np.random.uniform(0, 10),
            'blur': np.random.uniform(100, 5000),
            'brightness': np.random.uniform(50, 200),
            'brightness_std': np.random.uniform(0, 30),
            'face_visibility': np.random.uniform(0, 0.5),
            'face_detected': np.random.choice([0, 1], p=[0.3, 0.7]),
            'smile': np.random.uniform(0, 1),
            'mouth_open': np.random.uniform(0, 0.5),
            'eye_contact': np.random.uniform(0, 1),
            'pupil_left': np.random.uniform(0, 1),
            'pupil_right': np.random.uniform(0, 1),
            'head_yaw': np.random.uniform(-0.5, 0.5),
            'head_pitch': np.random.uniform(-0.3, 0.3),
            'head_roll': np.random.uniform(-0.2, 0.2),
            'face_symmetry': np.random.uniform(0.3, 1.0),
            'face_clarity': np.random.uniform(0.5, 1.0),
        }
        features_list.append(features)

    # Simulate labels (user prefers high smile, eye_contact, clarity)
    labels = []
    for f in features_list:
        score = (
            2.0  # base
            + 1.5 * f['smile']
            + 1.0 * f['eye_contact']
            + 0.5 * f['face_clarity']
            - 0.5 * f['motion_energy'] / 30
            + np.random.normal(0, 0.5)
        )
        score = np.clip(round(score), 1, 5)
        labels.append(int(score))

    # Train
    trainer = PreferenceModelTrainer()
    X, y = trainer.prepare_data(features_list, labels)
    result = trainer.train(X, y, model_type='random_forest')

    # Save
    output_path = './data/preference_model.pkl'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(output_path)

    # Test prediction
    print(f"\n[TEST PREDICTION]")
    test_features = [{
        'motion_energy': 10.0,
        'motion_peak': 15.0,
        'motion_variance': 2.0,
        'blur': 3000.0,
        'brightness': 120.0,
        'brightness_std': 10.0,
        'face_visibility': 0.3,
        'face_detected': 1.0,
        'smile': 0.8,
        'mouth_open': 0.2,
        'eye_contact': 0.9,
        'pupil_left': 0.5,
        'pupil_right': 0.5,
        'head_yaw': 0.1,
        'head_pitch': 0.0,
        'head_roll': 0.0,
        'face_symmetry': 0.8,
        'face_clarity': 0.9,
    }]

    X_test = np.array([[test_features[0].get(name, 0.0) for name in trainer.FEATURE_NAMES]])
    prediction = trainer.predict(X_test)
    print(f"  Score for high-quality clip: {prediction[0]:.2f}/5")

    # Save results
    results_path = './data/training_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] Results to: {results_path}")

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Train Preference Model')
    parser.add_argument('--features', type=str,
                       help='Path to features JSON file')
    parser.add_argument('--labels', type=str,
                       help='Path to labels JSON file')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting', 'ridge'],
                       help='Model type')
    parser.add_argument('--output', type=str, default='./data/preference_model.pkl',
                       help='Output model path')
    parser.add_argument('--simulate', action='store_true',
                       help='Run simulation')

    args = parser.parse_args()

    if args.simulate or (not args.features and not args.labels):
        # Simulation mode
        result = simulate_training()
    else:
        # Real training
        print("="*60)
        print("PREFERENCE MODEL TRAINING")
        print("="*60)

        # Load data
        with open(args.features, 'r') as f:
            features_list = json.load(f)

        with open(args.labels, 'r') as f:
            labels_data = json.load(f)

        labels = [l['rating'] for l in labels_data['labels']]

        # Train
        trainer = PreferenceModelTrainer()
        X, y = trainer.prepare_data(features_list, labels)
        result = trainer.train(X, y, model_type=args.model)

        # Save
        trainer.save(args.output)

        results_path = args.output.replace('.pkl', '_results.json')
        with open(results_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
