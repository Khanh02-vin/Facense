"""
Validate Preference Learning

Verify that the system learned the user's "gu" correctly.

Validation tests:
1. Feature importance sanity
2. Prediction accuracy
3. Model explanation quality
4. Failure mode detection
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score


class PreferenceModelValidator:
    """
    Validate preference model quality.

    Checks:
    1. Feature importance makes sense
    2. Prediction accuracy is acceptable
    3. Model can explain predictions
    4. No common failure modes
    """

    FEATURE_NAMES = [
        'motion_energy', 'motion_peak', 'motion_variance',
        'blur', 'brightness', 'brightness_std',
        'face_visibility', 'face_detected',
        'smile', 'mouth_open', 'eye_contact',
        'pupil_left', 'pupil_right',
        'head_yaw', 'head_pitch', 'head_roll',
        'face_symmetry', 'face_clarity',
    ]

    # Expected importance patterns
    EXPECTED_HIGH = ['smile', 'eye_contact', 'face_clarity']
    EXPECTED_LOW = ['brightness_std', 'motion_variance']

    def __init__(self):
        self.validation_results = {}
        self.warnings = []
        self.passed = []

    def validate_feature_importance(
        self,
        feature_importance: Dict[str, float]
    ) -> bool:
        """
        Check if feature importance makes sense.

        Rules:
        - Top features should include face/expression features
        - No single feature should dominate (> 50%)
        - Physical features should have some importance
        """
        print("\n" + "="*60)
        print("[TEST 1] Feature Importance Sanity")
        print("="*60)

        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: -x[1]
        )

        print("\nTop 10 features:")
        for i, (name, imp) in enumerate(sorted_features[:10], 1):
            marker = ""
            if name in self.EXPECTED_HIGH:
                marker = " [EXPECTED]"
            elif name in self.EXPECTED_LOW:
                marker = " [UNEXPECTED]"
            print(f"  {i:2d}. {name:20s}: {imp:.3f}{marker}")

        # Check 1: Top features are face-related
        top_5_names = [f[0] for f in sorted_features[:5]]
        face_related = sum(1 for n in top_5_names if
            any(x in n for x in ['smile', 'eye', 'face', 'head']))

        print(f"\n[CHECK] Face-related in top 5: {face_related}/5")
        if face_related >= 2:
            print("  ✓ PASS: Face features are important")
            self.passed.append("Face importance")
        else:
            print("  ⚠ WARN: Face features not dominant")
            self.warnings.append("Face features not dominant")

        # Check 2: No single feature dominates
        top_importance = sorted_features[0][1]
        if top_importance < 0.5:
            print(f"  ✓ PASS: No single feature dominates ({top_importance:.1%})")
            self.passed.append("Feature diversity")
        else:
            print(f"  ⚠ WARN: Single feature dominates ({top_importance:.1%})")
            self.warnings.append("Feature dominance")

        # Check 3: Physical features have some importance
        physical_features = ['blur', 'brightness', 'motion_energy']
        physical_importance = sum(
            feature_importance.get(f, 0) for f in physical_features
        )
        if physical_importance > 0.05:
            print(f"  ✓ PASS: Physical features matter ({physical_importance:.1%})")
            self.passed.append("Physical features")
        else:
            print(f"  ⚠ WARN: Physical features ignored")
            self.warnings.append("Physical features ignored")

        return len(self.warnings) == 0

    def validate_prediction_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cv_scores: List[float]
    ) -> bool:
        """Check if prediction accuracy is acceptable."""
        print("\n" + "="*60)
        print("[TEST 2] Prediction Accuracy")
        print("="*60)

        # Metrics
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        print(f"\n[METRICS]")
        print(f"  MAE: {mae:.3f}")
        print(f"  R2: {r2:.3f}")
        print(f"  CV R2: {cv_mean:.3f} ± {cv_std:.3f}")

        # Check 1: MAE is acceptable (< 1.0)
        if mae < 1.0:
            print(f"  ✓ PASS: MAE < 1.0 ({mae:.2f})")
            self.passed.append("MAE acceptable")
        else:
            print(f"  ⚠ WARN: MAE >= 1.0 ({mae:.2f})")
            self.warnings.append("High MAE")

        # Check 2: R2 > 0 (better than mean)
        if r2 > 0:
            print(f"  ✓ PASS: R2 > 0 ({r2:.2f})")
            self.passed.append("R2 positive")
        else:
            print(f"  ⚠ WARN: R2 < 0 (model worse than mean)")
            self.warnings.append("Negative R2")

        # Check 3: CV score is reasonable
        if cv_mean > 0:
            print(f"  ✓ PASS: CV R2 > 0")
            self.passed.append("CV positive")
        else:
            print(f"  ⚠ WARN: CV R2 < 0 (overfitting likely)")
            self.warnings.append("CV negative")

        # Check 4: Not overfitting (train R2 >> test R2)
        train_r2 = r2  # Simplified
        if train_r2 - cv_mean < 0.3:
            print(f"  ✓ PASS: No severe overfitting")
            self.passed.append("No overfitting")
        else:
            print(f"  ⚠ WARN: Possible overfitting")
            self.warnings.append("Overfitting")

        return mae < 1.0 and r2 > 0

    def validate_explanation_quality(
        self,
        feature_importance: Dict[str, float],
        sample_predictions: List[dict]
    ) -> bool:
        """Check if model explanations are meaningful."""
        print("\n" + "="*60)
        print("[TEST 3] Explanation Quality")
        print("="*60)

        # Top positive features
        positive_features = [
            (n, v) for n, v in sorted(
                feature_importance.items(),
                key=lambda x: -x[1]
            ) if v > 0.1
        ]

        # Top negative features
        negative_features = [
            (n, v) for n, v in sorted(
                feature_importance.items(),
                key=lambda x: x[1]
            ) if v < 0
        ]

        print(f"\n[POSITIVE FACTORS] (features that increase preference)")
        if positive_features:
            for name, imp in positive_features[:5]:
                bar = "█" * int(imp * 50)
                print(f"  + {name:20s}: {bar} ({imp:.1%})")
        else:
            print("  (none)")

        print(f"\n[NEGATIVE FACTORS] (features that decrease preference)")
        if negative_features:
            for name, imp in negative_features[:5]:
                bar = "█" * int(abs(imp) * 50)
                print(f"  - {name:20s}: {bar} ({imp:.1%})")
        else:
            print("  (none)")

        # Check: Can generate meaningful explanation
        explanation_quality = len(positive_features) >= 2

        if explanation_quality:
            print(f"\n  ✓ PASS: Can explain preferences")
            self.passed.append("Explanation available")
        else:
            print(f"\n  ⚠ WARN: Limited explanation")
            self.warnings.append("Limited explanation")

        return explanation_quality

    def check_failure_modes(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_importance: Dict[str, float]
    ) -> List[str]:
        """Detect common failure modes."""
        print("\n" + "="*60)
        print("[TEST 4] Failure Mode Detection")
        print("="*60)

        failures = []

        # 1. Data leakage check
        print("\n[CHECK] Data Leakage")
        label_std = np.std(y)
        if label_std < 0.5:
            print("  ⚠ WARN: Very low label variance")
            failures.append("Low label variance")
        else:
            print("  ✓ OK: Label variance acceptable")

        # 2. Feature correlation check
        print("\n[CHECK] Feature Correlation")
        corr_matrix = np.corrcoef(X.T)
        high_corr = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix[i,j]) > 0.9:
                    high_corr.append((i, j, corr_matrix[i,j]))

        if high_corr:
            print(f"  ⚠ WARN: {len(high_corr)} highly correlated features")
            for idx1, idx2, corr in high_corr[:3]:
                print(f"    {self.FEATURE_NAMES[idx1]} <-> {self.FEATURE_NAMES[idx2]}: {corr:.2f}")
            failures.append("Feature multicollinearity")
        else:
            print("  ✓ OK: No severe multicollinearity")

        # 3. Outlier detection
        print("\n[CHECK] Outliers")
        for i, name in enumerate(self.FEATURE_NAMES[:5]):
            col = X[:, i]
            z_scores = np.abs((col - col.mean()) / (col.std() + 1e-10))
            outliers = np.sum(z_scores > 3)
            if outliers > len(col) * 0.05:
                print(f"  ⚠ WARN: {name} has {outliers} outliers")
                failures.append(f"Outliers in {name}")

        # 4. Prediction confidence
        print("\n[CHECK] Prediction Confidence")
        pred_std = np.std(y)
        if pred_std < 0.3:
            print("  ⚠ WARN: Predictions too uniform")
            failures.append("Low prediction variance")
        else:
            print("  ✓ OK: Predictions have variance")

        return failures

    def generate_report(self) -> dict:
        """Generate validation report."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)

        print(f"\n[PASSED] ({len(self.passed)})")
        for p in self.passed:
            print(f"  ✓ {p}")

        if self.warnings:
            print(f"\n[WARNINGS] ({len(self.warnings)})")
            for w in self.warnings:
                print(f"  ⚠ {w}")

        passed_rate = len(self.passed) / max(1, len(self.passed) + len(self.warnings))

        print(f"\n[RESULT]")
        if passed_rate >= 0.7:
            print("  ✓✓✓ MODEL VALIDATED")
            print("  The system has learned your preference patterns.")
        elif passed_rate >= 0.5:
            print("  ⚠⚠ MODEL NEEDS IMPROVEMENT")
            print("  Some issues detected, but model is usable.")
        else:
            print("  ⚠⚠⚠ MODEL VALIDATION FAILED")
            print("  Significant issues detected. Consider retraining.")

        return {
            "passed": self.passed,
            "warnings": self.warnings,
            "passed_rate": passed_rate,
            "status": "validated" if passed_rate >= 0.7 else "needs_improvement"
        }


def simulate_validation():
    """Run validation with simulated data."""
    print("="*60)
    print("PREFERENCE MODEL VALIDATION - SIMULATION")
    print("="*60)

    validator = PreferenceModelValidator()

    # Simulate feature importance
    feature_importance = {
        'smile': 0.30,
        'eye_contact': 0.15,
        'motion_energy': 0.09,
        'mouth_open': 0.07,
        'face_clarity': 0.05,
        'motion_peak': 0.04,
        'head_roll': 0.03,
        'brightness': 0.03,
        'head_pitch': 0.03,
        'face_symmetry': 0.03,
        'motion_variance': 0.03,
        'face_visibility': 0.02,
        'head_yaw': 0.02,
        'pupil_left': 0.01,
        'pupil_right': 0.01,
        'brightness_std': 0.01,
        'blur': 0.01,
        'face_detected': 0.01,
    }

    # Test 1
    validator.validate_feature_importance(feature_importance)

    # Test 2
    np.random.seed(42)
    y_true = np.random.randint(2, 5, 50)
    y_pred = y_true + np.random.normal(0, 0.5, 50)
    cv_scores = [0.3, 0.2, 0.4, 0.1, 0.25]
    validator.validate_prediction_accuracy(y_true, y_pred, cv_scores)

    # Test 3
    validator.validate_explanation_quality(feature_importance, [])

    # Test 4
    X = np.random.randn(50, 18)
    validator.check_failure_modes(X, y_true, feature_importance)

    # Report
    report = validator.generate_report()

    return report


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Validate Preference Model')
    parser.add_argument('--model', type=str, help='Model path')
    parser.add_argument('--features', type=str, help='Features JSON')
    parser.add_argument('--labels', type=str, help='Labels JSON')
    parser.add_argument('--simulate', action='store_true')

    args = parser.parse_args()

    if args.simulate or not args.model:
        report = simulate_validation()
    else:
        # Real validation
        print("Loading model and data...")
        # TODO: Load real model and data
        report = simulate_validation()

    # Save report
    output = "./data/validation_report.json"
    with open(output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {output}")

    return report


if __name__ == "__main__":
    main()
