"""
Evaluation Script - Run full evaluation pipeline

Usage:
    python -m evaluation.run_evaluation --data_dir ./data --output ./results
    python -m evaluation.run_evaluation --synthetic   # demo / testing only

Without --synthetic, all phases require real data and stop if unavailable.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np


def _check_real_data(data_dir: str) -> bool:
    """Return True when real evaluation artifacts are present."""
    data_path = Path(data_dir)
    processed = data_path / "processed"
    annotations = data_path / "annotations"

    has_embeddings = (processed / "embeddings.npz").exists()
    has_annotations = (annotations / "annotations_result.json").exists()
    return has_embeddings and has_annotations


def run_evaluation(
    data_dir: str = "./data",
    output_dir: str = "./results",
    embedding_model: str = "siglip",
    n_permutations: int = 1000,
    synthetic: bool = False,
):
    """Run full evaluation pipeline.

    Args:
        data_dir: Path to data directory.
        output_dir: Path to output directory.
        embedding_model: Model to use for embeddings.
        n_permutations: Number of permutations for null tests.
        synthetic: When True, use synthetic data for demo/testing.
    """
    print("=" * 60)
    print("Face Project - Preference Signal Discovery Evaluation")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Mode: {'SYNTHETIC (demo only)' if synthetic else 'REAL DATA'}")
    print()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "embedding_model": embedding_model,
            "n_permutations": n_permutations,
            "synthetic": synthetic,
        },
        "phases": {},
        "summary": {"recommendation": ""},
    }

    # ---------- Data Availability Check ----------
    data_ok = _check_real_data(data_dir)
    results["phases"]["data_check"] = {
        "data_dir": data_dir,
        "real_data_available": data_ok,
        "synthetic_mode": synthetic,
    }

    if not data_ok and not synthetic:
        print("[!] No real evaluation artifacts found (embeddings.npz + annotations_result.json).")
        print("[!] Use --synthetic for demo mode.")
        print("[!] Stopping evaluation.")
        results["summary"]["recommendation"] = "data_unavailable"
        _save_results(results, output_path)
        return results

    if synthetic:
        print("[!] SYNTHETIC mode — results are NOT meaningful for real conclusions.")
        print()

    # ---------- Phase 1: Representation Validation ----------
    print("Phase 1: Representation Validation")
    print("-" * 40)

    from src.analysis_evalution.validation import RepresentationValidator

    if synthetic or not data_ok:
        np.random.seed(42)
        n_samples = 200
        n_features = 128
        embeddings = {
            "siglip": np.random.randn(n_samples, n_features),
            "dinov2": np.random.randn(n_samples, n_features),
        }
        for model_name in embeddings:
            embeddings[model_name] = (
                embeddings[model_name] /
                np.linalg.norm(embeddings[model_name], axis=1, keepdims=True)
            )
        labels = (embeddings["siglip"][:, 0] > 0).astype(int)
    else:
        # Real data path — future integration
        embeddings = {}
        labels = np.array([])
        print("[SKIP] Real embedding loading not yet wired — skipping Phase 1.")

    if embeddings:
        validator = RepresentationValidator(
            stability_threshold=0.85, cross_model_threshold=0.4
        )
        validation_results = validator.validate(embeddings, labels)
        results["phases"]["representation_validation"] = {
            "status": "completed",
            "n_samples": n_samples if synthetic else len(labels),
            "details": {
                k: str(v) if not isinstance(v, dict) else v
                for k, v in validation_results.items()
            },
        }
        print("[+] Representation validation completed")
    else:
        print("[-] Phase 1 skipped — no embeddings loaded.")
    print()

    # ---------- Phase 2: Null Models ----------
    print("Phase 2: Null Models Testing")
    print("-" * 40)

    from src.preference_learning.null_models import NullModelSuite, interpret_null_results

    if synthetic or (data_ok and len(labels) > 0):
        suite = NullModelSuite(n_permutations=n_permutations, alpha=0.05)
        if synthetic:
            user_ids = np.array([f"user_{i % 20}" for i in range(n_samples)])
            null_results = suite.run_all(
                embeddings["siglip"], labels, user_ids
            )
        else:
            null_results = {"summary": {"n_tests": 0, "n_rejected": 0, "all_rejected": False}}

        interpretation = interpret_null_results(null_results)
        results["phases"]["null_models"] = {
            "status": "completed",
            "synthetic": synthetic,
            "interpretation": interpretation,
            "summary": null_results.get("summary", {}),
        }
        print(interpretation)
    else:
        print("[-] Phase 2 skipped — no labels available.")
    print()

    # ---------- Phase 3: Pairwise Preference ----------
    print("Phase 3: Pairwise Preference Learning")
    print("-" * 40)

    if synthetic:
        from src.preference_learning.preference import BradleyTerryModel, PairwiseSample

        pairs = []
        for i in range(50):
            winner = "A" if np.random.rand() > 0.5 else "B"
            pairs.append(PairwiseSample(
                user_id=f"user_{i % 10}",
                image_A=f"img_{i}",
                image_B=f"img_{i + 50}",
                winner=winner,
            ))

        bt_model = BradleyTerryModel()
        bt_result = bt_model.fit(pairs)

        results["phases"]["preference_learning"] = {
            "status": "completed",
            "synthetic": True,
            "bradley_terry": {
                "n_items": len(bt_result.item_scores),
                "converged": bt_result.convergence,
                "n_iterations": bt_result.n_iterations,
            },
        }
        print(f"[+] Bradley-Terry: {len(bt_result.item_scores)} items, converged: {bt_result.convergence}")
    else:
        print("[-] Phase 3 skipped — requires pairwise annotations.")
    print()

    # ---------- Summary ----------
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if synthetic:
        print("[WARN] All results are SYNTHETIC — no conclusions about real data.")
        results["summary"]["recommendation"] = "synthetic_only"
    else:
        nulls = results.get("phases", {}).get("null_models", {}).get("summary", {})
        all_rejected = nulls.get("all_rejected", False)
        if all_rejected:
            print("[PASS] All null models rejected — signal detected.")
            results["summary"]["recommendation"] = "signal_detected"
        else:
            print("[WARN] Some null models not rejected — signal may be weak.")
            results["summary"]["recommendation"] = "weak_signal"

    _save_results(results, output_path)
    print(f"\nCompleted: {datetime.now().isoformat()}")
    return results


def _save_results(results: dict, output_path: Path):
    output_file = output_path / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline")
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--output_dir', type=str, default="./results")
    parser.add_argument('--embedding_model', type=str, default="siglip")
    parser.add_argument('--n_permutations', type=int, default=1000)
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data (demo mode). Without this flag, '
                             'evaluation requires real data.')
    args = parser.parse_args()

    run_evaluation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        n_permutations=args.n_permutations,
        synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
