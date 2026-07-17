"""
Canonical paths for Facense evaluation.

All evaluation scripts import from here instead of hardcoding paths.
"""

from pathlib import Path

# Root data directory (override with FACENSE_DATA_DIR env var)
import os
_DATA_ROOT = Path(os.getenv("FACENSE_DATA_DIR", "./data"))

# Processed artifacts (embeddings, BT scores)
PROCESSED_DIR = _DATA_ROOT / "processed"
EMBEDDINGS_NPZ = PROCESSED_DIR / "embeddings.npz"
BT_SCORES_JSON = PROCESSED_DIR / "bradley_terry_scores.json"

# Annotations
ANNOTATIONS_DIR = _DATA_ROOT / "annotations"
ANNOTATIONS_RESULT = ANNOTATIONS_DIR / "annotations_result.json"

# Results output
RESULTS_DIR = Path(os.getenv("FACENSE_RESULTS_DIR", "./results"))

# Raw dataset videos
DATASET_DIR = Path(os.getenv("DATASET_DIR", "./data/raw"))

# Labels / pairwise data
LABELS_FILE = _DATA_ROOT / "dataset_processed" / "features.json"
LABELING_PAIRS_FILE = _DATA_ROOT / "dataset_processed" / "labeling_pairs.json"
