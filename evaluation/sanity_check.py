"""
Sanity Check Experiments for Video Preference Model

Three critical tests to validate scientific validity:
(A) Frame Ablation: 1 frame vs 16 frames vs max-pooling
(B) Identity Leakage: Remove identity overlap → measure drop
(C) Random Baseline: Shuffle labels → must get ~0.5 AUC
"""

import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import spearmanr
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load data
embeddings_file = './data/processed/embeddings_siglip_multiframe.npy'
identity_file = './data/processed/image_to_identity_multiframe.json'
annotations_file = './data/annotations/annotations_result.json'

embeddings = np.load(embeddings_file)
print(f"Embeddings shape: {embeddings.shape}")

# Handle 4D embeddings (N, 1, patches, dim) → mean pool patches to (N, dim)
if embeddings.ndim == 4:
    embeddings = embeddings.mean(axis=2)  # (N, 1, dim)
    embeddings = embeddings.squeeze(1)    # (N, dim)
    print(f"Reshaped to: {embeddings.shape}")

with open(identity_file, 'r', encoding='utf-8') as f:
    identity_map = json.load(f)

with open(annotations_file, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# Build identity → list of indices mapping
identity_to_indices = {}
for idx_str, ident in identity_map.items():
    idx = int(idx_str)
    if ident not in identity_to_indices:
        identity_to_indices[ident] = []
    identity_to_indices[ident].append(idx)

# Build identity → mean embedding
identity_embeddings = {}
for ident, indices in identity_to_indices.items():
    identity_embeddings[ident] = embeddings[indices].mean(axis=0)

unique_identities = list(identity_embeddings.keys())
print(f"Unique identities: {len(unique_identities)}")

# Build preference pairs - proper Bradley-Terry format
# (identity_A, identity_B, prefer_A) where prefer_A is 1 if A preferred, 0 if B
pairs = []
for ann in annotations:
    ident_A = ann['identity_A']
    ident_B = ann['identity_B']

    # Only use explicit preferences (skip "equal")
    if ann['choice'] == 'A':
        prefer_A = 1  # A is preferred
    elif ann['choice'] == 'B':
        prefer_A = 0  # B is preferred (A loses)
    else:
        continue  # Skip "equal"

    if ident_A in identity_embeddings and ident_B in identity_embeddings:
        pairs.append((ident_A, ident_B, prefer_A))

print(f"Valid preference pairs: {len(pairs)}")

# Check class distribution
n_prefer_A = sum(1 for _, _, p in pairs if p == 1)
n_prefer_B = sum(1 for _, _, p in pairs if p == 0)
print(f"  Prefer A: {n_prefer_A}")
print(f"  Prefer B: {n_prefer_B}")

# ============================================================
# (A) FRAME ABLATION TEST
# ============================================================
print("\n" + "="*60)
print("(A) FRAME ABLATION TEST")
print("="*60)

# Method 1: Single best frame (highest norm)
def get_best_frame_embedding(ident, embeddings, identity_map):
    """Get embedding of frame with highest norm."""
    indices = identity_to_indices.get(ident, [])
    if not indices:
        return None
    frames = embeddings[indices]
    norms = np.linalg.norm(frames, axis=1)
    best_idx = indices[np.argmax(norms)]
    return embeddings[best_idx]

# Method 2: Mean pooling (all frames)
def get_mean_embedding(ident):
    """Get mean embedding across all frames."""
    return identity_embeddings.get(ident)

# Method 3: Max pooling (element-wise max)
def get_max_pool_embedding(ident):
    """Get max-pooled embedding across frames."""
    indices = identity_to_indices.get(ident, [])
    if not indices:
        return None
    frames = embeddings[indices]
    return frames.max(axis=0)

# Test each method
def evaluate_method(method_name, get_emb_fn, pairs):
    """Evaluate preference prediction with given embedding method."""
    X = []
    y = []

    for winner, loser, label in pairs:
        emb_w = get_emb_fn(winner)
        emb_l = get_emb_fn(loser)

        if emb_w is None or emb_l is None:
            continue

        # Difference vector
        diff = emb_w - emb_l
        X.append(diff)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    if len(X) < 5:
        return None, None, 0

    if len(np.unique(y)) < 2:
        return None, None, 0

    # For small datasets, use leave-one-out or simple evaluation
    if len(X) < 10:
        # Simple majority baseline
        majority = np.bincount(y.astype(int)).argmax()
        acc = np.mean(y.astype(int) == majority)
        return acc, 0.5, len(X)

    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        majority = np.bincount(y.astype(int)).argmax()
        acc = np.mean(y.astype(int) == majority)
        return acc, 0.5, len(X)

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5

    return acc, auc, len(X)

results = {}

print("\nComparing embedding strategies:")
for name, fn in [
    ("1_frame (best)", lambda i: get_best_frame_embedding(i, embeddings, identity_map)),
    ("Mean (all frames)", lambda i: get_mean_embedding(i)),
    ("Max-pool (all frames)", lambda i: get_max_pool_embedding(i)),
]:
    acc, auc, n = evaluate_method(name, fn, pairs)
    if acc is not None:
        results[name] = {"acc": acc, "auc": auc, "n": n}
        print(f"  {name}: Accuracy={acc:.2%}, AUC={auc:.2f}, N={n}")
    else:
        print(f"  {name}: Not enough data")

# Interpret results
print("\n>>> INTERPRETATION:")
if len(results) >= 2:
    best_method = max(results.keys(), key=lambda k: results[k]['auc'])
    worst_method = min(results.keys(), key=lambda k: results[k]['auc'])
    diff = results[best_method]['auc'] - results[worst_method]['auc']

    if diff < 0.05:
        print("  → Temporal aggregation (mean/max) ≈ single frame")
        print("  → Suggestion: Temporal signal is weak, focus on frame quality")
    else:
        print(f"  → Best: {best_method} ({results[best_method]['auc']:.2f})")
        print(f"  → Temporal signal IS useful")

# ============================================================
# (B) IDENTITY LEAKAGE TEST
# ============================================================
print("\n" + "="*60)
print("(B) IDENTITY LEAKAGE TEST")
print("="*60)

# Count how many pairs share identity in training/test
# True test: if identity appears in train, can it predict in test?

def evaluate_identity_leakage(pairs, test_frac=0.3):
    """Test identity leakage by measuring performance drop."""
    np.random.seed(42)

    # Check pair structure
    all_idents = set()
    for w, l, _ in pairs:
        all_idents.add(w)
        all_idents.add(l)

    print(f"  Total unique identities in pairs: {len(all_idents)}")

    # If identities appear multiple times, there's potential leakage
    ident_counts = {}
    for w, l, _ in pairs:
        ident_counts[w] = ident_counts.get(w, 0) + 1
        ident_counts[l] = ident_counts.get(l, 0) + 1

    multi_appear = sum(1 for c in ident_counts.values() if c > 1)
    print(f"  Identities appearing >1 time: {multi_appear}/{len(ident_counts)}")

    # Standard train/test (random split)
    X = []
    y = []
    for winner, loser, label in pairs:
        emb_w = identity_embeddings.get(winner)
        emb_l = identity_embeddings.get(loser)
        if emb_w is None or emb_l is None:
            continue
        X.append(emb_w - emb_l)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"  Total pairs for evaluation: {len(X)}")
    print(f"  Class distribution: {np.bincount(y.astype(int))}")

    if len(X) < 10:
        print("  ⚠ Not enough pairs for meaningful evaluation")
        return

    if len(np.unique(y)) < 2:
        print("  ⚠ Only one class in labels - cannot evaluate")
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_frac, random_state=42, stratify=y
        )
    except:
        # Fallback without stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_frac, random_state=42
        )

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print("  ⚠ Train or test has only one class")
        return

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc_random = accuracy_score(y_test, y_pred)
    print(f"  Random split accuracy: {acc_random:.2%}")

    # Strict split: no identity overlap between train/test
    np.random.seed(42)
    n_pairs = len(pairs)
    indices = np.random.permutation(n_pairs)
    split = int((1 - test_frac) * n_pairs)

    train_pairs = [pairs[i] for i in indices[:split]]
    test_pairs = [pairs[i] for i in indices[split:]]

    # Check overlap
    train_idents = set()
    for w, l, _ in train_pairs:
        train_idents.add(w)
        train_idents.add(l)

    test_idents = set()
    for w, l, _ in test_pairs:
        test_idents.add(w)
        test_idents.add(l)

    overlap = train_idents & test_idents
    print(f"  Identity overlap in strict split: {len(overlap)}/{len(test_idents)}")

    # Train on strict split
    X_train_s = []
    y_train_s = []
    for winner, loser, label in train_pairs:
        emb_w = identity_embeddings.get(winner)
        emb_l = identity_embeddings.get(loser)
        if emb_w is None or emb_l is None:
            continue
        X_train_s.append(emb_w - emb_l)
        y_train_s.append(label)

    X_test_s = []
    y_test_s = []
    for winner, loser, label in test_pairs:
        emb_w = identity_embeddings.get(winner)
        emb_l = identity_embeddings.get(loser)
        if emb_w is None or emb_l is None:
            continue
        X_test_s.append(emb_w - emb_l)
        y_test_s.append(label)

    if len(X_train_s) > 10 and len(X_test_s) > 5:
        if len(set(y_train_s)) >= 2 and len(set(y_test_s)) >= 2:
            clf_s = LogisticRegression(max_iter=1000, random_state=42)
            clf_s.fit(X_train_s, y_train_s)

            y_pred_s = clf_s.predict(X_test_s)
            acc_strict = accuracy_score(y_test_s, y_pred_s)

            print(f"\n  Random split accuracy: {acc_random:.2%}")
            print(f"  Strict (no overlap) accuracy: {acc_strict:.2%}")
            print(f"  Drop: {acc_random - acc_strict:.2%}")

            print("\n>>> INTERPRETATION:")
            if acc_random - acc_strict > 0.15:
                print("  → Model IS learning identity, not preference")
                print("  → Significant leakage detected")
            elif acc_random - acc_strict > 0.05:
                print("  → Partial leakage, some identity signal")
            else:
                print("  → No significant identity leakage")
                print("  → Model learns preference, not identity")
        else:
            print("  → Not enough test diversity for strict test")
    else:
        print("  → Not enough pairs for strict test")

evaluate_identity_leakage(pairs)

# ============================================================
# (C) RANDOM BASELINE SANITY CHECK
# ============================================================
print("\n" + "="*60)
print("(C) RANDOM BASELINE SANITY CHECK")
print("="*60)

def evaluate_random_baseline(pairs, n_shuffles=10):
    """Shuffle labels and verify model gets ~0.5 AUC."""
    np.random.seed(42)

    # Prepare data
    X = []
    y_true = []
    for winner, loser, label in pairs:
        emb_w = identity_embeddings.get(winner)
        emb_l = identity_embeddings.get(loser)
        if emb_w is None or emb_l is None:
            continue
        X.append(emb_w - emb_l)
        y_true.append(label)

    X = np.array(X)
    y_true = np.array(y_true)

    if len(X) < 5:
        print("  ⚠ Not enough pairs for random baseline test")
        return

    print(f"  True labels accuracy (baseline): ", end="")

    # True labels
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_true, test_size=0.3, random_state=42, stratify=y_true
        )
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_true, test_size=0.3, random_state=42
        )

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print("Not enough classes")
        return

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    true_acc = accuracy_score(y_test, y_pred)
    print(f"{true_acc:.2%}")

    # Shuffled labels
    shuffled_accs = []
    for i in range(n_shuffles):
        y_shuffled = np.random.permutation(y_true)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_shuffled, test_size=0.3, random_state=42, stratify=y_shuffled
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_shuffled, test_size=0.3, random_state=42
            )

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        shuffled_accs.append(accuracy_score(y_test, y_pred))

    if not shuffled_accs:
        print("  ⚠ Cannot compute random baseline (class issues)")
        return

    mean_shuffled = np.mean(shuffled_accs)
    std_shuffled = np.std(shuffled_accs)

    print(f"  Shuffled labels accuracy: {mean_shuffled:.2%} ± {std_shuffled:.2%}")

    print("\n>>> INTERPRETATION:")
    if abs(mean_shuffled - 0.5) < 0.05:
        print("  ✓ Shuffled ≈ 50% → Model has NO signal (labels are noise)")
        print("  ✓ True accuracy > shuffled → Model learns SOMETHING")
    elif mean_shuffled > 0.55:
        print("  ⚠ Shuffled > 55% → Potential label bias")
    else:
        print("  ✓ Random baseline is correct")

    if true_acc > mean_shuffled + 2 * std_shuffled:
        print(f"  ✓ True ({true_acc:.2%}) > Shuffled ({mean_shuffled:.2%})")
        print("  → Model learns real signal, not just noise")
    else:
        print(f"  ⚠ True ({true_acc:.2%}) ≈ Shuffled ({mean_shuffled:.2%})")
        print("  → Model may be learning noise")

evaluate_random_baseline(pairs)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("SUMMARY: Scientific Validity Assessment")
print("="*60)

print("""
Next steps based on results:

1. If (A) shows weak temporal signal:
   → Focus on frame quality selection, not temporal aggregation

2. If (B) shows identity leakage:
   → Need to control for identity in annotations
   → Or use identity-conditioned evaluation

3. If (C) shows shuffled ≈ true:
   → Labels are noise, need better annotation protocol
   → Current annotations don't measure true preference

4. If all tests pass:
   → Model is scientifically valid
   → Can proceed to counterfactual evaluation
""")

print("="*60)
print("DONE")
print("="*60)
