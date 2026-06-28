"""
Identity-Stratified Evaluation

Priority 1: Disentangle identity vs preference

Key question: "Does preference exist as a signal independent of identity?"

Design:
- Train: identities A, B, C, D
- Test: identities E, F, G, H (completely unseen)
- If performance → random: identity memorization, no preference
- If performance stays: genuine preference signal exists
"""

import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import defaultdict
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load data
embeddings = np.load('./data/processed/embeddings_siglip_multiframe.npy')
if embeddings.ndim == 4:
    embeddings = embeddings.mean(axis=2).squeeze(1)
print(f"Embeddings: {embeddings.shape}")

with open('./data/processed/image_to_identity_multiframe.json', 'r', encoding='utf-8') as f:
    identity_map = json.load(f)

with open('./data/annotations/annotations_result.json', 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# Build identity → mean embedding
identity_to_indices = defaultdict(list)
for idx_str, ident in identity_map.items():
    identity_to_indices[ident].append(int(idx_str))

identity_embeddings = {}
for ident, indices in identity_to_indices.items():
    identity_embeddings[ident] = embeddings[indices].mean(axis=0)

# Build preference pairs
pairs = []
for ann in annotations:
    ident_A = ann['identity_A']
    ident_B = ann['identity_B']

    if ann['choice'] == 'A':
        prefer_A = 1
    elif ann['choice'] == 'B':
        prefer_A = 0
    else:
        continue

    if ident_A in identity_embeddings and ident_B in identity_embeddings:
        pairs.append((ident_A, ident_B, prefer_A))

print(f"\n{'='*60}")
print("IDENTITY-STRATIFIED EVALUATION")
print(f"{'='*60}")
print(f"Total pairs: {len(pairs)}")

# ============================================================
# ANALYSIS 1: Identity appearance frequency
# ============================================================
print(f"\n[1] Identity Distribution Analysis")

ident_counts = defaultdict(int)
for a, b, _ in pairs:
    ident_counts[a] += 1
    ident_counts[b] += 1

print(f"  Unique identities in pairs: {len(ident_counts)}")
print(f"  Identities appearing >1 time: {sum(1 for c in ident_counts.values() if c > 1)}")
print(f"  Max appearances: {max(ident_counts.values())}")
print(f"  Mean appearances: {np.mean(list(ident_counts.values())):.2f}")

# ============================================================
# ANALYSIS 2: Leave-One-Identity-Out Cross-Validation
# ============================================================
print(f"\n{'='*60}")
print("[2] Leave-One-Identity-Out Cross-Validation (LOIO-CV)")
print(f"{'='*60}")

# Prepare data with identity groups
X = []
y = []
groups = []  # Which identity pair this belongs to

for winner, loser, label in pairs:
    emb_w = identity_embeddings.get(winner)
    emb_l = identity_embeddings.get(loser)
    if emb_w is None or emb_l is None:
        continue
    # Difference vector
    diff = emb_w - emb_l
    X.append(diff)
    y.append(label)
    # Group by the identities involved (both must be unseen to be a true test)
    groups.append((winner, loser))

X = np.array(X)
y = np.array(y)

# Leave-One-Group-Out CV
# Group = unique pair of identities (A, B)
unique_pairs = list(set(groups))
print(f"  Total unique identity-pairs: {len(unique_pairs)}")

if len(unique_pairs) > 5:
    logo = LeaveOneGroupOut()

    # Use pair index as group (each pair is its own group for strict testing)
    pair_to_idx = {p: i for i, p in enumerate(groups)}

    # Alternative: Use combined identity set as group
    group_ids = []
    for w, l in groups:
        # Sort to ensure consistent grouping
        combined = tuple(sorted([w, l]))
        # Hash to numeric
        group_id = hash(combined) % 10000
        group_ids.append(group_id)

    group_ids = np.array(group_ids)

    print(f"  Testing with {len(set(group_ids))} unique identity groups")

    # Cross-validation
    cv_accuracies = []
    cv_aucs = []

    for train_idx, test_idx in logo.split(X, y, groups=group_ids):
        if len(np.unique(y[train_idx])) < 2:
            continue
        if len(np.unique(y[test_idx])) < 2:
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv_accuracies.append(acc)

        if len(np.unique(y_test)) > 1:
            try:
                y_prob = clf.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
                cv_aucs.append(auc)
            except:
                pass

    if cv_accuracies:
        print(f"\n  Leave-One-Pair-Out Results:")
        print(f"    Mean Accuracy: {np.mean(cv_accuracies):.2%}")
        print(f"    Std Accuracy:  {np.std(cv_accuracies):.2%}")
        if cv_aucs:
            print(f"    Mean AUC:      {np.mean(cv_aucs):.2f}")
            print(f"    Std AUC:       {np.std(cv_aucs):.2f}")
else:
    print(f"  ⚠ Not enough unique pairs for proper CV")

# ============================================================
# ANALYSIS 3: Strict Unseen Identity Evaluation
# ============================================================
print(f"\n{'='*60}")
print("[3] Strict Unseen Identity Evaluation")
print(f"{'='*60}")

# Goal: Train on identities A,B,C,D → Test on E,F,G,H (never seen)

# Group pairs by which identities they contain
identity_to_pairs = defaultdict(list)
for i, (w, l, label) in enumerate(pairs):
    identity_to_pairs[w].append(i)
    identity_to_pairs[l].append(i)

all_idents = list(set(list(ident_counts.keys())))
np.random.seed(42)
np.random.shuffle(all_idents)

# Split identities
n_test_idents = max(2, len(all_idents) // 4)
test_idents = set(all_idents[:n_test_idents])
train_idents = set(all_idents[n_test_idents:])

print(f"  Train identities: {len(train_idents)}")
print(f"  Test identities: {len(test_idents)}")
print(f"  Overlap: {len(train_idents & test_idents)}")

# Check pair distribution
train_pairs = []
test_pairs = []

for i, (w, l, label) in enumerate(pairs):
    # Strict: both identities must be in the same split
    if w in test_idents and l in test_idents:
        test_pairs.append((w, l, label))
    elif w in train_idents and l in train_idents:
        train_pairs.append((w, l, label))
    # Mixed pairs (one in train, one in test) → exclude

print(f"  Train pairs (same-identity-split): {len(train_pairs)}")
print(f"  Test pairs (same-identity-split): {len(test_pairs)}")
print(f"  Excluded (mixed): {len(pairs) - len(train_pairs) - len(test_pairs)}")

# Evaluate
def evaluate_strict_split(train_pairs, test_pairs, name):
    """Evaluate with strict identity separation."""
    if len(train_pairs) < 5 or len(test_pairs) < 3:
        return None

    # Prepare data
    X_train = []
    y_train = []
    for w, l, label in train_pairs:
        emb_w = identity_embeddings.get(w)
        emb_l = identity_embeddings.get(l)
        if emb_w is None or emb_l is None:
            continue
        X_train.append(emb_w - emb_l)
        y_train.append(label)

    X_test = []
    y_test = []
    for w, l, label in test_pairs:
        emb_w = identity_embeddings.get(w)
        emb_l = identity_embeddings.get(l)
        if emb_w is None or emb_l is None:
            continue
        X_test.append(emb_w - emb_l)
        y_test.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    # Train model
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5

    print(f"\n  [{name}]")
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"    Accuracy: {acc:.2%}")
    print(f"    AUC: {auc:.2f}")

    return {"acc": acc, "auc": auc, "n_train": len(X_train), "n_test": len(X_test)}

print("\n  Evaluating with multiple random splits...")
results = []
for seed in range(5):
    np.random.seed(seed)
    shuffled_idents = all_idents.copy()
    np.random.shuffle(shuffled_idents)

    n_test = max(2, len(shuffled_idents) // 4)
    test_set = set(shuffled_idents[:n_test])
    train_set = set(shuffled_idents[n_test:])

    train_p = [(w, l, label) for w, l, label in pairs
               if (w in train_set and l in train_set)]
    test_p = [(w, l, label) for w, l, label in pairs
              if (w in test_set and l in test_set)]

    r = evaluate_strict_split(train_p, test_p, f"Split {seed+1}")
    if r:
        results.append(r)

if results:
    mean_acc = np.mean([r['acc'] for r in results])
    std_acc = np.std([r['acc'] for r in results])
    mean_auc = np.mean([r['auc'] for r in results])
    std_auc = np.std([r['auc'] for r in results])

    print(f"\n  >>> AGGREGATE RESULTS:")
    print(f"      Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
    print(f"      AUC: {mean_auc:.2f} ± {std_auc:.2f}")

# ============================================================
# ANALYSIS 4: Baseline Comparison
# ============================================================
print(f"\n{'='*60}")
print("[4] Baseline Comparisons")
print(f"{'='*60}")

# Random baseline
np.random.seed(42)
random_preds = np.random.rand(len(pairs)) > 0.5
y_true = np.array([p[2] for p in pairs])
random_acc = accuracy_score(y_true, random_preds)
print(f"  Random baseline accuracy: {random_acc:.2%}")

# Majority baseline
majority_class = np.bincount(y_true).argmax()
majority_acc = (y_true == majority_class).mean()
print(f"  Majority baseline accuracy: {majority_acc:.2%}")

# Embedding similarity baseline
print(f"\n  Embedding similarity as predictor:")

sim_correct = 0
for w, l, label in pairs:
    emb_w = identity_embeddings.get(w)
    emb_l = identity_embeddings.get(l)
    if emb_w is None or emb_l is None:
        continue
    # If embedding similarity predicts preference
    sim = np.dot(emb_w, emb_l) / (np.linalg.norm(emb_w) * np.linalg.norm(emb_l) + 1e-10)
    # Higher similarity = prefer A
    pred = 1 if sim > 0.5 else 0
    if pred == label:
        sim_correct += 1

sim_acc = sim_correct / len(pairs)
print(f"    Cosine similarity accuracy: {sim_acc:.2%}")

# ============================================================
# FINAL INTERPRETATION
# ============================================================
print(f"\n{'='*60}")
print("FINAL INTERPRETATION")
print(f"{'='*60}")

# What do the results mean?
print("""
KEY QUESTION: Does preference exist as a signal independent of identity?

EVIDENCE ANALYSIS:
""")

if results:
    if mean_acc > 0.7:
        print(f"✓ Strict unseen-identity accuracy: {mean_acc:.2%}")
        print(f"  → Preference signal EXISTS independent of identity")
        print(f"  → Model generalizes to new identities")
    elif mean_acc > 0.55:
        print(f"~ Partial generalization: {mean_acc:.2%}")
        print(f"  → Some preference signal, mixed with identity effects")
    else:
        print(f"✗ Near-random on unseen identities: {mean_acc:.2%}")
        print(f"  → Model relies heavily on identity memorization")
        print(f"  → Preference signal is WEAK or DOES NOT EXIST independently")

print(f"""
RANDOM BASELINE: {random_acc:.2%}
MAJORITY BASELINE: {majority_acc:.2%}
COSINE SIMILARITY: {sim_acc:.2%}

NEXT STEPS:
1. If near-random: Need attribute-controlled experiments
2. If signal exists: Proceed with preference decomposition
3. Design counterfactual: "Would preference change with identity swap?"
""")

print(f"{'='*60}")
print("DONE")
print(f"{'='*60}")
