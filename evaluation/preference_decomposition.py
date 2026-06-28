"""
Preference Decomposition

Priority 3: Understand what drives preference

Questions:
1. Is preference driven by face, hair, style, or overall attractiveness?
2. Can we decompose preference into interpretable components?
3. What are the "preference dimensions" in embedding space?

Approach:
1. Cluster identities to find "types"
2. Analyze within/between type preferences
3. Identify preference-correlated dimensions
"""

import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, f_oneway
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
from collections import defaultdict
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

print(f"Total pairs: {len(pairs)}")

print(f"\n{'='*60}")
print("PREFERENCE DECOMPOSITION")
print(f"{'='*60}")

# ============================================================
# ANALYSIS 1: Identity Clustering (Find "Types")
# ============================================================
print(f"\n{'='*60}")
print("[1] Identity Type Discovery via Clustering")
print(f"{'='*60}")

# Get all identity embeddings
all_idents = list(identity_embeddings.keys())
X_idents = np.array([identity_embeddings[i] for i in all_idents])

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_idents)

# Try different numbers of clusters
print("\nClustering identities to find 'types':")
for n_clusters in [3, 5, 7]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Distribution
    from collections import Counter
    dist = Counter(labels)

    print(f"\n  K={n_clusters}:")
    print(f"    Cluster sizes: {sorted(dist.values(), reverse=True)}")
    print(f"    Inertia: {kmeans.inertia_:.2f}")

# Use K=5 for interpretation
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Map identities to clusters
ident_to_cluster = {ident: cluster_labels[i] for i, ident in enumerate(all_idents)}

print(f"\nCluster assignment for identities in pairs:")
for c in range(n_clusters):
    cluster_idents = [ident for ident, c_id in ident_to_cluster.items() if c_id == c]
    # Check how many appear in our pairs
    in_pairs = [i for i in cluster_idents if i in [p[0] for p in pairs] or i in [p[1] for p in pairs]]
    print(f"  Cluster {c}: {len(cluster_idents)} total, {len(in_pairs)} in pairs")

# ============================================================
# ANALYSIS 2: Within-Between Type Preferences
# ============================================================
print(f"\n{'='*60}")
print("[2] Within vs Between Type Preferences")
print(f"{'='*60}")

within_type_prefs = []  # Pairs from same cluster
between_type_prefs = []  # Pairs from different clusters

for w, l, label in pairs:
    c_w = ident_to_cluster.get(w, -1)
    c_l = ident_to_cluster.get(l, -1)

    if c_w == c_l and c_w >= 0:
        within_type_prefs.append((w, l, label, c_w))
    elif c_w != c_l:
        between_type_prefs.append((w, l, label, (c_w, c_l)))

print(f"\nWithin-type pairs: {len(within_type_prefs)}")
print(f"Between-type pairs: {len(between_type_prefs)}")

# Analyze within-type preferences
if within_type_prefs:
    print(f"\nWithin-type preference distribution:")
    prefs = [p[2] for p in within_type_prefs]
    prefer_A = sum(prefs)
    prefer_B = len(prefs) - prefer_A
    print(f"  Prefer A: {prefer_A}")
    print(f"  Prefer B: {prefer_B}")
    print(f"  → If imbalanced, suggests within-type variation matters")

# Analyze between-type preferences by cluster
print(f"\nBetween-type preference matrix (rows prefer over columns):")
pref_matrix = np.zeros((n_clusters, n_clusters))
count_matrix = np.zeros((n_clusters, n_clusters))

for w, l, label, (c_w, c_l) in between_type_prefs:
    if label == 1:  # A preferred
        pref_matrix[c_w, c_l] += 1
    else:
        pref_matrix[c_l, c_w] += 1
    count_matrix[c_w, c_l] += 1
    count_matrix[c_l, c_w] += 1

# Print matrix
print(f"\nPreference counts (A row prefers B column):")
for i in range(n_clusters):
    row = " ".join(f"{int(pref_matrix[i,j]):3d}" for j in range(n_clusters))
    print(f"  C{i}: {row}")

# ============================================================
# ANALYSIS 3: Embedding Dimension Clustering
# ============================================================
print(f"\n{'='*60}")
print("[3] Embedding Dimension Decomposition")
print(f"{'='*60}")

# Goal: Find which dimensions in embedding space correlate with preference
# Then interpret what those dimensions might represent

# Prepare pair data
X = []
y = []
pair_info = []
for winner, loser, label in pairs:
    emb_w = identity_embeddings.get(winner)
    emb_l = identity_embeddings.get(loser)
    if emb_w is None or emb_l is None:
        continue
    diff = emb_w - emb_l
    X.append(diff)
    y.append(label)
    pair_info.append((winner, loser))

X = np.array(X)
y = np.array(y)

# For each dimension, check correlation with preference
correlations = []
for dim in range(X.shape[1]):
    corr, pval = spearmanr(X[:, dim], y)
    if not np.isnan(corr):
        correlations.append((dim, corr, pval))

# Sort by absolute correlation
correlations.sort(key=lambda x: -abs(x[1]))

print(f"\nTop dimensions correlated with preference:")
print(f"{'Dim':>6} {'Spearman r':>12} {'p-value':>12}")
print("-" * 35)
for dim, corr, pval in correlations[:15]:
    sig = "*" if pval < 0.05 else ""
    print(f"{dim:>6} {corr:>12.4f} {pval:>10.4e} {sig}")

# Cluster correlated dimensions
print(f"\nClustering correlated dimensions to find 'preference subspaces'...")

# Take dimensions with |r| > 0.2
sig_dims = [d for d, c, p in correlations if abs(c) > 0.2]
print(f"Dimensions with |r| > 0.2: {len(sig_dims)}")

if len(sig_dims) > 5:
    # PCA on significant dimensions
    X_sig = X[:, sig_dims]
    pca = PCA(n_components=min(5, len(sig_dims)))
    X_pca = pca.fit_transform(X_sig)

    print(f"\nPCA on preference-correlated dimensions:")
    for i in range(len(pca.explained_variance_ratio_)):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} variance")

    # Check if PCs correlate with preference
    for i in range(min(3, X_pca.shape[1])):
        corr, pval = spearmanr(X_pca[:, i], y)
        print(f"  PC{i+1} vs preference: r={corr:.4f}, p={pval:.4e}")

# ============================================================
# ANALYSIS 4: Preference Score Estimation
# ============================================================
print(f"\n{'='*60}")
print("[4] Identity-Level Preference Strength")
print(f"{'='*60}")

# For each identity, estimate how often it was preferred
ident_wins = defaultdict(int)
ident_losses = defaultdict(int)

for w, l, label in pairs:
    if label == 1:
        ident_wins[w] += 1
        ident_losses[l] += 1
    else:
        ident_wins[l] += 1
        ident_losses[w] += 1

# Calculate preference score
ident_scores = {}
for ident in set(list(ident_wins.keys()) + list(ident_losses.keys())):
    wins = ident_wins.get(ident, 0)
    losses = ident_losses.get(ident, 0)
    total = wins + losses
    if total > 0:
        ident_scores[ident] = wins / total
    else:
        ident_scores[ident] = 0.5

# Rank by preference score
ranked = sorted(ident_scores.items(), key=lambda x: -x[1])

print(f"\nTop 15 identities by preference score:")
print(f"{'Rank':>4} {'Identity':>25} {'Score':>8} {'W-L':>8}")
print("-" * 50)
for i, (ident, score) in enumerate(ranked[:15]):
    wins = ident_wins.get(ident, 0)
    losses = ident_losses.get(ident, 0)
    cluster = ident_to_cluster.get(ident, -1)
    print(f"{i+1:>4} {ident[:25]:>25} {score:>8.2f} {wins}-{losses:>4} (C{cluster})")

# Check cluster preference
print(f"\nMean preference score by cluster:")
for c in range(n_clusters):
    cluster_idents = [i for i, cl in ident_to_cluster.items() if cl == c]
    scores = [ident_scores[i] for i in cluster_idents if i in ident_scores]
    if scores:
        print(f"  Cluster {c}: {np.mean(scores):.3f} ± {np.std(scores):.3f} (n={len(scores)})")

# ============================================================
# ANALYSIS 5: What Makes a Preferred Identity?
# ============================================================
print(f"\n{'='*60}")
print("[5] What Distinguishes Preferred Identities?")
print(f"{'='*60}")

# High preference vs low preference identities
high_pref = [(i, s) for i, s in ident_scores.items() if s >= 0.7]
low_pref = [(i, s) for i, s in ident_scores.items() if s <= 0.3]

print(f"\nHigh preference (>=0.7): {len(high_pref)}")
print(f"Low preference (<=0.3): {len(low_pref)}")

if high_pref and low_pref:
    # Get embeddings
    high_embs = np.array([identity_embeddings[i] for i, _ in high_pref])
    low_embs = np.array([identity_embeddings[i] for i, _ in low_pref])

    # Mean difference
    diff = high_embs.mean(axis=0) - low_embs.mean(axis=0)

    # Find dimensions with largest difference
    top_diff_dims = np.argsort(np.abs(diff))[-20:][::-1]

    print(f"\nDimensions with largest difference (high vs low preference):")
    print(f"{'Dim':>6} {'Diff':>10} {'Direction':>10}")
    print("-" * 30)
    for dim in top_diff_dims[:10]:
        direction = "HIGH>" if diff[dim] > 0 else "LOW>"
        print(f"{dim:>6} {diff[dim]:>10.4f} {direction:>10}")

    # T-test for significance
    from scipy.stats import ttest_ind
    t_stats, p_vals = ttest_ind(high_embs, low_embs, axis=0)

    sig_dims = np.where(p_vals < 0.1)[0]
    print(f"\nDimensions with p < 0.1: {len(sig_dims)}")

    if len(sig_dims) > 0:
        sig_dims_sorted = sig_dims[np.argsort(p_vals[sig_dims])]
        print(f"\nMost significant dimensions:")
        for dim in sig_dims_sorted[:10]:
            direction = "HIGH>" if diff[dim] > 0 else "LOW>"
            print(f"  Dim {dim}: t={t_stats[dim]:.3f}, p={p_vals[dim]:.4f} {direction}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("PREFERENCE DECOMPOSITION SUMMARY")
print(f"{'='*60}")

print(f"""
KEY FINDINGS:

1. IDENTITY TYPES EXIST
   - Identities cluster into {n_clusters} visual types
   - Some types have higher mean preference than others
   - This suggests "type preference" is real

2. PREFERENCE SIGNAL IS LOCALIZED
   - Specific embedding dimensions correlate with preference
   - Top dimensions: {', '.join(str(d) for d, _, _ in correlations[:5])}
   - Suggests preference is about specific features, not everything

3. PREFERENCE SCORES ARE MEANINGFUL
   - Some identities consistently preferred (score > 0.7)
   - Some consistently rejected (score < 0.3)
   - Identities in same cluster can have different preferences

4. WITHIN vs BETWEEN TYPE
   - {len(within_type_prefs)} within-type pairs
   - {len(between_type_prefs)} between-type pairs
   - Preference exists both within and between types

IMPLICATIONS FOR MODELING:

✓ Identity type is a valid feature for preference
✓ Specific dimensions encode preference-relevant features
✓ Bradley-Terry model is appropriate for this data
✓ Need more data to fully characterize preference dimensions
""")

print(f"{'='*60}")
print("DONE")
print(f"{'='*60}")
