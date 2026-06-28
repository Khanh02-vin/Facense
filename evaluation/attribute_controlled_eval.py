"""
Attribute-Controlled Evaluation

Priority 2: Disentangle face features vs identity features

Key question: "Is preference driven by identity or by face/visual attributes?"

Approach:
1. Feature decomposition: What in the embedding correlates with preference?
2. Identity-similarity correlation: Does similarity predict preference?
3. Within-identity analysis: Same identity, different frames → does preference vary?
"""

import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
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

# Build identity → embeddings
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
print("ATTRIBUTE-CONTROLLED EVALUATION")
print(f"{'='*60}")

# ============================================================
# ANALYSIS 1: Feature Importance via Mutual Information
# ============================================================
print(f"\n{'='*60}")
print("[1] Feature Importance Analysis")
print(f"{'='*60}")

# Prepare data
X = []
y = []
for winner, loser, label in pairs:
    emb_w = identity_embeddings.get(winner)
    emb_l = identity_embeddings.get(loser)
    if emb_w is None or emb_l is None:
        continue
    diff = emb_w - emb_l
    X.append(diff)
    y.append(label)

X = np.array(X)
y = np.array(y)
print(f"Data shape: {X.shape}")

# Mutual information for feature importance
print("\nComputing mutual information for each embedding dimension...")
mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)

# Top features
top_k = 20
top_indices = np.argsort(mi_scores)[-top_k:][::-1]

print(f"\nTop {top_k} most informative dimensions:")
print(f"{'Dim':>6} {'MI Score':>10}")
print("-" * 20)
for idx in top_indices[:10]:
    print(f"{idx:>6} {mi_scores[idx]:>10.4f}")

print(f"\nTotal MI sum: {mi_scores.sum():.4f}")
print(f"Top 20 dims account for: {mi_scores[top_indices].sum() / mi_scores.sum() * 100:.1f}% of total MI")

# ============================================================
# ANALYSIS 2: PCA to find latent structure
# ============================================================
print(f"\n{'='*60}")
print("[2] PCA Latent Structure Analysis")
print(f"{'='*60}")

# PCA on difference vectors
pca = PCA(n_components=min(20, X.shape[0]))
X_pca = pca.fit_transform(X)

print(f"Explained variance ratio:")
for i in range(min(5, len(pca.explained_variance_ratio_))):
    print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} ({pca.explained_variance_ratio_[:i+1].sum()*100:.1f}% cumulative)")

# Check if PC1 correlates with preference
corr_pc1 = spearmanr(X_pca[:, 0], y)
print(f"\nPC1 vs preference correlation: rho={corr_pc1.statistic:.4f}, p={corr_pc1.pvalue:.4e}")

# ============================================================
# ANALYSIS 3: Embedding Similarity as Predictor
# ============================================================
print(f"\n{'='*60}")
print("[3] Embedding Similarity Analysis")
print(f"{'='*60}")

# For each pair, compute:
# 1. Cosine similarity between A and B
# 2. Euclidean distance between A and B
# 3. Correlation between preference and similarity

cos_sims = []
euc_dists = []
for w, l, label in pairs:
    emb_w = identity_embeddings.get(w)
    emb_l = identity_embeddings.get(l)
    if emb_w is None or emb_l is None:
        continue
    cos_sim = np.dot(emb_w, emb_l) / (np.linalg.norm(emb_w) * np.linalg.norm(emb_l) + 1e-10)
    euc_dist = np.linalg.norm(emb_w - emb_l)
    cos_sims.append(cos_sim)
    euc_dists.append(euc_dist)

cos_sims = np.array(cos_sims)
euc_dists = np.array(euc_dists)

# Correlations
corr_cos = spearmanr(cos_sims, y)
corr_euc = spearmanr(euc_dists, y)

print(f"\nEmbedding similarity vs preference:")
print(f"  Cosine similarity: rho={corr_cos.statistic:.4f}, p={corr_cos.pvalue:.4e}")
print(f"  Euclidean distance: rho={corr_euc.statistic:.4f}, p={corr_euc.pvalue:.4e}")

# Does higher similarity predict preference?
# If cosine similarity predicts who is preferred (regardless of who is A or B)
# we need to think about this differently

print(f"\nMean cosine similarity: {cos_sims.mean():.4f} ± {cos_sims.std():.4f}")
print(f"Mean euclidean distance: {euc_dists.mean():.4f} ± {euc_dists.std():.4f}")

# ============================================================
# ANALYSIS 4: Within-Identity Analysis
# ============================================================
print(f"\n{'='*60}")
print("[4] Within-Identity Analysis")
print(f"{'='*60}")

# Key question: Do different frames of the SAME identity show different preference patterns?
# This would indicate that identity alone doesn't determine preference

# Find identities with multiple frames
multi_frame_idents = {ident: indices for ident, indices in identity_to_indices.items()
                      if len(indices) > 1}

print(f"\nIdentities with multiple frames: {len(multi_frame_idents)}")

if multi_frame_idents:
    # For each multi-frame identity, compute frame variance
    frame_variances = []
    for ident, indices in multi_frame_idents.items():
        frames = embeddings[indices]
        # Variance across frames
        var = frames.var(axis=0).mean()  # Mean variance across dimensions
        frame_variances.append((ident, var, len(frames)))

    frame_variances.sort(key=lambda x: -x[1])

    print(f"\nTop 5 identities by frame variance:")
    print(f"{'Identity':>25} {'Variance':>10} {'Frames':>6}")
    print("-" * 45)
    for ident, var, n_frames in frame_variances[:5]:
        print(f"{ident[:25]:>25} {var:>10.4f} {n_frames:>6}")

    # Overall variance
    overall_var = np.mean([v for _, v, _ in frame_variances])
    print(f"\nMean frame variance: {overall_var:.4f}")

    # Compare to between-identity variance
    all_embs = np.array([identity_embeddings[i] for i in identity_embeddings])
    between_var = all_embs.var(axis=0).mean()
    print(f"Between-identity variance: {between_var:.4f}")
    print(f"Ratio (within/between): {overall_var/between_var:.4f}")

    if overall_var/between_var < 0.1:
        print(f"\n→ Within-identity variance is LOW (10% of between)")
        print(f"→ Embeddings are identity-dominated")
    else:
        print(f"\n→ Within-identity variance is NON-TRIVIAL")
        print(f"→ There is variation within identities")

# ============================================================
# ANALYSIS 5: Preference Prediction with Different Feature Sets
# ============================================================
print(f"\n{'='*60}")
print("[5] Feature Set Ablation")
print(f"{'='*60}")

from sklearn.model_selection import cross_val_score

# Method 1: Full embedding
print("\n[Full Embedding]")
clf_full = LogisticRegression(max_iter=1000, random_state=42)
scores_full = cross_val_score(clf_full, X, y, cv=min(5, len(y)), scoring='accuracy')
print(f"  Accuracy: {scores_full.mean():.2%} ± {scores_full.std():.2%}")

# Method 2: Top MI features only
print("\n[Top MI Features Only]")
top_k = min(50, X.shape[1])
top_indices = np.argsort(mi_scores)[-top_k:]
X_top_mi = X[:, top_indices]
scores_mi = cross_val_score(LogisticRegression(max_iter=1000, random_state=42),
                            X_top_mi, y, cv=min(5, len(y)), scoring='accuracy')
print(f"  Top {top_k} MI features: {scores_mi.mean():.2%} ± {scores_mi.std():.2%}")

# Method 3: PCA components
print("\n[PCA Components]")
n_components = min(10, X.shape[0]-1)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
scores_pca = cross_val_score(LogisticRegression(max_iter=1000, random_state=42),
                              X_pca, y, cv=min(5, len(y)), scoring='accuracy')
print(f"  Top {n_components} PCs: {scores_pca.mean():.2%} ± {scores_pca.std():.2%}")

# Method 6: Cosine similarity only
print("\n[Cosine Similarity Only]")
cos_features = cos_sims.reshape(-1, 1)
scores_cos = cross_val_score(LogisticRegression(max_iter=1000, random_state=42),
                              cos_features, y, cv=min(5, len(y)), scoring='accuracy')
print(f"  Cosine similarity: {scores_cos.mean():.2%} ± {scores_cos.std():.2%}")

# ============================================================
# FINAL INTERPRETATION
# ============================================================
print(f"\n{'='*60}")
print("FINAL INTERPRETATION")
print(f"{'='*60}")

print("""
KEY QUESTIONS:
1. What drives preference? Identity or visual attributes?
2. Can we separate identity signal from preference signal?
""")

# Summarize
print(f"\n[Evidence Summary]")
print(f"  - Top MI dimensions: {top_k} dims account for {mi_scores[top_indices].sum()/mi_scores.sum()*100:.1f}% of MI")
print(f"  - PC1 preference correlation: rho={corr_pc1.statistic:.4f}")

if multi_frame_idents:
    ratio = overall_var / between_var
    print(f"  - Within/between identity variance ratio: {ratio:.4f}")
    if ratio < 0.1:
        print(f"  → Embedding space is IDENTITY-DOMINATED")
    else:
        print(f"  → Embedding space has NON-TRIVIAL within-identity variation")

print(f"\n[Feature Ablation Results]")
print(f"  Full embedding: {scores_full.mean():.2%}")
print(f"  Top MI features: {scores_mi.mean():.2%}")
print(f"  PCA components: {scores_pca.mean():.2%}")
print(f"  Cosine similarity: {scores_cos.mean():.2%}")

# Interpretation
print(f"\n{'='*60}")
print("CONCLUSIONS")
print(f"{'='*60}")

# Determine dominant factor
if scores_full.mean() > scores_cos.mean() + 0.15:
    print("""
→ Preference is driven by FULL IDENTITY FEATURES, not just similarity

→ The embedding captures identity-level attractiveness
→ This is consistent with "type preference" hypothesis
  (people prefer certain TYPES of people, not specific individuals)

→ Next step: Disentangle identity type from individual identity
""")
else:
    print("""
→ Cosine similarity is a reasonable predictor
→ Preference may be driven by visual similarity
→ Consider: "Similarity to preferred type" as signal
""")

print(f"""
CAVEATS:
1. Small dataset (23 pairs) limits conclusions
2. Cannot definitively separate identity from attributes
3. Need counterfactual experiments with controlled stimuli
""")

print(f"{'='*60}")
print("DONE")
print(f"{'='*60}")
