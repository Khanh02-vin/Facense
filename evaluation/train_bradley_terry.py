"""
Train Bradley-Terry Model with Real Preferences

Uses the 23 preference edges from human annotations to fit Bradley-Terry model.
"""

import json
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load annotations
with open('./data/annotations/annotations_result.json', 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# Load SigLIP embeddings
embeddings_file = './data/processed/embeddings_siglip_multiframe.npy'
identity_file = './data/processed/image_to_identity_multiframe.json'

embeddings = np.load(embeddings_file)
if embeddings.ndim == 4:
    embeddings = embeddings.mean(axis=2).squeeze(1)

# Normalize
norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
embeddings = embeddings / norms

with open(identity_file, 'r', encoding='utf-8') as f:
    identity_map = json.load(f)

# Build identity -> index mapping (first occurrence)
identity_to_idx = {}
idx_to_identity = {}
for idx, ident in identity_map.items():
    idx_int = int(idx)
    if ident not in identity_to_idx:
        identity_to_idx[ident] = idx_int
        idx_to_identity[idx_int] = ident

print("="*50)
print("BRADLEY-TERRY MODEL TRAINING")
print("="*50)

# Build preference pairs from annotations
pairs = []
for a in annotations:
    if a['choice'] == 'A':
        winner = a['identity_A']
        loser = a['identity_B']
    elif a['choice'] == 'B':
        winner = a['identity_B']
        loser = a['identity_A']
    else:
        continue  # Skip equal

    if winner in identity_to_idx and loser in identity_to_idx:
        pairs.append((identity_to_idx[winner], identity_to_idx[loser]))

print("\n[1] Data Summary")
print("    Preference pairs: %d" % len(pairs))
print("    Unique identities: %d" % len(identity_to_idx))

# Get unique identities in pairs
unique_ids = set()
for w, l in pairs:
    unique_ids.add(w)
    unique_ids.add(l)
print("    Identities in pairs: %d" % len(unique_ids))

# Bradley-Terry Negative Log-Likelihood
def nll(theta, n_items, pairs, wins, losses):
    """Negative log-likelihood for Bradley-Terry."""
    theta = np.exp(theta)  # Ensure positive strengths
    total = 0.0
    for i, j in pairs:
        if i >= n_items or j >= n_items:
            continue
        prob = theta[i] / (theta[i] + theta[j])
        total -= np.log(prob + 1e-10)
    return total

# Prepare data
n_items = len(embeddings)
theta_init = np.ones(n_items)

# Build counts: wins[i] = times i beat j, losses[i] = times i lost to j
wins = defaultdict(int)
losses = defaultdict(int)
pair_counts = defaultdict(int)

for w, l in pairs:
    pair_counts[(w, l)] += 1
    wins[w] += 1
    losses[l] += 1

print("\n[2] Pair Counts")
print("    Total unique pairs: %d" % len(pair_counts))
for (w, l), count in sorted(pair_counts.items(), key=lambda x: -x[1])[:5]:
    print("    %s > %s: %d" % (idx_to_identity[w][:15], idx_to_identity[l][:15], count))

# Fit Bradley-Terry using gradient descent
print("\n[3] Fitting Bradley-Terry Model")

def objective(theta):
    theta = np.exp(theta)
    total = 0.0
    for w, l in pairs:
        prob = theta[w] / (theta[w] + theta[l] + 1e-10)
        total -= np.log(prob + 1e-10)
    # Add L2 regularization
    total += 0.001 * np.sum(theta**2)
    return total

def gradient(theta):
    theta = np.exp(theta)
    grad = np.zeros_like(theta)
    for w, l in pairs:
        denom = theta[w] + theta[l] + 1e-10
        grad[w] -= 1/theta[w] * (1 - theta[w]/denom)
        grad[l] -= 1/theta[l] * (-theta[l]/denom)
    grad += 0.002 * theta
    return grad * np.exp(theta)

# Optimize
result = minimize(
    objective,
    np.zeros(n_items),
    method='L-BFGS-B',
    jac=gradient,
    options={'maxiter': 1000, 'disp': False}
)

theta_hat = np.exp(result.x)

print("    Optimization converged: %s" % result.success)
print("    Final NLL: %.4f" % result.fun)

# Rank by strength
strengths = []
for i in range(n_items):
    if i in unique_ids:
        strengths.append((idx_to_identity[i], theta_hat[i], wins[i], losses[i]))

strengths.sort(key=lambda x: -x[1])

print("\n[4] Top 20 by Bradley-Terry Strength")
print("    %-25s %8s %4s %4s" % ("Identity", "Strength", "W", "L"))
print("    " + "-"*45)
for name, strength, w, l in strengths[:20]:
    print("    %-25s %8.3f %4d %4d" % (name[:25], strength, w, l))

# Evaluate: predict held-out pairs
print("\n[5] Model Evaluation")

# Split pairs for train/test
np.random.seed(42)
indices = np.random.permutation(len(pairs))
train_size = int(0.7 * len(pairs))
train_pairs = [pairs[i] for i in indices[:train_size]]
test_pairs = [pairs[i] for i in indices[train_size:]]

print("    Train pairs: %d" % len(train_pairs))
print("    Test pairs: %d" % len(test_pairs))

# Retrain on train only
def objective_train(theta):
    theta = np.exp(theta)
    total = 0.0
    for w, l in train_pairs:
        prob = theta[w] / (theta[w] + theta[l] + 1e-10)
        total -= np.log(prob + 1e-10)
    return total

result_train = minimize(objective_train, np.zeros(n_items), method='L-BFGS-B')
theta_train = np.exp(result_train.x)

# Predict test pairs
correct = 0
for w, l in test_pairs:
    if theta_train[w] > theta_train[l]:
        correct += 1

if len(test_pairs) > 0:
    accuracy = correct / len(test_pairs)
    print("    Test accuracy: %.1f%%" % (100 * accuracy))
else:
    print("    Not enough pairs for train/test split")

# Save model
print("\n[6] Saving Model")
model_data = {
    'theta': theta_hat.tolist(),
    'n_items': n_items,
    'n_pairs': len(pairs),
    'idx_to_identity': idx_to_identity,
    'rankings': [
        {'identity': name, 'strength': float(strength), 'wins': w, 'losses': l}
        for name, strength, w, l in strengths[:50]
    ]
}

with open('./data/annotations/bradley_terry_model.json', 'w', encoding='utf-8') as f:
    json.dump(model_data, f, ensure_ascii=False, indent=2)

print("    Saved to: data/annotations/bradley_terry_model.json")

# Save embeddings for ranked identities
ranked_indices = [identity_to_idx[s[0]] for s in strengths[:50] if s[0] in identity_to_idx]
ranked_embeddings = embeddings[ranked_indices]

np.save('./data/annotations/top_preferred_embeddings.npy', ranked_embeddings)
print("    Saved embeddings: data/annotations/top_preferred_embeddings.npy")

print("\n" + "="*50)
print("DONE!")
print("="*50)
