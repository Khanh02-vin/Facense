# -*- coding: utf-8 -*-
import json
import sys
from collections import Counter, defaultdict

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

with open('./data/annotations/annotations_result.json', 'r', encoding='utf-8') as f:
    annotations = json.load(f)

print('PAIRWISE ANNOTATION ANALYSIS')
print('='*40)

# Basic stats
valid = [a for a in annotations if a['choice'] in ['A', 'B', 'equal']]
skipped = [a for a in annotations if a['choice'] == 'skip']

print('\n[1] Basic Stats')
print('Total: %d, Valid: %d, Skipped: %d' % (len(annotations), len(valid), len(skipped)))

# Choice distribution
choices = Counter(a['choice'] for a in valid)
print('\n[2] Choice Distribution')
for choice, count in sorted(choices.items()):
    pct = 100 * count / len(valid)
    print('  %s: %d (%.1f%%)' % (choice, count, pct))

# Same vs different
same = [a for a in valid if a['same_identity']]
diff = [a for a in valid if not a['same_identity']]
print('\n[3] Same vs Different')
print('Same: %d, Different: %d' % (len(same), len(diff)))

# Build preference edges
edges = []
for a in annotations:
    if a['choice'] == 'A':
        edges.append((a['identity_A'], a['identity_B']))
    elif a['choice'] == 'B':
        edges.append((a['identity_B'], a['identity_A']))

print('\n[4] Preference Edges: %d' % len(edges))

# Win counts
wins = Counter()
losses = Counter()
for winner, loser in edges:
    wins[winner] += 1
    losses[loser] += 1

all_ids = set(wins.keys()) | set(losses.keys())
print('Unique identities: %d' % len(all_ids))

# Simple ranking
rankings = []
for identity in all_ids:
    total = wins[identity] + losses[identity]
    if total > 0:
        win_rate = wins[identity] / total
        rankings.append({
            'name': identity,
            'wins': wins[identity],
            'losses': losses[identity],
            'total': total,
            'win_rate': win_rate
        })

rankings.sort(key=lambda x: -x['win_rate'])

print('\n[5] Top Rankings')
for i, r in enumerate(rankings[:10]):
    print('  %d. %s' % (i+1, r['name'][:20]))

# Cycle analysis
print('\n[6] Cycle Analysis')
adj = defaultdict(set)
for winner, loser in edges:
    adj[winner].add(loser)

contradictions = [(w, l) for w, l in edges if w in adj[l]]
print('Contradictions: %d' % len(contradictions))

three_cycles = []
for a, b in edges:
    for c in adj[b]:
        if a in adj[c]:
            three_cycles.append((a, b, c))

print('3-cycles: %d' % len(three_cycles))

if not contradictions and not three_cycles:
    print('\n==> Preferences are TRANSITIVE - Good for Bradley-Terry!')
else:
    print('\n==> Some inconsistencies found')

# Save results
results = {
    'n_annotations': len(annotations),
    'n_valid': len(valid),
    'n_skipped': len(skipped),
    'choice_distribution': dict(choices),
    'n_edges': len(edges),
    'n_identities': len(all_ids),
    'rankings': rankings[:20],
    'edges': edges,
    'contradictions': contradictions,
    'three_cycles': three_cycles
}

with open('./data/annotations/analysis_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print('\nResults saved!')
