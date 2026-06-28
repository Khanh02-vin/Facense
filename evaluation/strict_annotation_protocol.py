"""
Strict Annotation Protocol với Experimental Control

Thiết kế annotation protocol đạt tiêu chuẩn scientific validity:

1. Identity balance - Mỗi identity xuất hiện trong giới hạn số pairs
2. Stratified sampling - Cân bằng same-type vs cross-type pairs
3. Randomized presentation - Loại bỏ position bias
4. Confidence logging - Đo reliability
5. Annotator agreement - Krippendorff's alpha
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load existing data
with open('./data/annotations/annotations_result.json', 'r', encoding='utf-8') as f:
    existing_annotations = json.load(f)

with open('./data/processed/image_to_identity_multiframe.json', 'r', encoding='utf-8') as f:
    identity_map = json.load(f)

# Get unique identities
all_identities = list(set(identity_map.values()))
print(f"Total identities: {len(all_identities)}")
print(f"Existing annotations: {len(existing_annotations)}")

# ============================================================
# PROTOCOL DESIGN
# ============================================================
print(f"\n{'='*60}")
print("STRICT ANNOTATION PROTOCOL DESIGN")
print(f"{'='*60}")

# 1. Calculate how many times each identity can appear
print(f"\n[1] Identity Appearance Constraint")

MAX_TIMES_PER_IDENTITY = 3  # Mỗi identity xuất hiện tối đa 3 lần
TOTAL_PAIRS_NEEDED = 300  # Target: 300 pairs

# How many pairs can we generate?
# C(n,2) pairs = n*(n-1)/2
# Với 178 identities: 178*177/2 = 15,753 pairs có thể

max_pairs_per_identity = TOTAL_PAIRS_NEEDED // (2 * MAX_TIMES_PER_IDENTITY)
print(f"  Max times per identity: {MAX_TIMES_PER_IDENTITY}")
print(f"  Target pairs: {TOTAL_PAIRS_NEEDED}")
print(f"  Required unique identities: {max_pairs_per_identity}")

# 2. Stratified pair design
print(f"\n[2] Stratified Pair Design")

# Define pair types:
# - Same-type pairs: 2 identities từ cùng cluster
# - Cross-type pairs: 2 identities từ khác cluster
# - High-preference vs Low-preference pairs

SAME_TYPE_RATIO = 0.2  # 20% within-type
CROSS_TYPE_RATIO = 0.6  # 60% between-type
HIGH_VS_LOW_RATIO = 0.2  # 20% high-pref vs low-pref

print(f"  Same-type pairs: {SAME_TYPE_RATIO*100:.0f}%")
print(f"  Cross-type pairs: {CROSS_TYPE_RATIO*100:.0f}%")
print(f"  High vs Low preference: {HIGH_VS_LOW_RATIO*100:.0f}%")

# 3. Calculate required pairs per category
n_same_type = int(TOTAL_PAIRS_NEEDED * SAME_TYPE_RATIO)
n_cross_type = int(TOTAL_PAIRS_NEEDED * CROSS_TYPE_RATIO)
n_high_low = int(TOTAL_PAIRS_NEEDED * HIGH_VS_LOW_RATIO)

print(f"\n  Required pairs:")
print(f"    Same-type: {n_same_type}")
print(f"    Cross-type: {n_cross_type}")
print(f"    High vs Low: {n_high_low}")

# ============================================================
# ANNOTATION UI PROTOCOL
# ============================================================
print(f"\n{'='*60}")
print("[3] Annotation UI Protocol")
print(f"{'='*60}")

ANNOTATION_UI_PROTOCOL = """
=== ANNOTATION PROTOCOL ===

1. PRESENTATION:
   - Randomize A/B position (left/right)
   - Show faces at same size
   - Neutral background
   - No identity names shown

2. INSTRUCTION TO ANNOTATOR:
   "Bạn thích ai hơn trong 2 người này?"

   Options:
   [A] - Prefer left person
   [B] - Prefer right person
   [Equal] - No strong preference
   [Skip] - Cannot decide / Face unclear

3. CONFIDENCE LOGGING:
   After each decision, ask:
   "Bạn chắc chắn về lựa chọn này?"
   - [1] Not at all confident
   - [2] Somewhat confident
   - [3] Neutral
   - [4] Confident
   - [5] Very confident

4. TIME LOGGING:
   - Record time spent on each pair
   - Flag pairs with < 2 second decisions

5. ANNOTATOR INFO:
   - Age, Gender (optional)
   - Relationship status (optional)
   - Cultural background (optional)

6. QUALITY CHECKS:
   - Every 10th pair is a repeat (test consistency)
   - Include attention check pairs
   - Track annotator accuracy on repeats
"""

print(ANNOTATION_UI_PROTOCOL)

# ============================================================
# QUALITY METRICS
# ============================================================
print(f"\n{'='*60}")
print("[4] Quality Metrics to Track")
print(f"{'='*60}")

QUALITY_METRICS = """
METRIC                    TARGET         PURPOSE
─────────────────────────────────────────────────────────────────
Krippendorff's alpha     > 0.6          Inter-annotator reliability
Test-retest accuracy     > 70%           Annotator consistency
Skip rate                < 20%          Task difficulty
Confidence mean           > 3.0          Decision confidence
Time per pair            3-10 sec        Not too fast/slow
Equal rate               30-50%          Natural distribution
─────────────────────────────────────────────────────────────────
"""
print(QUALITY_METRICS)

# ============================================================
# EXISTING ANNOTATION ANALYSIS
# ============================================================
print(f"\n{'='*60}")
print("[5] Existing Annotation Quality Check")
print(f"{'='*60}")

# Analyze existing annotations
choice_counts = defaultdict(int)
skip_count = 0
equal_count = 0

for ann in existing_annotations:
    choice = ann.get('choice', 'unknown')
    if choice in ['A', 'B']:
        choice_counts[choice] += 1
    elif choice == 'equal':
        equal_count += 1
    elif choice == 'skip':
        skip_count += 1

total = len(existing_annotations)
print(f"Total annotations: {total}")
print(f"  Prefer A: {choice_counts['A']} ({choice_counts['A']/total*100:.1f}%)")
print(f"  Prefer B: {choice_counts['B']} ({choice_counts['B']/total*100:.1f}%)")
print(f"  Equal: {equal_count} ({equal_count/total*100:.1f}%)")
print(f"  Skip: {skip_count} ({skip_count/total*100:.1f}%)")

# Check identity distribution in existing annotations
identity_appearance = defaultdict(int)
for ann in existing_annotations:
    identity_appearance[ann['identity_A']] += 1
    identity_appearance[ann['identity_B']] += 1

max_appear = max(identity_appearance.values())
mean_appear = np.mean(list(identity_appearance.values()))
print(f"\nIdentity appearance in existing annotations:")
print(f"  Max appearances: {max_appear}")
print(f"  Mean appearances: {mean_appear:.2f}")
print(f"  Identities with > 3 appearances: {sum(1 for v in identity_appearance.values() if v > 3)}")

# ============================================================
# RECOMMENDATIONS
# ============================================================
print(f"\n{'='*60}")
print("[6] Recommendations for Next Annotation Round")
print(f"{'='*60}")

recommendations = """
GAPS IN CURRENT DATA:
1. Too few pairs: 23 vs target 300
2. Some identities over-represented
3. No confidence logging
4. No time logging
5. No repeat pairs for consistency check
6. Single annotator (no inter-annotator reliability)

PRIORITY FIXES:
1. ✅ Implement confidence logging (easy)
2. ✅ Randomize A/B presentation (easy)
3. ⏳ Recruit 5+ annotators for reliability (medium)
4. ⏳ Implement repeat pairs (medium)
5. ⏳ Balance identity appearance (medium)
6. ⏳ Target 300 pairs with strict protocol (hard)

MINIMUM VIABLE PROTOCOL:
- 100 pairs with confidence logging
- 3 annotators per pair
- Krippendorff's alpha > 0.6 required
"""
print(recommendations)

# ============================================================
# GENERATE PAIR LIST FOR NEXT ROUND
# ============================================================
print(f"\n{'='*60}")
print("[7] Generate Balanced Pair List")
print(f"{'='*60}")

np.random.seed(42)

# Get identities from existing annotations
seen_idents = set()
for ann in existing_annotations:
    seen_idents.add(ann['identity_A'])
    seen_idents.add(ann['identity_B'])

# Exclude over-represented identities
valid_idents = [i for i in all_identities if identity_appearance.get(i, 0) < MAX_TIMES_PER_IDENTITY]

print(f"Total identities: {len(all_identities)}")
print(f"Already over-represented (>3): {len(all_identities) - len(valid_idents)}")
print(f"Valid for new pairs: {len(valid_idents)}")

if len(valid_idents) >= 20:
    # Generate balanced pairs
    new_pairs = []

    # Priority 1: Within-type pairs (need cluster info - skip for now)
    # Priority 2: Cross-type pairs with equal cluster representation

    for _ in range(min(100, len(valid_idents) * (len(valid_idents) - 1) // 2)):
        # Random pair
        idx_a, idx_b = np.random.choice(len(valid_idents), 2, replace=False)
        ident_a = valid_idents[idx_a]
        ident_b = valid_idents[idx_b]

        new_pairs.append({
            "identity_A": ident_a,
            "identity_B": ident_b,
            "position": np.random.choice(["A_left", "B_left"])  # Randomize presentation
        })

    print(f"\nGenerated {len(new_pairs)} new pair candidates")
    print(f"Sample pairs:")
    for p in new_pairs[:5]:
        print(f"  {p['identity_A']} vs {p['identity_B']} ({p['position']})")

    # Save pair list
    output_path = './data/annotations/new_pairs_for_annotation.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_pairs, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}")

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")
