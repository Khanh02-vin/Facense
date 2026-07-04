"""
Auto Label Nhanh - Dùng features đã extract sẵn
Không cần chạy lại video, dùng motion_energy, blur, brightness từ features.json
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict
import sys
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = Path("D:/Project/Face_project/data")
FEATURES_FILE = DATA_DIR / "dataset_processed/features.json"
OUTPUT_FILE = DATA_DIR / "auto_labels_quality.json"


def calculate_preference_score_v2(features: Dict) -> tuple:
    """
    Tính preference score (1-5) và confidence từ features đã có.
    Dựa trên quality metrics + attractiveness heuristics.
    """
    score = 0.0
    factors = []

    # 1. Face visibility (quan trọng) - 25%
    face_vis = features.get('face_visibility', 0)
    face_vis_score = min(face_vis * 5, 2.5)  # 0-2.5 điểm
    score += face_vis_score
    factors.append(("Face visibility", face_vis_score, face_vis))

    # 2. Smile (hấp dẫn) - 20%
    smile = features.get('smile', 0.5)
    smile_score = smile * 1.5  # 0-1.5 điểm
    score += smile_score
    factors.append(("Smile", smile_score, smile))

    # 3. Eye contact (thu hút) - 20%
    eye_contact = features.get('eye_contact', 0.5)
    eye_score = eye_contact * 1.0  # 0-1.0 điểm
    score += eye_score
    factors.append(("Eye contact", eye_score, eye_contact))

    # 4. Face symmetry (đẹp) - 15%
    symmetry = features.get('face_symmetry', 0.5)
    sym_score = symmetry * 0.75  # 0-0.75 điểm
    score += sym_score
    factors.append(("Face symmetry", sym_score, symmetry))

    # 5. Motion energy (tự nhiên) - 10%
    motion = features.get('motion_energy', 25)
    # Tối ưu: không quá ít (ứng chết) không quá nhiều (hỗn loạn)
    if 5 < motion < 50:
        motion_score = 0.5
    elif 1 < motion < 100:
        motion_score = 0.3
    else:
        motion_score = 0.1
    score += motion_score
    factors.append(("Motion energy", motion_score, motion))

    # 6. Blur quality (nét) - 5%
    blur = features.get('blur_score', 100)
    if blur > 100:
        blur_score = 0.25
    elif blur > 50:
        blur_score = 0.15
    else:
        blur_score = 0.05
    score += blur_score
    factors.append(("Blur quality", blur_score, blur))

    # 7. Brightness (ánh sáng) - 5%
    bright = features.get('brightness', 100)
    if 40 < bright < 180:
        bright_score = 0.25
    else:
        bright_score = 0.1
    score += bright_score
    factors.append(("Brightness", bright_score, bright))

    # Normalize to 1-5 scale
    # Max possible: 2.5 + 1.5 + 1.0 + 0.75 + 0.5 + 0.25 + 0.25 = 6.75
    normalized_score = 1 + (score / 6.75) * 4
    normalized_score = max(1, min(5, normalized_score))

    # Confidence dựa trên độ đầy đủ của features
    has_all_features = all([
        features.get('face_visibility', 0) > 0,
        features.get('smile', 0) > 0,
        features.get('face_symmetry', 0) > 0,
    ])
    confidence = 0.8 if has_all_features else 0.5

    return round(normalized_score), confidence, factors


def main():
    print("="*60)
    print("AUTO LABEL (Fast Version - Dùng Features đã có)")
    print("="*60)

    # Load features
    with open(FEATURES_FILE, 'r', encoding='utf-8') as f:
        features_list = json.load(f)

    print(f"Loaded {len(features_list)} clips")

    results = []
    identity_scores = {}

    for feat in features_list:
        clip_id = feat.get('clip_name', feat.get('video_id', ''))
        identity = feat.get('identity', 'unknown')

        score, confidence, factors = calculate_preference_score_v2(feat)

        results.append({
            'clip_id': clip_id,
            'predicted_rating': score,
            'rounded_rating': score,
            'confidence': confidence,
            'identity': identity,
            'auto_labeled': True,
            'method': 'quality_features'
        })

        # Aggregate identity scores
        if identity not in identity_scores:
            identity_scores[identity] = {'scores': [], 'confidences': []}
        identity_scores[identity]['scores'].append(score)
        identity_scores[identity]['confidences'].append(confidence)

    # Calculate identity averages
    identity_avg = {}
    for name, data in identity_scores.items():
        identity_avg[name] = {
            'score': np.mean(data['scores']),
            'confidence': np.mean(data['confidences']),
            'count': len(data['scores'])
        }

    # Sort by score
    sorted_identities = sorted(identity_avg.items(), key=lambda x: -x[1]['score'])

    # Statistics
    ratings = [r['rounded_rating'] for r in results]
    print(f"\n[RESULTS]")
    print(f"  Total clips: {len(results)}")
    print(f"  Unique identities: {len(identity_scores)}")
    print(f"\n  Rating distribution:")
    for r in range(1, 6):
        count = ratings.count(r)
        pct = count / len(ratings) * 100 if ratings else 0
        bar = "█" * int(pct / 5)
        print(f"    {r}: {bar} {count} ({pct:.1f}%)")

    print(f"\n{'='*60}")
    print("TOP 10 IDENTITIES (Được thích nhất):")
    print("="*60)
    for i, (name, data) in enumerate(sorted_identities[:10], 1):
        print(f"  {i:2d}. {name:30s} Score: {data['score']:.2f} ({data['count']} clips)")

    print(f"\n{'='*60}")
    print("BOTTOM 10 IDENTITIES (Ít được thích nhất):")
    print("="*60)
    for i, (name, data) in enumerate(sorted_identities[-10:], 1):
        print(f"  {i:2d}. {name:30s} Score: {data['score']:.2f} ({data['count']} clips)")

    # Save
    output_data = {
        'total': len(results),
        'identity_scores': identity_avg,
        'sorted_identities': [(n, d['score']) for n, d in sorted_identities],
        'predictions': results
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to: {OUTPUT_FILE}")
    print("\n✅ AUTO LABELING COMPLETE!")


if __name__ == "__main__":
    main()
