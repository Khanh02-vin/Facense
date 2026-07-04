#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Liked-Only Analysis Pipeline
=====================================
Dataset: Videos của người bạn THÍCH (không có disliked)
Goal: Tìm điểm chung và preference patterns

Pipeline:
1. Extract frames từ videos
2. Extract face embeddings (SigLIP)
3. Clustering + Visualization
4. Phân tích điểm chung

Run: python run_liked_only_analysis.py
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Config
DATASET_DIR = "D:/Dataset/Face_project_datset"
OUTPUT_DIR = "./data/liked_only_analysis"
FRAMES_DIR = f"{OUTPUT_DIR}/frames"
EMBEDDINGS_DIR = f"{OUTPUT_DIR}/embeddings"
RESULTS_DIR = f"{OUTPUT_DIR}/results"


def print_header():
    print("\n" + "=" * 70)
    print("[*] LIKED-ONLY PREFERENCE ANALYSIS PIPELINE [*]")
    print("=" * 70)
    print(f"\nDataset: {DATASET_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Time:    {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()


def check_dependencies():
    """Check required packages."""
    print("[1/5] Checking dependencies...")
    required = ["cv2", "numpy", "sklearn", "torch", "transformers"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg if pkg != "cv2" else "cv2")
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} - MISSING")
            missing.append(pkg)

    if missing:
        print(f"\n[ERROR] Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False

    print("  ✓ All dependencies OK!")
    return True


def count_dataset():
    """Count videos in dataset."""
    print("\n[2/5] Analyzing dataset...")

    dataset_path = Path(DATASET_DIR)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {DATASET_DIR}")
        return None

    folders = [d for d in dataset_path.iterdir() if d.is_dir() and d.name != "temp_frames"]
    total_videos = 0
    video_counts = []

    for folder in sorted(folders):
        videos = list(folder.glob("*.mp4"))
        video_counts.append((folder.name, len(videos)))
        total_videos += len(videos)

    print(f"  Total identities: {len(folders)}")
    print(f"  Total videos: {total_videos}")

    if len(folders) < 10:
        print("  [WARNING] Dataset too small for meaningful analysis")

    # Save summary
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {
        "n_identities": len(folders),
        "n_videos": total_videos,
        "top_10_by_videos": sorted(video_counts, key=lambda x: -x[1])[:10]
    }

    with open(f"{OUTPUT_DIR}/dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def extract_frames():
    """Extract frames from videos."""
    print("\n[3/5] Extracting frames...")

    os.makedirs(FRAMES_DIR, exist_ok=True)

    dataset_path = Path(DATASET_DIR)
    folders = [d for d in dataset_path.iterdir() if d.is_dir() and d.name != "temp_frames"]

    import cv2

    total_extracted = 0
    failed = []

    for i, folder in enumerate(sorted(folders)):
        videos = list(folder.glob("*.mp4"))

        if not videos:
            continue

        # Output folder for this identity
        output_folder = Path(FRAMES_DIR) / folder.name
        os.makedirs(output_folder, exist_ok=True)

        # Sample 5 frames from first video (or all videos if few)
        n_frames_to_extract = min(5, len(videos) * 2)

        for video in videos[:3]:  # Max 3 videos per person
            cap = cv2.VideoCapture(str(video))

            if not cap.isOpened():
                failed.append(video.name)
                continue

            # Get video length
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Sample frames evenly
            n_samples = min(3, total_frames)
            frame_indices = [int(total_frames * j / n_samples) for j in range(n_samples)]

            for j, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    # Save frame
                    output_path = output_folder / f"{video.stem}_{j:03d}.jpg"
                    cv2.imwrite(str(output_path), frame)
                    total_extracted += 1

            cap.release()

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(folders)} identities ({total_extracted} frames)")

    print(f"  ✓ Extracted {total_extracted} frames")
    if failed:
        print(f"  ✗ Failed: {len(failed)} videos")

    return total_extracted


def extract_embeddings():
    """Extract embeddings from frames using SigLIP."""
    print("\n[4/5] Extracting embeddings (SigLIP)...")

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    try:
        from transformers import AutoProcessor, AutoModel
        import torch
        from PIL import Image
        import numpy as np
    except ImportError as e:
        print(f"  [ERROR] Missing: {e}")
        print("  Install: pip install transformers torch")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load model
    print("  Loading SigLIP model...")
    try:
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        return None

    # Extract embeddings
    frames_path = Path(FRAMES_DIR)
    embeddings = {}
    identities = {}

    frame_files = list(frames_path.glob("*/*.jpg"))
    print(f"  Processing {len(frame_files)} frames...")

    for i, frame_path in enumerate(frame_files):
        identity = frame_path.parent.name

        try:
            image = Image.open(frame_path).convert("RGB")

            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model.get_image_features(**inputs)
                embedding = outputs.cpu().numpy().flatten()

            # Normalize
            embedding = embedding / np.linalg.norm(embedding)

            frame_id = f"{identity}/{frame_path.name}"
            embeddings[frame_id] = embedding.tolist()
            identities[frame_id] = identity

        except Exception as e:
            print(f"  [WARN] Failed: {frame_path.name}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(frame_files)} frames")

    # Save
    output_file = f"{EMBEDDINGS_DIR}/embeddings.json"
    with open(output_file, "w") as f:
        json.dump({
            "embeddings": embeddings,
            "identities": identities
        }, f)

    print(f"  ✓ Extracted {len(embeddings)} embeddings")
    return embeddings


def run_clustering_analysis():
    """Run clustering and analysis."""
    print("\n[5/5] Running clustering analysis...")

    try:
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import json
    except ImportError:
        print("  [ERROR] sklearn required")
        return None

    # Load embeddings
    emb_file = f"{EMBEDDINGS_DIR}/embeddings.json"
    if not os.path.exists(emb_file):
        print("  [ERROR] Embeddings not found. Run extraction first.")
        return None

    with open(emb_file) as f:
        data = json.load(f)

    embeddings = data["embeddings"]
    identities = data["identities"]

    # Convert to array
    ids = list(embeddings.keys())
    X = np.array([embeddings[i] for i in ids])

    print(f"  Dataset: {X.shape}")

    # PCA for visualization
    pca = PCA(n_components=min(50, X.shape[0], X.shape[1]))
    X_pca = pca.fit_transform(X)
    print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # K-Means clustering
    n_clusters = min(8, len(set(identities.values())) // 3)
    print(f"  Clustering into {n_clusters} groups...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    # Group by identity
    identity_to_cluster = {}
    for frame_id, label in zip(ids, labels):
        identity = identities[frame_id]
        if identity not in identity_to_cluster:
            identity_to_cluster[identity] = []
        identity_to_cluster[identity].append(label)

    # Majority vote
    from collections import Counter
    identity_clusters = {}
    for identity, cluster_labels in identity_to_cluster.items():
        most_common = Counter(cluster_labels).most_common(1)[0][0]
        identity_clusters[identity] = most_common

    # Cluster summary
    cluster_summary = {}
    for cluster_id in range(n_clusters):
        members = [i for i, c in identity_clusters.items() if c == cluster_id]
        cluster_summary[f"cluster_{cluster_id}"] = {
            "n_members": len(members),
            "members": members[:10]  # Top 10
        }

    # Find common traits
    print("\n  Analyzing patterns...")

    # Group identities by nationality/ethnicity
    asian_keywords = ["Kim", "Lee", "Park", " Choi", "Han", "Jang", "Bae", "Aoi", "Aragaki",
                      "Arimura", "Asuka", "Hamabe", "Hirose", "Komatsu", "Miyazawa",
                      "Toda", "Tsuyu", "Yoshitaka", "Điền", "Tưởng", "Triệu", "Trương",
                      "Vương", "Tô", "Ân", "Lý", "Lưu", "Dương", "Cao", "Hoài", "Trịnh",
                      "Băng", "Chu", "Diệp", "Hữu", "Lam", "Bảo", "Thượng", "Cúc", "Cổ",
                      "Đại", "Đồng", "Địch", "Mai", "Mạnh", "Mao", "Matsuzaka", "Nagasawa",
                      "Nagano", "Yui", "Wakana", "Ikeda", "Imada", "Imoto", "Ikuta", "Fujiyoshi"]

    western_keywords = ["Anne", "Alexandra", "Ana", "Blake", "Charlize", "Catherine",
                        "Diana", "Elle", "Emma", "Gisele", "Grace", "Jennifer", "Keira",
                        "Lily", "Margot", "Marion", "Megan", "Michelle", "Monica",
                        "Natalie", "Scarlett", "Taylor", "Uma", "Winona", "Heidi",
                        "Kylie", "Miriam", "Lynne", "Tina", "Catherine", "Kristen"]

    asian_count = sum(1 for i in identity_clusters if any(k in i for k in asian_keywords))
    western_count = len(identity_clusters) - asian_count

    results = {
        "n_identities": len(identity_clusters),
        "n_clusters": n_clusters,
        "cluster_summary": cluster_summary,
        "region_breakdown": {
            "east_asian_approx": asian_count,
            "western_approx": western_count
        },
        "identity_clusters": identity_clusters,
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum())
    }

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/clustering_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Analyzed {len(identity_clusters)} identities into {n_clusters} clusters")
    print(f"\n  Region breakdown:")
    print(f"    East Asian: {asian_count}")
    print(f"    Western: {western_count}")

    return results


def print_summary():
    """Print final summary."""
    print("\n" + "=" * 70)
    print("[*] ANALYSIS COMPLETE [*]")
    print("=" * 70)

    results_file = f"{RESULTS_DIR}/clustering_results.json"
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)

        print(f"\n📊 Dataset Summary:")
        print(f"   Identities: {results['n_identities']}")
        print(f"   Clusters: {results['n_clusters']}")
        print(f"   PCA Variance: {results['pca_variance_explained']:.1%}")

        print(f"\n📁 Region Breakdown (approximate):")
        for region, count in results["region_breakdown"].items():
            pct = count / results["n_identities"] * 100
            print(f"   {region}: {count} ({pct:.0f}%)")

        print(f"\n📁 Output files:")
        print(f"   {RESULTS_DIR}/clustering_results.json")

    print("\n✅ Next steps:")
    print("   1. Review clustering results")
    print("   2. For deeper analysis, add 'disliked' dataset")
    print("   3. Run trait-based analysis with universal_taste_engine")


def main():
    print_header()

    # Step 1: Check dependencies
    if not check_dependencies():
        return

    # Step 2: Count dataset
    summary = count_dataset()
    if not summary:
        return

    # Step 3: Extract frames
    n_frames = extract_frames()
    if n_frames == 0:
        print("[ERROR] No frames extracted")
        return

    # Step 4: Extract embeddings
    embeddings = extract_embeddings()
    if not embeddings:
        print("[WARNING] Embedding extraction failed. Running simplified analysis...")
        # Fallback to histogram features
        print("  Skipping embeddings. Use existing frames for analysis.")

    # Step 5: Clustering
    results = run_clustering_analysis()

    # Print summary
    print_summary()


if __name__ == "__main__":
    main()
