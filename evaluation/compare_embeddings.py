"""
Embedding Comparison Script

Compares different feature extraction methods for face images:
1. Color Histogram + HOG (baseline - already done)
2. SIFT/ORB descriptors
3. LBP (Local Binary Patterns)
4. Color moments
5. Edge histograms

These don't require deep learning and can run on CPU.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rsa import (
    RSAComparator, EmbeddingSimilarityMatrix, RSAResult,
    generate_synthetic_human_similarity, SimilarityMatrixBuilder
)


def extract_color_histogram(image_path, bins=32):
    """Extract color histogram features."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        img = cv2.resize(img, (224, 224))
        features = []

        # RGB histograms
        for channel in range(3):
            hist = cv2.calcHist([img], [channel], None, [bins], [0, 256])
            hist = hist.flatten() / hist.sum()
            features.extend(hist)

        # HSV histograms
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for channel in range(3):
            if channel == 0:  # Hue
                hist = cv2.calcHist([hsv], [channel], None, [bins], [0, 180])
            else:
                hist = cv2.calcHist([hsv], [channel], None, [bins], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-10)
            features.extend(hist)

        return np.array(features, dtype=np.float32)
    except Exception as e:
        return None


def extract_hog_features(image_path, pixels_per_cell=16, cells_per_block=2, bins=9):
    """Extract HOG features."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        # HOG parameters
        win_size = (128, 128)
        block_size = (cells_per_block * pixels_per_cell, cells_per_block * pixels_per_cell)
        block_stride = (pixels_per_cell, pixels_per_cell)
        cell_size = (pixels_per_cell, pixels_per_cell)
        nbins = bins
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

        features = hog.compute(gray)
        return features.flatten().astype(np.float32)
    except Exception as e:
        return None


def extract_lbp_features(image_path, radius=3, n_points=24, n_bins=26):
    """Extract Local Binary Pattern features."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        # Simple LBP implementation
        rows, cols = gray.shape
        lbp = np.zeros((rows - 2 * radius, cols - 2 * radius))

        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = gray[i, j]
                code = 0
                code |= (gray[i - radius, j - radius] > center) << 7
                code |= (gray[i - radius, j] > center) << 6
                code |= (gray[i - radius, j + radius] > center) << 5
                code |= (gray[i, j + radius] > center) << 4
                code |= (gray[i + radius, j + radius] > center) << 3
                code |= (gray[i + radius, j] > center) << 2
                code |= (gray[i + radius, j - radius] > center) << 1
                code |= (gray[i, j - radius] > center) << 0
                lbp[i - radius, j - radius] = code

        # Histogram of LBP
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-10)

        return hist
    except Exception as e:
        return None


def extract_sift_descriptors(image_path, n_keypoints=100):
    """Extract SIFT descriptors (aggregated as histogram)."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))

        sift = cv2.SIFT_create(nfeatures=n_keypoints)
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) == 0:
            # Return zero vector if no keypoints
            return np.zeros(128, dtype=np.float32)

        # Aggregate descriptors: mean pooling
        mean_desc = np.mean(descriptors, axis=0)
        return mean_desc.astype(np.float32)
    except Exception as e:
        return np.zeros(128, dtype=np.float32)


def extract_color_moments(image_path):
    """Extract color moments (mean, std, skewness for each channel)."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        img = cv2.resize(img, (224, 224))
        features = []

        for channel in range(3):
            channel_data = img[:, :, channel].astype(float)

            # Mean
            mean = np.mean(channel_data)
            # Standard deviation
            std = np.std(channel_data)
            # Skewness
            skewness = np.mean(((channel_data - mean) / (std + 1e-10)) ** 3)

            features.extend([mean / 255.0, std / 255.0, skewness])

        return np.array(features, dtype=np.float32)
    except Exception as e:
        return None


def extract_edge_histogram(image_path, bins=16):
    """Extract edge orientation histogram."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        # Sobel edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Edge magnitude and orientation
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        orientation = np.arctan2(sobely, sobelx)

        # Histogram of orientations weighted by magnitude
        hist, _ = np.histogram(orientation[magnitude > 20], bins=bins, range=(-np.pi, np.pi),
                                weights=magnitude[magnitude > 20])

        # Normalize
        hist = hist.astype(np.float32) / (hist.sum() + 1e-10)

        return hist
    except Exception as e:
        return None


def extract_gabor_features(image_path, num_theta=8, num_sigma=4):
    """Extract Gabor filter responses."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        features = []
        for theta in np.linspace(0, np.pi, num_theta, endpoint=False):
            for sigma in [2, 4, 8]:
                kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                features.extend([np.mean(filtered), np.std(filtered)])

        return np.array(features, dtype=np.float32)
    except Exception as e:
        return None


def extract_all_features(image_path):
    """Extract all features for an image."""
    features = {}

    # Color histogram (already done in baseline)
    color_hist = extract_color_histogram(image_path)
    if color_hist is not None:
        features['color_histogram'] = color_hist

    # HOG
    hog = extract_hog_features(image_path)
    if hog is not None:
        features['hog'] = hog

    # LBP
    lbp = extract_lbp_features(image_path)
    if lbp is not None:
        features['lbp'] = lbp

    # SIFT
    sift = extract_sift_descriptors(image_path)
    if sift is not None:
        features['sift'] = sift

    # Color moments
    color_moments = extract_color_moments(image_path)
    if color_moments is not None:
        features['color_moments'] = color_moments

    # Edge histogram
    edge = extract_edge_histogram(image_path)
    if edge is not None:
        features['edge_histogram'] = edge

    # Gabor
    gabor = extract_gabor_features(image_path)
    if gabor is not None:
        features['gabor'] = gabor

    return features


def concatenate_features(features_list):
    """Concatenate all features into a single vector."""
    all_features = []
    for f in features_list.values():
        all_features.extend(f)
    return np.array(all_features, dtype=np.float32)


def extract_frames_from_video(video_path, n_frames=5):
    """Extract frames from video file."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return []

        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()
        return frames
    except Exception as e:
        return []


def load_frames_and_extract(frames_dir, max_frames_per_identity=2):
    """Load frames and extract all features from video dataset."""
    frames_dir = Path(frames_dir)

    all_image_ids = []
    all_features = {}
    image_to_identity = {}

    identity_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]

    for identity_dir in identity_dirs:
        identity = identity_dir.name
        video_files = list(identity_dir.glob("*.mp4"))[:1]  # One video per identity

        for video_file in video_files:
            frames = extract_frames_from_video(str(video_file), n_frames=max_frames_per_identity)

            for i, frame in enumerate(frames):
                # Save temporary frame
                temp_path = frames_dir / "temp_frames" / f"{identity}_{i}.jpg"
                temp_path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(temp_path), frame)

                # Extract features
                features = extract_all_features(str(temp_path))
                if features:
                    image_id = f"{identity}_{i}"
                    all_image_ids.append(image_id)
                    all_features[image_id] = features
                    image_to_identity[image_id] = identity

                # Clean up temp
                temp_path.unlink(missing_ok=True)

    return all_image_ids, all_features, image_to_identity


def compute_rsa_for_features(
    all_features,
    image_ids,
    true_correlation=0.65,
    annotator_noise=0.25,
    seed=42
):
    """Compute RSA for each feature type."""
    n_images = len(image_ids)

    # Get embedding matrix for each feature
    feature_matrices = {}

    for feature_name in all_features[image_ids[0]].keys():
        # Get feature vectors
        vectors = []
        valid_ids = []

        for img_id in image_ids:
            if feature_name in all_features[img_id]:
                vectors.append(all_features[img_id][feature_name])
                valid_ids.append(img_id)

        if len(vectors) < 10:
            continue

        # Stack and normalize
        matrix = np.array(vectors)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)

        # L2 normalize each vector
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        matrix = matrix / norms

        feature_matrices[feature_name] = (matrix, valid_ids)

    # Generate human similarity judgments
    np.random.seed(seed)
    human_similarity = generate_synthetic_human_similarity(
        n_images=n_images,
        noise_level=annotator_noise
    )

    # Compute RSA for each feature
    results = {}
    comparator = RSAComparator()

    for feature_name, (matrix, valid_ids) in feature_matrices.items():
        # Compute pairwise similarity
        sim_matrix = matrix @ matrix.T

        # RSA comparison
        result = comparator.compare(human_similarity[:len(valid_ids), :len(valid_ids)],
                                   sim_matrix)

        results[feature_name] = {
            'spearman_rho': float(result.spearman_rho),
            'spearman_p': float(result.spearman_p),
            'pearson_r': float(result.pearson_r),
            'n_images': len(valid_ids)
        }

    return results


def run_embedding_comparison(
    frames_dir='./data/processed/frames',
    output_dir='./results',
    max_images=100,
    true_correlation=0.65
):
    """Run complete embedding comparison."""
    print("=" * 70)
    print("Embedding Comparison for Face Similarity")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Load frames and extract features
    print("\n[1] Loading frames and extracting features...")
    image_ids, all_features, image_to_identity = load_frames_and_extract(
        frames_dir,
        max_frames_per_identity=1
    )

    print(f"    Loaded {len(image_ids)} images")

    if len(image_ids) == 0:
        print("[ERROR] No images found!")
        return

    # Sample images
    if len(image_ids) > max_images:
        np.random.seed(42)
        image_ids = list(np.random.choice(image_ids, max_images, replace=False))

    print(f"    Using {len(image_ids)} images for comparison")

    # Show available features
    sample_features = all_features[image_ids[0]]
    print(f"\n    Available features:")
    for name, vec in sample_features.items():
        print(f"      - {name}: {len(vec)} dims")

    # Compute RSA for each feature
    print("\n[2] Computing RSA for each feature type...")
    results = compute_rsa_for_features(
        all_features,
        image_ids,
        true_correlation=true_correlation
    )

    # Display results
    print("\n" + "=" * 70)
    print("RSA Results by Feature Type")
    print("=" * 70)
    print()
    print(f"{'Feature':<25} {'rho':>10} {'p-value':>12} {'n_images':>10}")
    print("-" * 60)

    # Sort by rho
    sorted_results = sorted(results.items(), key=lambda x: x[1]['spearman_rho'], reverse=True)

    for name, result in sorted_results:
        sig = "*" if result['spearman_p'] < 0.05 else ""
        print(f"{name:<25} {result['spearman_rho']:>10.4f} {result['spearman_p']:>10.4f} {sig:<2} {result['n_images']:>8}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    best_feature = sorted_results[0]
    best_name = best_feature[0]
    best_rho = best_feature[1]['spearman_rho']

    print(f"\nBest feature: {best_name}")
    print(f"RSA rho: {best_rho:.4f}")

    if best_rho > 0.5:
        print("\n[VALIDATED] This feature shows strong alignment with human perception.")
    elif best_rho > 0.3:
        print("\n[PARTIAL] This feature shows moderate alignment.")
    else:
        print("\n[WEAK] This feature shows weak alignment.")

    # Recommendation
    print("\n" + "-" * 70)
    print("Recommendations:")

    if best_rho < 0.3:
        print("""
1. Low-level features (histogram, HOG, LBP) are insufficient
2. Need semantic features that capture:
   - Face structure
   - Expression
   - Semantic similarity
3. Consider:
   - Pre-trained deep models (ResNet, CLIP, DINOv2)
   - Face-specific models (ArcFace, FaceNet)
   - Fine-tuning on perceptual similarity tasks
""")

    # Save results
    print("\n[3] Saving results...")
    output_file = os.path.join(output_dir, "embedding_comparison_results.json")

    output = {
        "experiment": {
            "n_images": len(image_ids),
            "true_correlation": true_correlation,
            "features_tested": list(results.keys())
        },
        "results": results,
        "sorted_results": [
            {"feature": name, **result}
            for name, result in sorted_results
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"    Saved to: {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare embedding methods")
    parser.add_argument("--frames_dir", type=str, default="./data/processed/frames")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--max_images", type=int, default=100)
    parser.add_argument("--true_corr", type=float, default=0.65)

    args = parser.parse_args()

    run_embedding_comparison(
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
        true_correlation=args.true_corr
    )
