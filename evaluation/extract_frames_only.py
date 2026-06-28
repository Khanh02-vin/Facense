"""
Generate embeddings without PyTorch - using histogram-based features
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


def extract_histogram_features(image_path, bins=32):
    """Extract color histogram features from image."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)

        features = []

        # Color histograms for each channel
        for channel in range(3):
            hist, _ = np.histogram(img_array[:, :, channel], bins=bins, range=(0, 256))
            hist = hist.astype(np.float32) / hist.sum()  # Normalize
            features.extend(hist)

        # Spatial color distribution (divide image into 4 quadrants)
        h, w = img_array.shape[:2]
        quadrants = [
            img_array[:h//2, :w//2],
            img_array[:h//2, w//2:],
            img_array[h//2:, :w//2],
            img_array[h//2:, w//2:]
        ]

        for quad in quadrants:
            for channel in range(3):
                mean_val = np.mean(quad[:, :, channel])
                std_val = np.std(quad[:, :, channel])
                features.extend([mean_val / 255.0, std_val / 255.0])

        return np.array(features, dtype=np.float32)
    except Exception as e:
        print(f"[WARN] Failed to extract features from {image_path}: {e}")
        return None


def extract_hog_features(image_path, pixels_per_cell=32, cells_per_block=2, bins=9):
    """Extract HOG-like features without skimage."""
    try:
        from skimage import color, feature
        from skimage.transform import resize

        img = Image.open(image_path).convert("RGB")
        img = img.resize((128, 128))
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Convert to grayscale
        gray = color.rgb2gray(img_array)

        # Extract HOG
        hog_features = feature.hog(
            gray,
            orientations=bins,
            pixels_per_cell=(pixels_per_cell, pixels_per_cell),
            cells_per_block=(cells_per_block, cells_per_block),
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )

        return hog_features.astype(np.float32)
    except ImportError:
        # Fallback to simple edge features
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((64, 64))
            img_array = np.array(img, dtype=np.float32) / 255.0

            # Simple edge detection using gradient
            gray = np.mean(img_array, axis=2)
            dx = np.diff(gray, axis=1)
            dy = np.diff(gray, axis=0)

            # Histogram of gradients
            hist_dx, _ = np.histogram(dx.flatten(), bins=16, range=(-1, 1))
            hist_dy, _ = np.histogram(dy.flatten(), bins=16, range=(-1, 1))

            features = np.concatenate([
                hist_dx.astype(np.float32) / (hist_dx.sum() + 1e-10),
                hist_dy.astype(np.float32) / (hist_dy.sum() + 1e-10)
            ])

            return features
        except:
            return None
    except Exception as e:
        return None


def extract_combined_features(image_path):
    """Extract combined color and texture features."""
    color_feats = extract_histogram_features(image_path, bins=32)
    texture_feats = extract_hog_features(image_path)

    if color_feats is None and texture_feats is None:
        return None

    if color_feats is None:
        return texture_feats
    if texture_feats is None:
        return color_feats

    return np.concatenate([color_feats, texture_feats])


def generate_embeddings_from_frames(frames_dir, output_dir, batch_size=32):
    """Generate embeddings from extracted frames."""
    print("[INFO] Scanning for frames...")
    frames_dir = Path(frames_dir)

    all_frames = []
    identity_mapping = {}
    image_id_to_path = {}

    # Collect all frame paths
    for identity_dir in frames_dir.iterdir():
        if not identity_dir.is_dir():
            continue

        identity_name = identity_dir.name
        for frame_path in identity_dir.glob("*.jpg"):
            image_id = frame_path.stem
            all_frames.append(frame_path)
            identity_mapping[image_id] = identity_name
            image_id_to_path[image_id] = str(frame_path)

    print(f"[INFO] Found {len(all_frames)} frames from {len(set(identity_mapping.values()))} identities")

    if not all_frames:
        print("[WARN] No frames found!")
        return

    # Extract features
    print("[INFO] Extracting features...")
    features = []
    valid_ids = []
    valid_paths = []

    for i, frame_path in enumerate(all_frames):
        feat = extract_combined_features(frame_path)
        if feat is not None:
            features.append(feat)
            valid_ids.append(frame_path.stem)
            valid_paths.append(str(frame_path))

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_frames)} frames")

    if not features:
        print("[ERROR] No valid features extracted!")
        return

    print(f"[INFO] Extracted features from {len(features)} frames")

    # Stack features
    X = np.array(features)
    print(f"[INFO] Feature matrix shape: {X.shape}")

    # Apply PCA for dimensionality reduction
    print("[INFO] Applying PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Keep enough components to explain 95% variance
    n_components = min(min(X_scaled.shape) - 1, 256)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"[INFO] PCA reduced to {X_pca.shape[1]} components, explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # Normalize embeddings
    embeddings = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-10)

    # Save
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"[INFO] Saved embeddings to {embeddings_path}")

    # Save mappings
    idx_to_id = {i: vid for i, vid in enumerate(valid_ids)}
    id_to_idx = {vid: i for i, vid in enumerate(valid_ids)}

    with open(os.path.join(output_dir, "image_to_idx.json"), 'w') as f:
        json.dump({vid: i for i, vid in enumerate(valid_ids)}, f, indent=2)

    with open(os.path.join(output_dir, "image_to_identity.json"), 'w', encoding='utf-8') as f:
        json.dump(identity_mapping, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "image_to_path.json"), 'w', encoding='utf-8') as f:
        json.dump(image_id_to_path, f, indent=2, ensure_ascii=False)

    # Save metadata
    metadata = {
        "n_images": len(valid_ids),
        "embedding_dim": embeddings.shape[1],
        "n_identities": len(set(identity_mapping.values())),
        "feature_type": "color_histogram + hog",
        "pca_components": n_components,
        "explained_variance": float(pca.explained_variance_ratio_.sum())
    }

    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Saved metadata: {metadata}")
    print()
    print("[DONE] Embedding generation complete!")

    return metadata


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", type=str, default="./data/processed/frames")
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    args = parser.parse_args()

    generate_embeddings_from_frames(args.frames_dir, args.output_dir)


if __name__ == "__main__":
    main()
