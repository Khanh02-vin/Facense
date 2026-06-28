#!/bin/bash
# Run Embedding Extraction in Docker Container

set -e

echo "========================================"
echo "Face Embedding Extraction"
echo "========================================"

# Build image
docker build -t face-embeddings-cpu -f Dockerfile .

# Run extraction
docker run --rm \
    -v $(pwd)/data/processed:/workspace/data/processed \
    -v $(pwd)/results:/workspace/results \
    face-embeddings-cpu \
    python evaluation/extract_deep_embeddings.py \
        --dataset_dir /data/Face_project_datset \
        --output_dir /workspace/data/processed \
        --models clip siglip dinov2 \
        --max_per_identity 1

echo "========================================"
echo "Extraction Complete!"
echo "========================================"
