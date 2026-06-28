@echo off
REM Run Embedding Extraction in Docker Container

echo ========================================
echo Face Embedding Extraction
echo ========================================

REM Build image (if not built)
docker build -t face-embeddings-cpu -f Dockerfile .

REM Run extraction
docker run --rm ^
    -v D:/Dataset/Face_project_datset:/data/Face_project_datset:ro ^
    -v D:/Project/Face_project/data/processed:/workspace/data/processed ^
    -v D:/Project/Face_project/results:/workspace/results ^
    face-embeddings-cpu ^
    python evaluation/extract_deep_embeddings.py ^
        --dataset_dir /data/Face_project_datset ^
        --output_dir /workspace/data/processed ^
        --models clip siglip dinov2 ^
        --max_per_identity 1

echo ========================================
echo Extraction Complete!
echo ========================================
