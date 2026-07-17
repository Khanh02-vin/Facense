# Facense — Hệ thống xếp hạng sở thích thị giác

> Pipeline end-to-end học xếp hạng sở thích cá nhân hóa từ dữ liệu video thô, dùng embedding vision–language hiện đại, mô hình preference cổ điển và phân tích tương đồng biểu diễn (RSA).

---

## 🎯 Bài toán

Cho một tập video khuôn mặt chưa gán nhãn, hệ thống cần giải 4 câu hỏi:

1. Khám phá tín hiệu sở thích tiềm ẩn trong dữ liệu
2. Kiểm chứng không gian embedding có ý nghĩa tri giác (không chỉ thuận tiện về số học)
3. Học mô hình xếp hạng theo từng item, tổng quát hóa được sang đối tượng chưa thấy
4. Đánh giá nghiêm ngặt — kiểm soát theo đối tượng, thuộc tính và so với phân phối ngẫu nhiên

Đây là dạng bài toán mà hệ thống gợi ý nội dung, ứng dụng hẹn hò hay feed cá nhân hóa đều phải giải.

---

## 🧠 Điểm kỹ thuật nổi bật

- **Embedding SigLIP / CLIP / DINOv2**: trích 768-d đặc trưng thị giác mỗi frame. Đây là các foundation model hiện đại, cùng backbone được dùng trong hệ thống gợi ý production.
- **Tổng hợp đa frame**: lấy mẫu frame thông tin, loại bỏ nhòe và trùng lặp. Thể hiện tư duy pipeline CV chuẩn production.
- **Bradley–Terry preference model**: công thức `P(i > j) = σ(wᵢ - wⱼ)`, học điểm sức mạnh từng item từ nhãn cặp. Baseline cổ điển nhưng thường thắng nhiều mô hình neural trên dữ liệu thưa.
- **RSA (Representational Similarity Analysis)**: kiểm chứng tương đồng giữa embedding và tri giác con người qua Spearman ρ. Kỹ thuật nền tảng từ neuroscience (Kriegeskorte 2008), hiếm gặp ngoài phòng thí nghiệm nghiên cứu.
- **Đánh giá identity-stratified**: holdout theo đối tượng, không chỉ theo mẫu. Đây là cách đúng đắn để đánh giá mô hình cá nhân hóa và tránh data leakage.
- **Attribute-controlled evaluation**: tách preference khỏi đặc trưng cấp thấp, thể hiện nhận thức về bias và fairness.
- **Null-model permutation test**: so sánh điểm số học được với baseline ngẫu nhiên, đảm bảo tính rigor thống kê.
- **Docker pipeline đầy đủ**: 6 profile gồm pipeline, frame-extract, embeddings, rsa, evaluate và shell. Orchestration chuẩn production.

---

## 🏗️ Kiến trúc

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Video thô      │ ──► │  Lấy mẫu frame   │ ──► │  Kiểm tra chất  │
│  (mp4 / ID)     │     │  (adaptive)      │     │  lượng (blur)   │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Annotation     │ ◄── │  Embedding       │ ◄── │  Crop khuôn mặt │
│  cặp (pairwise) │     │  (SigLIP/CLIP/   │     │  (face detect)  │
│                 │     │   DINOv2)        │     │                 │
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Bradley–Terry  │ ──► │  RSA Validation  │ ──► │  Item Scores    │
│  (preference)   │     │  (Spearman ρ)    │     │  + ranking      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Đánh giá: identity-stratified · attribute-controlled · null    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

**Core ML**: PyTorch 2.3, torchvision, Hugging Face Transformers (SigLIP, CLIP), timm (DINOv2), scikit-learn, scipy.

**Computer Vision**: OpenCV (headless), MediaPipe (face detection).

**Inference / Serving**: Docker và Docker Compose với 6 profile.

**Data / Eval**: pandas, numpy, Matplotlib, framework RSA và null-model tự xây dựng.

---

## 📂 Cấu trúc dự án

- **src/frame_extraction**: adaptive thresholding, quality checker, sampling, prefilter
- **src/feature_extraction**: pipeline embedding SigLIP, CLIP, DINOv2
- **src/preference_learning**: Bradley-Terry, RSA, null models, universal taste engine
- **src/retrieval_recommend**: retrieval, UI gán nhãn, bộ thu thập preference
- **src/analysis_evalution**: validation, annotation, cycle diagnostics
- **src/utility**: preference discovery, video segmentation
- **src/run_pipeline.py**: orchestrator end-to-end chính
- **evaluation/**: 17 script đánh giá (RSA, BT trainer, stratified eval)
- **start/**: threshold calibrator, sample ranker
- **docker/**: Dockerfile và docker-compose.yml
- **TUTORIAL/**: giáo trình 7 phần từ Python đến preference learning
- **data/**: CSV và dataset đã xử lý
- **results/**: output đánh giá

---

## 🚀 Bắt đầu nhanh

### Cách A — Docker (khuyến nghị)

Bước 1: Clone repo và vào thư mục dự án.

Bước 2: Cấu hình bằng cách copy file `.env.example` thành `.env` trong thư mục `docker/`, sau đó sửa biến `DATASET_DIR` trỏ tới folder video của bạn.

Bước 3: Build image bằng lệnh `docker compose -f docker/docker-compose.yml build`.

Bước 4: Chạy pipeline end-to-end bằng lệnh `docker compose -f docker/docker-compose.yml --profile full up pipeline`.

Bước 5: Chạy đánh giá sau khi có embeddings:
- `docker compose -f docker/docker-compose.yml --profile rsa up rsa`
- `docker compose -f docker/docker-compose.yml --profile evaluate up evaluate`

Bước 6: Mở shell tương tác để debug: `docker compose -f docker/docker-compose.yml --profile shell up shell`.

### Cách B — Local Python

Bước 1: Tạo venv bằng `python -m venv .venv` rồi kích hoạt.

Bước 2: Cài đặt thư viện bằng `pip install -r requirements.txt`.

Bước 3: Cấu hình đường dẫn dataset qua biến môi trường `DATASET_DIR`.

Bước 4: Chạy pipeline bằng `python src/run_pipeline.py --max 100 --pairs 50`.

Bước 5: Chạy đánh giá bằng `python evaluation/run_evaluation.py`.

---

## 🧪 Phương pháp luận

### Bước 1 — Trích xuất frame thông tin

Từ mỗi video, lấy các frame chất lượng cao (loại bỏ nhòe, trùng, dư thừa) bằng adaptive thresholding và quality checker.

### Bước 2 — Trích embedding đa tầng

Dùng SigLIP (chính), CLIP và DINOv2 để trích vector 768 chiều trên từng frame. Tổng hợp đa frame để giảm nhiễu.

### Bước 3 — Bradley–Terry preference model

Áp dụng công thức `P(i thắng j) = σ(wᵢ - wⱼ)`:
- Ước lượng maximum-likelihood lặp đến hội tụ
- Regularization chống overfit khi nhãn thưa
- Trả về điểm sức mạnh từng item, từ đó suy ra ranking toàn cục

### Bước 4 — Kiểm chứng thống kê

- **Permutation null test** với 1000 lần: ranking học được có tốt hơn ngẫu nhiên không
- **Identity-stratified holdout**: ranking có tổng quát hóa qua đối tượng chưa thấy không
- **Attribute-controlled evaluation**: ranking có phụ thuộc vào một đặc trưng đơn lẻ không, hay nắm bắt được tín hiệu phong phú hơn

---

## 📈 Kết quả đánh giá

Sau khi chạy `evaluation/run_evaluation.py` và `evaluation/run_rsa.py`, output nằm trong thư mục `results/`:

- **rsa_spearman.csv**: hệ số ρ từng cặp giữa similarity người và embedding
- **bradley_terry_scores.json**: điểm sức mạnh từng item đã học
- **identity_stratified.json**: hiệu năng trên identity chưa thấy
- **null_model_pvalues.json**: kết quả permutation test
- **attribute_controlled.json**: phân tích ranking theo từng attribute

---

## 🧪 Tính tái lập

- Mọi randomness đều được seed qua `np.random.seed(42)`
- Dataset mount chế độ read-only trong Docker
- Checkpoint embedding model cache trong `models/`, đã được gitignore
- File `.env.example` document mọi biến cấu hình

---

## 🗺️ Hướng phát triển

- Serving layer FastAPI với endpoint `POST /rank` trả về top-K items
- Benchmark latency và throughput cho mô hình embedding
- Tích hợp MLflow để theo dõi experiment
- Export ONNX cho embedding model nhằm tăng tốc inference
- Identity-aware debiasing bằng kỹ thuật re-weighting

---

## 📚 Tài liệu nền

Repo đi kèm giáo trình 7 phần trong thư mục `TUTORIAL/`, đi từ Python, NumPy, xác suất, deep learning, embedding, preference learning đến kiến trúc phần mềm. Hữu ích cho người mới tham gia dự án.

---

## 👤 Tác giả

**Khanh Nguyen** — AI / ML Engineer

Xây dựng như một nghiên cứu thực hành về pipeline preference learning end-to-end, đi từ video thô đến ranking được kiểm chứng thống kê.

---

## 📄 License

MIT
