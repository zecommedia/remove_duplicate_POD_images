# POD Mockup Duplicate Detector

## Mô tả
Pipeline hoàn chỉnh để phát hiện các ảnh mockup POD (Print on Demand) dùng chung design gốc, dù đã:
- Đổi model/người mẫu
- Đổi background
- Đổi màu áo
- Thêm watermark
- Thêm sale badge

## Quy trình xử lý

### Bước 0: Chuẩn hóa ảnh đầu vào
- Resize về cạnh dài 512px, giữ tỷ lệ
- Convert sang RGB
- Giữ nguyên watermark, badge, text sale

### Bước 1: Lọc nhanh bằng Perceptual Hash (pHash)
- Tính pHash cho tất cả ảnh
- Distance ≤ 8: trùng tuyệt đối
- Distance 9-12: có khả năng trùng
- Distance > 12: cho qua tầng tiếp theo

### Bước 2: CLIP Embedding (QUAN TRỌNG NHẤT)
- Dùng OpenCLIP để sinh embedding vector
- Cosine ≥ 0.93: TRÙNG DESIGN
- 0.88 ≤ Cosine < 0.93: NGHI VẤN
- Cosine < 0.88: KHÁC DESIGN

### Bước 3: ORB Keypoint Matching (cho vùng nghi vấn)
- Chỉ áp dụng khi CLIP nằm vùng 0.88-0.93
- Match ratio ≥ 0.4: coi là trùng

### Bước 4: Ra quyết định cuối cùng
Một cặp ảnh được coi là TRÙNG nếu:
- pHash ≤ 8
- HOẶC CLIP cosine ≥ 0.93
- HOẶC (CLIP 0.88-0.93 VÀ ORB match pass)

## Cài đặt

```bash
pip install -r requirements.txt
```

**Requirements:**
- Pillow
- imagehash
- opencv-python
- numpy
- torch
- open-clip-torch
- scikit-learn
- requests

## Sử dụng

### Cách 1: Chạy trực tiếp với file cấu hình sẵn

Chỉnh sửa đường dẫn trong `run_detector.py` rồi chạy:

```bash
python run_detector.py
```

### Cách 2: Chạy qua command line

```bash
python pod_duplicate_detector.py -i input.json -o output_deduplicated.json -r removed.json
```

**Các options:**
- `-i, --input`: File JSON đầu vào (bắt buộc)
- `-o, --output`: File JSON đầu ra đã lọc trùng (bắt buộc)
- `-r, --removed`: File JSON chứa các item bị loại (optional)
- `--phash-exact`: Ngưỡng pHash exact (mặc định: 8)
- `--phash-likely`: Ngưỡng pHash likely (mặc định: 12)
- `--clip-dup`: Ngưỡng CLIP duplicate (mặc định: 0.93)
- `--clip-suspect`: Ngưỡng CLIP suspect (mặc định: 0.88)
- `--orb-ratio`: Ngưỡng ORB match ratio (mặc định: 0.4)
- `-q, --quiet`: Chế độ im lặng

### Cách 3: Import như module trong code khác

```python
from pod_duplicate_detector import PODDuplicateDetector, DuplicateConfig, process_json_file

# Sử dụng config mặc định
deduplicated, removed, stats = process_json_file(
    input_path="input.json",
    output_path="output.json"
)

# Hoặc custom config
config = DuplicateConfig(
    clip_duplicate_threshold=0.95,  # Ngưỡng cao hơn = ít false positive
    clip_suspect_threshold=0.90
)
detector = PODDuplicateDetector(config=config)
deduplicated, removed, stats = detector.deduplicate(items)
```

### Cách 4: Phiên bản Lightweight (không cần CLIP/GPU)

Nếu không có GPU hoặc cần chạy nhanh:

```bash
python pod_duplicate_lightweight.py input.json output.json removed.json
```

**Lưu ý:** Phiên bản lightweight có độ chính xác thấp hơn, chỉ phù hợp khi ảnh chủ yếu là resize/nén/đổi sáng nhẹ.

## Format JSON đầu vào

```json
[
  {
    "title": "Product Title",
    "image": "https://example.com/image.jpg",
    "link": "https://example.com/product",
    "seller": "Seller Name"
  },
  ...
]
```

## Format JSON đầu ra

Giống format đầu vào, nhưng đã loại bỏ các item có design trùng.

## Điều chỉnh ngưỡng

Tùy vào use case, bạn có thể điều chỉnh ngưỡng:

### Nếu muốn ít false positive hơn (ít bị loại nhầm):
```python
config = DuplicateConfig(
    clip_duplicate_threshold=0.95,  # Tăng lên
    clip_suspect_threshold=0.92,
    phash_exact_threshold=6
)
```

### Nếu muốn bắt nhiều duplicate hơn (kể cả không chắc chắn):
```python
config = DuplicateConfig(
    clip_duplicate_threshold=0.90,  # Giảm xuống
    clip_suspect_threshold=0.85,
    phash_exact_threshold=10
)
```

## Ứng dụng thực tế

- Block design quá giống khi redesign
- Cảnh báo trước khi upload Amazon/Etsy
- Group các listing dùng chung design gốc
- Deduplicate khi crawl đối thủ
- Chống reuse graphic nội bộ team

## Performance

- **CPU only:** ~0.5-1s/ảnh (phụ thuộc vào kích thước)
- **GPU (CUDA):** ~0.1-0.2s/ảnh
- **Lightweight mode:** ~0.2-0.4s/ảnh

## Troubleshooting

### CLIP không load được
```
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install open-clip-torch
```

### Lỗi memory khi xử lý nhiều ảnh
Chia nhỏ file JSON và xử lý từng phần.

### Quá nhiều false positive
Tăng ngưỡng `clip_duplicate_threshold` lên 0.95-0.97.
