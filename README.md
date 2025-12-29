# POD Mockup Duplicate Detector

Phát hiện ảnh mockup POD (Print on Demand) dùng chung design, hỗ trợ GPU/CPU tự động.

## 🚀 Quick Start (3 bước)

```bash
# 1. Clone repo
git clone <repo-url>
cd match_case

# 2. Tạo virtual environment (khuyến khích)
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Cài đặt dependencies
pip install -r requirements.txt          # CPU (mọi máy)
# pip install -r requirements-cuda.txt   # GPU NVIDIA (nhanh hơn 5-10x)
```

**Kiểm tra cài đặt:**
```bash
python setup_check.py
```

## 📖 Mô tả

Pipeline phát hiện các ảnh mockup POD dùng chung design gốc, dù đã:
- ✅ Đổi model/người mẫu
- ✅ Đổi background
- ✅ Đổi màu áo
- ✅ Thêm watermark
- ✅ Thêm sale badge

## ⚙️ Cài đặt chi tiết

### Yêu cầu hệ thống
- Python 3.8+ 
- RAM: 4GB minimum (8GB recommended)
- GPU: Optional (NVIDIA CUDA 11.8+)

### Option 1: CPU Mode (Mọi máy)

```bash
pip install -r requirements.txt
```

### Option 2: GPU Mode (Nhanh hơn 5-10x)

**Yêu cầu:** NVIDIA GPU với CUDA 11.8+

```bash
pip install -r requirements-cuda.txt
```

**Kiểm tra CUDA:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Option 3: Conda (Recommended cho GPU)

```bash
# Tạo môi trường
conda create -n pod-detector python=3.10 -y
conda activate pod-detector

# Cài PyTorch với CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Cài các package còn lại
pip install open-clip-torch Pillow imagehash opencv-python scikit-learn requests tqdm
```

## 🎯 Sử dụng

### Cách 1: Script nhanh (Recommended)

Chỉnh sửa đường dẫn trong `run_detector.py`:

```python
INPUT_FILE = "input.json"           # File JSON đầu vào
OUTPUT_FILE = "output_clean.json"   # File đã lọc trùng
REMOVED_FILE = "removed.json"       # File bị loại (optional)
```

Chạy:
```bash
python run_detector.py
```

### Cách 2: Command Line

```bash
python pod_duplicate_detector.py -i input.json -o output.json -r removed.json -p pairs.json
```

**Options:**
| Flag | Mô tả | Mặc định |
|------|-------|----------|
| `-i, --input` | File JSON đầu vào | Bắt buộc |
| `-o, --output` | File JSON đầu ra | Bắt buộc |
| `-r, --removed` | File chứa items bị loại | None |
| `-p, --pairs` | File chi tiết các cặp trùng | None |
| `--clip-dup` | Ngưỡng CLIP duplicate | 0.86 |
| `--clip-suspect` | Ngưỡng CLIP suspect | 0.75 |
| `-q, --quiet` | Chế độ im lặng | False |

### Cách 3: Import như module

```python
from pod_duplicate_detector import PODDuplicateDetector, DuplicateConfig, process_json_file

# Sử dụng config mặc định
deduplicated, removed, stats, pairs = process_json_file(
    input_path="input.json",
    output_path="output.json"
)

# Custom config
config = DuplicateConfig(
    clip_full_threshold=0.90,      # Ngưỡng cao hơn = ít false positive
    clip_center_threshold=0.87
)
detector = PODDuplicateDetector(config=config)
```

### Cách 4: Lightweight Mode (Không cần GPU/CLIP)

```bash
python pod_duplicate_lightweight.py input.json output.json removed.json
```

⚠️ **Lưu ý:** Độ chính xác thấp hơn, chỉ dùng khi không có GPU và cần chạy nhanh.

## 📄 Format JSON

**Input:**
```json
[
  {
    "title": "Product Title",
    "image": "https://example.com/image.jpg",
    "link": "https://example.com/product",
    "seller": "Seller Name"
  }
]
```

**Output:** Giống format input, đã loại bỏ items có design trùng.

## 🔧 Quy trình xử lý

```
Ảnh gốc → Chuẩn hóa (512px) → pHash Filter → CLIP Embedding → ORB Matching → Kết quả
```

1. **Chuẩn hóa:** Resize về 512px, convert RGB
2. **pHash:** Lọc nhanh ảnh giống pixel (distance ≤ 3)
3. **CLIP:** So sánh semantic similarity (≥ 0.86 = trùng)
   - **Full image:** So sánh toàn bộ ảnh
   - **Center crop:** So sánh 65% vùng giữa (loại bỏ watermark góc)
4. **ORB:** Xác nhận vùng nghi vấn (0.75-0.86)

### Cấu hình nâng cao

```python
CONFIG = DuplicateConfig(
    target_size=512,              # Resize về cạnh dài này
    
    # pHash thresholds (pre-filter)
    phash_exact_threshold=3,      # ≤ 3: gần như giống pixel
    phash_likely_threshold=10,    # 4-10: có khả năng trùng
    
    # CLIP thresholds (logic chính)
    clip_full_threshold=0.86,     # Full image >= 0.86: DUPLICATE
    clip_center_threshold=0.83,   # Center >= 0.83: cần boost + ORB
    clip_min_center_boost=0.04,   # Center cao hơn full ≥4% mới dùng
    clip_suspect_threshold=0.75,  # Vùng nghi vấn cho ORB
    
    # ORB threshold
    orb_match_ratio_threshold=0.15,
    
    # Center crop (loại bỏ watermark góc)
    use_center_crop=True,
    center_crop_ratio=0.65,       # Giữ 65% vùng giữa
)
```

## ⚡ Performance

| Mode | Tốc độ | Độ chính xác |
|------|--------|--------------|
| GPU (CUDA) | ~0.1-0.2s/ảnh | Cao nhất |
| CPU | ~0.5-1s/ảnh | Cao nhất |
| Lightweight | ~0.2-0.4s/ảnh | Trung bình |

## 🛠️ Troubleshooting

### CLIP không load được

```bash
# Reinstall PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install open-clip-torch
```

### CUDA không nhận

```bash
# Kiểm tra CUDA version
nvidia-smi

# Cài PyTorch matching CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Lỗi memory

- Chia nhỏ file JSON
- Giảm `target_size` xuống 384 hoặc 256
- Dùng CPU mode nếu GPU memory < 4GB

### Quá nhiều false positive

```python
config = DuplicateConfig(
    clip_full_threshold=0.92,    # Tăng lên
    clip_center_threshold=0.88
)
```

## 📁 Cấu trúc Project

```
match_case/
├── pod_duplicate_detector.py    # Main detector (CLIP + ORB)
├── pod_duplicate_lightweight.py # Lightweight version (pHash + ORB only)
├── run_detector.py              # Quick run script
├── setup_check.py               # Kiểm tra môi trường
├── sample_input.json            # File mẫu để test
├── requirements.txt             # Dependencies (CPU)
├── requirements-cuda.txt        # Dependencies (GPU CUDA)
├── .gitignore                   # Git ignore config
└── README.md                    # Documentation
```

## 📝 License

MIT License
