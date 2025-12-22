"""
POD Duplicate Detector - Script chạy nhanh
==========================================
File này giúp bạn chạy nhanh detector mà không cần command line arguments.

Sử dụng:
1. Chỉnh sửa các đường dẫn INPUT_FILE và OUTPUT_FILE bên dưới
2. Chạy: python run_detector.py
"""

from pod_duplicate_detector import PODDuplicateDetector, DuplicateConfig, process_json_file
import os

# =============================================================================
# CẤU HÌNH - CHỈNH SỬA TẠI ĐÂY
# =============================================================================

# Đường dẫn file đầu vào
INPUT_FILE = r"D:\Zecom AutoAgents\VPTEEK Project\match_case\output(1).json"

# Đường dẫn file đầu ra (đã lọc trùng)
OUTPUT_FILE = r"D:\Zecom AutoAgents\VPTEEK Project\match_case\output_deduplicated(1).json"

# Đường dẫn file chứa các item bị loại bỏ (optional, set None nếu không cần)
REMOVED_FILE = r"D:\Zecom AutoAgents\VPTEEK Project\match_case\output_removed(1).json"

# Đường dẫn file chứa chi tiết các cặp trùng (optional, set None nếu không cần)
PAIRS_FILE = r"D:\Zecom AutoAgents\VPTEEK Project\match_case\output_duplicate_pairs(1).json"

# Cấu hình ngưỡng detect (có thể điều chỉnh)
CONFIG = DuplicateConfig(
    # Bước 0: Chuẩn hóa
    target_size=512,  # Resize về cạnh dài này
    
    # Bước 1: pHash thresholds (CHỈ LÀ PRE-FILTER)
    phash_exact_threshold=3,      # ≤ 3: gần như giống pixel
    phash_likely_threshold=10,    # 4-10: có khả năng trùng
    
    # Bước 2: CLIP thresholds (LOGIC MỚI)
    # Full image threshold cao để tránh false positive
    clip_full_threshold=0.86,         # Full >= 0.86: DUPLICATE chắc chắn
    clip_center_threshold=0.83,       # Center >= 0.83: cần boost + ORB confirm
    clip_min_center_boost=0.04,       # Center phải cao hơn full ≥4% mới được dùng
    clip_suspect_threshold=0.75,      # Vùng nghi vấn cho ORB
    
    # Bước 3: ORB threshold (hạ xuống để dễ confirm hơn)
    orb_match_ratio_threshold=0.15,   # >= 0.15: trùng (center boost cases cần ORB confirm)
    
    # Center crop để loại bỏ watermark góc
    use_center_crop=True,             # Bật center crop
    center_crop_ratio=0.65,           # Crop 65% vùng giữa (bỏ 17.5% mỗi cạnh)
    
    # CLIP model (có thể đổi sang model khác nếu cần)
    clip_model_name="ViT-B-32",
    clip_pretrained="openai"
)

# =============================================================================
# CHẠY DETECTOR
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🎨 POD MOCKUP DUPLICATE DETECTOR")
    print("="*60)
    print(f"\n📁 Input:  {INPUT_FILE}")
    print(f"📁 Output: {OUTPUT_FILE}")
    if REMOVED_FILE:
        print(f"📁 Removed: {REMOVED_FILE}")
    if PAIRS_FILE:
        print(f"📁 Pairs:   {PAIRS_FILE}")
    print()
    
    # Kiểm tra file tồn tại
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Không tìm thấy file input: {INPUT_FILE}")
        exit(1)
    
    # Chạy detector
    deduplicated, removed, stats, duplicate_pairs = process_json_file(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        removed_path=REMOVED_FILE,
        pairs_path=PAIRS_FILE,
        config=CONFIG,
        verbose=True
    )
    
    print("🎉 HOÀN THÀNH!")
    print(f"   Đã lọc {stats['duplicates_removed']} ảnh trùng")
    print(f"   Tìm thấy {stats['duplicate_pairs_count']} cặp trùng")
    print(f"   Còn lại {stats['output_count']} ảnh unique")
