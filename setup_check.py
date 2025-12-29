#!/usr/bin/env python
"""
Setup Script - POD Duplicate Detector
======================================
Script tự động cài đặt và kiểm tra môi trường.

Sử dụng:
    python setup_check.py         # Kiểm tra môi trường
    python setup_check.py --test  # Kiểm tra + chạy test nhỏ
"""

import subprocess
import sys
import os

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_ok(text):
    print(f"  ✅ {text}")

def print_warn(text):
    print(f"  ⚠️  {text}")

def print_error(text):
    print(f"  ❌ {text}")

def check_python():
    """Kiểm tra Python version"""
    print_header("KIỂM TRA PYTHON")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error("Cần Python 3.8 trở lên!")
        return False
    print_ok("Python version OK")
    return True

def check_packages():
    """Kiểm tra các package cần thiết"""
    print_header("KIỂM TRA PACKAGES")
    
    required = [
        ("PIL", "Pillow"),
        ("imagehash", "imagehash"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("open_clip", "open-clip-torch"),
        ("sklearn", "scikit-learn"),
        ("requests", "requests"),
    ]
    
    missing = []
    for module, package in required:
        try:
            __import__(module)
            print_ok(f"{package}")
        except ImportError:
            print_error(f"{package} - CHƯA CÀI")
            missing.append(package)
    
    return missing

def check_torch():
    """Kiểm tra PyTorch và CUDA"""
    print_header("KIỂM TRA PYTORCH & CUDA")
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print_ok(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"      CUDA version: {torch.version.cuda}")
            print(f"      GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return "cuda"
        else:
            print_warn("CUDA không khả dụng - Sử dụng CPU mode")
            print("      (Vẫn hoạt động bình thường, chỉ chậm hơn)")
            return "cpu"
    except ImportError:
        print_error("PyTorch chưa được cài đặt!")
        return None

def check_clip():
    """Kiểm tra OpenCLIP"""
    print_header("KIỂM TRA OPENCLIP MODEL")
    
    try:
        import open_clip
        import torch
        
        print("  Đang load model ViT-B-32...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', 
            pretrained='openai',
            device=device
        )
        print_ok(f"Model loaded successfully on {device.upper()}")
        
        # Cleanup
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print_error(f"Không load được model: {e}")
        return False

def run_quick_test():
    """Chạy test nhanh với ảnh mẫu"""
    print_header("CHẠY TEST NHANH")
    
    try:
        from pod_duplicate_detector import PODDuplicateDetector, DuplicateConfig
        
        print("  Khởi tạo detector...")
        detector = PODDuplicateDetector()
        print_ok("Detector khởi tạo thành công!")
        
        # Test với ảnh giả
        print("  Test xử lý ảnh...")
        from PIL import Image
        import numpy as np
        
        # Tạo ảnh test
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        
        # Test các hàm cơ bản
        normalized = detector._normalize_image(img)
        print_ok("Normalize image OK")
        
        phash = detector._compute_phash(normalized)
        print_ok(f"pHash computed: {phash}")
        
        embedding = detector._compute_clip_embedding(normalized)
        print_ok(f"CLIP embedding shape: {embedding.shape}")
        
        print_ok("Tất cả tests passed!")
        return True
        
    except Exception as e:
        print_error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "🔧 "*20)
    print("    POD DUPLICATE DETECTOR - SETUP CHECK")
    print("🔧 "*20)
    
    # Check Python
    if not check_python():
        sys.exit(1)
    
    # Check packages
    missing = check_packages()
    
    if missing:
        print_header("CÀI ĐẶT PACKAGES THIẾU")
        print(f"  Chạy lệnh sau để cài đặt:")
        print(f"  pip install -r requirements.txt")
        print()
        print("  Hoặc nếu có GPU NVIDIA:")
        print(f"  pip install -r requirements-cuda.txt")
        sys.exit(1)
    
    # Check PyTorch/CUDA
    device = check_torch()
    if not device:
        sys.exit(1)
    
    # Check CLIP
    clip_ok = check_clip()
    
    # Run test if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_quick_test()
    
    # Summary
    print_header("TỔNG KẾT")
    print_ok("Môi trường đã sẵn sàng!")
    print()
    print(f"  Device: {device.upper()}")
    print(f"  CLIP:   {'OK' if clip_ok else 'FAILED'}")
    print()
    print("  Để chạy detector:")
    print("    python run_detector.py")
    print()
    print("  Hoặc qua command line:")
    print("    python pod_duplicate_detector.py -i input.json -o output.json")
    print()

if __name__ == "__main__":
    main()
