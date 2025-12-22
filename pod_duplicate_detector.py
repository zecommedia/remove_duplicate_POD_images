"""
POD Mockup Duplicate Design Detector
=====================================
Pipeline hoàn chỉnh để phát hiện các ảnh POD dùng chung design gốc
dù đã đổi model, background, màu áo, thêm watermark, sale badge.

Quy trình:
- Bước 0: Chuẩn hóa ảnh đầu vào
- Bước 1: Lọc nhanh bằng Perceptual Hash (pHash)
- Bước 2: Detect trùng design bằng CLIP embedding
- Bước 3: Xử lý vùng nghi vấn bằng ORB keypoint matching
- Bước 4: Ra quyết định cuối cùng
"""

import os
import json
import hashlib
import requests
from io import BytesIO
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
import imagehash
import cv2

# CLIP imports
try:
    import torch
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ open_clip không khả dụng. Cài đặt: pip install open-clip-torch")

# Sklearn for cosine similarity
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DuplicateConfig:
    """Cấu hình các ngưỡng cho việc detect trùng"""
    # Bước 0: Chuẩn hóa
    target_size: int = 512  # Resize về cạnh dài này
    
    # Bước 1: pHash thresholds (CHỈ DÙNG ĐỂ LỌC SƠ BỘ, KHÔNG KẾT LUẬN)
    phash_exact_threshold: int = 3      # ≤ 3: gần như giống pixel (cần CLIP xác nhận)
    phash_likely_threshold: int = 10    # 4-10: có khả năng trùng (cần CLIP xác nhận)
    
    # Bước 2: CLIP thresholds (TIÊU CHUẨN CHÍNH)
    # Logic mới: Full image threshold cao, center crop với boost requirement
    clip_full_threshold: float = 0.86         # Full image >= 0.86: TRÙNG chắc chắn
    clip_center_threshold: float = 0.83       # Center crop >= 0.83: có thể trùng (cần boost)
    clip_min_center_boost: float = 0.04       # Center phải cao hơn full ≥4% mới được dùng
    
    # Legacy thresholds (để tương thích)
    clip_duplicate_threshold: float = 0.86    # Mapped to clip_full_threshold
    clip_suspect_threshold: float = 0.75      # Vùng nghi vấn cho ORB
    
    # Bước 3: ORB threshold
    orb_match_ratio_threshold: float = 0.15   # >= 0.15: trùng (hạ để ORB dễ confirm hơn)
    orb_num_features: int = 500
    
    # Center crop để focus vào thiết kế, bỏ watermark góc
    use_center_crop: bool = True             # Bật/tắt center crop
    center_crop_ratio: float = 0.65          # Crop 65% vùng giữa (bỏ 17.5% mỗi cạnh)
    
    # CLIP model config
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"


@dataclass
class ImageData:
    """Dữ liệu của một ảnh đã xử lý"""
    index: int
    url: str
    pil_image: Optional[Image.Image] = None
    cv2_image: Optional[np.ndarray] = None
    phash: Optional[imagehash.ImageHash] = None
    clip_embedding: Optional[np.ndarray] = None
    clip_center_embedding: Optional[np.ndarray] = None  # CLIP embedding của phần center crop
    is_valid: bool = True
    error: Optional[str] = None


class PODDuplicateDetector:
    """
    Detector chính để phát hiện ảnh POD trùng design
    """
    
    def __init__(self, config: Optional[DuplicateConfig] = None):
        self.config = config or DuplicateConfig()
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if CLIP_AVAILABLE else "cpu"
        
        if CLIP_AVAILABLE:
            self._load_clip_model()
    
    def _load_clip_model(self):
        """Load CLIP model"""
        print(f"🔄 Đang load CLIP model ({self.config.clip_model_name})...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            self.config.clip_model_name,
            pretrained=self.config.clip_pretrained,
            device=self.device
        )
        self.clip_model.eval()
        print(f"✅ CLIP model loaded on {self.device}")
    
    # =========================================================================
    # BƯỚC 0: Chuẩn hóa ảnh đầu vào
    # =========================================================================
    
    def download_and_normalize_image(self, url: str) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
        """
        Download và chuẩn hóa ảnh:
        - Resize về cạnh dài target_size, giữ tỷ lệ
        - Convert sang RGB
        - Giữ nguyên watermark, badge, text sale
        """
        try:
            # Fix malformed URLs: Various patterns that should be data URLs
            # Pattern 1: "https:data:image/..." -> "data:image/..."
            # Pattern 2: "https://data:image/..." -> "data:image/..."
            # Pattern 3: "http:data:image/..." -> "data:image/..."
            if 'data:image' in url:
                # Find the position of 'data:image' and extract from there
                data_pos = url.find('data:image')
                if data_pos > 0:
                    url = url[data_pos:]
            
            # Handle base64 images
            if url.startswith('data:image'):
                import base64
                # Extract base64 data
                header, data = url.split(',', 1)
                image_data = base64.b64decode(data)
                pil_image = Image.open(BytesIO(image_data))
            else:
                # Download from URL
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                pil_image = Image.open(BytesIO(response.content))
            
            # Convert to RGB
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Resize giữ tỷ lệ
            width, height = pil_image.size
            if max(width, height) > self.config.target_size:
                if width > height:
                    new_width = self.config.target_size
                    new_height = int(height * (self.config.target_size / width))
                else:
                    new_height = self.config.target_size
                    new_width = int(width * (self.config.target_size / height))
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to CV2 format for ORB
            cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return pil_image, cv2_image
            
        except Exception as e:
            print(f"❌ Lỗi download/normalize ảnh: {e}")
            return None, None
    
    # =========================================================================
    # BƯỚC 1: Lọc nhanh bằng Perceptual Hash
    # =========================================================================
    
    def compute_phash(self, pil_image: Image.Image) -> imagehash.ImageHash:
        """Tính perceptual hash cho ảnh"""
        return imagehash.phash(pil_image)
    
    def compare_phash(self, hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> int:
        """Tính Hamming distance giữa 2 hash"""
        return hash1 - hash2
    
    def classify_phash_distance(self, distance: int) -> str:
        """
        Phân loại dựa trên khoảng cách pHash:
        - ≤ 3: EXACT (gần như giống pixel - nhưng vẫn cần CLIP xác nhận!)
        - 4-10: LIKELY (có khả năng trùng - cần CLIP xác nhận)
        - > 10: DIFFERENT (khác nhau)
        
        LƯU Ý: pHash CHỈ là pre-filter, KHÔNG kết luận trùng đơn lẻ vì:
        - Mockup áo POD thường có layout giống nhau (áo đen/trắng, hình ở giữa)
        - pHash không hiểu nội dung, chỉ so sánh cấu trúc pixel
        """
        if distance <= self.config.phash_exact_threshold:
            return "EXACT"
        elif distance <= self.config.phash_likely_threshold:
            return "LIKELY"
        else:
            return "DIFFERENT"
    
    # =========================================================================
    # BƯỚC 2: CLIP Embedding (QUAN TRỌNG NHẤT)
    # =========================================================================
    
    def center_crop_image(self, pil_image: Image.Image) -> Image.Image:
        """
        Crop phần trung tâm của ảnh để focus vào thiết kế, loại bỏ watermark ở góc.
        Sử dụng center_crop_ratio (mặc định 0.65 = giữ 65% vùng giữa)
        """
        width, height = pil_image.size
        crop_ratio = self.config.center_crop_ratio
        
        new_width = int(width * crop_ratio)
        new_height = int(height * crop_ratio)
        
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        
        return pil_image.crop((left, top, right, bottom))
    
    def compute_clip_embedding(self, pil_image: Image.Image) -> Optional[np.ndarray]:
        """Tính CLIP embedding vector cho ảnh"""
        if not CLIP_AVAILABLE or self.clip_model is None:
            return None
        
        try:
            image_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.clip_model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f"❌ Lỗi compute CLIP embedding: {e}")
            return None
    
    def compute_clip_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Tính cosine similarity giữa 2 CLIP embeddings"""
        return float(cosine_similarity([emb1], [emb2])[0][0])
    
    def classify_clip_similarity(self, similarity: float) -> str:
        """
        Phân loại dựa trên CLIP cosine similarity:
        - ≥ 0.93: DUPLICATE (trùng design)
        - 0.88-0.93: SUSPECT (nghi vấn)
        - < 0.88: DIFFERENT (khác design)
        """
        if similarity >= self.config.clip_duplicate_threshold:
            return "DUPLICATE"
        elif similarity >= self.config.clip_suspect_threshold:
            return "SUSPECT"
        else:
            return "DIFFERENT"
    
    # =========================================================================
    # BƯỚC 3: ORB Keypoint Matching (cho vùng nghi vấn)
    # =========================================================================
    
    def compute_orb_match_ratio(self, cv2_img1: np.ndarray, cv2_img2: np.ndarray) -> float:
        """
        Tính tỷ lệ keypoint match giữa 2 ảnh bằng ORB
        Chỉ dùng khi CLIP nằm vùng nghi vấn (0.88-0.93)
        """
        try:
            # Initialize ORB
            orb = cv2.ORB_create(nfeatures=self.config.orb_num_features)
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(cv2_img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(cv2_img2, cv2.COLOR_BGR2GRAY)
            
            # Detect keypoints and compute descriptors
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                return 0.0
            
            # BFMatcher with Hamming distance
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # Calculate match ratio
            min_keypoints = min(len(kp1), len(kp2))
            if min_keypoints == 0:
                return 0.0
            
            match_ratio = len(good_matches) / min_keypoints
            return match_ratio
            
        except Exception as e:
            print(f"❌ Lỗi ORB matching: {e}")
            return 0.0
    
    # =========================================================================
    # BƯỚC 4: Ra quyết định cuối cùng
    # =========================================================================
    
    def is_duplicate_pair(
        self,
        img1: ImageData,
        img2: ImageData,
        verbose: bool = False
    ) -> Tuple[bool, Dict]:
        """
        Quyết định cuối cùng xem 2 ảnh có trùng design không.
        
        Một cặp ảnh được coi là TRÙNG nếu:
        - CLIP cosine ≥ threshold (dùng MAX giữa full image và center crop)
        - HOẶC (CLIP suspect VÀ ORB match ≥ threshold)
        """
        result = {
            "phash_distance": None,
            "phash_classification": None,
            "clip_similarity": None,
            "clip_center_similarity": None,
            "clip_max_similarity": None,
            "clip_classification": None,
            "orb_match_ratio": None,
            "orb_used": False,
            "final_decision": None,
            "reason": None
        }
        
        # Bước 1: Check pHash (CHỈ LÀ PRE-FILTER, KHÔNG KẾT LUẬN TRÙNG ĐƠN LẺ)
        phash_passed = False
        if img1.phash is not None and img2.phash is not None:
            phash_dist = self.compare_phash(img1.phash, img2.phash)
            phash_class = self.classify_phash_distance(phash_dist)
            result["phash_distance"] = phash_dist
            result["phash_classification"] = phash_class
            
            # pHash chỉ đánh dấu "có khả năng" - KHÔNG kết luận trùng!
            if phash_class in ["EXACT", "LIKELY"]:
                phash_passed = True
        
        # Bước 2: Check CLIP (TIÊU CHUẨN CHÍNH - BẮT BUỘC)
        # Logic: Full image threshold cao, center crop với boost + ORB confirm
        # Tránh false positive khi cùng chủ đề (Stranger Things) nhưng khác thiết kế
        clip_sim_full = None
        clip_sim_center = None
        
        if img1.clip_embedding is not None and img2.clip_embedding is not None:
            clip_sim_full = self.compute_clip_similarity(img1.clip_embedding, img2.clip_embedding)
            result["clip_similarity"] = round(clip_sim_full, 4)
        
        # Tính similarity cho center crop nếu có
        center_boost = 0.0
        if (self.config.use_center_crop and 
            img1.clip_center_embedding is not None and 
            img2.clip_center_embedding is not None):
            clip_sim_center = self.compute_clip_similarity(img1.clip_center_embedding, img2.clip_center_embedding)
            result["clip_center_similarity"] = round(clip_sim_center, 4)
            center_boost = clip_sim_center - clip_sim_full if clip_sim_full else 0
            result["center_boost"] = round(center_boost, 4)
        
        # LOGIC QUYẾT ĐỊNH:
        # 1. Nếu full image >= clip_full_threshold: DUPLICATE chắc chắn
        # 2. Nếu center >= clip_center_threshold VÀ boost >= min_center_boost VÀ ORB confirm: DUPLICATE
        # 3. Else: NOT duplicate
        
        is_duplicate = False
        reason = ""
        
        # Rule 1: Full image rất cao - YÊU CẦU ORB XÁC NHẬN
        # Vì cùng brand (Morgan Wallen, Stranger Things) có thể có CLIP cao nhưng thiết kế khác
        if clip_sim_full and clip_sim_full >= self.config.clip_full_threshold:
            result["orb_used"] = True
            if img1.cv2_image is not None and img2.cv2_image is not None:
                orb_ratio = self.compute_orb_match_ratio(img1.cv2_image, img2.cv2_image)
                result["orb_match_ratio"] = round(orb_ratio, 4)
                
                # ORB phải >= threshold để confirm
                if orb_ratio >= self.config.orb_match_ratio_threshold:
                    is_duplicate = True
                    reason = f"CLIP full + ORB (similarity={clip_sim_full:.4f}, orb={orb_ratio:.4f})"
                    result["final_decision"] = True
                    result["reason"] = reason
                    result["clip_classification"] = "DUPLICATE"
                    return True, result
                else:
                    # CLIP cao nhưng ORB thấp → có thể là cùng brand nhưng khác design
                    result["reason"] = f"CLIP high but ORB low - likely same brand, different design (clip={clip_sim_full:.4f}, orb={orb_ratio:.4f})"
                    result["clip_classification"] = "SUSPECT"
        
        # Rule 2: Center boost - CẦN ORB XÁC NHẬN để tránh false positive
        # Vì center crop có thể cao dù thiết kế khác (cùng layout áo đen + graphic giữa)
        if (clip_sim_center and 
              clip_sim_center >= self.config.clip_center_threshold and 
              center_boost >= self.config.clip_min_center_boost):
            
            # Yêu cầu ORB confirm cho center boost cases
            result["orb_used"] = True
            if img1.cv2_image is not None and img2.cv2_image is not None:
                orb_ratio = self.compute_orb_match_ratio(img1.cv2_image, img2.cv2_image)
                result["orb_match_ratio"] = round(orb_ratio, 4)
                
                # ORB phải >= threshold để confirm
                if orb_ratio >= self.config.orb_match_ratio_threshold:
                    is_duplicate = True
                    reason = f"CLIP center + ORB (center={clip_sim_center:.4f}, boost=+{center_boost*100:.1f}%, orb={orb_ratio:.4f})"
                    result["final_decision"] = True
                    result["reason"] = reason
                    result["clip_classification"] = "DUPLICATE"
                    return True, result
                else:
                    # Center cao nhưng ORB thấp → có thể là false positive
                    result["reason"] = f"Center high but ORB low (center={clip_sim_center:.4f}, orb={orb_ratio:.4f})"
        
        # Rule 3: Vùng nghi vấn thông thường (full + center đều trong vùng suspect)
        clip_sim = max(clip_sim_full or 0, clip_sim_center or 0)
        result["clip_max_similarity"] = round(clip_sim, 4)
        
        if clip_sim >= self.config.clip_suspect_threshold and not result.get("orb_used"):
            result["clip_classification"] = "SUSPECT"
            result["orb_used"] = True
            if img1.cv2_image is not None and img2.cv2_image is not None:
                orb_ratio = self.compute_orb_match_ratio(img1.cv2_image, img2.cv2_image)
                result["orb_match_ratio"] = round(orb_ratio, 4)
                
                if orb_ratio >= self.config.orb_match_ratio_threshold:
                    result["final_decision"] = True
                    result["reason"] = f"CLIP suspect + ORB confirm (sim={clip_sim:.4f}, orb={orb_ratio:.4f})"
                    return True, result
        else:
            result["clip_classification"] = "DIFFERENT"
        
        result["final_decision"] = False
        result["reason"] = "No duplicate detected"
        return False, result
    
    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================
    
    def process_images(self, items: List[Dict], verbose: bool = True) -> List[ImageData]:
        """Xử lý tất cả ảnh: download, normalize, compute features"""
        images = []
        total = len(items)
        
        print(f"\n📥 Đang xử lý {total} ảnh...")
        
        for i, item in enumerate(items):
            url = item.get("image", "")
            if verbose and (i + 1) % 5 == 0:
                print(f"   Xử lý ảnh {i+1}/{total}...")
            
            img_data = ImageData(index=i, url=url)
            
            # Download và normalize
            pil_img, cv2_img = self.download_and_normalize_image(url)
            
            if pil_img is None:
                img_data.is_valid = False
                img_data.error = "Failed to download/process"
                images.append(img_data)
                continue
            
            img_data.pil_image = pil_img
            img_data.cv2_image = cv2_img
            
            # Compute pHash
            img_data.phash = self.compute_phash(pil_img)
            
            # Compute CLIP embedding (full image)
            if CLIP_AVAILABLE:
                img_data.clip_embedding = self.compute_clip_embedding(pil_img)
                
                # Compute CLIP embedding for center crop (để loại bỏ watermark góc)
                if self.config.use_center_crop:
                    center_cropped = self.center_crop_image(pil_img)
                    img_data.clip_center_embedding = self.compute_clip_embedding(center_cropped)
            
            images.append(img_data)
        
        valid_count = sum(1 for img in images if img.is_valid)
        print(f"✅ Xử lý xong: {valid_count}/{total} ảnh hợp lệ")
        
        return images
    
    def find_duplicates(
        self,
        images: List[ImageData],
        items: List[Dict] = None,
        verbose: bool = True
    ) -> Tuple[List[Set[int]], List[Dict]]:
        """
        Tìm các nhóm ảnh trùng nhau.
        Trả về:
        - danh sách các set, mỗi set chứa index của các ảnh trùng nhau
        - danh sách chi tiết các cặp trùng
        """
        n = len(images)
        valid_images = [img for img in images if img.is_valid]
        valid_indices = [img.index for img in valid_images]
        
        print(f"\n🔍 Đang so sánh {len(valid_images)} ảnh hợp lệ...")
        
        # Union-Find để group các ảnh trùng
        parent = {i: i for i in valid_indices}
        
        # Lưu chi tiết các cặp trùng
        duplicate_pairs = []

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # So sánh từng cặp
        comparisons = 0
        duplicates_found = 0
        total_pairs = len(valid_images) * (len(valid_images) - 1) // 2
        
        # Debug: lưu top similarities để phân tích
        top_similarities = []
        
        for i, img1 in enumerate(valid_images):
            for j, img2 in enumerate(valid_images[i+1:], start=i+1):
                comparisons += 1
                
                if verbose and comparisons % 100 == 0:
                    print(f"   So sánh {comparisons}/{total_pairs} cặp...")
                
                # Tính CLIP similarity để debug
                clip_sim = None
                if img1.clip_embedding is not None and img2.clip_embedding is not None:
                    clip_sim = self.compute_clip_similarity(img1.clip_embedding, img2.clip_embedding)
                    top_similarities.append((img1.index, img2.index, clip_sim))
                
                is_dup, details = self.is_duplicate_pair(img1, img2, verbose=False)
                
                # DEBUG: In ra khi CLIP sim cao
                if clip_sim and clip_sim >= 0.80:
                    print(f"   🔍 DEBUG: Ảnh {img1.index} & {img2.index} - CLIP={clip_sim:.4f}, is_dup={is_dup}")
                
                if is_dup:
                    union(img1.index, img2.index)
                    duplicates_found += 1
                    
                    # Lấy image URL (rút gọn nếu là base64)
                    img1_url = items[img1.index].get("image", "") if items else ""
                    img2_url = items[img2.index].get("image", "") if items else ""
                    
                    # Rút gọn base64 để dễ đọc
                    if img1_url and 'base64' in img1_url:
                        img1_url_display = img1_url[:80] + "...[base64]"
                    else:
                        img1_url_display = img1_url
                    
                    if img2_url and 'base64' in img2_url:
                        img2_url_display = img2_url[:80] + "...[base64]"
                    else:
                        img2_url_display = img2_url
                    
                    # Lưu thông tin chi tiết cặp trùng
                    pair_info = {
                        "image1_index": img1.index,
                        "image2_index": img2.index,
                        "image1_title": items[img1.index].get("title", "") if items else "",
                        "image2_title": items[img2.index].get("title", "") if items else "",
                        "image1_url": img1_url,
                        "image2_url": img2_url,
                        "image1_link": items[img1.index].get("link", "") if items else "",
                        "image2_link": items[img2.index].get("link", "") if items else "",
                        "reason": details['reason'],
                        "phash_distance": details.get('phash_distance'),
                        "clip_similarity": details.get('clip_similarity'),
                        "orb_match_ratio": details.get('orb_match_ratio'),
                    }
                    duplicate_pairs.append(pair_info)
                    
                    if verbose:
                        print(f"   ⚠️ Trùng: ảnh {img1.index} & {img2.index} - {details['reason']}")
        
        # Group các ảnh theo parent
        groups = defaultdict(set)
        for idx in valid_indices:
            groups[find(idx)].add(idx)
        
        # Chỉ giữ các group có > 1 ảnh (là duplicates)
        duplicate_groups = [group for group in groups.values() if len(group) > 1]
        
        # Debug: In top 10 similarities
        if verbose and top_similarities:
            top_similarities.sort(key=lambda x: x[2], reverse=True)
            print(f"\n📊 TOP 10 CLIP SIMILARITIES (để debug ngưỡng):")
            for idx, (i1, i2, sim) in enumerate(top_similarities[:10]):
                title1 = items[i1].get("title", "")[:40] if items else ""
                title2 = items[i2].get("title", "")[:40] if items else ""
                print(f"   {idx+1}. Ảnh {i1} & {i2}: {sim:.4f}")
                print(f"      - {title1}")
                print(f"      - {title2}")
        
        print(f"✅ Tìm thấy {len(duplicate_groups)} nhóm trùng, {duplicates_found} cặp trùng")
        
        return duplicate_groups, duplicate_pairs
    
    def deduplicate(
        self,
        items: List[Dict],
        strategy: str = "keep_first",
        verbose: bool = True
    ) -> Tuple[List[Dict], List[Dict], Dict, List[Dict]]:
        """
        Lọc trùng danh sách items.
        
        Args:
            items: Danh sách các item (dict có key "image")
            strategy: "keep_first" hoặc "keep_last" - giữ ảnh nào trong mỗi nhóm trùng
            verbose: In chi tiết quá trình
        
        Returns:
            - deduplicated_items: Danh sách items đã lọc trùng
            - removed_items: Danh sách items bị loại bỏ
            - stats: Thống kê quá trình
            - duplicate_pairs: Danh sách chi tiết các cặp trùng
        """
        print("\n" + "="*60)
        print("🚀 BẮT ĐẦU QUY TRÌNH LỌC TRÙNG MOCKUP POD")
        print("="*60)
        
        stats = {
            "total_input": len(items),
            "valid_images": 0,
            "duplicate_groups": 0,
            "duplicates_removed": 0,
            "output_count": 0,
            "duplicate_pairs_count": 0
        }
        
        # Xử lý tất cả ảnh
        images = self.process_images(items, verbose=verbose)
        stats["valid_images"] = sum(1 for img in images if img.is_valid)
        
        # Tìm các nhóm trùng
        duplicate_groups, duplicate_pairs = self.find_duplicates(images, items=items, verbose=verbose)
        stats["duplicate_groups"] = len(duplicate_groups)
        stats["duplicate_pairs_count"] = len(duplicate_pairs)
        
        # Xác định index cần loại bỏ
        indices_to_remove = set()
        for group in duplicate_groups:
            sorted_indices = sorted(group)
            if strategy == "keep_first":
                # Giữ ảnh đầu tiên, loại các ảnh còn lại
                indices_to_remove.update(sorted_indices[1:])
            else:  # keep_last
                # Giữ ảnh cuối cùng, loại các ảnh đầu
                indices_to_remove.update(sorted_indices[:-1])
        
        stats["duplicates_removed"] = len(indices_to_remove)
        
        # Tạo output
        deduplicated_items = []
        removed_items = []
        
        for i, item in enumerate(items):
            if i in indices_to_remove:
                removed_items.append(item)
            else:
                deduplicated_items.append(item)
        
        stats["output_count"] = len(deduplicated_items)
        
        # In thống kê
        print("\n" + "="*60)
        print("📊 THỐNG KÊ KẾT QUẢ")
        print("="*60)
        print(f"   Tổng ảnh đầu vào:     {stats['total_input']}")
        print(f"   Ảnh hợp lệ:           {stats['valid_images']}")
        print(f"   Nhóm trùng phát hiện: {stats['duplicate_groups']}")
        print(f"   Số cặp trùng:         {stats['duplicate_pairs_count']}")
        print(f"   Ảnh bị loại bỏ:       {stats['duplicates_removed']}")
        print(f"   Ảnh đầu ra:           {stats['output_count']}")
        print("="*60)
        
        # In chi tiết các cặp trùng
        if duplicate_pairs and verbose:
            print("\n" + "="*80)
            print("🔗 CHI TIẾT CÁC CẶP ẢNH TRÙNG")
            print("="*80)
            for i, pair in enumerate(duplicate_pairs, 1):
                print(f"\n{'─'*80}")
                print(f"   Cặp #{i}:")
                print(f"   ├─ Ảnh {pair['image1_index']}: {pair['image1_title'][:70]}..." if len(pair['image1_title']) > 70 else f"   ├─ Ảnh {pair['image1_index']}: {pair['image1_title']}")
                
                # Hiển thị URL ảnh 1
                img1_url = pair.get('image1_url', '')
                if img1_url:
                    if 'base64' in img1_url:
                        print(f"   │  🖼️  [Base64 Image]")
                    else:
                        print(f"   │  🖼️  {img1_url[:100]}..." if len(img1_url) > 100 else f"   │  🖼️  {img1_url}")
                
                print(f"   │")
                print(f"   ├─ Ảnh {pair['image2_index']}: {pair['image2_title'][:70]}..." if len(pair['image2_title']) > 70 else f"   ├─ Ảnh {pair['image2_index']}: {pair['image2_title']}")
                
                # Hiển thị URL ảnh 2
                img2_url = pair.get('image2_url', '')
                if img2_url:
                    if 'base64' in img2_url:
                        print(f"   │  🖼️  [Base64 Image]")
                    else:
                        print(f"   │  🖼️  {img2_url[:100]}..." if len(img2_url) > 100 else f"   │  🖼️  {img2_url}")
                
                print(f"   │")
                print(f"   ├─ Lý do: {pair['reason']}")
                if pair.get('clip_similarity'):
                    print(f"   ├─ CLIP Similarity: {pair['clip_similarity']}")
                if pair.get('phash_distance') is not None:
                    print(f"   ├─ pHash Distance: {pair['phash_distance']}")
                if pair.get('orb_match_ratio'):
                    print(f"   └─ ORB Match Ratio: {pair['orb_match_ratio']}")
            print(f"\n{'='*80}\n")
        
        return deduplicated_items, removed_items, stats, duplicate_pairs


def process_json_file(
    input_path: str,
    output_path: str,
    removed_path: Optional[str] = None,
    pairs_path: Optional[str] = None,
    config: Optional[DuplicateConfig] = None,
    verbose: bool = True
):
    """
    Xử lý file JSON đầu vào và xuất kết quả đã lọc trùng.
    
    Args:
        input_path: Đường dẫn file JSON đầu vào
        output_path: Đường dẫn file JSON đầu ra (đã lọc trùng)
        removed_path: Đường dẫn file JSON chứa các item bị loại (optional)
        pairs_path: Đường dẫn file JSON chứa chi tiết các cặp trùng (optional)
        config: Cấu hình ngưỡng detect
        verbose: In chi tiết quá trình
    """
    # Load input
    print(f"📂 Đang đọc file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    
    # Initialize detector
    detector = PODDuplicateDetector(config=config)
    
    # Process
    deduplicated, removed, stats, duplicate_pairs = detector.deduplicate(items, verbose=verbose)
    
    # Save deduplicated output
    print(f"💾 Đang lưu kết quả đã lọc: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(deduplicated, f, ensure_ascii=False, indent=2)
    
    # Save removed items if path provided
    if removed_path and removed:
        print(f"💾 Đang lưu các item bị loại: {removed_path}")
        with open(removed_path, 'w', encoding='utf-8') as f:
            json.dump(removed, f, ensure_ascii=False, indent=2)
    
    # Save duplicate pairs if path provided
    if pairs_path and duplicate_pairs:
        print(f"💾 Đang lưu chi tiết các cặp trùng: {pairs_path}")
        with open(pairs_path, 'w', encoding='utf-8') as f:
            json.dump(duplicate_pairs, f, ensure_ascii=False, indent=2)
    
    return deduplicated, removed, stats, duplicate_pairs


# =============================================================================
# DEMO / MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="POD Mockup Duplicate Detector")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path (deduplicated)")
    parser.add_argument("--removed", "-r", help="Optional: Output JSON file for removed items")
    parser.add_argument("--phash-exact", type=int, default=8, help="pHash exact threshold (default: 8)")
    parser.add_argument("--phash-likely", type=int, default=12, help="pHash likely threshold (default: 12)")
    parser.add_argument("--clip-dup", type=float, default=0.93, help="CLIP duplicate threshold (default: 0.93)")
    parser.add_argument("--clip-suspect", type=float, default=0.88, help="CLIP suspect threshold (default: 0.88)")
    parser.add_argument("--orb-ratio", type=float, default=0.4, help="ORB match ratio threshold (default: 0.4)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    
    args = parser.parse_args()
    
    # Build config
    config = DuplicateConfig(
        phash_exact_threshold=args.phash_exact,
        phash_likely_threshold=args.phash_likely,
        clip_duplicate_threshold=args.clip_dup,
        clip_suspect_threshold=args.clip_suspect,
        orb_match_ratio_threshold=args.orb_ratio
    )
    
    # Process
    process_json_file(
        input_path=args.input,
        output_path=args.output,
        removed_path=args.removed,
        config=config,
        verbose=not args.quiet
    )
