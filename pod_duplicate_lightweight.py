"""
POD Duplicate Detector - Lightweight Version (Không cần CLIP)
==============================================================
Phiên bản nhẹ chỉ sử dụng pHash và ORB, không cần GPU/CLIP.
Phù hợp khi:
- Không có GPU
- Cần chạy nhanh
- Ảnh chủ yếu là resize/nén/đổi sáng nhẹ

Lưu ý: Độ chính xác thấp hơn version full với CLIP
"""

import os
import json
import requests
from io import BytesIO
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
import imagehash
import cv2


@dataclass
class LightweightConfig:
    """Cấu hình cho phiên bản nhẹ"""
    target_size: int = 512
    
    # pHash thresholds - điều chỉnh thấp hơn vì không có CLIP backup
    phash_duplicate_threshold: int = 10    # ≤ 10: coi là trùng
    phash_suspect_threshold: int = 16      # 11-16: nghi vấn, check ORB
    
    # ORB threshold
    orb_match_ratio_threshold: float = 0.35
    orb_num_features: int = 500


@dataclass
class ImageDataLight:
    """Dữ liệu ảnh đã xử lý"""
    index: int
    url: str
    pil_image: Optional[Image.Image] = None
    cv2_image: Optional[np.ndarray] = None
    phash: Optional[imagehash.ImageHash] = None
    ahash: Optional[imagehash.ImageHash] = None  # Average hash bổ sung
    dhash: Optional[imagehash.ImageHash] = None  # Difference hash bổ sung
    is_valid: bool = True
    error: Optional[str] = None


class LightweightDuplicateDetector:
    """
    Detector nhẹ không cần CLIP - chỉ dùng multiple hash + ORB
    """
    
    def __init__(self, config: Optional[LightweightConfig] = None):
        self.config = config or LightweightConfig()
    
    def download_and_normalize(self, url: str) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
        """Download và chuẩn hóa ảnh"""
        try:
            # Fix malformed URLs: "https:data:image" -> "data:image"
            if 'data:image' in url and url.startswith('https:data:'):
                url = url.replace('https:data:', 'data:', 1)
            elif 'data:image' in url and url.startswith('http:data:'):
                url = url.replace('http:data:', 'data:', 1)
            
            if url.startswith('data:image'):
                import base64
                header, data = url.split(',', 1)
                image_data = base64.b64decode(data)
                pil_image = Image.open(BytesIO(image_data))
            else:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                pil_image = Image.open(BytesIO(response.content))
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Resize
            width, height = pil_image.size
            if max(width, height) > self.config.target_size:
                if width > height:
                    new_width = self.config.target_size
                    new_height = int(height * (self.config.target_size / width))
                else:
                    new_height = self.config.target_size
                    new_width = int(width * (self.config.target_size / height))
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return pil_image, cv2_image
            
        except Exception as e:
            return None, None
    
    def compute_hashes(self, pil_image: Image.Image) -> Tuple:
        """Tính multiple perceptual hashes"""
        phash = imagehash.phash(pil_image)
        ahash = imagehash.average_hash(pil_image)
        dhash = imagehash.dhash(pil_image)
        return phash, ahash, dhash
    
    def compute_combined_distance(
        self,
        img1: ImageDataLight,
        img2: ImageDataLight
    ) -> float:
        """
        Tính khoảng cách kết hợp từ nhiều hash
        Trả về giá trị chuẩn hóa 0-64
        """
        if img1.phash is None or img2.phash is None:
            return 64  # Max distance
        
        # Weighted average của các hash distances
        phash_dist = img1.phash - img2.phash
        ahash_dist = img1.ahash - img2.ahash if img1.ahash and img2.ahash else 64
        dhash_dist = img1.dhash - img2.dhash if img1.dhash and img2.dhash else 64
        
        # pHash được weight cao hơn vì robust hơn
        combined = (phash_dist * 0.5 + ahash_dist * 0.25 + dhash_dist * 0.25)
        return combined
    
    def compute_orb_match(self, cv2_img1: np.ndarray, cv2_img2: np.ndarray) -> float:
        """Tính ORB match ratio"""
        try:
            orb = cv2.ORB_create(nfeatures=self.config.orb_num_features)
            
            gray1 = cv2.cvtColor(cv2_img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(cv2_img2, cv2.COLOR_BGR2GRAY)
            
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                return 0.0
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            
            good_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            min_kp = min(len(kp1), len(kp2))
            return len(good_matches) / min_kp if min_kp > 0 else 0.0
            
        except:
            return 0.0
    
    def is_duplicate_pair(self, img1: ImageDataLight, img2: ImageDataLight) -> Tuple[bool, Dict]:
        """Quyết định 2 ảnh có trùng không"""
        result = {
            "combined_hash_distance": None,
            "orb_match_ratio": None,
            "orb_used": False,
            "final_decision": False,
            "reason": None
        }
        
        # Tính combined hash distance
        combined_dist = self.compute_combined_distance(img1, img2)
        result["combined_hash_distance"] = round(combined_dist, 2)
        
        # Trùng rõ ràng
        if combined_dist <= self.config.phash_duplicate_threshold:
            result["final_decision"] = True
            result["reason"] = f"Hash match (combined_dist={combined_dist:.2f})"
            return True, result
        
        # Vùng nghi vấn - check ORB
        if combined_dist <= self.config.phash_suspect_threshold:
            result["orb_used"] = True
            if img1.cv2_image is not None and img2.cv2_image is not None:
                orb_ratio = self.compute_orb_match(img1.cv2_image, img2.cv2_image)
                result["orb_match_ratio"] = round(orb_ratio, 4)
                
                if orb_ratio >= self.config.orb_match_ratio_threshold:
                    result["final_decision"] = True
                    result["reason"] = f"Hash suspect + ORB confirm (dist={combined_dist:.2f}, orb={orb_ratio:.4f})"
                    return True, result
        
        result["reason"] = "No duplicate detected"
        return False, result
    
    def process_images(self, items: List[Dict], verbose: bool = True) -> List[ImageDataLight]:
        """Xử lý tất cả ảnh"""
        images = []
        total = len(items)
        
        print(f"\n📥 Đang xử lý {total} ảnh (lightweight mode)...")
        
        for i, item in enumerate(items):
            url = item.get("image", "")
            if verbose and (i + 1) % 10 == 0:
                print(f"   Xử lý ảnh {i+1}/{total}...")
            
            img_data = ImageDataLight(index=i, url=url)
            
            pil_img, cv2_img = self.download_and_normalize(url)
            
            if pil_img is None:
                img_data.is_valid = False
                images.append(img_data)
                continue
            
            img_data.pil_image = pil_img
            img_data.cv2_image = cv2_img
            
            phash, ahash, dhash = self.compute_hashes(pil_img)
            img_data.phash = phash
            img_data.ahash = ahash
            img_data.dhash = dhash
            
            images.append(img_data)
        
        valid_count = sum(1 for img in images if img.is_valid)
        print(f"✅ Xử lý xong: {valid_count}/{total} ảnh hợp lệ")
        
        return images
    
    def find_duplicates(self, images: List[ImageDataLight], verbose: bool = True) -> List[Set[int]]:
        """Tìm các nhóm trùng"""
        valid_images = [img for img in images if img.is_valid]
        valid_indices = [img.index for img in valid_images]
        
        print(f"\n🔍 Đang so sánh {len(valid_images)} ảnh...")
        
        parent = {i: i for i in valid_indices}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        duplicates_found = 0
        
        for i, img1 in enumerate(valid_images):
            for j, img2 in enumerate(valid_images[i+1:], start=i+1):
                is_dup, details = self.is_duplicate_pair(img1, img2)
                
                if is_dup:
                    union(img1.index, img2.index)
                    duplicates_found += 1
                    if verbose:
                        print(f"   ⚠️ Trùng: ảnh {img1.index} & {img2.index} - {details['reason']}")
        
        groups = defaultdict(set)
        for idx in valid_indices:
            groups[find(idx)].add(idx)
        
        duplicate_groups = [g for g in groups.values() if len(g) > 1]
        
        print(f"✅ Tìm thấy {len(duplicate_groups)} nhóm trùng")
        return duplicate_groups
    
    def deduplicate(self, items: List[Dict], verbose: bool = True) -> Tuple[List[Dict], List[Dict], Dict]:
        """Lọc trùng danh sách items"""
        print("\n" + "="*60)
        print("🚀 LIGHTWEIGHT DUPLICATE DETECTOR (không CLIP)")
        print("="*60)
        
        stats = {
            "total_input": len(items),
            "valid_images": 0,
            "duplicate_groups": 0,
            "duplicates_removed": 0,
            "output_count": 0
        }
        
        images = self.process_images(items, verbose=verbose)
        stats["valid_images"] = sum(1 for img in images if img.is_valid)
        
        duplicate_groups = self.find_duplicates(images, verbose=verbose)
        stats["duplicate_groups"] = len(duplicate_groups)
        
        indices_to_remove = set()
        for group in duplicate_groups:
            sorted_indices = sorted(group)
            indices_to_remove.update(sorted_indices[1:])
        
        stats["duplicates_removed"] = len(indices_to_remove)
        
        deduplicated = [item for i, item in enumerate(items) if i not in indices_to_remove]
        removed = [items[i] for i in sorted(indices_to_remove)]
        
        stats["output_count"] = len(deduplicated)
        
        print(f"\n📊 Kết quả: {stats['total_input']} → {stats['output_count']} (loại {stats['duplicates_removed']})")
        
        return deduplicated, removed, stats


def process_json_lightweight(input_path: str, output_path: str, removed_path: Optional[str] = None):
    """Xử lý file JSON với phiên bản lightweight"""
    print(f"📂 Đọc file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    
    detector = LightweightDuplicateDetector()
    deduplicated, removed, stats = detector.deduplicate(items)
    
    print(f"💾 Lưu kết quả: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(deduplicated, f, ensure_ascii=False, indent=2)
    
    if removed_path and removed:
        with open(removed_path, 'w', encoding='utf-8') as f:
            json.dump(removed, f, ensure_ascii=False, indent=2)
    
    return deduplicated, removed, stats


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pod_duplicate_lightweight.py <input.json> <output.json> [removed.json]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    removed_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    process_json_lightweight(input_file, output_file, removed_file)
