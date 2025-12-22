# Debug script to find Morgan Wallen shirts and check their similarity
import json

# Load data
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== TÌM ẢNH MORGAN WALLEN ===")
morgan_indices = []
for i, item in enumerate(data):
    title = item.get('title', '').lower()
    if 'wallen' in title or 'morgan' in title:
        print(f"{i}: {item.get('title', '')[:70]}")
        morgan_indices.append(i)

print(f"\nTìm thấy {len(morgan_indices)} ảnh Morgan Wallen: {morgan_indices}")

# Now check CLIP similarity between these
print("\n=== CHECK CLIP SIMILARITY ===")
from pod_duplicate_detector import PODDuplicateDetector, DuplicateConfig
import os

config = DuplicateConfig()
detector = PODDuplicateDetector(config)

# Process only Morgan Wallen images
if morgan_indices:
    images = detector.process_images(data)
    
    print("\n=== CLIP SIMILARITIES BETWEEN MORGAN WALLEN IMAGES ===")
    for i, idx1 in enumerate(morgan_indices):
        for idx2 in morgan_indices[i+1:]:
            img1 = images[idx1]
            img2 = images[idx2]
            
            if img1.is_valid and img2.is_valid:
                if img1.clip_embedding is not None and img2.clip_embedding is not None:
                    sim = detector.compute_clip_similarity(img1.clip_embedding, img2.clip_embedding)
                    phash_dist = detector.compare_phash(img1.phash, img2.phash) if img1.phash and img2.phash else None
                    print(f"Ảnh {idx1} vs {idx2}: CLIP={sim:.4f}, pHash={phash_dist}")
                else:
                    print(f"Ảnh {idx1} vs {idx2}: CLIP embedding missing!")
            else:
                print(f"Ảnh {idx1} vs {idx2}: Invalid image!")
