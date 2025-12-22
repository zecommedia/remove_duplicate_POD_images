# Debug: Check similarity between first 3 images
import json
from pod_duplicate_detector import PODDuplicateDetector, DuplicateConfig

# Load data
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== 3 ẢNH ĐẦU TIÊN ===")
for i in range(min(3, len(data))):
    print(f"{i}: {data[i].get('title', 'No title')[:70]}")

print("\n=== LOADING DETECTOR ===")
config = DuplicateConfig()
detector = PODDuplicateDetector(config)

# Process images
images = detector.process_images(data)

print("\n=== CLIP SIMILARITIES GIỮA 3 ẢNH ĐẦU ===")
pairs = [(0, 1), (0, 2), (1, 2)]
for idx1, idx2 in pairs:
    img1, img2 = images[idx1], images[idx2]
    if img1.is_valid and img2.is_valid:
        if img1.clip_embedding is not None and img2.clip_embedding is not None:
            clip_sim = detector.compute_clip_similarity(img1.clip_embedding, img2.clip_embedding)
            phash_dist = detector.compare_phash(img1.phash, img2.phash) if img1.phash and img2.phash else None
            print(f"Ảnh {idx1} vs {idx2}: CLIP={clip_sim:.4f}, pHash={phash_dist}")
        else:
            print(f"Ảnh {idx1} vs {idx2}: CLIP embedding missing!")
    else:
        print(f"Ảnh {idx1} vs {idx2}: Invalid image(s)!")
