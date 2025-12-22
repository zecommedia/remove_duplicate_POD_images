# Debug script to check why pairs are not being detected
import json
from pod_duplicate_detector import PODDuplicateDetector, DuplicateConfig

# Load data
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== LOADING ===")
config = DuplicateConfig()
print(f"CLIP duplicate threshold: {config.clip_duplicate_threshold}")
print(f"CLIP suspect threshold: {config.clip_suspect_threshold}")

detector = PODDuplicateDetector(config)

# Process images
images = detector.process_images(data)

# Check specific pair (34 & 36) that had high similarity
idx1, idx2 = 34, 36
img1 = images[idx1]
img2 = images[idx2]

print(f"\n=== CHECKING PAIR {idx1} & {idx2} ===")
print(f"Image {idx1} valid: {img1.is_valid}")
print(f"Image {idx2} valid: {img2.is_valid}")

if img1.is_valid and img2.is_valid:
    # Compute CLIP similarity
    if img1.clip_embedding is not None and img2.clip_embedding is not None:
        clip_sim = detector.compute_clip_similarity(img1.clip_embedding, img2.clip_embedding)
        clip_class = detector.classify_clip_similarity(clip_sim)
        print(f"CLIP similarity: {clip_sim:.4f}")
        print(f"CLIP classification: {clip_class}")
        
        # Check is_duplicate_pair
        is_dup, details = detector.is_duplicate_pair(img1, img2, verbose=True)
        print(f"\nis_duplicate_pair result: {is_dup}")
        print(f"Details: {details}")
    else:
        print("CLIP embeddings missing!")
        print(f"img1.clip_embedding: {img1.clip_embedding is not None}")
        print(f"img2.clip_embedding: {img2.clip_embedding is not None}")
