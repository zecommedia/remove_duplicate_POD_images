"""
Test logic mới: Center crop chỉ được dùng khi full image đạt ngưỡng tối thiểu
"""
import torch
import open_clip
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load CLIP model
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
model.eval()
print(f"CLIP loaded on {device}")

def download_image(url):
    """Download image from URL"""
    if 'data:image' in url:
        if url.startswith('https:data:'):
            url = url.replace('https:data:', 'data:', 1)
        import base64
        header, data = url.split(',', 1)
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data)).convert('RGB')
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=30)
    return Image.open(BytesIO(response.content)).convert('RGB')

def center_crop(img, ratio=0.65):
    """Crop center of image"""
    w, h = img.size
    new_w, new_h = int(w * ratio), int(h * ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))

def get_clip_embedding(img):
    """Get CLIP embedding for image"""
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()

def compute_similarity(emb1, emb2):
    return float(cosine_similarity([emb1], [emb2])[0][0])

# Thresholds - TINH CHỈNH
FULL_THRESHOLD = 0.86       # Ngưỡng cho full image (cao hơn để tránh false positive)
CENTER_THRESHOLD = 0.83     # Ngưỡng cho center crop
MIN_CENTER_BOOST = 0.04     # Center phải cao hơn full ít nhất 4% mới được dùng

def check_duplicate_new_logic(img1, img2, name):
    """
    Check duplicate với logic tinh chỉnh:
    1. Nếu full image >= FULL_THRESHOLD: DUPLICATE (rất chắc chắn)
    2. Nếu center >= CENTER_THRESHOLD VÀ center > full + MIN_BOOST: DUPLICATE (watermark case)
    3. Else: NOT duplicate
    """
    # Full image
    emb1_full = get_clip_embedding(img1)
    emb2_full = get_clip_embedding(img2)
    sim_full = compute_similarity(emb1_full, emb2_full)
    
    # Center crop
    emb1_center = get_clip_embedding(center_crop(img1))
    emb2_center = get_clip_embedding(center_crop(img2))
    sim_center = compute_similarity(emb1_center, emb2_center)
    
    boost = sim_center - sim_full
    
    # LOGIC TINH CHỈNH
    is_duplicate = False
    reason = ""
    
    if sim_full >= FULL_THRESHOLD:
        # Full image rất cao → DUPLICATE chắc chắn
        is_duplicate = True
        reason = f"full >= {FULL_THRESHOLD}"
    elif sim_center >= CENTER_THRESHOLD and boost >= MIN_CENTER_BOOST:
        # Center cao + boost đáng kể → watermark case
        is_duplicate = True
        reason = f"center boost (+{boost*100:.1f}%)"
    else:
        reason = "not enough evidence"
    
    print(f"\n{name}:")
    print(f"  Full image:   {sim_full:.4f}")
    print(f"  Center crop:  {sim_center:.4f}")
    print(f"  Boost:        {boost*100:+.1f}%")
    print(f"  Decision:     {'✅ DUPLICATE' if is_duplicate else '❌ NOT duplicate'} ({reason})")
    
    return is_duplicate, sim_full, sim_center, boost
    
    return is_duplicate, sim_full, sim_center, sim_final

# Load images
with open('output.json', 'r', encoding='utf-8') as f:
    items = json.load(f)

print(f"\nLoading images...")
images = {}
for idx in [0, 1, 2, 9, 22]:
    url = items[idx].get('image', '')
    images[idx] = download_image(url)
    print(f"  [{idx}] {items[idx].get('title', '')[:40]}...")

print("\n" + "="*70)
print("TEST VỚI LOGIC MỚI")
print(f"FULL threshold: {FULL_THRESHOLD}")
print(f"CENTER threshold: {CENTER_THRESHOLD}")
print(f"Min center boost: {MIN_CENTER_BOOST*100:.0f}%")
print("="*70)

# Test các cặp
print("\n--- 3 ẢNH ĐẦU (nên là DUPLICATE) ---")
check_duplicate_new_logic(images[0], images[1], "0 vs 1")
check_duplicate_new_logic(images[0], images[2], "0 vs 2")
check_duplicate_new_logic(images[1], images[2], "1 vs 2")

print("\n--- CẶP FALSE POSITIVE (không nên là DUPLICATE) ---")
check_duplicate_new_logic(images[9], images[22], "9 vs 22 (Demogorgon vs Text Logo)")
