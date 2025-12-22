"""
Debug ORB ratio cho các cặp 0,1,2 và các false positive
"""
import json
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import base64

def download_image(url):
    """Download và convert sang CV2"""
    if 'data:image' in url:
        if url.startswith('https:data:'):
            url = url.replace('https:data:', 'data:', 1)
        header, data = url.split(',', 1)
        image_data = base64.b64decode(data)
        pil_img = Image.open(BytesIO(image_data)).convert('RGB')
    else:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=30)
        pil_img = Image.open(BytesIO(response.content)).convert('RGB')
    
    # Resize
    w, h = pil_img.size
    if max(w, h) > 512:
        if w > h:
            new_w, new_h = 512, int(h * 512 / w)
        else:
            new_h, new_w = 512, int(w * 512 / h)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def compute_orb_match_ratio(img1, img2, nfeatures=500):
    """Compute ORB match ratio"""
    orb = cv2.ORB_create(nfeatures=nfeatures)
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0.0
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    max_possible = min(len(kp1), len(kp2))
    if max_possible == 0:
        return 0.0
    
    return len(good_matches) / max_possible

# Load data
with open('output.json', 'r', encoding='utf-8') as f:
    items = json.load(f)

# Load images
print("Loading images...")
images = {}
indices_to_check = [0, 1, 2, 9, 22, 27, 32, 34]
for idx in indices_to_check:
    url = items[idx].get('image', '')
    images[idx] = download_image(url)
    print(f"  [{idx}] {items[idx].get('title', '')[:40]}...")

print("\n" + "="*70)
print("ORB MATCH RATIOS")
print("="*70)

pairs_to_check = [
    (0, 1, "TRUE POSITIVE - cùng design"),
    (0, 2, "TRUE POSITIVE - cùng design"),
    (1, 2, "TRUE POSITIVE - cùng design"),
    (9, 22, "FALSE POSITIVE - khác design"),
    (22, 32, "FALSE POSITIVE - khác design"),
    (22, 34, "FALSE POSITIVE - khác design"),
    (27, 32, "FALSE POSITIVE - khác design"),
]

for i, j, label in pairs_to_check:
    orb = compute_orb_match_ratio(images[i], images[j])
    status = "✅ >= 0.25" if orb >= 0.25 else "❌ < 0.25"
    print(f"\n{i} vs {j} ({label}):")
    print(f"  ORB Match Ratio: {orb:.4f} {status}")
