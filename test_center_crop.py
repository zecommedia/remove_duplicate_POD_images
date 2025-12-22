"""
Test center crop với 3 ảnh Morgan Wallen
So sánh CLIP similarity giữa full image vs center crop
"""
import torch
import open_clip
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load CLIP model
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
model.eval()
print(f"CLIP loaded on {device}")

def download_image(url):
    """Download image from URL"""
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
    """Compute cosine similarity"""
    return float(cosine_similarity([emb1], [emb2])[0][0])

# 3 ảnh Morgan Wallen (giả sử là 3 URL đầu tiên trong dataset)
# Thực tế cần URL thật, tạm thời test với output.json
import json

with open('output.json', 'r', encoding='utf-8') as f:
    items = json.load(f)

print(f"\nLoading first 3 images from dataset...")
images = []
for i in range(min(3, len(items))):
    url = items[i].get('image', '')
    print(f"  [{i}] {items[i].get('title', 'No title')[:50]}...")
    
    # Handle data:image URLs
    if 'data:image' in url:
        if url.startswith('https:data:'):
            url = url.replace('https:data:', 'data:', 1)
        import base64
        header, data = url.split(',', 1)
        image_data = base64.b64decode(data)
        img = Image.open(BytesIO(image_data)).convert('RGB')
    else:
        img = download_image(url)
    
    images.append(img)

print("\n" + "="*70)
print("SO SÁNH CLIP SIMILARITY")
print("="*70)

for i in range(len(images)):
    for j in range(i+1, len(images)):
        img1, img2 = images[i], images[j]
        
        # Full image embeddings
        emb1_full = get_clip_embedding(img1)
        emb2_full = get_clip_embedding(img2)
        sim_full = compute_similarity(emb1_full, emb2_full)
        
        # Center crop embeddings
        emb1_center = get_clip_embedding(center_crop(img1))
        emb2_center = get_clip_embedding(center_crop(img2))
        sim_center = compute_similarity(emb1_center, emb2_center)
        
        # MAX
        sim_max = max(sim_full, sim_center)
        
        print(f"\nImage {i} vs Image {j}:")
        print(f"  Full image:   {sim_full:.4f}")
        print(f"  Center crop:  {sim_center:.4f}")
        print(f"  MAX:          {sim_max:.4f} {'✅ HIGHER!' if sim_center > sim_full else ''}")
        
        # Cải thiện
        improvement = (sim_center - sim_full) / sim_full * 100
        if improvement > 0:
            print(f"  Improvement:  +{improvement:.1f}%")
