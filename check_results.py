import json

# Kiểm tra kết quả
with open('output_deduplicated.json', 'r', encoding='utf-8') as f:
    unique = json.load(f)
    
with open('output_removed.json', 'r', encoding='utf-8') as f:
    removed = json.load(f)
    
with open('output_duplicate_pairs.json', 'r', encoding='utf-8') as f:
    pairs = json.load(f)
    
with open('output.json', 'r', encoding='utf-8') as f:
    original = json.load(f)

print(f"Original: {len(original)}")
print(f"Unique:   {len(unique)}")
print(f"Removed:  {len(removed)}")
print(f"Pairs:    {len(pairs)}")

# Kiểm tra xem index 0, 1, 2 có được group cùng nhau không
# Tìm trong pairs
print("\n=== Pairs liên quan đến index 0, 1, 2 ===")
for p in pairs:
    i = p['image1_index']
    j = p['image2_index']
    if i in [0, 1, 2] or j in [0, 1, 2]:
        print(f"  Index {i} & {j}: CLIP={p['clip_similarity']:.4f}")

# Kiểm tra 3 ảnh đầu đi vào unique hay removed
print("\n=== 3 ảnh đầu ở đâu? ===")
# Lưu ý: output_deduplicated và output_removed không lưu original index
# Cần tìm cách khác
