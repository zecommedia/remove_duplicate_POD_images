"""
Preview các nhóm duplicate đã được gom bởi Union-Find
"""
import json
from collections import defaultdict

# Load data
with open('output_duplicate_pairs.json', 'r', encoding='utf-8') as f:
    pairs = json.load(f)

with open('output.json', 'r', encoding='utf-8') as f:
    original = json.load(f)

# Xây dựng Union-Find từ pairs
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

# Gom nhóm
n = len(original)
uf = UnionFind(n)

for p in pairs:
    i, j = p['image1_index'], p['image2_index']
    uf.union(i, j)

# Tạo groups
groups = defaultdict(list)
for i in range(n):
    root = uf.find(i)
    groups[root].append(i)

# Lọc chỉ các nhóm có > 1 phần tử (nhóm trùng)
duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}

print(f"=== TỔNG QUAN ===")
print(f"Tổng ảnh: {n}")
print(f"Số nhóm trùng: {len(duplicate_groups)}")
print(f"Số ảnh unique (không trùng): {len([v for v in groups.values() if len(v) == 1])}")
print()

# In chi tiết từng nhóm
print(f"=== CHI TIẾT CÁC NHÓM TRÙNG ===\n")

for group_id, (root, members) in enumerate(sorted(duplicate_groups.items(), key=lambda x: -len(x[1])), 1):
    print(f"{'='*60}")
    print(f"NHÓM {group_id}: {len(members)} ảnh trùng")
    print(f"{'='*60}")
    
    for idx in sorted(members):
        item = original[idx]
        title = item.get('title', 'No title')[:60]
        print(f"  [{idx:2d}] {title}")
    
    # In CLIP similarity giữa các cặp trong nhóm
    print(f"\n  Độ tương đồng CLIP trong nhóm:")
    member_set = set(members)
    for p in pairs:
        i, j = p['image1_index'], p['image2_index']
        if i in member_set and j in member_set:
            print(f"    {i} ↔ {j}: CLIP={p['clip_similarity']:.4f}")
    print()

# Thống kê
print(f"\n{'='*60}")
print(f"THỐNG KÊ")
print(f"{'='*60}")
sizes = [len(v) for v in duplicate_groups.values()]
print(f"Nhóm lớn nhất: {max(sizes)} ảnh")
print(f"Nhóm nhỏ nhất: {min(sizes)} ảnh")
print(f"Trung bình: {sum(sizes)/len(sizes):.1f} ảnh/nhóm")
