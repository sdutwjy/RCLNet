#!/usr/bin/env python3

def read_file_paths(file_path):
    """读取文件中的路径列表"""
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f if line.strip())

# 读取四个域的训练文件
domains = ['street', 'fog', 'snow', 'stadium']
domain_files = {}

base_path = '/home/jianyong/exp/MPCount/jhu_domains'
for domain in domains:
    file_path = f"{base_path}/jhu_{domain}_train.txt"
    domain_files[domain] = read_file_paths(file_path)
    print(f"域 {domain} 有 {len(domain_files[domain])} 张图片")

# 查找street域与其他域的重复图片
street_paths = domain_files['street']
for domain in ['fog', 'snow', 'stadium']:
    other_paths = domain_files[domain]
    duplicates = street_paths.intersection(other_paths)
    
    print(f"\n街道域与{domain}域的重复图片 ({len(duplicates)}张):")
    for path in sorted(duplicates):
        print(f"  {path}")

# 保存重复图片到文件
with open('street_duplicates.txt', 'w') as f:
    f.write("# 街道域与其他域的重复图片\n\n")
    
    for domain in ['fog', 'snow', 'stadium']:
        other_paths = domain_files[domain]
        duplicates = street_paths.intersection(other_paths)
        
        f.write(f"## 街道域与{domain}域的重复图片 ({len(duplicates)}张)\n")
        for path in sorted(duplicates):
            f.write(f"{path}\n")
        f.write("\n")

print("\n重复图片已保存到 street_duplicates.txt")

# 计算所有重复图片的总数
all_duplicates = set()
for domain in ['fog', 'snow', 'stadium']:
    other_paths = domain_files[domain]
    duplicates = street_paths.intersection(other_paths)
    all_duplicates.update(duplicates)

print(f"\n街道域与其他域的所有重复图片总数: {len(all_duplicates)}张") 