#!/usr/bin/env python3

def read_file_paths(file_path):
    """读取文件中的路径列表"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# 读取四个域的训练文件
domains = ['street', 'fog', 'snow', 'stadium']
domain_files = {}

base_path = '/home/jianyong/exp/MPCount/jhu_domains'
for domain in domains:
    file_path = f"{base_path}/jhu_{domain}_train.txt"
    domain_files[domain] = set(read_file_paths(file_path))
    print(f"域 {domain} 有 {len(domain_files[domain])} 张图片")

# 找出所有重复的图片
street_paths = read_file_paths(f"{base_path}/jhu_street_train.txt")  # 保持原始顺序
street_paths_set = set(street_paths)  # 转换为集合以便进行集合操作
all_duplicates = set()

for domain in ['fog', 'snow', 'stadium']:
    other_paths = domain_files[domain]
    duplicates = street_paths_set.intersection(other_paths)
    all_duplicates.update(duplicates)
    print(f"街道域与{domain}域有 {len(duplicates)} 张重复图片")

print(f"总共有 {len(all_duplicates)} 张重复图片需要移除")

# 创建不包含重复图片的新街道域训练文件
cleaned_street_paths = [path for path in street_paths if path not in all_duplicates]
print(f"清理后的街道域训练集有 {len(cleaned_street_paths)} 张图片")

# 保存新的训练文件
output_file = f"{base_path}/jhu_street_train_cleaned.txt"
with open(output_file, 'w') as f:
    for path in cleaned_street_paths:
        f.write(f"{path}\n")

print(f"已将清理后的文件保存到: {output_file}")

# 可选：如果需要，还可以创建一个只包含重复图片的文件，用于检查
duplicates_file = f"{base_path}/jhu_street_duplicates.txt"
with open(duplicates_file, 'w') as f:
    for path in sorted(all_duplicates):
        f.write(f"{path}\n")

print(f"已将重复图片列表保存到: {duplicates_file}") 