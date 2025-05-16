#!/usr/bin/env python3

def read_file_paths(file_path):
    """读取文件中的路径列表"""
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"警告: 文件 {file_path} 不存在")
        return []

# 文件路径
base_path = '/home/jianyong/exp/MPCount/jhu_domains'

# 街道域验证集文件
street_val_file = f"{base_path}/jhu_street_val.txt"

# 读取已经找到的重复图片列表
duplicates_file = f"{base_path}/street_val_duplicates.txt"
all_duplicates = set()

try:
    with open(duplicates_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                all_duplicates.add(line)
    print(f"从 {duplicates_file} 读取了 {len(all_duplicates)} 张重复图片")
except FileNotFoundError:
    print(f"警告: 重复图片列表文件 {duplicates_file} 不存在")

# 如果没有找到重复图片列表，则自动检测重复
if len(all_duplicates) == 0:
    print("没有找到重复图片列表，将自动检测重复...")
    
    # 其他域文件
    domains = ['fog', 'snow', 'stadium']
    domain_files = {}

    # 读取所有域的训练、验证和测试文件
    for domain in domains:
        domain_files[domain] = {
            'train': set(read_file_paths(f"{base_path}/jhu_{domain}_train.txt")),
            'val': set(read_file_paths(f"{base_path}/jhu_{domain}_val.txt")),
            'test': set(read_file_paths(f"{base_path}/jhu_{domain}_test.txt"))
        }
    
    # 读取街道域的验证文件
    street_val_paths = read_file_paths(street_val_file)
    
    # 检查验证集中的重复
    val_duplicates = {}
    for domain in domains:
        # 与其他域的全部数据进行对比
        all_domain_paths = domain_files[domain]['train'] | domain_files[domain]['val'] | domain_files[domain]['test']
        dup = [path for path in street_val_paths if path in all_domain_paths]
        val_duplicates[domain] = dup
        print(f"街道域验证集与{domain}域的重复图片: {len(dup)}张")
    
    # 计算总重复数
    for domain in domains:
        all_duplicates.update(val_duplicates[domain])
    
    print(f"总共检测到 {len(all_duplicates)} 张与其他域重复的图片")

# 读取验证集文件
street_val_paths = read_file_paths(street_val_file)
print(f"街道域验证集原始图片数: {len(street_val_paths)}")

# 创建备份
import shutil
backup_file = f"{street_val_file}.bak"
shutil.copy2(street_val_file, backup_file)
print(f"已创建验证集原文件备份: {backup_file}")

# 清理验证集
cleaned_val_paths = [path for path in street_val_paths if path not in all_duplicates]
with open(street_val_file, 'w') as f:
    for path in cleaned_val_paths:
        f.write(f"{path}\n")

print(f"已清理并更新验证集文件: {street_val_file}")
print(f"验证集从 {len(street_val_paths)} 张减少到 {len(cleaned_val_paths)} 张")

# 保存移除的图片列表
removed_file = f"{base_path}/removed_val_duplicates.txt"
with open(removed_file, 'w') as f:
    f.write("# 从街道域验证集中移除的重复图片\n\n")
    for path in sorted(all_duplicates):
        f.write(f"{path}\n")

print(f"已将移除的验证集图片列表保存到: {removed_file}") 