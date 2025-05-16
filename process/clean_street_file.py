#!/usr/bin/env python3

def read_file_paths(file_path):
    """读取文件中的路径列表"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# 文件路径
base_path = '/home/jianyong/exp/MPCount/jhu_domains'
street_file = f"{base_path}/jhu_street_train.txt"
fog_file = f"{base_path}/jhu_fog_train.txt"
snow_file = f"{base_path}/jhu_snow_train.txt"
stadium_file = f"{base_path}/jhu_stadium_train.txt"

# 读取各个域的文件
print("读取各域文件...")
street_paths = read_file_paths(street_file)
fog_paths = set(read_file_paths(fog_file))
snow_paths = set(read_file_paths(snow_file))
stadium_paths = set(read_file_paths(stadium_file))

print(f"街道域原始图片数: {len(street_paths)}")
print(f"雾域图片数: {len(fog_paths)}")
print(f"雪域图片数: {len(snow_paths)}")
print(f"体育场域图片数: {len(stadium_paths)}")

# 找出所有重复的图片
duplicates = set()
for path in street_paths:
    if path in fog_paths or path in snow_paths or path in stadium_paths:
        duplicates.add(path)

print(f"找到 {len(duplicates)} 张重复图片")

# 创建不包含重复图片的新街道路径列表
cleaned_street_paths = [path for path in street_paths if path not in duplicates]
print(f"清理后的街道域图片数: {len(cleaned_street_paths)}")

# 创建备份
import shutil
backup_file = f"{street_file}.bak"
shutil.copy2(street_file, backup_file)
print(f"已创建原文件备份: {backup_file}")

# 将清理后的路径写回原文件
with open(street_file, 'w') as f:
    for path in cleaned_street_paths:
        f.write(f"{path}\n")

print(f"已清理并更新文件: {street_file}")

# 可选：保存被移除的重复图片路径
duplicates_file = f"{base_path}/removed_duplicates.txt"
with open(duplicates_file, 'w') as f:
    f.write("# 从街道域训练集中移除的重复图片\n\n")
    for path in sorted(duplicates):
        f.write(f"{path}\n")

print(f"已将移除的图片列表保存到: {duplicates_file}") 