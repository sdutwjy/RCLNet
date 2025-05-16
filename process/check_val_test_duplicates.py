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

# 街道域文件
street_val_file = f"{base_path}/jhu_street_val.txt"
street_test_file = f"{base_path}/jhu_street_test.txt"

# 其他域文件
domains = ['fog', 'snow', 'stadium']
domain_files = {}

# 读取所有域的训练、验证和测试文件
print("读取所有域文件...")
for domain in domains:
    domain_files[domain] = {
        'train': set(read_file_paths(f"{base_path}/jhu_{domain}_train.txt")),
        'val': set(read_file_paths(f"{base_path}/jhu_{domain}_val.txt")),
        'test': set(read_file_paths(f"{base_path}/jhu_{domain}_test.txt"))
    }
    
    total = len(domain_files[domain]['train']) + len(domain_files[domain]['val']) + len(domain_files[domain]['test'])
    print(f"{domain}域总图片数: {total} (训练: {len(domain_files[domain]['train'])}, " + 
          f"验证: {len(domain_files[domain]['val'])}, 测试: {len(domain_files[domain]['test'])})")

# 读取街道域的验证和测试文件
street_val_paths = read_file_paths(street_val_file)
street_test_paths = read_file_paths(street_test_file)

print(f"\n街道域验证集图片数: {len(street_val_paths)}")
print(f"街道域测试集图片数: {len(street_test_paths)}")

# 检查验证集中的重复
print("\n== 验证集重复检查 ==")
val_duplicates = {}
for domain in domains:
    # 与其他域的全部数据进行对比
    all_domain_paths = domain_files[domain]['train'] | domain_files[domain]['val'] | domain_files[domain]['test']
    dup = [path for path in street_val_paths if path in all_domain_paths]
    val_duplicates[domain] = dup
    print(f"街道域验证集与{domain}域的重复图片: {len(dup)}张")
    if len(dup) > 0 and len(dup) <= 10:
        for path in dup:
            print(f"  {path}")

# 检查测试集中的重复
print("\n== 测试集重复检查 ==")
test_duplicates = {}
for domain in domains:
    # 与其他域的全部数据进行对比
    all_domain_paths = domain_files[domain]['train'] | domain_files[domain]['val'] | domain_files[domain]['test']
    dup = [path for path in street_test_paths if path in all_domain_paths]
    test_duplicates[domain] = dup
    print(f"街道域测试集与{domain}域的重复图片: {len(dup)}张")
    if len(dup) > 0 and len(dup) <= 10:
        for path in dup:
            print(f"  {path}")

# 计算总重复数
all_val_duplicates = set()
for domain in domains:
    all_val_duplicates.update(val_duplicates[domain])

all_test_duplicates = set()
for domain in domains:
    all_test_duplicates.update(test_duplicates[domain])

print(f"\n街道域验证集中共有 {len(all_val_duplicates)} 张与其他域重复的图片")
print(f"街道域测试集中共有 {len(all_test_duplicates)} 张与其他域重复的图片")

# 保存重复图片列表
if len(all_val_duplicates) > 0:
    val_duplicates_file = f"{base_path}/street_val_duplicates.txt"
    with open(val_duplicates_file, 'w') as f:
        f.write("# 街道域验证集中与其他域重复的图片\n\n")
        for path in sorted(all_val_duplicates):
            f.write(f"{path}\n")
    print(f"\n已将验证集重复图片列表保存到: {val_duplicates_file}")

if len(all_test_duplicates) > 0:
    test_duplicates_file = f"{base_path}/street_test_duplicates.txt"
    with open(test_duplicates_file, 'w') as f:
        f.write("# 街道域测试集中与其他域重复的图片\n\n")
        for path in sorted(all_test_duplicates):
            f.write(f"{path}\n")
    print(f"已将测试集重复图片列表保存到: {test_duplicates_file}")

# 如果需要清理重复图片，请取消下面的注释
"""
# 清理验证集
if len(all_val_duplicates) > 0:
    import shutil
    # 创建备份
    backup_file = f"{street_val_file}.bak"
    shutil.copy2(street_val_file, backup_file)
    print(f"\n已创建验证集原文件备份: {backup_file}")

    # 清理验证集
    cleaned_val_paths = [path for path in street_val_paths if path not in all_val_duplicates]
    with open(street_val_file, 'w') as f:
        for path in cleaned_val_paths:
            f.write(f"{path}\n")
    print(f"已清理并更新验证集文件: {street_val_file}")
    print(f"验证集从 {len(street_val_paths)} 张减少到 {len(cleaned_val_paths)} 张")

# 清理测试集
if len(all_test_duplicates) > 0:
    import shutil
    # 创建备份
    backup_file = f"{street_test_file}.bak"
    shutil.copy2(street_test_file, backup_file)
    print(f"已创建测试集原文件备份: {backup_file}")

    # 清理测试集
    cleaned_test_paths = [path for path in street_test_paths if path not in all_test_duplicates]
    with open(street_test_file, 'w') as f:
        for path in cleaned_test_paths:
            f.write(f"{path}\n")
    print(f"已清理并更新测试集文件: {street_test_file}")
    print(f"测试集从 {len(street_test_paths)} 张减少到 {len(cleaned_test_paths)} 张")
""" 