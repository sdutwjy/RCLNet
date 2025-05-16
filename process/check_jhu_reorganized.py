#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from collections import Counter

def count_images(base_dir):
    """统计重组后数据集中各个域和分割的图像数量"""
    stats = {}
    total_images = 0
    total_dmaps = 0
    
    # 检查基础目录
    if not os.path.exists(base_dir):
        print(f"错误: 目录不存在 {base_dir}")
        return
    
    # 遍历域和分割
    domains = ['street', 'stadium', 'snow', 'fog']
    splits = ['train', 'val']
    
    for domain in domains:
        domain_stats = {}
        domain_total_images = 0
        domain_total_dmaps = 0
        
        for split in splits:
            path = os.path.join(base_dir, domain, split)
            
            if not os.path.exists(path):
                print(f"警告: 目录不存在 {path}")
                domain_stats[split] = {"images": 0, "dmaps": 0}
                continue
            
            # 统计图像数量
            images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_count = len(images)
            
            # 统计密度图数量
            dmaps = [f for f in os.listdir(path) if f.lower().endswith('_dmap.npy')]
            dmap_count = len(dmaps)
            
            domain_stats[split] = {"images": image_count, "dmaps": dmap_count}
            domain_total_images += image_count
            domain_total_dmaps += dmap_count
            
            print(f"{domain}/{split}: {image_count} 张图像, {dmap_count} 个密度图")
            
            # 检查是否每张图像都有对应的密度图
            if image_count != dmap_count:
                print(f"  注意: 图像数量 ({image_count}) 与密度图数量 ({dmap_count}) 不匹配!")
        
        stats[domain] = domain_stats
        stats[domain]['total_images'] = domain_total_images
        stats[domain]['total_dmaps'] = domain_total_dmaps
        total_images += domain_total_images
        total_dmaps += domain_total_dmaps
        
        print(f"{domain} 总计: {domain_total_images} 张图像, {domain_total_dmaps} 个密度图")
        print("-" * 40)
    
    print(f"数据集总计: {total_images} 张图像, {total_dmaps} 个密度图")
    
    # 输出总体匹配率
    match_rate = 100.0 if total_images == 0 else (total_dmaps / total_images * 100)
    print(f"图像-密度图匹配率: {match_rate:.2f}%")
    
    return stats

def check_symlinks(base_dir):
    """检查有多少是符号链接以及多少是实际文件"""
    symlink_count = 0
    file_count = 0
    
    image_links = 0
    image_files = 0
    dmap_links = 0
    dmap_files = 0
    
    # 检查基础目录
    if not os.path.exists(base_dir):
        print(f"错误: 目录不存在 {base_dir}")
        return
    
    # 遍历所有文件
    for root, _, files in os.walk(base_dir):
        for filename in files:
            full_path = os.path.join(root, filename)
            
            # 判断文件类型
            is_image = filename.lower().endswith(('.jpg', '.jpeg', '.png'))
            is_dmap = filename.lower().endswith('_dmap.npy')
            
            if os.path.islink(full_path):
                symlink_count += 1
                if is_image:
                    image_links += 1
                elif is_dmap:
                    dmap_links += 1
            else:
                file_count += 1
                if is_image:
                    image_files += 1
                elif is_dmap:
                    dmap_files += 1
    
    print(f"\n符号链接总数: {symlink_count}")
    print(f"  图像符号链接: {image_links}")
    print(f"  密度图符号链接: {dmap_links}")
    
    print(f"\n实际文件总数: {file_count}")
    print(f"  图像文件: {image_files}")
    print(f"  密度图文件: {dmap_files}")
    
    return {
        "symlinks": {
            "total": symlink_count,
            "images": image_links,
            "dmaps": dmap_links
        },
        "files": {
            "total": file_count,
            "images": image_files,
            "dmaps": dmap_files
        }
    }

def main():
    parser = argparse.ArgumentParser(description='检查重组JHU数据集的统计信息')
    parser.add_argument('--dir', type=str, default='jhu_reorganized',
                        help='重组数据集的目录路径')
    parser.add_argument('--check_symlinks', action='store_true',
                        help='是否检查符号链接和实际文件的数量')
    
    args = parser.parse_args()
    
    # 统计图像数量
    print(f"统计 {args.dir} 中的图像和密度图...")
    print("=" * 50)
    stats = count_images(args.dir)
    
    # 如果需要，检查符号链接
    if args.check_symlinks:
        print("\n检查符号链接和实际文件...")
        print("=" * 50)
        link_stats = check_symlinks(args.dir)

if __name__ == "__main__":
    main() 