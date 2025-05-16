#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
import argparse
import glob

def parse_domain_file(file_path):
    """解析域文件并返回图像路径列表"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def get_corresponding_dmaps(img_path):
    """获取对应的密度图文件路径（可能有多个）"""
    # 从图像路径中提取文件名（不带扩展名）
    file_base = os.path.splitext(os.path.basename(img_path))[0]
    img_dir = os.path.dirname(img_path)
    
    dmaps = []
    
    # 检查常见的两种命名格式
    dmap1_path = os.path.join(img_dir, f"{file_base}_dmap.npy")
    dmap2_path = os.path.join(img_dir, f"{file_base}.npy")
    
    if os.path.exists(dmap1_path):
        dmaps.append(dmap1_path)
    
    if os.path.exists(dmap2_path):
        dmaps.append(dmap2_path)
    
    # 如果没有找到任何密度图，尝试模糊匹配
    if not dmaps:
        potential_dmaps = glob.glob(os.path.join(img_dir, f"{file_base}*.npy"))
        dmaps.extend(potential_dmaps)
    
    return dmaps

def create_symlinks(image_paths, target_dir, domain_name, split, with_dmap=True):
    """创建图像符号链接到目标目录"""
    target_path = os.path.join(target_dir, domain_name, split)
    os.makedirs(target_path, exist_ok=True)
    
    processed_count = 0
    dmap_count = 0
    missing_dmap_count = 0
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"警告: 图像不存在 {img_path}")
            continue
        
        # 处理图像文件
        filename = os.path.basename(img_path)
        symlink_path = os.path.join(target_path, filename)
        
        # 如果链接已存在则移除
        if os.path.exists(symlink_path):
            os.remove(symlink_path)
        
        # 创建符号链接
        try:
            os.symlink(img_path, symlink_path)
            processed_count += 1
        except Exception as e:
            print(f"创建链接失败: {img_path} -> {symlink_path}，错误: {e}")
            
        # 处理对应的密度图文件
        if with_dmap:
            dmap_paths = get_corresponding_dmaps(img_path)
            
            if dmap_paths:
                for dmap_path in dmap_paths:
                    dmap_filename = os.path.basename(dmap_path)
                    dmap_symlink_path = os.path.join(target_path, dmap_filename)
                    
                    # 如果链接已存在则移除
                    if os.path.exists(dmap_symlink_path):
                        os.remove(dmap_symlink_path)
                    
                    # 创建密度图的符号链接
                    try:
                        os.symlink(dmap_path, dmap_symlink_path)
                        dmap_count += 1
                    except Exception as e:
                        print(f"创建密度图链接失败: {dmap_path} -> {dmap_symlink_path}，错误: {e}")
            else:
                print(f"警告: 未找到对应的密度图，对应图像: {img_path}")
                missing_dmap_count += 1
    
    print(f"已创建 {processed_count} 个图像链接和 {dmap_count} 个密度图链接，缺失 {missing_dmap_count} 个密度图")

def copy_images(image_paths, target_dir, domain_name, split, with_dmap=True):
    """复制图像到目标目录"""
    target_path = os.path.join(target_dir, domain_name, split)
    os.makedirs(target_path, exist_ok=True)
    
    processed_count = 0
    dmap_count = 0
    missing_dmap_count = 0
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"警告: 图像不存在 {img_path}")
            continue
        
        # 处理图像文件
        filename = os.path.basename(img_path)
        target_file = os.path.join(target_path, filename)
        
        try:
            shutil.copy2(img_path, target_file)
            processed_count += 1
        except Exception as e:
            print(f"复制失败: {img_path} -> {target_file}，错误: {e}")
            
        # 处理对应的密度图文件
        if with_dmap:
            dmap_paths = get_corresponding_dmaps(img_path)
            
            if dmap_paths:
                for dmap_path in dmap_paths:
                    dmap_filename = os.path.basename(dmap_path)
                    dmap_target_file = os.path.join(target_path, dmap_filename)
                    
                    try:
                        shutil.copy2(dmap_path, dmap_target_file)
                        dmap_count += 1
                    except Exception as e:
                        print(f"复制密度图失败: {dmap_path} -> {dmap_target_file}，错误: {e}")
            else:
                print(f"警告: 未找到对应的密度图，对应图像: {img_path}")
                missing_dmap_count += 1
    
    print(f"已复制 {processed_count} 个图像和 {dmap_count} 个密度图，缺失 {missing_dmap_count} 个密度图")

def process_folder_directly(input_dir, target_dir, domain_name, split, with_dmap=True, use_symlinks=False):
    """直接处理文件夹中的所有图像和密度图"""
    source_path = os.path.join(input_dir, domain_name, split)
    if not os.path.exists(source_path):
        print(f"警告: 源目录不存在 {source_path}")
        return
        
    target_path = os.path.join(target_dir, domain_name, split)
    os.makedirs(target_path, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 获取所有密度图文件
    dmap_files = [f for f in os.listdir(source_path) if f.lower().endswith('.npy')]
    
    # 创建用于匹配的映射
    # 1. 基本文件名到_dmap.npy映射
    dmap_map = {}
    # 2. 基本文件名到.npy映射
    dmap_simple_map = {}
    
    for dmap_file in dmap_files:
        base_name = os.path.splitext(dmap_file)[0]
        # 去掉可能的_dmap后缀
        if base_name.endswith("_dmap"):
            original_base = base_name[:-5]
            dmap_map[original_base] = dmap_file
        else:
            # 不带_dmap后缀的.npy文件
            dmap_simple_map[base_name] = dmap_file
    
    processed_count = 0
    dmap_count = 0
    missing_dmap_count = 0
    
    for img_file in image_files:
        img_path = os.path.join(source_path, img_file)
        img_base = os.path.splitext(img_file)[0]
        target_img_path = os.path.join(target_path, img_file)
        
        # 处理图像
        if use_symlinks:
            if os.path.exists(target_img_path):
                os.remove(target_img_path)
            try:
                os.symlink(img_path, target_img_path)
                processed_count += 1
            except Exception as e:
                print(f"创建图像链接失败: {img_path} -> {target_img_path}，错误: {e}")
        else:
            try:
                shutil.copy2(img_path, target_img_path)
                processed_count += 1
            except Exception as e:
                print(f"复制图像失败: {img_path} -> {target_img_path}，错误: {e}")
        
        # 处理对应的密度图
        if with_dmap:
            dmap_processed = False
            
            # 1. 检查是否有_dmap.npy格式的文件
            if img_base in dmap_map:
                dmap_file = dmap_map[img_base]
                dmap_path = os.path.join(source_path, dmap_file)
                target_dmap_path = os.path.join(target_path, dmap_file)
                
                if use_symlinks:
                    if os.path.exists(target_dmap_path):
                        os.remove(target_dmap_path)
                    try:
                        os.symlink(dmap_path, target_dmap_path)
                        dmap_count += 1
                        dmap_processed = True
                    except Exception as e:
                        print(f"创建密度图链接失败: {dmap_path} -> {target_dmap_path}，错误: {e}")
                else:
                    try:
                        shutil.copy2(dmap_path, target_dmap_path)
                        dmap_count += 1
                        dmap_processed = True
                    except Exception as e:
                        print(f"复制密度图失败: {dmap_path} -> {target_dmap_path}，错误: {e}")
            
            # 2. 检查是否有简单.npy格式的文件
            if img_base in dmap_simple_map:
                dmap_file = dmap_simple_map[img_base]
                dmap_path = os.path.join(source_path, dmap_file)
                target_dmap_path = os.path.join(target_path, dmap_file)
                
                if use_symlinks:
                    if os.path.exists(target_dmap_path):
                        os.remove(target_dmap_path)
                    try:
                        os.symlink(dmap_path, target_dmap_path)
                        dmap_count += 1
                        dmap_processed = True
                    except Exception as e:
                        print(f"创建密度图链接失败: {dmap_path} -> {target_dmap_path}，错误: {e}")
                else:
                    try:
                        shutil.copy2(dmap_path, target_dmap_path)
                        dmap_count += 1
                        dmap_processed = True
                    except Exception as e:
                        print(f"复制密度图失败: {dmap_path} -> {target_dmap_path}，错误: {e}")
            
            # 3. 如果上述都没有找到，尝试模糊匹配
            if not dmap_processed:
                potential_matches = [df for df in dmap_files if img_base in df]
                if potential_matches:
                    for match in potential_matches:
                        dmap_path = os.path.join(source_path, match)
                        target_dmap_path = os.path.join(target_path, match)
                        
                        if use_symlinks:
                            if os.path.exists(target_dmap_path):
                                os.remove(target_dmap_path)
                            try:
                                os.symlink(dmap_path, target_dmap_path)
                                dmap_count += 1
                                dmap_processed = True
                            except Exception as e:
                                print(f"创建密度图链接失败: {dmap_path} -> {target_dmap_path}，错误: {e}")
                        else:
                            try:
                                shutil.copy2(dmap_path, target_dmap_path)
                                dmap_count += 1
                                dmap_processed = True
                            except Exception as e:
                                print(f"复制密度图失败: {dmap_path} -> {target_dmap_path}，错误: {e}")
            
            if not dmap_processed:
                print(f"警告: 未找到与 {img_file} 匹配的密度图")
                missing_dmap_count += 1
    
    action = "链接" if use_symlinks else "复制"
    print(f"已{action} {processed_count} 个图像和 {dmap_count} 个密度图，缺失 {missing_dmap_count} 个密度图")

def main():
    parser = argparse.ArgumentParser(description='重新组织JHU数据集按域分类')
    parser.add_argument('--domains_dir', type=str, default='/home/jianyong/exp/MPCount/jhu_domains', 
                        help='包含域文件的目录路径')
    parser.add_argument('--input_dir', type=str, default='', 
                        help='如果提供，直接从该目录处理文件而不是使用域文件中的路径')
    parser.add_argument('--output_dir', type=str, default='/scratch/jianyong/MPCount/data/jhu_re',
                        help='重组数据集的输出目录')
    parser.add_argument('--use_symlinks', action='store_true',
                        help='使用符号链接而不是复制文件')
    parser.add_argument('--without_dmap', action='store_true',
                        help='不包含密度图文件')
    parser.add_argument('--domains', type=str, default='street,stadium,snow,fog',
                        help='要处理的域，逗号分隔')
    parser.add_argument('--splits', type=str, default='train,val',
                        help='要处理的数据集分割，逗号分隔')
    parser.add_argument('--force', action='store_true',
                        help='强制重新创建目标目录')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细处理信息')
    
    args = parser.parse_args()
    
    # 解析域和分割参数
    domains = args.domains.split(',')
    splits = args.splits.split(',')
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每个域
    for domain in domains:
        for split in splits:
            target_dir = os.path.join(args.output_dir, domain, split)
            
            # 清空或创建目标目录
            if os.path.exists(target_dir) and args.force:
                print(f"清空目标目录: {target_dir}")
                shutil.rmtree(target_dir)
            
            os.makedirs(target_dir, exist_ok=True)
            
            if args.input_dir:
                # 直接从输入目录处理文件
                print(f"从目录处理 {domain} {split}...")
                process_folder_directly(args.input_dir, args.output_dir, domain, split, 
                                       not args.without_dmap, args.use_symlinks)
            else:
                # 从域文件处理
                file_path = os.path.join(args.domains_dir, f'jhu_{domain}_{split}.txt')
                
                if not os.path.exists(file_path):
                    print(f"警告: 文件不存在 {file_path}")
                    continue
                    
                print(f"从文件列表处理 {domain} {split}...")
                image_paths = parse_domain_file(file_path)
                
                if args.use_symlinks:
                    create_symlinks(image_paths, args.output_dir, domain, split, not args.without_dmap)
                else:
                    copy_images(image_paths, args.output_dir, domain, split, not args.without_dmap)
    
    print("数据集重组完成!")

if __name__ == "__main__":
    main() 