#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import shutil
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from datasets.jhu_domain_cls_dataset import JHUDomainClsDataset

def create_test_set(domain_path, test_ratio=0.3, min_samples=30, max_samples=40, seed=42):
    """
    从训练集中随机抽取图片及其对应的npy文件移动到测试集
    
    参数:
        domain_path: 域数据路径 (如fog、snow等)
        test_ratio: 抽取为测试集的比例
        min_samples: 最少抽取的样本数
        max_samples: 最多抽取的样本数
        seed: 随机种子，确保结果可重现
    """
    random.seed(seed)
    
    train_dir = os.path.join(domain_path, 'train')
    val_dir = os.path.join(domain_path, 'val')
    
    # 创建test目录
    test_dir = os.path.join(domain_path, 'test')
    os.makedirs(test_dir, exist_ok=True)
    
    # 获取所有jpg文件
    jpg_files = [f for f in os.listdir(train_dir) if f.endswith('.jpg')]
    
    # 确定要抽取的图片数量
    num_images = len(jpg_files)
    num_to_extract = max(min(int(num_images * test_ratio), max_samples), min_samples)
    num_to_extract = min(num_to_extract, num_images)  # 确保不超过总数
    
    # 随机选择图片
    selected_images = random.sample(jpg_files, num_to_extract)
    
    print(f"域 {os.path.basename(domain_path)}: 从{num_images}张图片中抽取{num_to_extract}张作为测试集")
    
    # 移动文件到test目录
    for img_file in selected_images:
        base_name = os.path.splitext(img_file)[0]
        dmap_file = f"{base_name}_dmap.npy"
        npy_file = f"{base_name}.npy"
        
        # 移动图片文件
        src_img = os.path.join(train_dir, img_file)
        dst_img = os.path.join(test_dir, img_file)
        shutil.move(src_img, dst_img)
        print(f"移动图片: {src_img} -> {dst_img}")
        
        # 移动对应的npy文件
        src_dmap = os.path.join(train_dir, dmap_file)
        dst_dmap = os.path.join(test_dir, dmap_file)
        
        sr_dmap = os.path.join(train_dir, npy_file)
        ds_dmap = os.path.join(test_dir, npy_file)
        
        if os.path.exists(src_dmap):
            shutil.move(src_dmap, dst_dmap)
            print(f"移动密度图: {src_dmap} -> {dst_dmap}")
        else:
            print(f"警告: 找不到对应的密度图文件 {dmap_file}")
            
        if os.path.exists(sr_dmap):
            shutil.move(sr_dmap, ds_dmap)
            print(f"移动密度图: {sr_dmap} -> {ds_dmap}")
        else:
            print(f"警告: 找不到对应的密度图文件 {npy_file}")
    
    # 创建测试集的文本文件记录
    test_txt_path = os.path.join('/home/jianyong/exp/MPCount/jhu_domains', f'jhu_{os.path.basename(domain_path)}_test.txt')
    with open(test_txt_path, 'w') as f:
        for img_file in selected_images:
            # 获取原始JHU数据集中的路径，以便与现有的train/val文本文件格式一致
            original_path = os.path.join('/scratch/jianyong/MPCount/data/jhu_reorganized', 
                                         os.path.basename(domain_path), 'test', img_file)
            f.write(f"{original_path}\n")
    
    print(f"创建测试集文本文件: {test_txt_path}")
    
    return selected_images

def validate_test_set(domain, test_txt_path):
    """
    验证创建的测试集是否可以被数据集类正确加载
    
    参数:
        domain: 域名称 (如fog、snow等)
        test_txt_path: 测试集文本文件路径
    """
    print(f"\n验证{domain}域测试集...")
    
    # 检查文件是否存在
    if not os.path.exists(test_txt_path):
        print(f"错误: 测试集文件 {test_txt_path} 不存在!")
        return False
    
    # 读取文件内容并检查行数
    with open(test_txt_path, 'r') as f:
        lines = f.readlines()
    
    line_count = len(lines)
    print(f"测试集文件包含 {line_count} 行")
    
    # 检查前几行图片是否存在
    for i, line in enumerate(lines[:5]):
        img_path = line.strip()
        if not os.path.exists(img_path):
            print(f"警告: 图片路径不存在: {img_path}")
        else:
            print(f"图片路径有效: {img_path}")
    
    # 尝试使用数据集类加载
    try:
        test_dataset = JHUDomainClsDataset(
            root="/scratch/jianyong/MPCount/data/jhu_reorganized",
            domain_label=f"jhu_{domain}",
            crop_size=320,
            domain_type="scene",
            domain=domain,
            downsample=4,
            method="test",
            is_grey=False,
            unit_size=16,
            pre_resize=1,
            split="test"
        )
        
        print(f"成功加载数据集，包含 {len(test_dataset)} 个样本")
        
        # 尝试获取一个样本
        if len(test_dataset) > 0:
            try:
                sample = test_dataset[0]
                print(f"成功获取第一个样本")
                return True
            except Exception as e:
                print(f"获取样本时出错: {str(e)}")
                return False
        else:
            print("警告: 数据集为空!")
            return False
            
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='从JHU训练集中创建测试集并验证')
    parser.add_argument('--jhu_root', type=str, default='/scratch/jianyong/MPCount/data/jhu_reorganized',
                        help='JHU重组数据集的根目录')
    parser.add_argument('--test_ratio', type=float, default=0.3,
                        help='抽取为测试集的比例')
    parser.add_argument('--min_samples', type=int, default=30,
                        help='每个域最少抽取的样本数')
    parser.add_argument('--max_samples', type=int, default=40,
                        help='每个域最多抽取的样本数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--skip_create', action='store_true',
                        help='跳过创建步骤，仅验证现有测试集')
    parser.add_argument('--domain', type=str, default=None,
                        help='只处理指定的域 (fog, snow, stadium, street)，默认处理所有域')
    parser.add_argument('--backup', action='store_true',
                        help='在移动前创建数据备份')
    
    args = parser.parse_args()
    
    # 处理的域
    all_domains = ['fog', 'snow', 'stadium', 'street']
    domains = [args.domain] if args.domain and args.domain in all_domains else all_domains
    
    # 创建备份
    if args.backup and not args.skip_create:
        print("正在创建数据备份...")
        backup_dir = '/scratch/jianyong/MPCount/data/jhu_backup'
        os.makedirs(backup_dir, exist_ok=True)
        
        for domain in domains:
            domain_path = os.path.join(args.jhu_root, domain)
            if not os.path.exists(domain_path):
                continue
                
            backup_domain_path = os.path.join(backup_dir, domain)
            if not os.path.exists(backup_domain_path):
                os.makedirs(backup_domain_path, exist_ok=True)
                
            # 备份训练集
            train_dir = os.path.join(domain_path, 'train')
            backup_train_dir = os.path.join(backup_domain_path, 'train')
            if os.path.exists(train_dir) and not os.path.exists(backup_train_dir):
                print(f"备份 {domain} 训练集...")
                shutil.copytree(train_dir, backup_train_dir)
                
        print("备份完成!")
    
    for domain in domains:
        domain_path = os.path.join(args.jhu_root, domain)
        if not os.path.exists(domain_path):
            print(f"警告: 找不到域 {domain} 的路径 {domain_path}")
            continue
        
        test_txt_path = os.path.join('/home/jianyong/exp/MPCount/jhu_domains', f'jhu_{domain}_test.txt')
        
        if not args.skip_create:
            print(f"\n处理{domain}域...")
            selected_images = create_test_set(domain_path, 
                                            args.test_ratio, 
                                            args.min_samples, 
                                            args.max_samples, 
                                            args.seed)
            
            print(f"域 {domain}: 成功创建测试集，包含 {len(selected_images)} 张图片")
        
        # 验证测试集
        validate_result = validate_test_set(domain, test_txt_path)
        if validate_result:
            print(f"域 {domain}: 测试集验证成功!")
        else:
            print(f"域 {domain}: 测试集验证失败!")
        
        print("-" * 50)

if __name__ == "__main__":
    main()
    print("完成所有域的测试集创建与验证!") 