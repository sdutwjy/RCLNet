#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import time
import copy
import random
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import join

from models.models import DGModel_mem, DGModel_final, DGModel_base

from datasets.den_dataset import DensityMapDataset
from datasets.den_cls_dataset import DenClsDataset  # 如果需要的话
from datasets.jhu_domain_dataset import JHUDomainDataset
from datasets.jhu_domain_cls_dataset import JHUDomainClsDataset 
from trainers.dgtrainer import DGTrainer
from utils.misc import seed_everything, seed_worker, get_seeded_generator

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def get_dataset(name, params, method):
    if name == 'den':
        dataset = DensityMapDataset(method=method, **params)
        collate = DensityMapDataset.collate
    elif name == 'den_cls':
        dataset = DenClsDataset(method=method, **params)
        collate = DenClsDataset.collate
    elif name == 'jhu_domain':
        dataset = JHUDomainDataset(method=method, **params)
        collate = JHUDomainDataset.collate
    elif name == 'jhu_domain_cls':
        dataset = JHUDomainClsDataset(method=method, **params)
        collate = JHUDomainClsDataset.collate
    else:
        raise ValueError('Unknown dataset: {}'.format(name))
    return dataset, collate

def get_loss():
    return nn.MSELoss()

def compute_count_loss(loss: nn.Module, pred_dmaps, gt_datas, weights=None, device='cuda', log_para=1000):
    if loss.__class__.__name__ == 'MSELoss':
        _, gt_dmaps, _ = gt_datas
        gt_dmaps = gt_dmaps.to(device)
        if weights is not None:
            pred_dmaps = pred_dmaps * weights
            gt_dmaps = gt_dmaps * weights
        loss_value = loss(pred_dmaps, gt_dmaps * log_para)
    else:
        raise ValueError('Unknown loss: {}'.format(loss))
    return loss_value

def get_model(model_type, params):
    if model_type == 'mem':
        return DGModel_mem(**params)
    elif model_type == 'final':
        return DGModel_final(**params)
    else:
        raise ValueError(f'Unknown model type: {model_type}')

def train_and_evaluate(cfg, args, device, seed):
    # 设置随机种子
    random.seed(seed)
    torch.manual_seed(seed)
    seed_everything(seed)
    
    # 获取数据集配置
    train_datasets = cfg['train_dataset']
    val_datasets = cfg['val_dataset']
    test_datasets = cfg['test_dataset']
    
    # 创建数据集
    train_dataset, collate = get_dataset(
            train_datasets['name'], 
            train_datasets['params'], 
            method='train'
        )
        
    val_dataset, _ = get_dataset(
            val_datasets['name'], 
            val_datasets['params'], 
            method='val'
        )
        
    test_dataset, _ = get_dataset(
            test_datasets['name'], 
            test_datasets['params'], 
            method='test'
        )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, collate_fn=collate, **cfg['train_loader'], worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, **cfg['val_loader'])
    test_loader = DataLoader(test_dataset, **cfg['test_loader'])
    
    # 创建模型
    model_params = cfg['model']['params'] if 'params' in cfg['model'] else {}
    model = get_model(args.model_type, model_params).to(device)
    
    # 创建训练器
    log_para = cfg.get('log_para', 1000)
    patch_size = cfg.get('patch_size', 10000)
    mode = cfg.get('mode', 'final')
    trainer = DGTrainer(
        seed=seed,
        version=cfg.get('version', 'single'),
        device=device,
        log_para=log_para,
        patch_size=patch_size,
        mode=mode
    )
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = get_loss()
    
    # 创建保存模型的目录
    os.makedirs('checkpoints/single', exist_ok=True)
    
    # 训练循环
    best_val_mae = float('inf')
    best_val_rmse = float('inf')
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            try:
                images, imgs2, gt_datas = batch
                images = images.to(device)
                imgs2 = imgs2.to(device)
                gt_cmaps = gt_datas[-1].to(device)
                
                optimizer.zero_grad()
                loss = get_loss()
                
                dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = model.forward_train(images, imgs2, gt_cmaps)
                loss_den = compute_count_loss(loss, dmaps1, gt_datas, device=device, log_para=log_para) + \
                          compute_count_loss(loss, dmaps2, gt_datas, device=device, log_para=log_para)
                loss_cls = nn.functional.binary_cross_entropy(cmaps1, gt_cmaps) + \
                          nn.functional.binary_cross_entropy(cmaps2, gt_cmaps)
                loss_total = loss_den + 10 * loss_cls + 10 * loss_con
                
                loss_total.backward()
                optimizer.step()
                
                epoch_loss += loss_total.item()
                num_batches += 1
                
            except Exception as e:
                print(f"训练时出错: {e}")
                if "CUDA" in str(e):
                    print("CUDA错误，尝试清理内存并继续")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
        
        # 验证阶段
        print(f"Val...")
        model.eval()
        mae_sum = 0
        mse_sum = 0
        val_count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                try:
                    # 使用trainer.val_step来评估
                    mae, metrics = trainer.val_step(model, batch)
                    mae_sum += mae
                    mse_sum += metrics['mse']
                    val_count += 1
                    
                    if val_count <= 2:  # 只打印前两个样本的详细信息
                        print(f"Sample {val_count}: MAE={mae:.2f}, MSE={metrics['mse']:.2f}")
                except Exception as e:
                    print(f"验证时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if val_count > 0:
            val_mae = mae_sum / val_count
            val_rmse = np.sqrt(mse_sum / val_count)
            
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"训练损失: {epoch_loss/max(1, num_batches):.4f}")
            print(f"验证 MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}")
            
            # 保存最佳模型
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_rmse = val_rmse
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': val_mae,
                    'val_rmse': val_rmse
                # }, f'checkpoints/single/{args.dataset_name}_best.pth')
                }, f'checkpoints/single/street_best.pth')
                print(f"保存最佳模型，验证 MAE: {val_mae:.2f}")
    
    # 测试阶段
    print("\n=== 测试阶段 ===")
    model.eval()
    mae_sum = 0
    mse_sum = 0
    test_count = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            try:
                # 使用trainer.test_step来评估
                metrics = trainer.test_step(model, batch)
                mae_sum += metrics['mae']
                mse_sum += metrics['mse']
                test_count += 1
                
                if test_count <= 2:  # 只打印前两个样本的详细信息
                    print(f"Sample {test_count}: MAE={metrics['mae']:.2f}, MSE={metrics['mse']:.2f}")
            except Exception as e:
                print(f"测试时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if test_count > 0:
        test_mae = mae_sum / test_count
        test_rmse = np.sqrt(mse_sum / test_count)
        
        print(f"\n测试结果:")
        print(f"MAE: {test_mae:.2f}")
        print(f"RMSE: {test_rmse:.2f}")
        
        # 保存最终结果
        results = {
            'best_val_mae': best_val_mae,
            'best_val_rmse': best_val_rmse,
            'test_mae': test_mae,
            'test_rmse': test_rmse
        }
        
        return results
    
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="jhu_domain",
                        help="数据集名称")
    parser.add_argument("--model_type", type=str, default="final",
                        choices=["mem", "final"], help="模型类型")
    parser.add_argument("--epochs", type=int, default=500,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--n_seeds", type=int, default=2,
                        help="随机种子数量")
    parser.add_argument("--config", type=str, default="configs/jhu_stadium_train.yml",
                        help="配置文件路径")
    parser.add_argument("--domain", type=str, default="fog",
                        choices=["fog", "snow", "stadium", "street"],
                        help="要训练的数据集域")
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 更新配置中的数据集路径
    for dataset_type in ['train_dataset', 'val_dataset', 'test_dataset']:
        if dataset_type in cfg:
            cfg[dataset_type]['params']['root'] = f'/scratch/jianyong/MPCount/data/jhu_reorganized/{args.domain}'
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"训练数据集: {args.domain}")
    
    # 运行多次实验
    all_results = []
    
    for seed in range(args.n_seeds):
        print(f"\n=== 实验 {seed+1}/{args.n_seeds} ===")
        master_seed = 1000 + seed
        
        results = train_and_evaluate(cfg, args, device, master_seed)
        if results:
            all_results.append(results)
    
    # 计算平均结果
    if all_results:
        avg_results = {
            'best_val_mae': np.mean([r['best_val_mae'] for r in all_results]),
            'best_val_rmse': np.mean([r['best_val_rmse'] for r in all_results]),
            'test_mae': np.mean([r['test_mae'] for r in all_results]),
            'test_rmse': np.mean([r['test_rmse'] for r in all_results])
        }
        
        std_results = {
            'best_val_mae': np.std([r['best_val_mae'] for r in all_results]),
            'best_val_rmse': np.std([r['best_val_rmse'] for r in all_results]),
            'test_mae': np.std([r['test_mae'] for r in all_results]),
            'test_rmse': np.std([r['test_rmse'] for r in all_results])
        }
        
        print("\n=== 最终结果 ===")
        print(f"最佳验证 MAE: {avg_results['best_val_mae']:.2f} ± {std_results['best_val_mae']:.2f}")
        print(f"最佳验证 RMSE: {avg_results['best_val_rmse']:.2f} ± {std_results['best_val_rmse']:.2f}")
        print(f"测试 MAE: {avg_results['test_mae']:.2f} ± {std_results['test_mae']:.2f}")
        print(f"测试 RMSE: {avg_results['test_rmse']:.2f} ± {std_results['test_rmse']:.2f}")

if __name__ == "__main__":
    main() 