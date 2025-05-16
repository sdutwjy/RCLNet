import os
import argparse
import time
import copy
import random
import numpy as np
import yaml
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from clearml import Task

from datasets.den_dataset import DensityMapDataset
from datasets.den_cls_dataset import DenClsDataset
from datasets.jhu_domain_dataset import JHUDomainDataset
from datasets.jhu_domain_cls_dataset import JHUDomainClsDataset

from models.models import DGModel_base, DGModel_mem, DGModel_final
from trainers.dgtrainer import DGTrainer
from tqdm import tqdm
from utils.misc import seed_worker, get_seeded_generator, seed_everything


class CustomClearML():
    def __init__(self, project_name, task_name):
        self.task = Task.init(project_name, task_name)
        self.logger = self.task.get_logger()

    def __call__(self, title, series, value, iteration):
        self.logger.report_scalar(title, series, value, iteration)
        
    def report_scalar(self, title, series, value, iteration):
        self.logger.report_scalar(title, series, value, iteration)


# 包装数据集，以便处理不同格式的输出
class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, idx):
        result = self.dataset[idx]
        if isinstance(result, tuple) and len(result) == 5:
            images1 = result[0] 
            images2 = result[1]
            dmaps = result[3]
            points = result[2]
            bmaps = result[4]
        else:
            raise ValueError(f"Unexpected dataset output with length={len(result)}")
        return images1, images2, points, dmaps, bmaps
    
    def __len__(self):
        return len(self.dataset)


# 创建源域（大数据集）和目标域（小数据集）
def create_source_target_datasets(cfg):
    """
    创建源域（预训练数据集）和目标域（few-shot数据集）
    """
    # 源域数据集（用于预训练）
    source_train = []
    source_val = []
    
    # 目标域数据集（用于few-shot学习）
    target_train = []
    target_val = []
    target_test = []
    
    # 加载源域数据集
    source_datasets = cfg['source_dataset']
    for i in range(len(source_datasets)):
        # 训练集
        train_dataset, collate = get_dataset(
            source_datasets[i]['name'], 
            source_datasets[i]['params'], 
            method='train'
        )
        
        # 验证集
        val_dataset, _ = get_dataset(
            source_datasets[i]['name'], 
            source_datasets[i]['params'], 
            method='val'
        )
        
        # 包装为IndexedDataset
        indexed_train = IndexedDataset(train_dataset)
        indexed_val = IndexedDataset(val_dataset)
        
        source_train.append((indexed_train, collate))
        source_val.append((indexed_val, collate))
    
    # 加载目标域数据集
    target_datasets = cfg['target_dataset']
    for i in range(len(target_datasets)):
        # 训练集（少量样本）
        train_dataset, collate = get_dataset(
            target_datasets[i]['name'], 
            target_datasets[i]['params'], 
            method='train'
        )
        
        # 验证集
        val_dataset, _ = get_dataset(
            target_datasets[i]['name'], 
            target_datasets[i]['params'], 
            method='val'
        )
        
        # 测试集
        test_dataset, _ = get_dataset(
            target_datasets[i]['name'], 
            target_datasets[i]['params'], 
            method='test'
        )
        
        # 包装为IndexedDataset
        indexed_train = IndexedDataset(train_dataset)
        indexed_val = IndexedDataset(val_dataset)
        indexed_test = IndexedDataset(test_dataset)
        
        target_train.append((indexed_train, collate))
        target_val.append((indexed_val, collate))
        target_test.append((indexed_test, collate))
    
    return source_train, source_val, target_train, target_val, target_test


# 只选择数据集的一小部分样本用于Few-shot学习
def create_few_shot_subset(dataset, n_samples, seed=42):
    """
    从数据集中随机选择n_samples个样本，用于Few-shot学习
    
    Args:
        dataset: 原始数据集
        n_samples: 要选择的样本数量
        seed: 随机种子，确保可重复性
    
    Returns:
        subset: 子数据集
    """
    # 设置随机种子
    random.seed(seed)
    
    # 获取数据集大小
    dataset_size = len(dataset)
    
    # 确保n_samples不超过数据集大小
    n_samples = min(n_samples, dataset_size)
    
    # 随机选择n_samples个样本索引
    indices = random.sample(range(dataset_size), n_samples)
    
    # 创建子数据集
    subset = Subset(dataset, indices)
    
    return subset, indices


# 训练源域模型（预训练阶段）
def train_source_model(cfg, args, device, model, source_train, source_val):
    """
    在源域数据集上训练模型（预训练阶段）
    
    Args:
        cfg: 配置文件
        args: 命令行参数
        device: 设备（CPU/GPU）
        model: 模型
        source_train: 源域训练数据集
        source_val: 源域验证数据集
    
    Returns:
        model: 预训练好的模型
    """
    print("\n=== 预训练阶段：在源域数据集上训练模型 ===")
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.source_lr)
    
    # 创建损失函数
    criterion = get_loss()
    
    # 创建DGTrainer实例用于评估
    log_para = cfg.get('log_para', 1000)
    patch_size = cfg.get('patch_size', 10000)
    mode = cfg.get('mode', 'final')
    trainer = DGTrainer(
        seed=cfg['seed'],
        version=cfg.get('version', 'cl'),
        device=device,
        log_para=log_para,
        patch_size=patch_size,
        mode=mode
    )
    
    # 合并所有源域数据集
    dataset, collate_fn = source_train[0]  # 假设都使用相同的collate_fn
    train_loader = DataLoader(dataset, collate_fn=collate_fn, **cfg['source_loader'], worker_init_fn=seed_worker)
    
    # 训练循环
    best_model = None
    best_loss = float('inf')
    
    for epoch in range(args.source_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"预训练轮次 {epoch+1}/{args.source_epochs}"):
            try:
                images, imgs2, gt_datas = batch
                images = images.to(device)
                imgs2 = imgs2.to(device)
                gt_cmaps = gt_datas[-1].to(device)
                
                optimizer.zero_grad()
                
                # 前向传播
                dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = model.forward_train(images, imgs2, gt_cmaps)
                
                # 计算损失
                loss_den = compute_count_loss(criterion, dmaps1, gt_datas, device=device, log_para=log_para) + \
                           compute_count_loss(criterion, dmaps2, gt_datas, device=device, log_para=log_para)
                loss_cls = nn.functional.binary_cross_entropy(cmaps1, gt_cmaps) + nn.functional.binary_cross_entropy(cmaps2, gt_cmaps)
                loss_total = loss_den + 10 * loss_cls + 10 * loss_con  # + loss_err
                
                # 反向传播
                loss_total.backward()
                optimizer.step()
                
                epoch_loss += loss_total.item()
                num_batches += 1
                
            except Exception as e:
                print(f"批处理时出错: {e}")
                if "CUDA" in str(e):
                    print("CUDA错误，尝试清理内存并继续")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
        
        # 计算平均损失
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"  轮次 {epoch+1}, 训练损失: {avg_loss:.4f}")
        
        # 在验证集上评估
        if epoch % args.eval_interval == 0:
            val_metrics = evaluate_model(model, source_val[0][0], device, trainer)
            print(f"  验证 MAE: {val_metrics['mae']:.2f}, RMSE: {val_metrics['rmse']:.2f}")
            
            # 保存最佳模型
            if val_metrics['mae'] < best_loss:
                best_loss = val_metrics['mae']
                best_model = copy.deepcopy(model.state_dict())
                if args.save_model:
                    torch.save(best_model, f"{args.model_dir}/source_best.pth")
    
    # 加载最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model


# Few-shot学习阶段：在目标域微调模型
def finetune_target_model(cfg, args, device, model, target_train, target_val, target_test):
    """
    在目标域数据集上微调模型（Few-shot学习阶段）
    
    Args:
        cfg: 配置文件
        args: 命令行参数
        device: 设备（CPU/GPU）
        model: 预训练好的模型
        target_train: 目标域训练数据集（少量样本）
        target_val: 目标域验证数据集
        target_test: 目标域测试数据集
    
    Returns:
        model: 微调后的模型
        metrics: 评估指标
    """
    print("\n=== Few-shot学习阶段：在目标域数据集上微调模型 ===")
    
    # 为目标域微调创建新模型
    if args.use_new_model:
        model_params = cfg['model']['params'] if 'params' in cfg['model'] else {}
        new_model = get_model(cfg['model']['name'], model_params)
        new_model.load_state_dict(model.state_dict())
        model = new_model.to(device)
    
    # 冻结特定层（可选，取决于args.freeze_layers）
    if args.freeze_backbone:
        # 冻结编码器部分（VGG16特征提取器）
        for param in model.enc1.parameters():
            param.requires_grad = False
        for param in model.enc2.parameters():
            param.requires_grad = False
        for param in model.enc3.parameters():
            param.requires_grad = False
        print("  已冻结骨干网络参数")
    
    # 创建优化器（只优化未冻结的参数）
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.target_lr)
    
    # 创建损失函数
    criterion = get_loss()
    
    # 创建DGTrainer实例用于评估
    log_para = cfg.get('log_para', 1000)
    patch_size = cfg.get('patch_size', 10000)
    mode = cfg.get('mode', 'final')
    trainer = DGTrainer(
        seed=cfg['seed'],
        version=cfg.get('version', 'cl'),
        device=device,
        log_para=log_para,
        patch_size=patch_size,
        mode=mode
    )
    
    # 从目标域训练集创建few-shot子集
    dataset, collate_fn = target_train[0]
    few_shot_dataset, few_shot_indices = create_few_shot_subset(dataset, args.n_shots, seed=cfg['seed'])
    
    print(f"  Few-shot学习: 使用 {args.n_shots} 个样本 (总共 {len(dataset)} 个样本)")
    
    # 创建Few-shot数据加载器
    few_shot_loader = DataLoader(few_shot_dataset, collate_fn=collate_fn, **cfg['target_loader'], worker_init_fn=seed_worker)
    
    # 微调循环
    best_model = None
    best_val_mae = float('inf')
    
    for epoch in range(args.target_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(few_shot_loader, desc=f"微调轮次 {epoch+1}/{args.target_epochs}"):
            try:
                images, imgs2, gt_datas = batch
                images = images.to(device)
                imgs2 = imgs2.to(device)
                gt_cmaps = gt_datas[-1].to(device)
                
                optimizer.zero_grad()
                
                # 前向传播
                dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = model.forward_train(images, imgs2, gt_cmaps)
                
                # 计算损失
                loss_den = compute_count_loss(criterion, dmaps1, gt_datas, device=device, log_para=log_para) + \
                           compute_count_loss(criterion, dmaps2, gt_datas, device=device, log_para=log_para)
                loss_cls = nn.functional.binary_cross_entropy(cmaps1, gt_cmaps) + nn.functional.binary_cross_entropy(cmaps2, gt_cmaps)
                loss_total = loss_den + 10 * loss_cls + 10 * loss_con  # + loss_err
                
                # 反向传播
                loss_total.backward()
                optimizer.step()
                
                epoch_loss += loss_total.item()
                num_batches += 1
                
            except Exception as e:
                print(f"批处理时出错: {e}")
                if "CUDA" in str(e):
                    print("CUDA错误，尝试清理内存并继续")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
        
        # 计算平均损失
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"  轮次 {epoch+1}, 训练损失: {avg_loss:.4f}")
        
        # 在验证集上评估
        if epoch % args.eval_interval == 0:
            val_metrics = evaluate_model(model, target_val[0][0], device, trainer)
            print(f"  验证 MAE: {val_metrics['mae']:.2f}, RMSE: {val_metrics['rmse']:.2f}")
            
            # 保存最佳模型
            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                best_model = copy.deepcopy(model.state_dict())
                if args.save_model:
                    torch.save(best_model, f"{args.model_dir}/target_best.pth")
    
    # 加载最佳模型进行测试
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # 在测试集上评估
    test_metrics = evaluate_model(model, target_test[0][0], device, trainer)
    print(f"\n最终测试结果 - MAE: {test_metrics['mae']:.2f}, RMSE: {test_metrics['rmse']:.2f}")
    
    return model, test_metrics


# 评估模型性能
def evaluate_model(model, dataset, device, trainer):
    """
    评估模型在给定数据集上的性能
    
    Args:
        model: 模型
        dataset: 数据集
        device: 设备（CPU/GPU）
        trainer: DGTrainer实例
    
    Returns:
        metrics: 包含MAE和RMSE的字典
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    mae_sum = 0
    mse_sum = 0
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                # 评估单个样本
                mae, metrics = trainer.val_step(model, batch)
                mae_sum += mae
                mse_sum += metrics['mse']
                sample_count += 1
            except Exception as e:
                print(f"评估时出错: {e}")
                continue
    
    if sample_count == 0:
        return {'mae': float('inf'), 'rmse': float('inf')}
    
    # 计算平均MAE和RMSE
    avg_mae = mae_sum / sample_count
    avg_rmse = np.sqrt(mse_sum / sample_count)
    
    return {'mae': avg_mae, 'rmse': avg_rmse}


# 主函数
def main():
    parser = argparse.ArgumentParser(description="Few-shot Learning for Crowd Counting")
    
    # 基本参数
    parser.add_argument("--config", type=str, default="configs/fewshot_config.yml", 
                       help="配置文件路径")
    parser.add_argument("--model_type", type=str, default="final", 
                       choices=["base", "mem", "final"], help="模型类型")
    parser.add_argument("--model_dir", type=str, default="saved_models",
                       help="模型保存目录")
    parser.add_argument("--save_model", action="store_true",
                       help="是否保存模型")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    # 源域训练参数（预训练阶段）
    parser.add_argument("--source_epochs", type=int, default=50,
                       help="源域训练轮次")
    parser.add_argument("--source_lr", type=float, default=1e-4,
                       help="源域学习率")
    parser.add_argument("--pretrained", action="store_true",
                       help="是否使用ImageNet预训练权重")
    parser.add_argument("--load_pretrained", type=str, default=None,
                       help="加载预训练模型的路径")
    
    # 目标域微调参数（Few-shot学习阶段）
    parser.add_argument("--n_shots", type=int, default=10,
                       help="每个类别的样本数（Few-shot学习）")
    parser.add_argument("--target_epochs", type=int, default=20,
                       help="目标域微调轮次")
    parser.add_argument("--target_lr", type=float, default=5e-5,
                       help="目标域学习率")
    parser.add_argument("--freeze_backbone", action="store_true",
                       help="是否冻结骨干网络")
    parser.add_argument("--use_new_model", action="store_true",
                       help="微调时是否创建新模型")
    parser.add_argument("--eval_interval", type=int, default=1,
                       help="评估间隔（每隔多少个epoch评估一次）")
    
    # 其他参数
    parser.add_argument("--safe_mode", action="store_true",
                       help="启用安全模式，使用较小的批处理大小和更保守的内存设置")
    parser.add_argument("--use_clearml", action="store_true",
                       help="是否使用ClearML进行实验跟踪")
    parser.add_argument("--clearml_project", type=str, default="MPCount",
                       help="ClearML项目名称")
    parser.add_argument("--clearml_task", type=str, default="FewShotLearning",
                       help="ClearML任务名称")
    
    args = parser.parse_args()
    
    # 加载配置文件
    cfg = load_config(args.config)
    
    # 设置随机种子
    seed_everything(args.seed)
    
    # 创建模型保存目录
    if args.save_model and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    # 安全模式设置
    if args.safe_mode:
        print("启用安全模式，降低批处理大小")
        # 减小batch_size以避免内存问题
        if 'source_loader' in cfg and 'batch_size' in cfg['source_loader']:
            cfg['source_loader']['batch_size'] = min(4, cfg['source_loader']['batch_size'])
        if 'target_loader' in cfg and 'batch_size' in cfg['target_loader']:
            cfg['target_loader']['batch_size'] = min(4, cfg['target_loader']['batch_size'])
        # 禁用pin_memory
        if 'source_loader' in cfg:
            cfg['source_loader']['pin_memory'] = False
        if 'target_loader' in cfg:
            cfg['target_loader']['pin_memory'] = False
        # 减少工作线程数
        if 'source_loader' in cfg and 'num_workers' in cfg['source_loader']:
            cfg['source_loader']['num_workers'] = min(2, cfg['source_loader']['num_workers'])
        if 'target_loader' in cfg and 'num_workers' in cfg['target_loader']:
            cfg['target_loader']['num_workers'] = min(2, cfg['target_loader']['num_workers'])
    
    # 使用GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化ClearML（如果启用）
    clearml_logger = None
    if args.use_clearml:
        task_name = f"{args.clearml_task}_{args.model_type}_{args.n_shots}shots"
        clearml_logger = CustomClearML(args.clearml_project, task_name)
    
    # 创建数据集
    source_train, source_val, target_train, target_val, target_test = create_source_target_datasets(cfg)
    
    # 创建模型
    model_params = cfg['model']['params'] if 'params' in cfg['model'] else {}
    model_params['pretrained'] = args.pretrained  # 是否使用ImageNet预训练权重
    
    if args.model_type == 'base':
        model = DGModel_base(**model_params)
    elif args.model_type == 'mem':
        model = DGModel_mem(**model_params)
    elif args.model_type == 'final':
        model = DGModel_final(**model_params)
    else:
        raise ValueError(f"未知的模型类型: {args.model_type}")
    
    model = model.to(device)
    
    # 加载预训练模型（如果指定）
    if args.load_pretrained:
        print(f"加载预训练模型: {args.load_pretrained}")
        model.load_state_dict(torch.load(args.load_pretrained, map_location=device))
    
    # 源域训练（预训练阶段）
    if not args.load_pretrained:
        model = train_source_model(cfg, args, device, model, source_train, source_val)
    
    # 目标域微调（Few-shot学习阶段）
    model, metrics = finetune_target_model(cfg, args, device, model, target_train, target_val, target_test)
    
    # 输出最终结果
    print("\n=== 实验结果总结 ===")
    print(f"Few-shot设置: {args.n_shots}个样本")
    print(f"最终MAE: {metrics['mae']:.2f}")
    print(f"最终RMSE: {metrics['rmse']:.2f}")
    
    # 记录实验结果（如果启用ClearML）
    if clearml_logger:
        clearml_logger.report_scalar(
            title="Few-shot Results",
            series=f"{args.n_shots}_shots",
            value=metrics['mae'],
            iteration=0
        )
        clearml_logger.report_scalar(
            title="Few-shot Results",
            series=f"{args.n_shots}_shots_rmse",
            value=metrics['rmse'],
            iteration=0
        )


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
        raise ValueError(f'未知数据集: {name}')
    return dataset, collate


def get_loss():
    return nn.MSELoss()


def get_model(name, params):
    if name == 'base':
        return DGModel_base(**params)
    elif name == 'mem':
        return DGModel_mem(**params)
    elif name == 'final':
        return DGModel_final(**params)
    else:
        raise ValueError(f'未知模型: {name}')


def compute_count_loss(loss: nn.Module, pred_dmaps, gt_datas, weights=None, device='cuda', log_para=1000):
    if loss.__class__.__name__ == 'MSELoss':
        _, gt_dmaps, _ = gt_datas
        gt_dmaps = gt_dmaps.to(device)
        if weights is not None:
            pred_dmaps = pred_dmaps * weights
            gt_dmaps = gt_dmaps * weights
        loss_value = loss(pred_dmaps, gt_dmaps * log_para)
    elif loss.__class__.__name__ == 'BL':
        gts, targs, st_sizes = gt_datas
        gts = [gt.to(device) for gt in gts]
        targs = [targ.to(device) for targ in targs]
        st_sizes = st_sizes.to(device)
        loss_value = loss(gts, st_sizes, targs, pred_dmaps)
    else:
        raise ValueError(f'未知损失函数: {loss.__class__.__name__}')
        
    return loss_value


if __name__ == "__main__":
    main() 