import os
import argparse
import time
import copy
import random
import numpy as np
import yaml
import torch.nn.functional as F
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from clearml import Task

from datasets.den_dataset import DensityMapDataset
from datasets.den_cls_dataset import DenClsDataset  # 如果需要的话
from datasets.jhu_domain_dataset import JHUDomainDataset
from datasets.jhu_domain_cls_dataset import JHUDomainClsDataset  # 如果需要的话

from models.models import DGModel_mem, DGModel_final, DGModel_base  # 使用您的模型
from datasets.den_dataset import DensityMapDataset  # 使用您的数据集
from datasets.jhu_domain_dataset import JHUDomainDataset
from trainers.dgtrainer import DGTrainer  # 使用您的训练器
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

# 保留IndexedDataset包装器，方便记忆缓冲区管理
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

# 创建Task，每个数据集视为一个Task
def create_crowd_counting_tasks(cfg, split='train'):
    """
    创建人群计数Task，每个数据集为一个Task
    cfg: 配置文件
    """
    tasks_train = []
    tasks_val = []
    tasks_test = []
    
    # 获取数据集配置
    train_datasets = cfg['train_dataset']
    val_datasets = cfg['val_dataset']
    test_datasets = cfg['test_dataset']
    
    for i in range(len(train_datasets)):
        # 使用get_dataset函数加载数据集
        train_dataset, collate = get_dataset(
            train_datasets[i]['name'], 
            train_datasets[i]['params'], 
            method='train'
        )
        
        val_dataset, _ = get_dataset(
            val_datasets[i]['name'], 
            val_datasets[i]['params'], 
            method='val'
        )
        
        test_dataset, _ = get_dataset(
            test_datasets[i]['name'], 
            test_datasets[i]['params'], 
            method='test'
        )
        
        # 包装为IndexedDataset
        indexed_train = IndexedDataset(train_dataset)
        indexed_val = IndexedDataset(val_dataset)
        indexed_test = IndexedDataset(test_dataset)
        
        tasks_train.append((indexed_train, collate))
        tasks_val.append((indexed_val, collate))
        tasks_test.append((indexed_test, collate))
    
    return tasks_train, tasks_val, tasks_test

# 修改记忆缓冲区更新函数
def reservoir_update(buffer, sample, counter, max_size):
    counter += 1
    if len(buffer) < max_size:
        buffer.append(sample)
    else:
        if random.random() < (max_size / counter):
            replace_idx = random.randint(0, max_size - 1)
            buffer[replace_idx] = sample
    return counter

# Sample from memory buffer
def sample_from_memory(memory_buffer, replay_batch_size, custom_probs, device):
    """
    Sample replay_batch_size examples from the memory buffer.

    Args:
        memory_buffer: List of memory samples.
        replay_batch_size: Number of samples to draw.
        custom_probs: Custom probability vector for sampling; if None, uniform sampling is used.
        device: Device (CPU/GPU) to move the sampled tensors to.

    Returns:
        sampled_imgs: Sampled image tensor.
        sampled_gt_dens: Sampled density map tensor.
        sampled_meta: Sampled metadata.
    """
    if replay_batch_size <= 0 or len(memory_buffer) == 0:
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device), {}

    if custom_probs is not None:
        # Use custom probability vector for sampling
        probs = custom_probs[:len(memory_buffer)]
        probs = probs / np.sum(probs)
    else:
        # Use uniform sampling
        probs = np.ones(len(memory_buffer)) / len(memory_buffer)

    chosen_indices = np.random.choice(
        len(memory_buffer),
        size=min(replay_batch_size, len(memory_buffer)),
        replace=(replay_batch_size > len(memory_buffer)),
        p=probs
    )

    sampled_imgs = []
    sampled_gt_dens = []
    sampled_meta_list = []

    for idx in chosen_indices:
        img, gt_den, meta = memory_buffer[idx]
        sampled_imgs.append(img.unsqueeze(0))
        sampled_gt_dens.append(gt_den.unsqueeze(0))
        sampled_meta_list.append(meta)

    sampled_imgs = torch.cat(sampled_imgs, dim=0).to(device)
    sampled_gt_dens = torch.cat(sampled_gt_dens, dim=0).to(device)

    # 处理meta数据 - 修复以处理不同类型的meta数据
    try:
        if isinstance(sampled_meta_list[0], dict) and 'count' in sampled_meta_list[0]:
            # 如果meta是包含count键的字典
            count_values = []
            for meta in sampled_meta_list:
                if isinstance(meta['count'], torch.Tensor):
                    # 如果是张量，确保它是标量
                    if meta['count'].numel() == 1:
                        count_values.append(meta['count'].item())
                    else:
                        count_values.append(meta['count'][0].item() if meta['count'].numel() > 0 else 0)
                else:
                    # 如果是标量值
                    count_values.append(meta['count'])
            sampled_meta = {'count': torch.tensor(count_values).to(device)}
        elif isinstance(sampled_meta_list[0], torch.Tensor):
            # 如果meta是张量
            meta_tensors = []
            for meta in sampled_meta_list:
                if meta.numel() == 1:
                    meta_tensors.append(meta.item())
                else:
                    meta_tensors.append(meta[0].item() if meta.numel() > 0 else 0)
            sampled_meta = {'count': torch.tensor(meta_tensors).to(device)}
        else:
            # 其他类型，尝试转换为张量
            sampled_meta = {'count': torch.tensor([float(m) for m in sampled_meta_list]).to(device)}
    except (TypeError, ValueError) as e:
        print(f"警告: 处理meta数据时出错: {e}")
        print(f"Meta数据类型: {type(sampled_meta_list[0])}")
        # 提供默认值
        sampled_meta = {'count': torch.zeros(len(sampled_meta_list)).to(device)}

    return sampled_imgs, sampled_gt_dens, sampled_meta

# 自定义collate函数，用于处理各种数据格式
def custom_collate_fn(batch):
    """
    自定义collate函数，处理不一致的数据格式
    """
    # 直接返回单个样本，避免stack问题
    if len(batch) == 1:
        return batch[0]
    
    # 只处理一个批次的第一个样本
    return batch[0]

# 修改评估函数，使用MAE和MSE
def evaluate_performance(model, dataset, device, collate_fn):
    """
    使用与DGTrainer.val_step类似的方法评估模型性能
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    mae_sum = 0
    mse_sum = 0
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            # 根据DGTrainer.val_step的输入格式解包数据
            img1, img2, gt, _, _ = batch
            img1 = img1.to(device)
            
            # 使用模型预测
            if hasattr(model, 'predict'):
                pred_count = model.predict(img1)
            else:
                # 如果模型没有predict方法，直接使用前向传播
                pred_dmap = model(img1)
                pred_count = pred_dmap.sum().item() / 1000  # 除以log_para (1000)
            
            # 获取真实人数（与DGTrainer.val_step一致）
            gt_count = gt.shape[1]
            
            # 计算MAE和MSE
            batch_mae = np.abs(pred_count - gt_count)
            batch_mse = (pred_count - gt_count) ** 2
            
            mae_sum += batch_mae
            mse_sum += batch_mse
            sample_count += 1
    
    if sample_count == 0:
        return {'mae': float('inf'), 'rmse': float('inf')}
        
    mae = mae_sum / sample_count
    rmse = np.sqrt(mse_sum / sample_count)
    
    return {'mae': mae, 'rmse': rmse}

# 训练循环
def train_and_evaluate_trial(cfg, args, device, master_seed, trial_custom_probs):
    # 设置随机种子
    random.seed(master_seed)
    torch.manual_seed(master_seed)
    seed_everything(master_seed)
    
    # 添加CUDA内存调试选项
    if device.type == 'cuda':
        try:
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(device)}")
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"CUDA memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        except Exception as e:
            print(f"Error getting CUDA info: {e}")

    # 创建Task
    datasets_list = args.datasets.split(',')
    tasks_train, tasks_val, tasks_test = create_crowd_counting_tasks(cfg)
    
    # 创建模型
    model_params = cfg['model']['params'] if 'params' in cfg['model'] else {}
    if args.model_type == 'mem':
        model = DGModel_mem(**model_params).to(device)
    elif args.model_type == 'final':
        model = DGModel_final(**model_params).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # 创建ClearML Logger（如果启用）
    clearml_logger = None
    if args.use_clearml and Task.current_task():
        clearml_logger = Task.current_task().get_logger()
    
    # 创建DGTrainer实例用于评估
    log_para = cfg.get('log_para', 1000)
    patch_size = cfg.get('patch_size', 10000)
    mode = cfg.get('mode', 'final')
    trainer = DGTrainer(
        seed=master_seed,
        version=cfg.get('version', 'cl'),
        device=device,
        log_para=log_para,
        patch_size=patch_size,
        mode=mode
    )
    
    # 获取优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = get_loss()
    generator = get_seeded_generator(cfg['seed'])
    
    # 记忆缓冲区
    memory_buffer = []
    memory_counter = 0
    
    best_perf_each_task = [{'mae': float('inf'), 'rmse': float('inf')} for _ in range(len(tasks_train))]
    final_perf_each_task = [{'mae': float('inf'), 'rmse': float('inf')} for _ in range(len(tasks_train))]
    
    print(f"Number of training epochs for each Task: {args.epochs_per_task}")
    
    # 为每个Task训练
    for task_id in range(len(tasks_train)):
        print(f"\n=== Train Task {task_id} ({datasets_list[task_id]}) ===")
        
        dataset, collate_fn = tasks_train[task_id]
        task_loader = DataLoader(dataset, collate_fn=collate_fn, **{**cfg['train_loader'], 'pin_memory': False}, worker_init_fn=seed_worker, generator=generator)
        
        for epoch in range(args.epochs_per_task):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in tqdm(task_loader, desc=f"Task {task_id} Epoch {epoch+1}"):
                # 保存当前样本用于更新记忆缓冲区
                try:
                    images, imgs2, gt_datas = batch
                    images = images.to(device)
                    imgs2 = imgs2.to(device)
                    gt_cmaps = gt_datas[-1].to(device)
                    new_images = images.clone()
                    new_gt_dens = gt_cmaps.clone()
                    optimizer.zero_grad()
                    loss = get_loss()
                    dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = model.forward_train(images, imgs2, gt_cmaps)
                    loss_den = compute_count_loss(loss, dmaps1, gt_datas, device=device, log_para=log_para) + compute_count_loss(loss, dmaps2, gt_datas, device=device, log_para=log_para)
                    loss_cls = F.binary_cross_entropy(cmaps1, gt_cmaps) + F.binary_cross_entropy(cmaps2, gt_cmaps)
                    loss_total = loss_den + 10 * loss_cls + 10 * loss_con # + loss_err 
     
                    loss_total.backward()
                    optimizer.step()
                    
                    epoch_loss += loss_total.item()
                    num_batches += 1
                    
                    # 记忆重放步骤
                    if len(memory_buffer) > 0 and args.replay_batch_size > 0:
                        # 从记忆中采样
                        replay_images, replay_gt_dens, replay_meta = sample_from_memory(
                            memory_buffer,
                            args.replay_batch_size,
                            trial_custom_probs,
                            device
                        )
                        
                        # 合并当前批次和重放样本
                        images = torch.cat([images.to(device), replay_images], dim=0)
                        gt_dens = torch.cat([gt_cmaps.to(device), replay_gt_dens], dim=0)
                    else:
                        images = images.to(device)
                        gt_dens = gt_cmaps.to(device)
                    
                    # 更新记忆缓冲区
                    for i in range(new_images.size(0)):
                        sample_img = new_images[i].cpu()
                        sample_den = new_gt_dens[i].cpu()
                        
                        # 从gt_datas中提取count信息
                        if isinstance(gt_datas, tuple) and len(gt_datas) >= 3:
                            # 如果gt_datas是元组，尝试获取count
                            count = None
                            if len(gt_datas) >= 4 and isinstance(gt_datas[3], (torch.Tensor, int, float)):
                                # 直接使用count数据
                                count = gt_datas[3][i].item() if isinstance(gt_datas[3], torch.Tensor) else gt_datas[3]
                            else:
                                # 从密度图或其他数据计算count
                                try:
                                    dmaps = gt_datas[1]
                                    if isinstance(dmaps, torch.Tensor):
                                        count = dmaps[i].sum().item()
                                except:
                                    count = 0  # 默认值
                        else:
                            # 如果gt_datas不是元组，使用默认值
                            count = 0
                        
                        sample_meta = {'count': count}
                        
                        memory_counter = reservoir_update(
                            memory_buffer, 
                            (sample_img, sample_den, sample_meta),
                            memory_counter, 
                            args.memory_size
                        )
                except Exception as e:
                    print(f"批处理时出错: {e}")
                    if "CUDA" in str(e):
                        print("CUDA错误，尝试清理内存并继续")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    continue
            
            # 记录每个epoch的损失
            if clearml_logger:
                clearml_logger.report_scalar(
                    title=f"Task_{task_id}_Loss",
                    series="Train",
                    value=epoch_loss / max(1, num_batches),
                    iteration=epoch
                )
            
            # 评估当前Task性能
            print(f"Val Task {task_id}...")
            val_dataset, collate_fn = tasks_val[task_id]
            
            # 创建有效的DataLoader，使用collate_fn，关闭pin_memory
            val_loader = DataLoader(val_dataset, **{**cfg['val_loader'], 'pin_memory': False})
            
            model.eval()
            mae_sum = 0
            mse_sum = 0
            val_count = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="evaling", leave=False):
                    try:
                        # 使用trainer.val_step来评估
                        mae, metrics = trainer.val_step(model, batch)
                        mae_sum += mae
                        mse_sum += metrics['mse']
                        val_count += 1
                        
                        if val_count <= 2:  # 只打印前两个样本的详细信息
                            print(f"Sample {val_count}: MAE={mae:.2f}, MSE={metrics['mse']:.2f}")
                    except Exception as e:
                        print(f"error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            if val_count > 0:
                perf_current = {
                    'mae': mae_sum / val_count,
                    'rmse': np.sqrt(mse_sum / val_count)
                }
                
                # Update best performance
                if perf_current['mae'] < best_perf_each_task[task_id]['mae']:
                    best_perf_each_task[task_id] = perf_current
                
                print(f"  Epoch {epoch+1}/{args.epochs_per_task}, Task {task_id}, MAE = {perf_current['mae']:.2f}, RMSE = {perf_current['rmse']:.2f}")
                
                # 记录验证指标
                if clearml_logger:
                    clearml_logger.report_scalar(
                        title=f"Task_{task_id}_MAE",
                        series="Validation",
                        value=perf_current['mae'],
                        iteration=epoch
                    )
                    clearml_logger.report_scalar(
                        title=f"Task_{task_id}_RMSE",
                        series="Validation",
                        value=perf_current['rmse'],
                        iteration=epoch
                    )
        
        # 在每个任务训练完后，显示该任务的最佳性能
        print(f"\n=== Task {task_id} ({datasets_list[task_id]}) training completed ===")
        print(f"Best validation performance - MAE: {best_perf_each_task[task_id]['mae']:.2f}, RMSE: {best_perf_each_task[task_id]['rmse']:.2f}")
    
    # 最终评估所有Task
    print("\n=== All tasks are finally evaluated ===")
    
    # 先显示每个任务的最佳验证性能
    print("\n=== Best validation performance for each task ===")
    for task_id in range(len(tasks_train)):
        print(f"Task {task_id} ({datasets_list[task_id]}) - Best MAE: {best_perf_each_task[task_id]['mae']:.2f}, Best RMSE: {best_perf_each_task[task_id]['rmse']:.2f}")
    
    for task_id in range(len(tasks_train)):
        print(f"\nval Task {task_id} ({datasets_list[task_id]})")
        
        # 使用测试集，创建有效的DataLoader，关闭pin_memory
        test_dataset, test_collate_fn = tasks_test[task_id]
        test_loader = DataLoader(test_dataset, **{**cfg['test_loader'], 'pin_memory': False}
        )
        
        model.eval()
        mae_sum = 0
        mse_sum = 0
        test_count = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="evaluating", leave=False):
                try:
                    # 使用trainer.test_step来评估
                    metrics = trainer.test_step(model, batch)
                    mae_sum += metrics['mae']
                    mse_sum += metrics['mse']
                    test_count += 1
                    
                    if test_count <= 2:  # 只打印前两个样本的详细信息
                        print(f"Sample {test_count}: MAE={metrics['mae']:.2f}, MSE={metrics['mse']:.2f}")
                except Exception as e:
                    print(f"error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if test_count > 0:
            final_perf = {
                'mae': mae_sum / test_count,
                'rmse': np.sqrt(mse_sum / test_count)
            }
            final_perf_each_task[task_id] = final_perf
            print(f"MAE = {final_perf['mae']:.2f}, RMSE = {final_perf['rmse']:.2f}")
            
            # 记录测试指标
            if clearml_logger:
                clearml_logger.report_scalar(
                    title="Final_Test_MAE",
                    series=f"Task_{task_id}",
                    value=final_perf['mae'],
                    iteration=0
                )
                clearml_logger.report_scalar(
                    title="Final_Test_RMSE",
                    series=f"Task_{task_id}",
                    value=final_perf['rmse'],
                    iteration=0
                )
    
    # 计算遗忘度
    forgetting_vals = [
        {
            'mae': best_perf_each_task[t]['mae'] - final_perf_each_task[t]['mae'],
            'rmse': best_perf_each_task[t]['rmse'] - final_perf_each_task[t]['rmse']
        }
        for t in range(len(tasks_train))
    ]
    
    # 汇Summary果
    avg_final_mae = np.mean([perf['mae'] for perf in final_perf_each_task])
    avg_final_rmse = np.mean([perf['rmse'] for perf in final_perf_each_task])
    
    # 记录总体指标
    if clearml_logger:
        clearml_logger.report_scalar(
            title="Overall_Performance",
            series="Avg_MAE",
            value=avg_final_mae,
            iteration=0
        )
        clearml_logger.report_scalar(
            title="Overall_Performance",
            series="Avg_RMSE",
            value=avg_final_rmse,
            iteration=0
        )
        
        # 记录遗忘度
        for t in range(len(tasks_train)):
            clearml_logger.report_scalar(
                title="Forgetting",
                series=f"Task_{t}_MAE",
                value=forgetting_vals[t]['mae'],
                iteration=0
            )
            clearml_logger.report_scalar(
                title="Forgetting",
                series=f"Task_{t}_RMSE",
                value=forgetting_vals[t]['rmse'],
                iteration=0
            )
    
    return {
        'final_perf_each_task': final_perf_each_task,
        'best_perf_each_task': best_perf_each_task,
        'forgetting_vals': forgetting_vals,
        'avg_final_mae': avg_final_mae,
        'avg_final_rmse': avg_final_rmse
    }

# 主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="SHA,SHB,QNRF", 
                        help="Comma-separated list of datasets to use")
    parser.add_argument("--model_type", type=str, default="final", 
                        choices=["mem", "final"], help="Model type")
    parser.add_argument("--memory_size", type=int, default=200, 
                        help="Replay memory size")
    parser.add_argument("--replay_batch_size", type=int, default=16,
                        help="Replay batch size per step")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Training batch size")
    parser.add_argument("--epochs_per_task", type=int, default=5, 
                        help="每个Task的训练Epoch数")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--n_seeds", type=int, default=3, 
                        help="Number of random seeds")
    parser.add_argument("--n_trials", type=int, default=5, 
                        help="Number of replay probability trials per seed")
    parser.add_argument("--config", type=str, default="configs/cl_config.yml", 
                        help="Config file path")
    parser.add_argument("--clearml_project", type=str, default="MPCount",
                        help="ClearML project name")
    parser.add_argument("--clearml_task", type=str, default="ContinualLearning",
                        help="ClearML task name")
    parser.add_argument("--use_clearml", action="store_true",
                        help="Whether to use ClearML for experiment tracking")
    parser.add_argument("--safe_mode", action="store_true",
                        help="启用安全模式，使用较小的批处理大小和更保守的内存设置")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    # 从配置文件读取epochs_per_task，如果没有则使用命令行参数的默认值
    args.epochs_per_task = cfg.get('num_epochs', args.epochs_per_task) // len(cfg['train_dataset'])
    
    # 安全模式设置
    if args.safe_mode:
        print("启用安全模式，降低批处理大小")
        # 减小batch_size以避免内存问题
        if 'train_loader' in cfg and 'batch_size' in cfg['train_loader']:
            cfg['train_loader']['batch_size'] = min(4, cfg['train_loader']['batch_size'])
        if 'val_loader' in cfg and 'batch_size' in cfg['val_loader']:
            cfg['val_loader']['batch_size'] = 1 
        if 'test_loader' in cfg and 'batch_size' in cfg['test_loader']:
            cfg['test_loader']['batch_size'] = 1
        # 禁用pin_memory
        if 'train_loader' in cfg:
            cfg['train_loader']['pin_memory'] = False
        if 'val_loader' in cfg:
            cfg['val_loader']['pin_memory'] = False
        if 'test_loader' in cfg:
            cfg['test_loader']['pin_memory'] = False
        # 减少工作线程数
        if 'train_loader' in cfg and 'num_workers' in cfg['train_loader']:
            cfg['train_loader']['num_workers'] = min(1, cfg['train_loader']['num_workers'])
        if 'val_loader' in cfg and 'num_workers' in cfg['val_loader']:
            cfg['val_loader']['num_workers'] = min(1, cfg['val_loader']['num_workers'])
        if 'test_loader' in cfg and 'num_workers' in cfg['test_loader']:
            cfg['test_loader']['num_workers'] = min(1, cfg['test_loader']['num_workers'])
    
    # 初始化ClearML
    task_name=join(args.clearml_task, args.model_type, args.datasets.replace(',', '_'))
    clearml_logger = CustomClearML('MPCount', task_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(vars(args))
    print("Using device:", device)
    
    overall_results = {}
    
    for seed in range(args.n_seeds):
        print("="*80)
        print(f"Start the experiment with {seed}")
        master_seed = 1000 + seed
        seed_trial_results = []
        
        for trial in range(args.n_trials):
            if trial == 0:
                # Baseline：均匀采样
                trial_custom_probs = None
                print(f"\n--- Seed {seed}, trial {trial} (Baseline: Uniform sampling) ---")
            else:
                # 自定义概率向量
                trial_custom_probs = np.random.rand(args.memory_size)
                print(f"\n--- Seed {seed}, trial {trial} (Custom sampling) ---")
            
            results = train_and_evaluate_trial(cfg, args, device, master_seed, trial_custom_probs)
            seed_trial_results.append(results['avg_final_mae'])
            
            # 在ClearML中记录每个trial的结果
            if clearml_logger:
                clearml_logger.report_scalar(
                    title="Trial_Results",
                    series=f"Seed_{seed}",
                    value=results['avg_final_mae'],
                    iteration=trial
                )
        
        # 汇总本种子下的结果
        baseline_mae = seed_trial_results[0]
        best_trial_mae = np.min(seed_trial_results)
        mean_trial_mae = np.mean(seed_trial_results)
        
        print(f"\nseed {seed} Summary:")
        print(f"  Baseline (task 0) MAE: {baseline_mae:.2f}")
        print(f"  Best trial MAE: {best_trial_mae:.2f}")
        print(f"  Mean trial MAE: {mean_trial_mae:.2f}")
        
        overall_results[seed] = {
            "baseline": baseline_mae,
            "best": best_trial_mae,
            "mean": mean_trial_mae
        }
        
        # 在ClearML中记录每个种子的汇总结果
        if clearml_logger:
            clearml_logger.report_scalar(
                title="Seed_Summary",
                series="Baseline_MAE",
                value=baseline_mae,
                iteration=seed
            )
            clearml_logger.report_scalar(
                title="Seed_Summary",
                series="Best_Trial_MAE",
                value=best_trial_mae,
                iteration=seed
            )
            clearml_logger.report_scalar(
                title="Seed_Summary",
                series="Mean_Trial_MAE",
                value=mean_trial_mae,
                iteration=seed
            )
    
    # 计算总体统计
    baseline_maes = [overall_results[s]["baseline"] for s in overall_results]
    best_maes = [overall_results[s]["best"] for s in overall_results]
    mean_maes = [overall_results[s]["mean"] for s in overall_results]
    
    print("\n==== Overall results for all seeds ====")
    print(f"Average baseline MAE: {np.mean(baseline_maes):.2f} ± {np.std(baseline_maes):.2f}")
    print(f"Average Best trial MAE: {np.mean(best_maes):.2f} ± {np.std(best_maes):.2f}")
    print(f"Average MAE: {np.mean(mean_maes):.2f} ± {np.std(mean_maes):.2f}")
    
    # 在ClearML中记录总体结果
    if clearml_logger:
        clearml_logger.report_scalar(
            title="Overall_Results",
            series="Avg_Baseline_MAE",
            value=np.mean(baseline_maes),
            iteration=0
        )
        clearml_logger.report_scalar(
            title="Overall_Results",
            series="Avg_Best_Trial_MAE",
            value=np.mean(best_maes),
            iteration=0
        )
        clearml_logger.report_scalar(
            title="Overall_Results",
            series="Avg_Mean_Trial_MAE",
            value=np.mean(mean_maes),
            iteration=0
        )
        clearml_logger.report_scalar(
            title="Overall_Results",
            series="Std_Baseline_MAE",
            value=np.std(baseline_maes),
            iteration=0
        )
        clearml_logger.report_scalar(
            title="Overall_Results",
            series="Std_Best_Trial_MAE",
            value=np.std(best_maes),
            iteration=0
        )
        clearml_logger.report_scalar(
            title="Overall_Results",
            series="Std_Mean_Trial_MAE",
            value=np.std(mean_maes),
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
        raise ValueError('Unknown dataset: {}'.format(name))
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
    # ...
    
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
            raise ValueError('Unknown loss: {}'.format(loss))
        
    return loss_value


if __name__ == "__main__":
    main()