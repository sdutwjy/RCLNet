import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from torch.utils.data import DataLoader
from clearml import Task

from models.models import DGModel_mem, DGModel_final, DGModel_base
from datasets.den_dataset import DensityMapDataset
from datasets.jhu_domain_dataset import JHUDomainDataset
from trainers.dgtrainer import DGTrainer
from tqdm import tqdm
from utils.misc import seed_worker, get_seeded_generator, seed_everything
from lwf_utils import distillation_loss, create_teacher_model, combine_losses

# 导入必要的其他模块
from main_cl_jhu import (CustomClearML, IndexedDataset, get_dataset, 
                        get_loss, get_model, compute_count_loss)

def lwf_train_and_evaluate_trial(cfg, args, device, master_seed):
    """
    使用LwF方法进行持续学习训练
    """
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
    from main_cl_jhu import create_crowd_counting_tasks
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
    
    best_perf_each_task = [{'mae': float('inf'), 'rmse': float('inf')} for _ in range(len(tasks_train))]
    final_perf_each_task = [{'mae': float('inf'), 'rmse': float('inf')} for _ in range(len(tasks_train))]
    
    # 创建保存模型的目录
    os.makedirs('checkpoints', exist_ok=True)
    
    print(f"Number of training epochs for each Task: {args.epochs_per_task}")
    
    # 创建教师模型（初始为None）
    teacher_model = None
    
    # 为每个Task训练
    for task_id in range(len(tasks_train)):
        print(f"\n=== Train Task {task_id} ({datasets_list[task_id]}) ===")
        
        # LwF: 如果不是第一个任务，创建教师模型
        if task_id > 0 and teacher_model is None:
            print("创建教师模型用于知识蒸馏...")
            teacher_model = create_teacher_model(model)
        
        dataset, collate_fn = tasks_train[task_id]
        task_loader = DataLoader(dataset, collate_fn=collate_fn, **{**cfg['train_loader'], 'pin_memory': False}, worker_init_fn=seed_worker, generator=generator)
        
        for epoch in range(args.epochs_per_task):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in tqdm(task_loader, desc=f"Task {task_id} Epoch {epoch+1}"):
                # 解析批次数据
                try:
                    images, imgs2, gt_datas = batch
                    images = images.to(device)
                    imgs2 = imgs2.to(device)
                    gt_cmaps = gt_datas[-1].to(device)
                    
                    optimizer.zero_grad()
                    
                    # 前向传播
                    if mode == 'final':
                        dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = model.forward_train(images, imgs2, gt_cmaps)
                        loss_den = compute_count_loss(criterion, dmaps1, gt_datas, device=device, log_para=log_para) + compute_count_loss(criterion, dmaps2, gt_datas, device=device, log_para=log_para)
                        loss_cls = F.binary_cross_entropy(cmaps1, gt_cmaps) + F.binary_cross_entropy(cmaps2, gt_cmaps)
                        
                        # 如果是第一个任务，或者没有教师模型，正常训练
                        if task_id == 0 or teacher_model is None:
                            loss_total = loss_den + 10 * loss_cls + 10 * loss_con
                        else:
                            # 如果有教师模型，添加知识蒸馏
                            # 使用教师模型获取旧知识的预测
                            with torch.no_grad():
                                teacher_dmaps1, teacher_dmaps2, teacher_cmaps1, teacher_cmaps2, _, _, _ = teacher_model.forward_train(images, imgs2, gt_cmaps)
                            
                            # 计算知识蒸馏损失
                            distill_den = (distillation_loss(dmaps1, teacher_dmaps1) + 
                                          distillation_loss(dmaps2, teacher_dmaps2))
                            distill_cls = (distillation_loss(cmaps1, teacher_cmaps1) + 
                                          distillation_loss(cmaps2, teacher_cmaps2))
                            
                            # 组合新任务损失和蒸馏损失
                            # 调整权重: 新任务(1-alpha)，旧知识(alpha)
                            alpha = 0.5  # 可以根据需要调整
                            loss_total = ((1-alpha) * (loss_den + 10 * loss_cls + 10 * loss_con) + 
                                         alpha * (distill_den + 10 * distill_cls))
                    else:
                        # 其他模式的处理可以基于原始代码添加
                        raise ValueError(f"LwF目前只支持 'final' 模式")
     
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
                    # 保存最佳模型
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'task_id': task_id,
                        'mae': perf_current['mae'],
                        'rmse': perf_current['rmse']
                    }, f'checkpoints/lwf/task_{task_id}_best.pth')
                    print(f"保存任务 {task_id} 的最佳模型，MAE: {perf_current['mae']:.2f}")
                
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
        
        # 在每个任务训练完后，更新教师模型为当前最佳模型
        if task_id < len(tasks_train) - 1:  # 如果不是最后一个任务
            print(f"更新教师模型为任务 {task_id} 的最佳模型...")
            # 加载当前任务的最佳模型
            best_model_path = f'checkpoints/lwf/task_{task_id}_best.pth'
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                # 更新教师模型
                teacher_model = create_teacher_model(model)
            else:
                # 如果没有找到最佳模型文件，使用当前模型作为教师模型
                teacher_model = create_teacher_model(model)
        
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
        print(f"\nTest Task {task_id} ({datasets_list[task_id]})")
        
        # 使用测试集，创建有效的DataLoader，关闭pin_memory
        test_dataset, test_collate_fn = tasks_test[task_id]
        test_loader = DataLoader(test_dataset, **{**cfg['test_loader'], 'pin_memory': False})
        
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
    
    # 汇总结果
    avg_final_mae = np.mean([perf['mae'] for perf in final_perf_each_task])
    avg_final_rmse = np.mean([perf['rmse'] for perf in final_perf_each_task])
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_perf_each_task': final_perf_each_task,
        'best_perf_each_task': best_perf_each_task,
        'forgetting_vals': forgetting_vals,
        'avg_final_mae': avg_final_mae,
        'avg_final_rmse': avg_final_rmse
    }, 'checkpoints/lwf/lwf_final_model.pth')
    print("\n保存最终模型到 checkpoints/lwf/lwf_final_model.pth")
    
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

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="fog,snow,stadium,street", 
                        help="Comma-separated list of datasets to use")
    parser.add_argument("--model_type", type=str, default="final", 
                        choices=["mem", "final"], help="Model type")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Training batch size")
    parser.add_argument("--epochs_per_task", type=int, default=5, 
                        help="每个Task的训练Epoch数")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--config", type=str, default="configs/jhu_domains_cl_config.yml", 
                        help="Config file path")
    parser.add_argument("--clearml_project", type=str, default="MPCount",
                        help="ClearML project name")
    parser.add_argument("--clearml_task", type=str, default="JHU_ContinualLearning_LwF",
                        help="ClearML task name")
    parser.add_argument("--use_clearml", action="store_true",
                        help="Whether to use ClearML for experiment tracking")
    parser.add_argument("--safe_mode", action="store_true",
                        help="启用安全模式，使用较小的批处理大小和更保守的内存设置")
    args = parser.parse_args()
    
    # 加载配置
    def load_config(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg
    
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
    task_name = join(args.clearml_task, args.model_type, args.datasets.replace(',', '_'))
    clearml_logger = CustomClearML('MPCount', task_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(vars(args))
    print("Using device:", device)
    
    # 使用LwF方法进行训练和评估
    master_seed = 1000  # 设置种子
    lwf_train_and_evaluate_trial(cfg, args, device, master_seed)
    
    print("\n==== 使用LwF方法的持续学习训练完成 ====")

if __name__ == "__main__":
    main() 