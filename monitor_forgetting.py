#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import time
import copy
import random
import numpy as np
import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.models import DGModel_mem, DGModel_final, DGModel_base
from trainers.dgtrainer import DGTrainer
from utils.misc import seed_everything

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def get_model(name, params):
    if name == 'base':
        return DGModel_base(**params)
    elif name == 'mem':
        return DGModel_mem(**params)
    elif name == 'final':
        return DGModel_final(**params)
    else:
        raise ValueError(f'未知模型: {name}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cl_config.yml", 
                        help="配置文件路径")
    parser.add_argument("--model_type", type=str, default="final", 
                        choices=["mem", "final"], help="模型类型")
    parser.add_argument("--datasets", type=str, default="fog,snow,stadium,street", 
                        help="逗号分隔的数据集列表")
    parser.add_argument("--device", type=str, default="cuda", help="设备(cuda或cpu)")
    parser.add_argument("--eval_task", type=int, default=-1, 
                        help="要评估的任务ID,-1表示评估所有任务")
    parser.add_argument("--eval_mode", type=str, default="best", 
                        choices=["best", "final", "last"], 
                        help="评估模式: best-使用每个任务的最优模型, final-使用final_model.pth, last-使用最后一个任务的最优模型")
    parser.add_argument("--metric", type=str, default="all",
                        choices=["mae", "rmse", "mse", "all"],
                        help="要显示的评估指标: mae-平均绝对误差, rmse-均方根误差, mse-均方误差, all-显示所有指标")
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    datasets_list = args.datasets.split(',')
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建DGTrainer实例用于评估
    log_para = cfg.get('log_para', 1000)
    patch_size = cfg.get('patch_size', 10000)
    mode = cfg.get('mode', 'final')
    trainer = DGTrainer(
        seed=cfg.get('seed', 2023),
        version=cfg.get('version', 'cl'),
        device=device,
        log_para=log_para,
        patch_size=patch_size,
        mode=mode
    )
    
    # 检查checkpoints目录
    if not os.path.exists('checkpoints'):
        print("错误: 没有找到checkpoints目录，请先训练模型")
        return
    
    # 创建模型
    model_params = cfg['model']['params'] if 'params' in cfg['model'] else {}
    model = get_model(args.model_type, model_params).to(device)
    
    # 从main_cl_jhu.py的create_crowd_counting_tasks函数中提取创建数据集的逻辑
    from main_cl_jhu import create_crowd_counting_tasks
    tasks_train, tasks_val, tasks_test = create_crowd_counting_tasks(cfg)
    
    print(f"任务数据集: {datasets_list}")
    
    # 跟踪每个任务的性能
    task_performances = {}
    
    if args.eval_mode == "best":
        # 检查任务特定的最佳模型文件
        task_files = []
        for task_id in range(len(datasets_list)):
            task_file = f'checkpoints/gradient_ac/task_{task_id}_best.pth'
            if os.path.exists(task_file):
                task_files.append(task_file)
            else:
                print(f"警告: 没有找到任务 {task_id} 的模型文件 {task_file}")
        
        if not task_files:
            print("错误: 没有找到任何任务模型文件")
            return
        
        print(f"找到 {len(task_files)} 个任务模型文件")
        
        # 遍历每个任务的模型文件
        for curr_task_id in range(len(task_files)):
            task_file = f'checkpoints/gradient_ac/task_{curr_task_id}_best.pth'
            print(f"\n=== 使用任务 {curr_task_id} ({datasets_list[curr_task_id]}) 的最佳模型评估 ===")
            
            # 加载当前任务的最佳模型
            checkpoint = torch.load(task_file, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"加载模型文件: {task_file}")
            print(f"该模型在任务 {curr_task_id} 上的验证 MAE: {checkpoint['mae']:.2f}")
            if 'rmse' in checkpoint:
                print(f"该模型在任务 {curr_task_id} 上的验证 RMSE: {checkpoint['rmse']:.2f}")
            if 'mse' in checkpoint:
                print(f"该模型在任务 {curr_task_id} 上的验证 MSE: {checkpoint['mse']:.2f}")
            
            # 评估模型在所有任务上的性能
            evaluate_model_performance(model, trainer, tasks_test, datasets_list, 
                                    curr_task_id, task_performances, args.eval_task, cfg)
    
    elif args.eval_mode == "final":
        final_model_path = 'checkpoints/gradient_ac/final_model.pth'
        if not os.path.exists(final_model_path):
            print(f"错误: 没有找到最终模型文件 {final_model_path}")
            return
            
        print("\n=== 使用最终模型 (final_model.pth) 评估 ===")
        checkpoint = torch.load(final_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"加载模型文件: {final_model_path}")
        evaluate_model_performance(model, trainer, tasks_test, datasets_list, 
                                "final", task_performances, args.eval_task, cfg)
    
    elif args.eval_mode == "last":
        last_task_id = len(datasets_list) - 1
        last_task_file = f'checkpoints/gradient_ac/task_{last_task_id}_best.pth'
        
        if not os.path.exists(last_task_file):
            print(f"错误: 没有找到最后一个任务的模型文件 {last_task_file}")
            return
            
        print(f"\n=== 使用最后一个任务 {last_task_id} ({datasets_list[last_task_id]}) 的最佳模型评估 ===")
        checkpoint = torch.load(last_task_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"加载模型文件: {last_task_file}")
        print(f"该模型在任务 {last_task_id} 上的验证 MAE: {checkpoint['mae']:.2f}")
        if 'rmse' in checkpoint:
            print(f"该模型在任务 {last_task_id} 上的验证 RMSE: {checkpoint['rmse']:.2f}")
        if 'mse' in checkpoint:
            print(f"该模型在任务 {last_task_id} 上的验证 MSE: {checkpoint['mse']:.2f}")
        
        evaluate_model_performance(model, trainer, tasks_test, datasets_list, 
                                "last", task_performances, args.eval_task, cfg)
    
    # 打印性能矩阵
    if args.metric in ['mae', 'both', 'all']:
        print("\n=== 性能矩阵 (MAE) ===")
        print("行: 模型训练到的任务, 列: 评估的任务")
        print(f"{'Task ID':>10}", end="")
        for eval_task_id in range(len(datasets_list)):
            print(f"{eval_task_id:>10}", end="")
        print()
        
        for curr_task_id in range(len(datasets_list)):
            print(f"{curr_task_id:>10}", end="")
            for eval_task_id in range(len(datasets_list)):
                key = (curr_task_id, eval_task_id)
                if key in task_performances:
                    print(f"{task_performances[key]['mae']:>10.2f}", end="")
                else:
                    print(f"{'N/A':>10}", end="")
            print()
    
    if args.metric in ['mse', 'all']:
        print("\n=== 性能矩阵 (MSE) ===")
        print("行: 模型训练到的任务, 列: 评估的任务")
        print(f"{'Task ID':>10}", end="")
        for eval_task_id in range(len(datasets_list)):
            print(f"{eval_task_id:>10}", end="")
        print()
        
        for curr_task_id in range(len(datasets_list)):
            print(f"{curr_task_id:>10}", end="")
            for eval_task_id in range(len(datasets_list)):
                key = (curr_task_id, eval_task_id)
                if key in task_performances:
                    print(f"{task_performances[key]['mse']:>10.2f}", end="")
                else:
                    print(f"{'N/A':>10}", end="")
            print()

    if args.metric in ['rmse', 'both', 'all']:
        print("\n=== 性能矩阵 (RMSE) ===")
        print("行: 模型训练到的任务, 列: 评估的任务")
        print(f"{'Task ID':>10}", end="")
        for eval_task_id in range(len(datasets_list)):
            print(f"{eval_task_id:>10}", end="")
        print()
        
        for curr_task_id in range(len(datasets_list)):
            print(f"{curr_task_id:>10}", end="")
            for eval_task_id in range(len(datasets_list)):
                key = (curr_task_id, eval_task_id)
                if key in task_performances:
                    print(f"{task_performances[key]['rmse']:>10.2f}", end="")
                else:
                    print(f"{'N/A':>10}", end="")
            print()
    
    # 计算并显示平均性能
    if args.metric in ['mae', 'both', 'all']:  # 显示MAE平均值
        avg_mae_values = []
        for eval_task_id in range(len(datasets_list)):
            task_mae_values = []
            for curr_task_id in range(len(datasets_list)):
                key = (curr_task_id, eval_task_id)
                if key in task_performances:
                    task_mae_values.append(task_performances[key]['mae'])
            if task_mae_values:
                avg_mae = sum(task_mae_values) / len(task_mae_values)
                avg_mae_values.append(avg_mae)
                print(f"任务 {eval_task_id} ({datasets_list[eval_task_id]}) 的平均 MAE: {avg_mae:.2f}")
        
        if avg_mae_values:
            overall_avg_mae = sum(avg_mae_values) / len(avg_mae_values)
            print(f"\n所有任务的总体平均 MAE: {overall_avg_mae:.2f}")
    
    if args.metric in ['mse', 'all']:  # 显示MSE平均值
        avg_mse_values = []
        for eval_task_id in range(len(datasets_list)):
            task_mse_values = []
            for curr_task_id in range(len(datasets_list)):
                key = (curr_task_id, eval_task_id)
                if key in task_performances:
                    task_mse_values.append(task_performances[key]['mse'])
            if task_mse_values:
                avg_mse = sum(task_mse_values) / len(task_mse_values)
                avg_mse_values.append(avg_mse)
                print(f"任务 {eval_task_id} ({datasets_list[eval_task_id]}) 的平均 MSE: {avg_mse:.2f}")
        
        if avg_mse_values:
            overall_avg_mse = sum(avg_mse_values) / len(avg_mse_values)
            print(f"\n所有任务的总体平均 MSE: {overall_avg_mse:.2f}")
    
    if args.metric in ['rmse', 'both', 'all']:  # 显示RMSE平均值
        avg_rmse_values = []
        for eval_task_id in range(len(datasets_list)):
            task_rmse_values = []
            for curr_task_id in range(len(datasets_list)):
                key = (curr_task_id, eval_task_id)
                if key in task_performances:
                    task_rmse_values.append(task_performances[key]['rmse'])
            if task_rmse_values:
                avg_rmse = sum(task_rmse_values) / len(task_rmse_values)
                avg_rmse_values.append(avg_rmse)
                print(f"任务 {eval_task_id} ({datasets_list[eval_task_id]}) 的平均 RMSE: {avg_rmse:.2f}")
        
        if avg_rmse_values:
            overall_avg_rmse = sum(avg_rmse_values) / len(avg_rmse_values)
            print(f"\n所有任务的总体平均 RMSE: {overall_avg_rmse:.2f}")

def evaluate_model_performance(model, trainer, tasks_test, datasets_list, 
                             curr_task_id, task_performances, eval_task, cfg):
    """评估模型在所有任务上的性能"""
    task_id_to_eval = eval_task if eval_task >= 0 else range(len(tasks_test))
    
    if isinstance(task_id_to_eval, int):
        task_id_to_eval = [task_id_to_eval]
    
    for eval_task_id in task_id_to_eval:
        if eval_task_id >= len(tasks_test):
            print(f"任务ID {eval_task_id} 超出范围，跳过")
            continue
            
        # 获取测试数据集
        test_dataset, test_collate_fn = tasks_test[eval_task_id]
        test_loader = DataLoader(
            test_dataset, 
            **{**cfg['test_loader'], 'pin_memory': False}
        )
        
        # 评估模型
        mae_sum = 0
        mse_sum = 0
        test_count = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"评估任务 {eval_task_id}", leave=False):
                try:
                    metrics = trainer.test_step(model, batch)
                    mae_sum += metrics['mae']
                    mse_sum += metrics['mse']
                    test_count += 1
                except Exception as e:
                    print(f"评估时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if test_count > 0:
            perf = {
                'mae': mae_sum / test_count,
                'mse': mse_sum / test_count,
                'rmse': np.sqrt(mse_sum / test_count)
            }
            
            # 保存性能数据
            key = (curr_task_id, eval_task_id)
            task_performances[key] = perf
            
            print(f"任务 {curr_task_id} 的模型在任务 {eval_task_id} ({datasets_list[eval_task_id]}) 上的性能:")
            print(f"  MAE: {perf['mae']:.2f}, MSE: {perf['mse']:.2f}, RMSE: {perf['rmse']:.2f}")

if __name__ == "__main__":
    main() 