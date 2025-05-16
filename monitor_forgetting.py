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
    parser.add_argument("--config", type=str, default="configs/jhu_domains_cl_config.yml", 
                        help="配置文件路径")
    parser.add_argument("--model_type", type=str, default="final", 
                        choices=["mem", "final"], help="模型类型")
    parser.add_argument("--datasets", type=str, default="fog,snow,stadium,street", 
                        help="逗号分隔的数据集列表")
    parser.add_argument("--device", type=str, default="cuda", help="设备(cuda或cpu)")
    parser.add_argument("--eval_task", type=int, default=-1, 
                        help="要评估的任务ID,-1表示评估所有任务")
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
    
    # 检查任务特定的最佳模型文件
    task_files = []
    for task_id in range(len(datasets_list)):
        task_file = f'checkpoints/lwf/task_{task_id}_best.pth'
        if os.path.exists(task_file):
            task_files.append(task_file)
        else:
            print(f"警告: 没有找到任务 {task_id} 的模型文件 {task_file}")
    
    if not task_files:
        print("错误: 没有找到任何任务模型文件")
        return
    
    # 创建模型
    model_params = cfg['model']['params'] if 'params' in cfg['model'] else {}
    model = get_model(args.model_type, model_params).to(device)
    
    # 从main_cl_jhu.py的create_crowd_counting_tasks函数中提取创建数据集的逻辑
    # 注意: 这里应当根据您的具体实现调整
    from main_cl_jhu import create_crowd_counting_tasks
    tasks_train, tasks_val, tasks_test = create_crowd_counting_tasks(cfg)
    
    print(f"找到 {len(task_files)} 个任务模型文件")
    print(f"任务数据集: {datasets_list}")
    
    # 跟踪每个任务的初始和当前性能
    task_performances = {}
    
    # 遍历每个任务的模型文件
    for curr_task_id in range(len(task_files)):
        task_file = f'checkpoints/lwf/task_{curr_task_id}_best.pth'
        
        print(f"\n=== 使用任务 {curr_task_id} ({datasets_list[curr_task_id]}) 的最佳模型评估 ===")
        
        # 加载当前任务的最佳模型
        checkpoint = torch.load(task_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"加载模型文件: {task_file}")
        print(f"该模型在任务 {curr_task_id} 上的验证 MAE: {checkpoint['mae']:.2f}")
        
        # 评估模型在所有任务上的性能
        task_id_to_eval = args.eval_task if args.eval_task >= 0 else range(len(tasks_test))
        
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
                        # 使用trainer.test_step来评估
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
                    'rmse': np.sqrt(mse_sum / test_count)
                }
                
                # 保存性能数据
                key = (curr_task_id, eval_task_id)
                task_performances[key] = perf
                
                print(f"任务 {curr_task_id} 的模型在任务 {eval_task_id} ({datasets_list[eval_task_id]}) 上的性能:")
                print(f"  MAE: {perf['mae']:.2f}, RMSE: {perf['rmse']:.2f}")
                
                # 计算遗忘度(如果评估的是先前学习的任务)
                if eval_task_id < curr_task_id:
                    # 找到之前该任务最佳的性能
                    prev_best_key = (eval_task_id, eval_task_id)  # 该任务自己的最佳模型在该任务上的性能
                    if prev_best_key in task_performances:
                        prev_best = task_performances[prev_best_key]
                        
                        forgetting_mae = perf['mae'] - prev_best['mae'] 
                        forgetting_rmse = perf['rmse'] - prev_best['rmse']
                        
                        print(f"  遗忘度 (MAE): {forgetting_mae:.2f}")
                        print(f"  遗忘度 (RMSE): {forgetting_rmse:.2f}")
                        print(f"  相对遗忘率: {(forgetting_mae / prev_best['mae'] * 100):.2f}%")
    
    print("\n=== 遗忘矩阵 (MAE) ===")
    print("行: 模型训练到的任务, 列: 评估的任务")
    print(f"{'Task ID':>10}", end="")
    for eval_task_id in range(len(datasets_list)):
        print(f"{eval_task_id:>10}", end="")
    print()
    
    for curr_task_id in range(len(task_files)):
        print(f"{curr_task_id:>10}", end="")
        for eval_task_id in range(len(datasets_list)):
            key = (curr_task_id, eval_task_id)
            if key in task_performances:
                print(f"{task_performances[key]['mae']:>10.2f}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()
    
    # 计算平均遗忘度
    total_forgetting_mae = 0.0
    count = 0
    
    for curr_task_id in range(1, len(task_files)):  # 从第二个任务开始
        for eval_task_id in range(curr_task_id):  # 只考虑先前的任务
            curr_key = (curr_task_id, eval_task_id)
            best_key = (eval_task_id, eval_task_id)
            
            if curr_key in task_performances and best_key in task_performances:
                forgetting = task_performances[curr_key]['mae'] - task_performances[best_key]['mae']
                total_forgetting_mae += forgetting
                count += 1
    
    if count > 0:
        avg_forgetting = total_forgetting_mae / count
        print(f"\n平均遗忘度 (MAE): {avg_forgetting:.2f}")
    else:
        print("\n无法计算平均遗忘度，数据不足")

if __name__ == "__main__":
    main() 