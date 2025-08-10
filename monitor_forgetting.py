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
import cv2
import scipy.spatial
import math
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
    parser.add_argument("--config", type=str, default="/home/jianyong/exp/MPCount/configs/jhu_domains_cl_config.yml", 
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
    parser.add_argument("--visual", action="store_true", help="是否生成可视化结果")
    parser.add_argument("--visual_path", type=str, default="./visual_results", help="可视化结果保存路径")
    # 移除 --image_dir 参数，因为我们将直接使用测试数据
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
            task_file = f'checkpoints/ewc/task_{task_id}_best.pth'
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
            task_file = f'checkpoints/ewc/task_{curr_task_id}_best.pth'
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
                                    curr_task_id, task_performances, args.eval_task, cfg, args)
    
    elif args.eval_mode == "final":
        final_model_path = 'checkpoints/ewc/final_model.pth'
        if not os.path.exists(final_model_path):
            print(f"错误: 没有找到最终模型文件 {final_model_path}")
            return
            
        print("\n=== 使用最终模型 (final_model.pth) 评估 ===")
        checkpoint = torch.load(final_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"加载模型文件: {final_model_path}")
        evaluate_model_performance(model, trainer, tasks_test, datasets_list, 
                                "final", task_performances, args.eval_task, cfg, args)
    
    elif args.eval_mode == "last":
        last_task_id = len(datasets_list) - 1
        last_task_file = f'checkpoints/ewc/task_{last_task_id}_best.pth'
        
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
                                "last", task_performances, args.eval_task, cfg, args)
    
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

def show_map(input):
    """将模型输出转换为热力图"""
    # 处理输出为元组的情况
    if isinstance(input, tuple):
        input = input[0]
    
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    
    # 处理可能的NaN值
    if np.isnan(fidt_map1).any() or np.max(fidt_map1) == 0:
        # 如果有NaN值或全为0，创建一个空白图像
        h, w = fidt_map1.shape
        fidt_map1 = np.zeros((h, w), dtype=np.uint8)
    else:
        fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
        fidt_map1 = fidt_map1.astype(np.uint8)
    
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1

def generate_point_map(density_map, rate=1):
    """根据密度图生成点图"""
    # 处理输出为元组的情况
    if isinstance(density_map, tuple):
        # 假设第一个元素是我们需要的密度图
        density_map = density_map[0]
    
    # 找到局部最大值作为预测点
    keep = torch.nn.functional.max_pool2d(density_map, (3, 3), stride=1, padding=1)
    keep = (keep == density_map).float()
    density_map = keep * density_map
    
    # 设置阈值
    input_max = torch.max(density_map).item()
    density_map[density_map < 100.0 / 255.0 * input_max] = 0
    density_map[density_map > 0] = 1
    
    # 转换为numpy数组
    kpoint = density_map.data.squeeze(0).squeeze(0).cpu().numpy()
    
    # 获取预测点坐标
    pred_coor = np.nonzero(kpoint)
    
    # 创建点图
    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)
    
    return point_map, kpoint

def generate_bounding_boxes(kpoint, img_path):
    """在原始图像上生成边界框"""
    try:
        # 读取原始图像
        Img_data = cv2.imread(img_path)
        if Img_data is None:
            print(f"警告: 无法读取图像 {img_path}")
            return None, None
            
        ori_Img_data = Img_data.copy()
        
        # 获取预测点坐标
        pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
        if len(pts) == 0:
            return ori_Img_data, ori_Img_data
            
        # 构建KD树
        leafsize = 2048
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
        
        # 为每个点计算sigma并绘制边界框
        distances, locations = tree.query(pts, k=min(4, len(pts)))
        for index, pt in enumerate(pts):
            pt2d = np.zeros(kpoint.shape, dtype=np.float32)
            pt2d[pt[1], pt[0]] = 1.
            if np.sum(kpoint) > 1 and len(distances[index]) > 1:
                sigma = np.mean(distances[index][1:]) * 0.1
            else:
                sigma = np.average(np.array(kpoint.shape)) / 2. / 2.
            sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.05)
            
            t = 2
            Img_data = cv2.rectangle(Img_data, (int(pt[0] - sigma), int(pt[1] - sigma)),
                                    (int(pt[0] + sigma), int(pt[1] + sigma)), (0, 255, 0), t)
        
        return ori_Img_data, Img_data
    except Exception as e:
        print(f"生成边界框时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def evaluate_model_performance(model, trainer, tasks_test, datasets_list, 
                             curr_task_id, task_performances, eval_task, cfg, args=None):
    """评估模型在所有任务上的性能并生成可视化结果"""
    task_id_to_eval = eval_task if eval_task >= 0 else range(len(tasks_test))
    
    if isinstance(task_id_to_eval, int):
        task_id_to_eval = [task_id_to_eval]
    
    # 创建可视化结果目录
    if args and args.visual:
        visual_path = args.visual_path
        if not os.path.exists(visual_path):
            os.makedirs(visual_path)
    
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
        
        # 为当前任务创建可视化子目录
        if args and args.visual:
            task_visual_path = os.path.join(visual_path, f"task_{eval_task_id}_{datasets_list[eval_task_id]}")
            if not os.path.exists(task_visual_path):
                os.makedirs(task_visual_path)
        
        # 评估模型
        mae_sum = 0
        mse_sum = 0
        test_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"评估任务 {eval_task_id}", leave=False)):
                try:
                    # 计算指标
                    metrics = trainer.test_step(model, batch)
                    mae_sum += metrics['mae']
                    mse_sum += metrics['mse']
                    test_count += 1
                    
                    # 生成可视化结果 - 为所有样本生成，不再限制每5个样本
                    if args and args.visual:
                        try:
                            # 检查batch的结构
                            if isinstance(batch, list) and len(batch) >= 4:
                                # 获取文件名
                                if isinstance(batch[0], list) and len(batch[0]) > 0:
                                    fname = batch[0][0]
                                elif isinstance(batch[0], str):
                                    fname = batch[0]
                                else:
                                    # 如果无法获取文件名，使用索引作为文件名
                                    fname = f"sample_{batch_idx}.jpg"
                                    
                                # 打印文件名用于调试
                                print(f"处理文件: {fname}")
                                
                                # 获取输入图像和目标
                                img = batch[1].to(trainer.device)
                                targets = batch[2].to(trainer.device)
                                
                                # 前向传播
                                outputs = model(img)
                                
                                # 处理outputs为元组的情况
                                if isinstance(outputs, tuple):
                                    output_tensor = outputs[0]
                                else:
                                    output_tensor = outputs
                                
                                # 计算预测人数和真实人数
                                if isinstance(outputs, tuple):
                                    pred_count = output_tensor.sum().cpu().item() / trainer.log_para
                                else:
                                    pred_count = outputs.sum().cpu().item() / trainer.log_para
                                    
                                gt_count = targets.sum().cpu().item() / trainer.log_para
                                
                                # 生成点图
                                point_map, kpoint = generate_point_map(outputs)
                                
                                # 生成热力图 - 使用正确处理的tensor
                                if isinstance(outputs, tuple):
                                    pred_heatmap = show_map(output_tensor.cpu().numpy())
                                else:
                                    pred_heatmap = show_map(outputs.data.cpu().numpy())
                                    
                                target_heatmap = show_map(targets.data.cpu().numpy())
                                
                                # 使用输入图像作为原始图像
                                # 将PyTorch张量转换为OpenCV格式的图像
                                input_img = img.cpu().numpy()
                                input_img = input_img.squeeze(0).transpose(1, 2, 0)  # 从[C,H,W]转换为[H,W,C]
                                
                                # 反归一化图像
                                mean = np.array([0.485, 0.456, 0.406])
                                std = np.array([0.229, 0.224, 0.225])
                                input_img = input_img * std + mean
                                input_img = np.clip(input_img * 255, 0, 255).astype(np.uint8)
                                
                                # 使用输入图像生成边界框图像
                                box_img = input_img.copy()
                                
                                # 在边界框图像上绘制边界框
                                pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
                                if len(pts) > 0:
                                    # 构建KD树
                                    leafsize = 2048
                                    tree = scipy.spatial.KDTree(pts.copy(), leafsize=min(leafsize, len(pts)))
                                    
                                    # 为每个点计算sigma并绘制边界框
                                    distances, locations = tree.query(pts, k=min(4, len(pts)))
                                    for index, pt in enumerate(pts):
                                        if np.sum(kpoint) > 1 and len(distances[index]) > 1:
                                            sigma = np.mean(distances[index][1:]) * 0.1
                                        else:
                                            sigma = np.average(np.array(kpoint.shape)) / 2. / 2.
                                        sigma = min(sigma, min(box_img.shape[0], box_img.shape[1]) * 0.05)
                                        
                                        t = 2
                                        box_img = cv2.rectangle(box_img, 
                                                              (int(pt[0] - sigma), int(pt[1] - sigma)),
                                                              (int(pt[0] + sigma), int(pt[1] + sigma)), 
                                                              (0, 255, 0), t)
                                
                                # 检查图像尺寸
                                print(f"原始图像尺寸: input_img={input_img.shape}, target_heatmap={target_heatmap.shape}, pred_heatmap={pred_heatmap.shape}, point_map={point_map.shape}")
                                
                                # 如果任何图像尺寸太小，创建一个更大的空白图像
                                min_height = 256
                                min_width = 256
                                
                                # 确保所有图像至少有最小尺寸
                                if input_img.shape[0] < min_height or input_img.shape[1] < min_width:
                                    # 创建更大的空白图像
                                    new_input_img = np.ones((min_height, min_width, 3), dtype=np.uint8) * 255
                                    # 如果原图不是空的，将其放在中心位置
                                    if input_img.size > 0:
                                        h, w = input_img.shape[:2]
                                        y_offset = (min_height - h) // 2
                                        x_offset = (min_width - w) // 2
                                        if y_offset >= 0 and x_offset >= 0 and y_offset + h <= min_height and x_offset + w <= min_width:
                                            new_input_img[y_offset:y_offset+h, x_offset:x_offset+w] = input_img
                                    input_img = new_input_img
                                    box_img = input_img.copy()
                                
                                if target_heatmap.shape[0] < min_height or target_heatmap.shape[1] < min_width:
                                    new_target = np.zeros((min_height, min_width, 3), dtype=np.uint8)
                                    h, w = target_heatmap.shape[:2]
                                    y_offset = (min_height - h) // 2
                                    x_offset = (min_width - w) // 2
                                    if y_offset >= 0 and x_offset >= 0 and y_offset + h <= min_height and x_offset + w <= min_width:
                                        new_target[y_offset:y_offset+h, x_offset:x_offset+w] = target_heatmap
                                    target_heatmap = new_target
                                
                                if pred_heatmap.shape[0] < min_height or pred_heatmap.shape[1] < min_width:
                                    new_pred = np.zeros((min_height, min_width, 3), dtype=np.uint8)
                                    h, w = pred_heatmap.shape[:2]
                                    y_offset = (min_height - h) // 2
                                    x_offset = (min_width - w) // 2
                                    if y_offset >= 0 and x_offset >= 0 and y_offset + h <= min_height and x_offset + w <= min_width:
                                        new_pred[y_offset:y_offset+h, x_offset:x_offset+w] = pred_heatmap
                                    pred_heatmap = new_pred
                                
                                if point_map.shape[0] < min_height or point_map.shape[1] < min_width:
                                    new_point = np.ones((min_height, min_width, 3), dtype=np.uint8) * 255
                                    h, w = point_map.shape[:2]
                                    y_offset = (min_height - h) // 2
                                    x_offset = (min_width - w) // 2
                                    if y_offset >= 0 and x_offset >= 0 and y_offset + h <= min_height and x_offset + w <= min_width:
                                        new_point[y_offset:y_offset+h, x_offset:x_offset+w] = point_map
                                    point_map = new_point
                                
                                # 确保所有图像具有相同的尺寸
                                # 找到最大的尺寸
                                max_h = max(input_img.shape[0], target_heatmap.shape[0], pred_heatmap.shape[0], point_map.shape[0])
                                max_w = max(input_img.shape[1], target_heatmap.shape[1], pred_heatmap.shape[1], point_map.shape[1])
                                
                                # 调整所有图像大小
                                input_img = cv2.resize(input_img, (max_w, max_h))
                                box_img = cv2.resize(box_img, (max_w, max_h))
                                target_heatmap = cv2.resize(target_heatmap, (max_w, max_h))
                                pred_heatmap = cv2.resize(pred_heatmap, (max_w, max_h))
                                point_map = cv2.resize(point_map, (max_w, max_h))
                                
                                # 检查调整后的图像尺寸
                                print(f"调整后的图像尺寸: input_img={input_img.shape}, target_heatmap={target_heatmap.shape}, pred_heatmap={pred_heatmap.shape}, point_map={point_map.shape}, box_img={box_img.shape}")
                                
                                # 创建标题区域，用于显示GT和预测值
                                title_height = 50  # 标题区域高度
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.7
                                font_thickness = 2
                                font_color = (0, 0, 255)  # 红色
                                
                                # 计算误差
                                error = abs(pred_count - gt_count)
                                
                                # 创建每个图像的标题区域
                                input_title = np.ones((title_height, max_w, 3), dtype=np.uint8) * 255
                                target_title = np.ones((title_height, max_w, 3), dtype=np.uint8) * 255
                                pred_title = np.ones((title_height, max_w, 3), dtype=np.uint8) * 255
                                point_title = np.ones((title_height, max_w, 3), dtype=np.uint8) * 255
                                box_title = np.ones((title_height, max_w, 3), dtype=np.uint8) * 255
                                
                                # 在标题区域添加文本
                                cv2.putText(input_title, "原始图像", (10, 30), font, font_scale, (0, 0, 0), font_thickness)
                                cv2.putText(target_title, f"目标热力图 (GT: {gt_count:.1f})", (10, 30), font, font_scale, (0, 0, 0), font_thickness)
                                cv2.putText(pred_title, f"预测热力图 (Pred: {pred_count:.1f})", (10, 30), font, font_scale, (0, 0, 0), font_thickness)
                                cv2.putText(point_title, "点图", (10, 30), font, font_scale, (0, 0, 0), font_thickness)
                                cv2.putText(box_title, f"边界框 (Error: {error:.1f})", (10, 30), font, font_scale, (0, 0, 0), font_thickness)
                                
                                # 将标题区域与图像垂直拼接
                                input_with_title = np.vstack((input_title, input_img))
                                target_with_title = np.vstack((target_title, target_heatmap))
                                pred_with_title = np.vstack((pred_title, pred_heatmap))
                                point_with_title = np.vstack((point_title, point_map))
                                box_with_title = np.vstack((box_title, box_img))
                                
                                # 水平拼接所有图像
                                result_img = np.hstack((input_with_title, target_with_title, pred_with_title, point_with_title, box_with_title))
                                
                                # 保存结果
                                if isinstance(fname, str):
                                    save_name = os.path.basename(fname)
                                else:
                                    save_name = f"sample_{batch_idx}.jpg"
                                    
                                save_path = os.path.join(task_visual_path, save_name)
                                cv2.imwrite(save_path, result_img)
                                print(f"已保存可视化结果: {save_path}")
                            else:
                                print(f"警告: 无法识别的batch结构: {type(batch)}")
                                
                        except Exception as e:
                            print(f"生成可视化结果时出错: {e}")
                            import traceback
                            traceback.print_exc()
                
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