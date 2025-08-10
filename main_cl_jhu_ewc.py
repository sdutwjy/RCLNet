import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from main_cl_jhu import (CustomClearML, IndexedDataset, get_dataset, 
                        get_loss, get_model, compute_count_loss)

# ================= EWC工具类 ===================
class EWC:
    def __init__(self, model, dataloader, device, fisher_sample_num=200):
        self.model = copy.deepcopy(model)
        self.device = device
        self.fisher_sample_num = fisher_sample_num
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        count = 0
        for i, batch in enumerate(dataloader):
            if count >= self.fisher_sample_num:
                break
            images, imgs2, gt_datas = batch
            images = images.to(self.device)
            imgs2 = imgs2.to(self.device)
            gt_cmaps = gt_datas[-1].to(self.device)
            self.model.zero_grad()
            outputs = self.model.forward_train(images, imgs2, gt_cmaps)
            # 只用主损失
            if isinstance(outputs, (tuple, list)):
                loss = outputs[-1] if isinstance(outputs[-1], torch.Tensor) else outputs[0]
            else:
                loss = outputs
            loss = loss.mean()
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None and p.requires_grad:
                    fisher[n] += p.grad.detach() ** 2
            count += 1
        for n in fisher:
            fisher[n] /= max(count, 1)
        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                _loss = self.fisher[n] * (p - self.params[n]) ** 2
                loss += _loss.sum()
        return loss

# =================== 主训练流程 =====================
def ewc_train_and_evaluate_trial(cfg, args, device, master_seed):
    random.seed(master_seed)
    torch.manual_seed(master_seed)
    seed_everything(master_seed)

    datasets_list = args.datasets.split(',')
    from main_cl_jhu import create_crowd_counting_tasks
    tasks_train, tasks_val, tasks_test = create_crowd_counting_tasks(cfg)

    model_params = cfg['model']['params'] if 'params' in cfg['model'] else {}
    if args.model_type == 'mem':
        model = DGModel_mem(**model_params).to(device)
    elif args.model_type == 'final':
        model = DGModel_final(**model_params).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    clearml_logger = None
    if args.use_clearml and Task.current_task():
        clearml_logger = Task.current_task().get_logger()

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

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = get_loss()
    generator = get_seeded_generator(cfg['seed'])

    best_perf_each_task = [{'mae': float('inf'), 'rmse': float('inf')} for _ in range(len(tasks_train))]
    final_perf_each_task = [{'mae': float('inf'), 'rmse': float('inf')} for _ in range(len(tasks_train))]

    os.makedirs('checkpoints/ewc_ab', exist_ok=True)
    print(f"Number of training epochs for each Task: {args.epochs_per_task}")

    ewc_list = []
    ewc_lambda = 15.0  # 可根据需要调整

    for task_id in range(len(tasks_train)):
        print(f"\n=== Train Task {task_id} ({datasets_list[task_id]}) ===")
        dataset, collate_fn = tasks_train[task_id]
        task_loader = DataLoader(dataset, collate_fn=collate_fn, **{**cfg['train_loader'], 'pin_memory': False}, worker_init_fn=seed_worker, generator=generator)

        for epoch in range(args.epochs_per_task):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            for batch in tqdm(task_loader, desc=f"Task {task_id} Epoch {epoch+1}"):
                try:
                    images, imgs2, gt_datas = batch
                    images = images.to(device)
                    imgs2 = imgs2.to(device)
                    gt_cmaps = gt_datas[-1].to(device)

                    optimizer.zero_grad()
                    # 只用常规损失
                    dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = model.forward_train(images, imgs2, gt_cmaps)
                    loss_den = compute_count_loss(criterion, dmaps1, gt_datas, device=device, log_para=log_para) + \
                             compute_count_loss(criterion, dmaps2, gt_datas, device=device, log_para=log_para)
                    loss_cls = F.binary_cross_entropy(cmaps1, gt_cmaps) + F.binary_cross_entropy(cmaps2, gt_cmaps)
                    loss_total = loss_den + 10 * loss_cls + 10 * loss_con

                    # EWC penalty
                    if len(ewc_list) > 0:
                        ewc_penalty = sum([ewc.penalty(model) for ewc in ewc_list])
                        loss_total = loss_total + ewc_lambda * ewc_penalty

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
            # 验证
            print(f"Val Task {task_id}...")
            val_dataset, collate_fn = tasks_val[task_id]
            val_loader = DataLoader(val_dataset, **{**cfg['val_loader'], 'pin_memory': False})
            model.eval()
            mae_sum = 0
            mse_sum = 0
            val_count = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="evaling", leave=False):
                    try:
                        mae, metrics = trainer.val_step(model, batch)
                        mae_sum += mae
                        mse_sum += metrics['mse']
                        val_count += 1
                        if val_count <= 2:
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
                if perf_current['mae'] < best_perf_each_task[task_id]['mae']:
                    best_perf_each_task[task_id] = perf_current
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'task_id': task_id,
                        'mae': perf_current['mae'],
                        'rmse': perf_current['rmse']
                    }, f'checkpoints/ewc_ab/task_{task_id}_best.pth')
                    print(f"保存任务 {task_id} 的最佳模型，MAE: {perf_current['mae']:.2f}")
                print(f"  Epoch {epoch+1}/{args.epochs_per_task}, Task {task_id}, MAE = {perf_current['mae']:.2f}, RMSE = {perf_current['rmse']:.2f}")
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
        # 训练完当前任务，保存EWC
        print(f"计算EWC Fisher信息矩阵并保存...")
        ewc = EWC(model, task_loader, device)
        ewc_list.append(ewc)
        print(f"=== Task {task_id} ({datasets_list[task_id]}) training completed ===")
        print(f"Best validation performance - MAE: {best_perf_each_task[task_id]['mae']:.2f}, RMSE: {best_perf_each_task[task_id]['rmse']:.2f}")

    # 最终评估所有Task
    print("\n=== All tasks are finally evaluated ===")
    print("\n=== Best validation performance for each task ===")
    for task_id in range(len(tasks_train)):
        print(f"Task {task_id} ({datasets_list[task_id]}) - Best MAE: {best_perf_each_task[task_id]['mae']:.2f}, Best RMSE: {best_perf_each_task[task_id]['rmse']:.2f}")
    for task_id in range(len(tasks_train)):
        print(f"\nEvaluating Task {task_id} ({datasets_list[task_id]}) on test set...")
        test_dataset, test_collate_fn = tasks_test[task_id]
        test_loader = DataLoader(test_dataset, **{**cfg['test_loader'], 'pin_memory': False})
        model.eval()
        mae_sum = 0
        mse_sum = 0
        test_count = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="evaluating", leave=False):
                try:
                    metrics = trainer.test_step(model, batch)
                    mae_sum += metrics['mae']
                    mse_sum += metrics['mse']
                    test_count += 1
                    if test_count <= 2:
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
    forgetting_vals = [
        {
            'mae': best_perf_each_task[t]['mae'] - final_perf_each_task[t]['mae'],
            'rmse': best_perf_each_task[t]['rmse'] - final_perf_each_task[t]['rmse']
        }
        for t in range(len(tasks_train))
    ]
    avg_final_mae = np.mean([perf['mae'] for perf in final_perf_each_task])
    avg_final_rmse = np.mean([perf['rmse'] for perf in final_perf_each_task])
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_perf_each_task': final_perf_each_task,
        'best_perf_each_task': best_perf_each_task,
        'forgetting_vals': forgetting_vals,
        'avg_final_mae': avg_final_mae,
        'avg_final_rmse': avg_final_rmse
    }, 'checkpoints/ewc_ab/final_model.pth')
    print("\n保存最终模型到 checkpoints/ewc_ab/final_model.pth")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="sta,stb,qnrf", 
                        help="Comma-separated list of datasets to use")
    parser.add_argument("--model_type", type=str, default="final", 
                        choices=["mem", "final"], help="Model type")
    parser.add_argument("--epochs_per_task", type=int, default=5, 
                        help="每个Task的训练Epoch数")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--config", type=str, default="configs/cl_config.yml", 
                        help="Config file path")
    parser.add_argument("--clearml_project", type=str, default="MPCount",
                        help="ClearML project name")
    parser.add_argument("--clearml_task", type=str, default="abqnrf_ContinualLearning_EWC",
                        help="ClearML task name")
    parser.add_argument("--use_clearml", action="store_true",
                        help="Whether to use ClearML for experiment tracking")
    parser.add_argument("--safe_mode", action="store_true",
                        help="启用安全模式，使用较小的批处理大小和更保守的内存设置")
    args = parser.parse_args()
    def load_config(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg
    cfg = load_config(args.config)
    args.epochs_per_task = cfg.get('num_epochs', args.epochs_per_task) // len(cfg['train_dataset'])
    if args.safe_mode:
        print("启用安全模式，降低批处理大小")
        if 'train_loader' in cfg and 'batch_size' in cfg['train_loader']:
            cfg['train_loader']['batch_size'] = min(4, cfg['train_loader']['batch_size'])
        if 'val_loader' in cfg and 'batch_size' in cfg['val_loader']:
            cfg['val_loader']['batch_size'] = 1 
        if 'test_loader' in cfg and 'batch_size' in cfg['test_loader']:
            cfg['test_loader']['batch_size'] = 1
        if 'train_loader' in cfg:
            cfg['train_loader']['pin_memory'] = False
        if 'val_loader' in cfg:
            cfg['val_loader']['pin_memory'] = False
        if 'test_loader' in cfg:
            cfg['test_loader']['pin_memory'] = False
        if 'train_loader' in cfg and 'num_workers' in cfg['train_loader']:
            cfg['train_loader']['num_workers'] = min(1, cfg['train_loader']['num_workers'])
        if 'val_loader' in cfg and 'num_workers' in cfg['val_loader']:
            cfg['val_loader']['num_workers'] = min(1, cfg['val_loader']['num_workers'])
        if 'test_loader' in cfg and 'num_workers' in cfg['test_loader']:
            cfg['test_loader']['num_workers'] = min(1, cfg['test_loader']['num_workers'])
    task_name=join(args.clearml_task, args.model_type, args.datasets.replace(',', '_'))
    clearml_logger = CustomClearML(args.clearml_project, task_name) if args.use_clearml else None
    device = torch.device("cuda")
    print(vars(args))
    print("Using device:", device)
    master_seed = 1000
    ewc_train_and_evaluate_trial(cfg, args, device, master_seed)

if __name__ == "__main__":
    main()