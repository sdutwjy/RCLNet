import os
import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from trainers.dgtrainer import DGTrainer
from models.models import DGModel_base, DGModel_mem, DGModel_final
from datasets.den_dataset import DensityMapDataset
from datasets.den_cls_dataset import DenClsDataset
from datasets.jhu_domain_dataset import JHUDomainDataset
from datasets.jhu_domain_cls_dataset import JHUDomainClsDataset
from utils.misc import seed_worker, get_seeded_generator, seed_everything


def get_model(name, params):
    if name == 'base':
        return DGModel_base(**params)
    elif name == 'mem':
        return DGModel_mem(**params)
    elif name == 'final':
        return DGModel_final(**params)
    else:
        raise ValueError(f'未知模型: {name}')


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


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/stb_test_sta.yml', 
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='预训练模型权重路径，如果为None则使用配置文件中的值')
    parser.add_argument('--device', type=str, default=None,
                        help='使用的设备 (cuda:0, cuda:1, ...)')
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)
    
    # 设置设备
    device_str = args.device if args.device else cfg['device']
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子
    seed_everything(cfg['seed'])
    
    # 创建模型
    model_params = cfg['model']['params'] if 'params' in cfg['model'] else {}
    model = get_model(cfg['model']['name'], model_params).to(device)
    
    # 加载预训练模型权重
    checkpoint_path = args.checkpoint if args.checkpoint else cfg['checkpoint']
    print(f"加载预训练模型: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # 创建测试数据集
    test_dataset, _ = get_dataset(
        cfg['test_dataset']['name'],
        cfg['test_dataset']['params'],
        method='test'
    )
    test_loader = DataLoader(test_dataset, **cfg['test_loader'])
    
    # 创建DGTrainer实例用于评估
    log_para = cfg.get('log_para', 1000)
    patch_size = cfg.get('patch_size', 10000)
    mode = cfg.get('mode', 'final')
    trainer = DGTrainer(
        seed=cfg['seed'],
        version=cfg.get('version', 'stb_test_sta'),
        device=device,
        log_para=log_para,
        patch_size=patch_size,
        mode=mode
    )
    
    # 进行测试
    print("\n开始测试 STB 模型在 STA 数据集上的性能...")
    model.eval()
    mae_sum = 0
    mse_sum = 0
    test_count = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试中"):
            try:
                metrics = trainer.test_step(model, batch)
                mae_sum += metrics['mae']
                mse_sum += metrics['mse']
                test_count += 1
                
                if test_count <= 5:  # 显示前5个样本的详细结果
                    print(f"样本 {test_count}: MAE={metrics['mae']:.2f}, MSE={metrics['mse']:.2f}")
            except Exception as e:
                print(f"错误: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if test_count > 0:
        final_mae = mae_sum / test_count
        final_rmse = np.sqrt(mse_sum / test_count)
        print(f"\n测试结果汇总:")
        print(f"测试样本数: {test_count}")
        print(f"平均 MAE: {final_mae:.2f}")
        print(f"平均 RMSE: {final_rmse:.2f}")
    else:
        print("没有成功评估任何测试样本!")


if __name__ == "__main__":
    main() 