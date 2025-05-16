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


# Wrap dataset to handle different output formats
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


# Reservoir sampling update function
def reservoir_update(buffer, sample, counter, max_size):
    """
    Update memory buffer using reservoir sampling algorithm
    
    Args:
        buffer: Memory buffer list
        sample: Sample to add
        counter: Sample counter
        max_size: Maximum buffer size
    
    Returns:
        counter: Updated counter
    """
    counter += 1
    if len(buffer) < max_size:
        buffer.append(sample)
    else:
        # Use reservoir sampling to decide whether to replace
        if random.random() < (max_size / counter):
            replace_idx = random.randint(0, max_size - 1)
            buffer[replace_idx] = sample
    return counter


# Sample from memory buffer
def sample_from_memory(memory_buffer, replay_batch_size, custom_probs, device):
    """
    Sample replay_batch_size samples from memory buffer
    
    Args:
        memory_buffer: Memory buffer
        replay_batch_size: Number of samples to draw
        custom_probs: Custom sampling probabilities, if None use uniform sampling
        device: Device (CPU/GPU)
    
    Returns:
        Sampled images, density maps and metadata
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

    # Process meta data
    try:
        if isinstance(sampled_meta_list[0], dict) and 'count' in sampled_meta_list[0]:
            # If meta is a dictionary with count key
            count_values = []
            for meta in sampled_meta_list:
                if isinstance(meta['count'], torch.Tensor):
                    # If tensor, ensure it's a scalar
                    if meta['count'].numel() == 1:
                        count_values.append(meta['count'].item())
                    else:
                        count_values.append(meta['count'][0].item() if meta['count'].numel() > 0 else 0)
                else:
                    # If scalar value
                    count_values.append(meta['count'])
            sampled_meta = {'count': torch.tensor(count_values).to(device)}
        elif isinstance(sampled_meta_list[0], torch.Tensor):
            # If meta is a tensor
            meta_tensors = []
            for meta in sampled_meta_list:
                if meta.numel() == 1:
                    meta_tensors.append(meta.item())
                else:
                    meta_tensors.append(meta[0].item() if meta.numel() > 0 else 0)
            sampled_meta = {'count': torch.tensor(meta_tensors).to(device)}
        else:
            # Other types, try to convert to tensor
            sampled_meta = {'count': torch.tensor([float(m) for m in sampled_meta_list]).to(device)}
    except (TypeError, ValueError) as e:
        print(f"Warning: Error processing meta data: {e}")
        print(f"Meta data type: {type(sampled_meta_list[0])}")
        # Provide default value
        sampled_meta = {'count': torch.zeros(len(sampled_meta_list)).to(device)}

    return sampled_imgs, sampled_gt_dens, sampled_meta


# Create source domain (large dataset) and target domain (small dataset)
def create_source_target_datasets(cfg):
    """
    Create source domain (pretraining dataset) and target domain (few-shot dataset)
    """
    # Source domain datasets (for pretraining)
    source_train = []
    source_val = []
    
    # Target domain datasets (for few-shot learning)
    target_train = []
    target_val = []
    target_test = []
    
    # Load source domain datasets
    source_datasets = cfg['source_dataset']
    for i in range(len(source_datasets)):
        # Training set
        train_dataset, collate = get_dataset(
            source_datasets[i]['name'], 
            source_datasets[i]['params'], 
            method='train'
        )
        
        # Validation set
        val_dataset, _ = get_dataset(
            source_datasets[i]['name'], 
            source_datasets[i]['params'], 
            method='val'
        )
        
        # Wrap as IndexedDataset
        indexed_train = IndexedDataset(train_dataset)
        indexed_val = IndexedDataset(val_dataset)
        
        source_train.append((indexed_train, collate))
        source_val.append((indexed_val, collate))
    
    # Load target domain datasets
    target_datasets = cfg['target_dataset']
    for i in range(len(target_datasets)):
        # Training set (few samples)
        train_dataset, collate = get_dataset(
            target_datasets[i]['name'], 
            target_datasets[i]['params'], 
            method='train'
        )
        
        # Validation set
        val_dataset, _ = get_dataset(
            target_datasets[i]['name'], 
            target_datasets[i]['params'], 
            method='val'
        )
        
        # Test set
        test_dataset, _ = get_dataset(
            target_datasets[i]['name'], 
            target_datasets[i]['params'], 
            method='test'
        )
        
        # Wrap as IndexedDataset
        indexed_train = IndexedDataset(train_dataset)
        indexed_val = IndexedDataset(val_dataset)
        indexed_test = IndexedDataset(test_dataset)
        
        target_train.append((indexed_train, collate))
        target_val.append((indexed_val, collate))
        target_test.append((indexed_test, collate))
    
    return source_train, source_val, target_train, target_val, target_test


# Select a small subset of samples for Few-shot learning
def create_few_shot_subset(dataset, n_samples, seed=42):
    """
    Randomly select n_samples from dataset for Few-shot learning
    
    Args:
        dataset: Original dataset
        n_samples: Number of samples to select
        seed: Random seed for reproducibility
    
    Returns:
        subset: Subset of dataset
    """
    # Set random seed
    random.seed(seed)
    
    # Get dataset size
    dataset_size = len(dataset)
    
    # Ensure n_samples doesn't exceed dataset size
    n_samples = min(n_samples, dataset_size)
    
    # Randomly select n_samples indices
    indices = random.sample(range(dataset_size), n_samples)
    
    # Create subset
    subset = Subset(dataset, indices)
    
    return subset, indices


# Evaluate model performance
def evaluate_model(model, dataset, device, trainer):
    """
    Evaluate model performance on given dataset
    
    Args:
        model: Model
        dataset: Dataset
        device: Device (CPU/GPU)
        trainer: DGTrainer instance
    
    Returns:
        metrics: Dictionary containing MAE and RMSE
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    mae_sum = 0
    mse_sum = 0
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                # Evaluate single sample
                mae, metrics = trainer.val_step(model, batch)
                mae_sum += mae
                mse_sum += metrics['mse']
                sample_count += 1
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue
    
    if sample_count == 0:
        return {'mae': float('inf'), 'rmse': float('inf')}
    
    # Calculate average MAE and RMSE
    avg_mae = mae_sum / sample_count
    avg_rmse = np.sqrt(mse_sum / sample_count)
    
    return {'mae': avg_mae, 'rmse': avg_rmse}


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
        raise ValueError(f'Unknown dataset: {name}')
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
        raise ValueError(f'Unknown model: {name}')


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
        raise ValueError(f'Unknown loss function: {loss.__class__.__name__}')
        
    return loss_value 


# Train source model (pretraining phase)
def train_source_model(cfg, args, device, model, source_train, source_val, memory_buffer=None):
    """
    Train model on source domain dataset (pretraining phase) and store samples in memory buffer
    
    Args:
        cfg: Configuration file
        args: Command line arguments
        device: Device (CPU/GPU)
        model: Model
        source_train: Source domain training dataset
        source_val: Source domain validation dataset
        memory_buffer: Optional, memory buffer to store source domain samples
    
    Returns:
        model: Pretrained model
        memory_buffer: Updated memory buffer
    """
    print("\n=== Pretraining Phase: Training Model on Source Domain Dataset ===")
    
    # Initialize memory counter
    memory_counter = 0
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.source_lr)
    
    # Create loss function
    criterion = get_loss()
    
    # Create DGTrainer instance for evaluation
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
    
    # Combine all source domain datasets
    dataset, collate_fn = source_train[0]  # Assume same collate_fn
    train_loader = DataLoader(dataset, collate_fn=collate_fn, **cfg['source_loader'], worker_init_fn=seed_worker)
    
    # Training loop
    best_model = None
    best_loss = float('inf')
    
    for epoch in range(args.source_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Pretraining Epoch {epoch+1}/{args.source_epochs}"):
            try:
                images, imgs2, gt_datas = batch
                images = images.to(device)
                imgs2 = imgs2.to(device)
                gt_cmaps = gt_datas[-1].to(device)
                
                # Save samples to memory buffer (if provided)
                if memory_buffer is not None and args.memory_size > 0:
                    for i in range(images.size(0)):
                        sample_img = images[i].cpu()
                        sample_den = gt_cmaps[i].cpu()
                        
                        # Extract count information from gt_datas
                        if isinstance(gt_datas, tuple) and len(gt_datas) >= 3:
                            count = None
                            if len(gt_datas) >= 4 and isinstance(gt_datas[3], (torch.Tensor, int, float)):
                                count = gt_datas[3][i].item() if isinstance(gt_datas[3], torch.Tensor) else gt_datas[3]
                            else:
                                try:
                                    dmaps = gt_datas[1]
                                    if isinstance(dmaps, torch.Tensor):
                                        count = dmaps[i].sum().item()
                                except:
                                    count = 0
                        else:
                            count = 0
                        
                        sample_meta = {'count': count, 'domain': 'source'}
                        
                        # Update memory buffer using reservoir sampling
                        memory_counter = reservoir_update(
                            memory_buffer, 
                            (sample_img, sample_den, sample_meta),
                            memory_counter, 
                            args.memory_size
                        )
                
                optimizer.zero_grad()
                
                # Forward pass
                dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = model.forward_train(images, imgs2, gt_cmaps)
                
                # Calculate loss
                loss_den = compute_count_loss(criterion, dmaps1, gt_datas, device=device, log_para=log_para) + \
                           compute_count_loss(criterion, dmaps2, gt_datas, device=device, log_para=log_para)
                loss_cls = nn.functional.binary_cross_entropy(cmaps1, gt_cmaps) + nn.functional.binary_cross_entropy(cmaps2, gt_cmaps)
                loss_total = loss_den + 10 * loss_cls + 10 * loss_con
                
                # Backward pass
                loss_total.backward()
                optimizer.step()
                
                epoch_loss += loss_total.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error during batch processing: {e}")
                if "CUDA" in str(e):
                    print("CUDA error, attempting to clear memory and continue")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
        
        # Calculate average loss
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"  Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")
        
        # Evaluate on validation set
        if epoch % args.eval_interval == 0:
            val_metrics = evaluate_model(model, source_val[0][0], device, trainer)
            print(f"  Validation MAE: {val_metrics['mae']:.2f}, RMSE: {val_metrics['rmse']:.2f}")
            
            # Save best model
            if val_metrics['mae'] < best_loss:
                best_loss = val_metrics['mae']
                best_model = copy.deepcopy(model.state_dict())
                if args.save_model:
                    os.makedirs(args.model_dir, exist_ok=True)
                    torch.save(best_model, f"{args.model_dir}/source_best.pth")
                    print(f"  Saved best source domain model, MAE: {val_metrics['mae']:.2f}")
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Output memory buffer information
    if memory_buffer is not None:
        print(f"Source domain pretraining completed, memory buffer contains {len(memory_buffer)} samples")
    
    return model, memory_buffer


# Traditional Few-shot learning: Fine-tune model on target domain
def finetune_target_model(cfg, args, device, model, target_train, target_val, target_test):
    """
    Fine-tune model on target domain dataset (standard Few-shot learning phase, without memory replay)
    
    Args:
        cfg: Configuration file
        args: Command line arguments
        device: Device (CPU/GPU)
        model: Pretrained model
        target_train: Target domain training dataset (few samples)
        target_val: Target domain validation dataset
        target_test: Target domain test dataset
    
    Returns:
        model: Fine-tuned model
        metrics: Evaluation metrics
    """
    print("\n=== Few-shot Learning Phase: Fine-tuning Model on Target Domain Dataset (Standard Version) ===")
    
    # Create new model for target domain fine-tuning
    if args.use_new_model:
        model_params = cfg['model']['params'] if 'params' in cfg['model'] else {}
        new_model = get_model(cfg['model']['name'], model_params)
        new_model.load_state_dict(model.state_dict())
        model = new_model.to(device)
    
    # Freeze specific layers (optional, depends on args.freeze_layers)
    if args.freeze_backbone:
        # Freeze encoder parts (VGG16 feature extractor)
        for param in model.enc1.parameters():
            param.requires_grad = False
        for param in model.enc2.parameters():
            param.requires_grad = False
        for param in model.enc3.parameters():
            param.requires_grad = False
        print("  Backbone network parameters frozen")
    
    # Create optimizer (only optimize unfrozen parameters)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.target_lr)
    
    # Create loss function
    criterion = get_loss()
    
    # Create DGTrainer instance for evaluation
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
    
    # Create few-shot subset from target domain training set
    dataset, collate_fn = target_train[0]
    few_shot_dataset, few_shot_indices = create_few_shot_subset(dataset, args.n_shots, seed=cfg['seed'])
    
    print(f"  Few-shot learning: Using {args.n_shots} samples (out of {len(dataset)} total samples)")
    
    # Create Few-shot data loader
    few_shot_loader = DataLoader(few_shot_dataset, collate_fn=collate_fn, **cfg['target_loader'], worker_init_fn=seed_worker)
    
    # Fine-tuning loop
    best_model = None
    best_val_mae = float('inf')
    
    for epoch in range(args.target_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(few_shot_loader, desc=f"Fine-tuning Epoch {epoch+1}/{args.target_epochs}"):
            try:
                images, imgs2, gt_datas = batch
                images = images.to(device)
                imgs2 = imgs2.to(device)
                gt_cmaps = gt_datas[-1].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = model.forward_train(images, imgs2, gt_cmaps)
                
                # Calculate loss
                loss_den = compute_count_loss(criterion, dmaps1, gt_datas, device=device, log_para=log_para) + \
                           compute_count_loss(criterion, dmaps2, gt_datas, device=device, log_para=log_para)
                loss_cls = nn.functional.binary_cross_entropy(cmaps1, gt_cmaps) + nn.functional.binary_cross_entropy(cmaps2, gt_cmaps)
                loss_total = loss_den + 10 * loss_cls + 10 * loss_con
                
                # Backward pass
                loss_total.backward()
                optimizer.step()
                
                epoch_loss += loss_total.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error during batch processing: {e}")
                if "CUDA" in str(e):
                    print("CUDA error, attempting to clear memory and continue")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
        
        # Calculate average loss
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"  Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")
        
        # Evaluate on validation set
        if epoch % args.eval_interval == 0:
            val_metrics = evaluate_model(model, target_val[0][0], device, trainer)
            print(f"  Validation MAE: {val_metrics['mae']:.2f}, RMSE: {val_metrics['rmse']:.2f}")
            
            # Save best model
            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                best_model = copy.deepcopy(model.state_dict())
                if args.save_model:
                    os.makedirs(args.model_dir, exist_ok=True)
                    torch.save(best_model, f"{args.model_dir}/target_best.pth")
                    print(f"  Saved best target domain model, MAE: {val_metrics['mae']:.2f}")
    
    # Load best model for testing
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, target_test[0][0], device, trainer)
    print(f"\nFinal Test Results - MAE: {test_metrics['mae']:.2f}, RMSE: {test_metrics['rmse']:.2f}")
    
    return model, test_metrics


# Continual Learning with Memory Replay: Fine-tune model on target domain
def finetune_target_model_with_replay(cfg, args, device, model, target_train, target_val, target_test, memory_buffer):
    """
    Fine-tune model on target domain dataset with memory replay to prevent catastrophic forgetting
    
    Args:
        cfg: Configuration file
        args: Command line arguments
        device: Device (CPU/GPU)
        model: Pretrained model
        target_train: Target domain training dataset (few samples)
        target_val: Target domain validation dataset
        target_test: Target domain test dataset
        memory_buffer: Memory buffer containing source domain samples
    
    Returns:
        model: Fine-tuned model
        metrics: Evaluation metrics
        memory_buffer: Updated memory buffer
    """
    print("\n=== Few-shot Learning Phase: Fine-tuning Model on Target Domain Dataset (with Memory Replay) ===")
    
    # Create new model for target domain fine-tuning
    if args.use_new_model:
        model_params = cfg['model']['params'] if 'params' in cfg['model'] else {}
        new_model = get_model(cfg['model']['name'], model_params)
        new_model.load_state_dict(model.state_dict())
        model = new_model.to(device)
    
    # Freeze specific layers (optional, depends on args.freeze_layers)
    if args.freeze_backbone:
        # Freeze encoder parts (VGG16 feature extractor)
        for param in model.enc1.parameters():
            param.requires_grad = False
        for param in model.enc2.parameters():
            param.requires_grad = False
        for param in model.enc3.parameters():
            param.requires_grad = False
        print("  Backbone network parameters frozen")
    
    # Create optimizer (only optimize unfrozen parameters)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.target_lr)
    
    # Create loss function
    criterion = get_loss()
    
    # Create DGTrainer instance for evaluation
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
    
    # Create few-shot subset from target domain training set
    dataset, collate_fn = target_train[0]
    few_shot_dataset, few_shot_indices = create_few_shot_subset(dataset, args.n_shots, seed=cfg['seed'])
    
    print(f"  Few-shot learning: Using {args.n_shots} samples (out of {len(dataset)} total samples)")
    print(f"  Memory buffer size: {len(memory_buffer)} samples")
    
    # Create Few-shot data loader
    few_shot_loader = DataLoader(few_shot_dataset, collate_fn=collate_fn, **cfg['target_loader'], worker_init_fn=seed_worker)
    
    # Fine-tuning loop
    best_model = None
    best_val_mae = float('inf')
    memory_counter = len(memory_buffer)  # Initialize memory counter
    
    for epoch in range(args.target_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(few_shot_loader, desc=f"Fine-tuning Epoch {epoch+1}/{args.target_epochs}"):
            try:
                images, imgs2, gt_datas = batch
                images = images.to(device)
                imgs2 = imgs2.to(device)
                gt_cmaps = gt_datas[-1].to(device)
                
                # Save target domain samples to memory buffer
                if args.memory_size > 0:
                    for i in range(images.size(0)):
                        sample_img = images[i].cpu()
                        sample_den = gt_cmaps[i].cpu()
                        
                        # Extract count information from gt_datas
                        if isinstance(gt_datas, tuple) and len(gt_datas) >= 3:
                            count = None
                            if len(gt_datas) >= 4 and isinstance(gt_datas[3], (torch.Tensor, int, float)):
                                count = gt_datas[3][i].item() if isinstance(gt_datas[3], torch.Tensor) else gt_datas[3]
                            else:
                                try:
                                    dmaps = gt_datas[1]
                                    if isinstance(dmaps, torch.Tensor):
                                        count = dmaps[i].sum().item()
                                except:
                                    count = 0
                        else:
                            count = 0
                        
                        sample_meta = {'count': count, 'domain': 'target'}
                        
                        # Update memory buffer using reservoir sampling
                        memory_counter = reservoir_update(
                            memory_buffer, 
                            (sample_img, sample_den, sample_meta),
                            memory_counter, 
                            args.memory_size
                        )
                
                optimizer.zero_grad()
                
                # Forward pass (target domain samples)
                dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = model.forward_train(images, imgs2, gt_cmaps)
                
                # Calculate target domain loss
                loss_den = compute_count_loss(criterion, dmaps1, gt_datas, device=device, log_para=log_para) + \
                           compute_count_loss(criterion, dmaps2, gt_datas, device=device, log_para=log_para)
                loss_cls = nn.functional.binary_cross_entropy(cmaps1, gt_cmaps) + nn.functional.binary_cross_entropy(cmaps2, gt_cmaps)
                loss_target = loss_den + 10 * loss_cls + 10 * loss_con
                
                # Memory replay (if enabled)
                loss_replay = 0
                if args.use_memory_replay and len(memory_buffer) > 0:
                    # Sample from memory buffer
                    replay_imgs, replay_dens, replay_meta = sample_from_memory(
                        memory_buffer, 
                        args.replay_batch_size, 
                        None,  # Use uniform sampling
                        device
                    )
                    
                    if replay_imgs.size(0) > 0:
                        # Forward pass (memory samples)
                        replay_dmaps1, replay_dmaps2, replay_cmaps1, replay_cmaps2, _, replay_loss_con, _ = model.forward_train(
                            replay_imgs, replay_imgs, replay_dens
                        )
                        
                        # Calculate replay loss
                        replay_loss_den = compute_count_loss(criterion, replay_dmaps1, (None, replay_dens, None), device=device, log_para=log_para) + \
                                        compute_count_loss(criterion, replay_dmaps2, (None, replay_dens, None), device=device, log_para=log_para)
                        replay_loss_cls = nn.functional.binary_cross_entropy(replay_cmaps1, replay_dens) + \
                                        nn.functional.binary_cross_entropy(replay_cmaps2, replay_dens)
                        loss_replay = replay_loss_den + 10 * replay_loss_cls + 10 * replay_loss_con
                
                # Total loss = target domain loss + replay loss (if enabled)
                loss_total = loss_target
                if args.use_memory_replay:
                    loss_total = loss_target + args.replay_weight * loss_replay
                
                # Backward pass
                loss_total.backward()
                optimizer.step()
                
                epoch_loss += loss_total.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error during batch processing: {e}")
                if "CUDA" in str(e):
                    print("CUDA error, attempting to clear memory and continue")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
        
        # Calculate average loss
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"  Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")
        
        # Evaluate on validation set
        if epoch % args.eval_interval == 0:
            val_metrics = evaluate_model(model, target_val[0][0], device, trainer)
            print(f"  Validation MAE: {val_metrics['mae']:.2f}, RMSE: {val_metrics['rmse']:.2f}")
            
            # Save best model
            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                best_model = copy.deepcopy(model.state_dict())
                if args.save_model:
                    os.makedirs(args.model_dir, exist_ok=True)
                    torch.save(best_model, f"{args.model_dir}/target_best_with_replay.pth")
                    print(f"  Saved best target domain model (with memory replay), MAE: {val_metrics['mae']:.2f}")
    
    # Load best model for testing
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, target_test[0][0], device, trainer)
    print(f"\nFinal Test Results - MAE: {test_metrics['mae']:.2f}, RMSE: {test_metrics['rmse']:.2f}")
    
    return model, test_metrics, memory_buffer


# Main function
def main():
    parser = argparse.ArgumentParser(description="Few-shot Learning for Crowd Counting with Continual Learning")
    
    # Basic parameters
    parser.add_argument("--config", type=str, default="configs/fewshot_cl_config.yml", 
                       help="Configuration file path")
    parser.add_argument("--model_type", type=str, default="final", 
                       choices=["base", "mem", "final"], help="Model type")
    parser.add_argument("--model_dir", type=str, default="saved_models",
                       help="Model save directory")
    parser.add_argument("--save_model", action="store_true",
                       help="Whether to save model")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Source domain training parameters (pretraining phase)
    parser.add_argument("--pretrained", action="store_true",
                       help="Whether to use ImageNet pretrained weights")
    parser.add_argument("--load_pretrained", type=str, default=None,
                       help="Path to load pretrained model")
    
    # Target domain fine-tuning parameters (Few-shot learning phase)
    parser.add_argument("--n_shots", type=int, default=10,
                       help="Number of samples per class (Few-shot learning)")
    parser.add_argument("--freeze_backbone", action="store_true",
                       help="Whether to freeze backbone network")
    parser.add_argument("--use_new_model", action="store_true",
                       help="Whether to create new model for fine-tuning")
    
    # Continual learning parameters
    parser.add_argument("--use_memory_replay", action="store_true",
                       help="Whether to use memory replay")
    parser.add_argument("--weighted_replay", action="store_true",
                       help="Whether to use weighted memory replay")
    
    # Other parameters
    parser.add_argument("--safe_mode", action="store_true",
                       help="Enable safe mode, use smaller batch size and more conservative memory settings")
    parser.add_argument("--use_clearml", action="store_true",
                       help="Whether to use ClearML for experiment tracking")
    parser.add_argument("--clearml_project", type=str, default="MPCount",
                       help="ClearML project name")
    parser.add_argument("--clearml_task", type=str, default="FewShotCL",
                       help="ClearML task name")
    
    args = parser.parse_args()
    
    # Load configuration file
    cfg = load_config(args.config)
    
    # Update parameters from configuration file
    args.source_epochs = int(cfg.get('source_epochs', 50))
    args.target_epochs = int(cfg.get('target_epochs', 20))
    args.eval_interval = int(cfg.get('eval_interval', 1))
    args.source_lr = float(cfg.get('source_lr', 1e-4))
    args.target_lr = float(cfg.get('target_lr', 5e-5))
    args.memory_size = int(cfg.get('memory_size', 200))
    args.replay_batch_size = int(cfg.get('replay_batch_size', 16))
    args.replay_weight = float(cfg.get('replay_weight', 0.5))
    
    # Set random seed
    seed_everything(args.seed)
    
    # Create model save directory
    if args.save_model and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    # Safe mode settings
    if args.safe_mode:
        print("Enabling safe mode, reducing batch size")
        # Reduce batch_size to avoid memory issues
        if 'source_loader' in cfg and 'batch_size' in cfg['source_loader']:
            cfg['source_loader']['batch_size'] = min(4, cfg['source_loader']['batch_size'])
        if 'target_loader' in cfg and 'batch_size' in cfg['target_loader']:
            cfg['target_loader']['batch_size'] = min(4, cfg['target_loader']['batch_size'])
        # Disable pin_memory
        if 'source_loader' in cfg:
            cfg['source_loader']['pin_memory'] = False
        if 'target_loader' in cfg:
            cfg['target_loader']['pin_memory'] = False
        # Reduce number of workers
        if 'source_loader' in cfg and 'num_workers' in cfg['source_loader']:
            cfg['source_loader']['num_workers'] = min(2, cfg['source_loader']['num_workers'])
        if 'target_loader' in cfg and 'num_workers' in cfg['target_loader']:
            cfg['target_loader']['num_workers'] = min(2, cfg['target_loader']['num_workers'])
    
    # Use GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize ClearML (if enabled)
    clearml_logger = None
    if args.use_clearml:
        task_name = f"{args.clearml_task}_{args.model_type}_{args.n_shots}shots"
        clearml_logger = CustomClearML(args.clearml_project, task_name)
    
    # Create datasets
    source_train, source_val, target_train, target_val, target_test = create_source_target_datasets(cfg)
    
    # Create model
    model_params = cfg['model']['params'] if 'params' in cfg['model'] else {}
    model_params['pretrained'] = args.pretrained  # Whether to use ImageNet pretrained weights
    
    if args.model_type == 'base':
        model = DGModel_base(**model_params)
    elif args.model_type == 'mem':
        model = DGModel_mem(**model_params)
    elif args.model_type == 'final':
        model = DGModel_final(**model_params)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    
    # Initialize memory buffer (if enabled)
    memory_buffer = [] if args.use_memory_replay else None
    
    # Load pretrained model (if specified)
    if args.load_pretrained:
        print(f"Loading pretrained model: {args.load_pretrained}")
        model.load_state_dict(torch.load(args.load_pretrained, map_location=device))
    
    # Source domain training (pretraining phase)
    if not args.load_pretrained:
        model, memory_buffer = train_source_model(cfg, args, device, model, source_train, source_val, memory_buffer)
    
    # Target domain fine-tuning (Few-shot learning phase)
    if args.use_memory_replay:
        # Use continual learning strategy (memory replay)
        model, metrics, memory_buffer = finetune_target_model_with_replay(
            cfg, args, device, model, target_train, target_val, target_test, memory_buffer
        )
    else:
        # Use standard Few-shot learning strategy (without memory replay)
        model, metrics = finetune_target_model(cfg, args, device, model, target_train, target_val, target_test)
    
    # Output final results
    print("\n=== Experimental Results Summary ===")
    print(f"Few-shot setting: {args.n_shots} samples")
    print(f"Memory buffer: {'Enabled' if args.use_memory_replay else 'Disabled'}")
    if args.use_memory_replay:
        print(f"Memory buffer size: {len(memory_buffer)}")
        print(f"Replay batch size: {args.replay_batch_size}")
    print(f"Final MAE: {metrics['mae']:.2f}")
    print(f"Final RMSE: {metrics['rmse']:.2f}")
    
    # Record experimental results (if ClearML enabled)
    if clearml_logger:
        clearml_logger.report_scalar(
            title="Few-shot Results",
            series=f"{args.n_shots}_shots{'_with_replay' if args.use_memory_replay else ''}",
            value=metrics['mae'],
            iteration=0
        )
        clearml_logger.report_scalar(
            title="Few-shot Results",
            series=f"{args.n_shots}_shots{'_with_replay' if args.use_memory_replay else ''}_rmse",
            value=metrics['rmse'],
            iteration=0
        )


if __name__ == "__main__":
    main() 