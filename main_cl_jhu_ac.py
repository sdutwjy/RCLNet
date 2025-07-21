import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import time
import copy
import random
import numpy as np
import yaml
import torch.nn.functional as F
from os.path import join
import gc  # Add garbage collection module

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from clearml import Task

from datasets.den_dataset import DensityMapDataset
from datasets.den_cls_dataset import DenClsDataset
from datasets.jhu_domain_dataset import JHUDomainDataset
from datasets.jhu_domain_cls_dataset import JHUDomainClsDataset

from models.models import DGModel_mem, DGModel_final, DGModel_base
from models.actor_critic import ActorCriticMemorySampler
from trainers.dgtrainer import DGTrainer
from tqdm import tqdm
from utils.misc import seed_worker, get_seeded_generator, seed_everything
import torchvision.transforms as transforms
from PIL import Image
import io
import traceback

# Custom ClearML Logger
class CustomClearML():
    def __init__(self, project_name, task_name):
        self.task = Task.init(project_name, task_name)
        self.logger = self.task.get_logger()

    def __call__(self, title, series, value, iteration):
        self.logger.report_scalar(title, series, value, iteration)
        
    def report_scalar(self, title, series, value, iteration):
        self.logger.report_scalar(title, series, value, iteration)

# IndexedDataset wrapper
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

# Create Task, each dataset is considered as a Task
def create_crowd_counting_tasks(cfg, split='train'):
    """
    Create crowd counting Tasks, each dataset as a Task
    cfg: configuration file
    """
    tasks_train = []
    tasks_val = []
    tasks_test = []
    
    # Get dataset configuration
    train_datasets = cfg['train_dataset']
    val_datasets = cfg['val_dataset']
    test_datasets = cfg['test_dataset']
    
    for i in range(len(train_datasets)):
        # Use get_dataset function to load dataset
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
        
        # Wrap as IndexedDataset
        indexed_train = IndexedDataset(train_dataset)
        indexed_val = IndexedDataset(val_dataset)
        indexed_test = IndexedDataset(test_dataset)
        
        tasks_train.append((indexed_train, collate))
        tasks_val.append((indexed_val, collate))
        tasks_test.append((indexed_test, collate))
    
    return tasks_train, tasks_val, tasks_test

# Modified memory buffer update function
def reservoir_update(buffer, sample, counter, max_size, resize=True, target_size=(128, 128)):
    """
    Update memory buffer using reservoir sampling, with option to resize images to reduce memory usage
    
    Args:
        buffer: Memory buffer
        sample: Sample (image, density map, metadata)
        counter: Counter
        max_size: Maximum buffer size
        resize: Whether to resize images
        target_size: Target image size
        
    Returns:
        Updated counter
    """
    counter += 1
    
    # Process sample, optionally resize images
    try:
        if resize:
            img, den, meta = sample
            
            # Check if image and density map are tensors
            if not isinstance(img, torch.Tensor) or not isinstance(den, torch.Tensor):
                print("Warning: Image or density map is not a tensor, skipping resize")
                return counter
            
            # Get original dimensions
            orig_img_size = img.shape
            orig_den_size = den.shape
            
            # Resize image
            if len(img.shape) == 3:  # [C, H, W]
                img_small = F.interpolate(img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            else:
                print(f"Warning: Image shape is abnormal {img.shape}, skipping resize")
                return counter
            
            # Resize density map
            if len(den.shape) == 2:  # [H, W]
                den_small = F.interpolate(den.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            elif len(den.shape) == 3 and den.shape[0] == 1:  # [1, H, W]
                den_small = F.interpolate(den.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            else:
                print(f"Warning: Density map shape is abnormal {den.shape}, skipping resize")
                return counter
            
            # Keep the sum of density map unchanged (maintain people count)
            if den.sum() > 0:
                scaling_factor = den.sum() / den_small.sum() if den_small.sum() > 0 else 1.0
                den_small = den_small * scaling_factor
            
            sample = (img_small, den_small, meta)
        
        # Update buffer
        if len(buffer) < max_size:
            buffer.append(sample)
        else:
            if random.random() < (max_size / counter):
                replace_idx = random.randint(0, max_size - 1)
                buffer[replace_idx] = sample
                
        return counter
    except Exception as e:
        print(f"Error updating memory buffer: {e}")
        # Don't update buffer on error, but still increment counter
        return counter

# Sample from multiple dataset buffers
def sample_from_dataset_buffers(dataset_buffers, replay_batch_size, device, max_batch_size=5):
    """
    Sample from buffers of multiple datasets
    
    Args:
        dataset_buffers: List of dataset buffers
        replay_batch_size: Number of samples to sample
        device: Device
        max_batch_size: Maximum batch size to process at once
    
    Returns:
        sampled_imgs: Sampled images
        sampled_gt_dens: Sampled density maps
        sampled_meta: Sampled metadata
    """
    if replay_batch_size <= 0:
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device), {}
    
    # Count samples in each buffer
    buffer_sizes = [len(buffer) for buffer in dataset_buffers]
    total_samples = sum(buffer_sizes)
    
    if total_samples == 0:
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device), {}
    
    # Extract all samples and their values
    all_samples = []
    dataset_indices = []  # Track which dataset each sample comes from
    
    for dataset_id, buffer in enumerate(dataset_buffers):
        for sample, _ in buffer:
            all_samples.append(sample)
            dataset_indices.append(dataset_id)
    
    # 使用均匀分布进行采样
    sampling_probs = np.ones(len(all_samples)) / len(all_samples)
    
    # Sample based on probabilities
    try:
        chosen_indices = np.random.choice(
            len(all_samples),
            size=min(replay_batch_size, len(all_samples)),
            replace=(replay_batch_size > len(all_samples)),
            p=sampling_probs
        )
    except ValueError:
        # If sampling fails, use uniform sampling
        print("Sampling failed, using uniform sampling")
        chosen_indices = np.random.choice(
            len(all_samples),
            size=min(replay_batch_size, len(all_samples)),
            replace=(replay_batch_size > len(all_samples))
        )
    
    # Process sampled results in batches to avoid loading too much data to GPU at once
    sampled_imgs_list = []
    sampled_gt_dens_list = []
    sampled_meta_list = []
    
    # Get size information from first sample to ensure all samples have same size
    try:
        first_sample = all_samples[chosen_indices[0]]
        first_img, first_gt_den, _ = first_sample
        target_img_size = first_img.shape
        target_den_size = first_gt_den.shape
    except Exception as e:
        print(f"Error getting size information from first sample: {e}")
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device), {}
    
    # Process sampled results in batches to avoid loading too much data to GPU at once
    for i in range(0, len(chosen_indices), max_batch_size):
        batch_indices = chosen_indices[i:min(i+max_batch_size, len(chosen_indices))]
        batch_imgs = []
        batch_gt_dens = []
        batch_meta = []
        
        for idx in batch_indices:
            try:
                img, gt_den, meta = all_samples[idx]
                
                # Ensure all images have the same size
                if img.shape != target_img_size:
                    if len(img.shape) == 3:  # [C, H, W]
                        img = F.interpolate(img.unsqueeze(0), size=target_img_size[1:], mode='bilinear', align_corners=False).squeeze(0)
                    else:
                        continue
                
                # Ensure all density maps have the same size
                if gt_den.shape != target_den_size:
                    if len(gt_den.shape) == 2:  # [H, W]
                        gt_den = F.interpolate(gt_den.unsqueeze(0).unsqueeze(0), size=target_den_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    elif len(gt_den.shape) == 3 and gt_den.shape[0] == 1:  # [1, H, W]
                        gt_den = F.interpolate(gt_den.unsqueeze(0), size=target_den_size[1:], mode='bilinear', align_corners=False).squeeze(0)
                    else:
                        continue
                    
                    # Keep the sum of density map unchanged (maintain people count)
                    original_sum = gt_den.sum().item()
                    if original_sum > 0:
                        gt_den = gt_den * (original_sum / gt_den.sum().item())
                
                batch_imgs.append(img.unsqueeze(0))
                batch_gt_dens.append(gt_den.unsqueeze(0))
                batch_meta.append(meta)
            except Exception:
                continue
        
        # Process current batch
        if batch_imgs:
            try:
                batch_imgs_tensor = torch.cat(batch_imgs, dim=0).to(device)
                batch_gt_dens_tensor = torch.cat(batch_gt_dens, dim=0).to(device)
                
                sampled_imgs_list.append(batch_imgs_tensor)
                sampled_gt_dens_list.append(batch_gt_dens_tensor)
                sampled_meta_list.extend(batch_meta)
            except RuntimeError:
                continue
            
            del batch_imgs, batch_gt_dens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Merge results from all batches
    if not sampled_imgs_list:
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device), {}
    
    try:
        # Check shapes of all batches
        img_batch_shapes = [imgs.shape for imgs in sampled_imgs_list]
        den_batch_shapes = [dens.shape for dens in sampled_gt_dens_list]
        
        if len(set(str(s[1:]) for s in img_batch_shapes)) > 1:
            first_batch_shape = sampled_imgs_list[0].shape[1:]
            for i in range(1, len(sampled_imgs_list)):
                if sampled_imgs_list[i].shape[1:] != first_batch_shape:
                    sampled_imgs_list[i] = F.interpolate(sampled_imgs_list[i], size=first_batch_shape[1:], mode='bilinear', align_corners=False)
        
        if len(set(str(s[1:]) for s in den_batch_shapes)) > 1:
            first_batch_shape = sampled_gt_dens_list[0].shape[1:]
            for i in range(1, len(sampled_gt_dens_list)):
                if sampled_gt_dens_list[i].shape[1:] != first_batch_shape:
                    sampled_gt_dens_list[i] = F.interpolate(sampled_gt_dens_list[i], size=first_batch_shape[1:], mode='bilinear', align_corners=False)
        
        sampled_imgs = torch.cat(sampled_imgs_list, dim=0)
        sampled_gt_dens = torch.cat(sampled_gt_dens_list, dim=0)
    except RuntimeError:
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device), {}
    
    # Process meta data
    try:
        if isinstance(sampled_meta_list[0], dict) and 'count' in sampled_meta_list[0]:
            count_values = []
            for meta in sampled_meta_list:
                if isinstance(meta['count'], torch.Tensor):
                    if meta['count'].numel() == 1:
                        count_values.append(meta['count'].item())
                    else:
                        count_values.append(meta['count'][0].item() if meta['count'].numel() > 0 else 0)
                else:
                    try:
                        count_values.append(float(meta['count']) if meta['count'] is not None else 0.0)
                    except (TypeError, ValueError):
                        count_values.append(0.0)
            sampled_meta = {'count': torch.tensor(count_values).to(device)}
        elif isinstance(sampled_meta_list[0], torch.Tensor):
            meta_tensors = []
            for meta in sampled_meta_list:
                if meta.numel() == 1:
                    meta_tensors.append(meta.item())
                else:
                    meta_tensors.append(meta[0].item() if meta.numel() > 0 else 0)
            sampled_meta = {'count': torch.tensor(meta_tensors).to(device)}
        else:
            try:
                sampled_meta = {'count': torch.tensor([float(m) if m is not None else 0.0 for m in sampled_meta_list]).to(device)}
            except (TypeError, ValueError):
                sampled_meta = {'count': torch.zeros(len(sampled_meta_list)).to(device)}
    except (TypeError, ValueError):
        sampled_meta = {'count': torch.zeros(len(sampled_meta_list)).to(device)}
    
    # Clean up memory
    del sampled_imgs_list, sampled_gt_dens_list
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return sampled_imgs, sampled_gt_dens, sampled_meta

# Custom collate function, used for handling various data formats
def custom_collate_fn(batch):
    """
    Custom collate function, handling inconsistent data formats
    """
    # Check if batch is empty
    if len(batch) == 0:
        return None
    
    # Return single sample directly, avoiding stack problem
    if len(batch) == 1:
        return batch[0]
    
    # Check element types in batch
    if isinstance(batch[0], tuple):
        # Check if each element in the tuple is of the same type
        lengths = [len(item) for item in batch]
        if not all(length == lengths[0] for length in lengths):
            print(f"Warning: Lengths of tuples in batch are inconsistent: {lengths}")
            # Return first sample
            return batch[0]
        
        # Try transposing batch
        try:
            transposed = list(zip(*batch))
            result = []
            
            for i, items in enumerate(transposed):
                # Check if all items are tensors
                if all(isinstance(item, torch.Tensor) for item in items):
                    try:
                        # Try stacking tensors
                        result.append(torch.stack(items, 0))
                    except:
                        # If stacking fails, keep original
                        result.append(items)
                else:
                    # If not all items are tensors, keep original
                    result.append(items)
            
            return tuple(result)
        except Exception as e:
            print(f"Error transposing batch: {e}")
            # Return first sample on failure
            return batch[0]
    else:
        # If batch elements are not tuples, try stacking directly
        try:
            if all(isinstance(item, torch.Tensor) for item in batch):
                return torch.stack(batch, 0)
            else:
                return batch
        except:
            # Return list on failure
            return batch

# Evaluation function, using MAE and MSE
def evaluate_performance(model, dataset, device, collate_fn):
    """
    Evaluate model performance using a method similar to DGTrainer.val_step
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    mae_sum = 0
    mse_sum = 0
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
            try:
                # Try different data format unpacking
                if isinstance(batch, tuple) and len(batch) == 5:
                    img1, img2, gt, dmaps, bmaps = batch
                    
                    # Check data type, ensure it's a tensor
                    if not isinstance(img1, torch.Tensor):
                        continue
                    if not isinstance(gt, torch.Tensor) and not isinstance(gt, list):
                        continue
                        
                    # Ensure img1 is 4D tensor [B, C, H, W]
                    if len(img1.shape) == 3:  # [C, H, W]
                        img1 = img1.unsqueeze(0)  # Add batch dimension, becomes [1, C, H, W]
                    
                    img1 = img1.to(device)
                    
                    # Use model to predict
                    try:
                        if hasattr(model, 'predict'):
                            pred_count = model.predict(img1)
                        else:
                            # If model doesn't have predict method, use forward propagation directly
                            model_output = model(img1)
                            
                            # Handle case where model returns a tuple
                            if isinstance(model_output, tuple):
                                # Assume first element is density map
                                pred_dmap = model_output[0]
                            else:
                                pred_dmap = model_output
                            
                            pred_count = pred_dmap.sum().item() / 1000  # Divide by log_para (1000)
                    except Exception as e:
                        print(f"Model inference error: {e}")
                        continue
                    
                    # Get true number of people
                    if isinstance(gt, torch.Tensor):
                        if gt.numel() > 0:
                            gt_count = gt.shape[1] if len(gt.shape) > 1 else gt.sum().item()
                        else:
                            gt_count = 0
                    elif isinstance(gt, list) and len(gt) > 0:
                        if isinstance(gt[0], torch.Tensor):
                            gt_count = len(gt)
                        else:
                            gt_count = len(gt)
                    else:
                        # Try getting count from dmaps
                        if isinstance(dmaps, torch.Tensor):
                            gt_count = dmaps.sum().item()
                        else:
                            gt_count = 0
                else:
                    # If data format doesn't match expectations, skip
                    continue
                
                # Calculate MAE and MSE
                batch_mae = np.abs(pred_count - gt_count)
                batch_mse = (pred_count - gt_count) ** 2
                
                mae_sum += batch_mae
                mse_sum += batch_mse
                sample_count += 1
                
            except Exception as e:
                print(f"Error evaluating samples: {e}")
                continue
    
    if sample_count == 0:
        return {'mae': float('inf'), 'rmse': float('inf')}
        
    mae = mae_sum / sample_count
    rmse = np.sqrt(mse_sum / sample_count)
    
    return {'mae': mae, 'rmse': rmse}

# Add memory monitoring function
def print_memory_stats(device, prefix=""):
    """
    Print current memory usage
    
    Args:
        device: Device
        prefix: Prefix for printing
    """
    if torch.cuda.is_available():
        # Get current GPU memory usage
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
        
        # Get total memory
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
        
        print(f"{prefix} Memory statistics (GB): "
              f"Allocated={allocated:.2f}, "
              f"Reserved={reserved:.2f}, "
              f"Peak={max_allocated:.2f}, "
              f"Total={total_memory:.2f}, "
              f"Usage={allocated/total_memory*100:.1f}%")
    
    # Print Python object memory usage
    print(f"{prefix} Number of Python objects: {len(gc.get_objects())}")

# Training loop
def train_and_evaluate_simple(cfg, args, device, master_seed):
    """
    使用简单的随机采样和行为克隆进行训练和评估
    """
    # Set random seed
    random.seed(master_seed)
    torch.manual_seed(master_seed)
    seed_everything(master_seed)
    
    # Add CUDA memory debugging option
    if device.type == 'cuda':
        try:
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(device)}")
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"CUDA memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        except Exception as e:
            print(f"Error getting CUDA info: {e}")

    # Create Task
    datasets_list = args.datasets.split(',')
    tasks_train, tasks_val, tasks_test = create_crowd_counting_tasks(cfg)
    
    # Create model
    model_params = cfg['model']['params'] if 'params' in cfg['model'] else {}
    if args.model_type == 'mem':
        model = DGModel_mem(**model_params).to(device)
    elif args.model_type == 'final':
        model = DGModel_final(**model_params).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Create ClearML Logger (if enabled)
    clearml_logger = None
    if args.use_clearml and Task.current_task():
        clearml_logger = Task.current_task().get_logger()
    
    # Create DGTrainer instance for evaluation
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
    
    # Get optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = get_loss()
    generator = get_seeded_generator(cfg['seed'])
    
    # Memory buffer
    memory_buffer = []
    memory_counter = 0
    
    # Create separate memory buffers for each dataset and set quotas
    dataset_memory_buffers = [[] for _ in range(len(tasks_train))]
    dataset_memory_counters = [0 for _ in range(len(tasks_train))]
    
    # Use memory quotas passed from main function
    memory_quotas = args.memory_quotas if hasattr(args, 'memory_quotas') else [args.memory_size // len(tasks_train)] * len(tasks_train)
    
    # 存储历史模型
    history_models = []
    history_model_class = None
    if args.model_type == 'mem':
        history_model_class = DGModel_mem
    elif args.model_type == 'final':
        history_model_class = DGModel_final
    
    best_perf_each_task = [{'mae': float('inf'), 'rmse': float('inf')} for _ in range(len(tasks_train))]
    final_perf_each_task = [{'mae': float('inf'), 'rmse': float('inf')} for _ in range(len(tasks_train))]
    
    # Create directory to save models
    os.makedirs('checkpoints/kelong', exist_ok=True)
    
    print(f"Number of training epochs for each Task: {args.epochs_per_task}")
    
    # Add memory monitoring function
    def get_gpu_memory_usage():
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        return 0
    
    # Function to adjust memory size based on current memory usage
    def adjust_memory_params(current_memory_usage, max_memory=0.8):
        """Adjust parameters based on current memory usage"""
        # Get total GPU memory
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
            memory_threshold = total_memory * max_memory  # Use threshold
            
            if current_memory_usage > memory_threshold:
                # If memory usage exceeds threshold, reduce parameters
                return {
                    'replay_batch_size': max(1, args.replay_batch_size // 2),
                    'resize_target': (args.memory_image_size // 2, args.memory_image_size // 2),
                    'max_samples': max(1, args.max_samples_per_forward // 2)
                }
            elif current_memory_usage < memory_threshold * 0.5:
                # If memory usage is much lower than threshold, can increase parameters
                return {
                    'replay_batch_size': min(32, args.replay_batch_size * 2),
                    'resize_target': (args.memory_image_size, args.memory_image_size),
                    'max_samples': min(10, args.max_samples_per_forward * 2)
                }
        
        # Default no adjustment
        return {
            'replay_batch_size': args.replay_batch_size,
            'resize_target': (args.memory_image_size, args.memory_image_size),
            'max_samples': args.max_samples_per_forward
        }
    
    # Initialize dynamic parameters
    dynamic_params = {
        'replay_batch_size': args.replay_batch_size,
        'resize_target': (args.memory_image_size, args.memory_image_size),
        'max_samples': args.max_samples_per_forward
    }
    
    # Print memory status before training starts
    print_memory_stats(device, "Training starts")
    
    # Train for each Task
    for task_id in range(len(tasks_train)):
        print(f"\n=== Train Task {task_id} ({datasets_list[task_id]}) ===")
        print_memory_stats(device, f"Task {task_id} starts")
        
        dataset, collate_fn = tasks_train[task_id]
        task_loader = DataLoader(dataset, collate_fn=collate_fn, **{**cfg['train_loader'], 'pin_memory': False}, worker_init_fn=seed_worker, generator=generator)
        
        # Evaluate performance before training for each task, used for calculating rewards
        if task_id > 0:
            pre_train_perf = {}
            for prev_task_id in range(task_id):
                val_dataset, val_collate_fn = tasks_val[prev_task_id]
                try:
                    pre_train_perf[prev_task_id] = evaluate_performance(model, val_dataset, device, val_collate_fn)
                    print(f"Task {prev_task_id} training performance: MAE={pre_train_perf[prev_task_id]['mae']:.2f}, RMSE={pre_train_perf[prev_task_id]['rmse']:.2f}")
                except Exception as e:
                    print(f"Error evaluating task {prev_task_id}: {e}")
                    pre_train_perf[prev_task_id] = {'mae': float('inf'), 'rmse': float('inf')}
        
        for epoch in range(args.epochs_per_task):
            # Print memory status before each epoch starts
            print_memory_stats(device, f"Task {task_id} Epoch {epoch+1} starts")
            
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # If memory optimization is enabled, check and adjust parameters at the start of each epoch
            if args.memory_optimization and epoch > 0:
                current_memory = get_gpu_memory_usage()
                print(f"Current GPU memory usage: {current_memory:.2f} GB")
                dynamic_params = adjust_memory_params(current_memory)
                print(f"Dynamic adjustment: replay batch size={dynamic_params['replay_batch_size']}, "
                      f"image size={dynamic_params['resize_target']}, "
                      f"max samples={dynamic_params['max_samples']}")
            
            # Initialize gradient accumulation counter
            accumulation_steps = 0
            optimizer.zero_grad()  # Zero gradients at the start of the epoch
            
            for batch_idx, batch in enumerate(tqdm(task_loader, desc=f"Task {task_id} Epoch {epoch+1}")):
                # Save current sample for updating memory buffer
                try:
                    images, imgs2, gt_datas = batch
                    images = images.to(device)
                    imgs2 = imgs2.to(device)
                    gt_cmaps = gt_datas[-1].to(device)
                    
                    # 检查批次大小是否一致，如果不一致则调整
                    if images.shape[0] != gt_cmaps.shape[0]:
                        print(f"Warning: Batch size mismatch - images: {images.shape[0]}, gt_cmaps: {gt_cmaps.shape[0]}")
                        # 使用较小的批次大小
                        min_batch_size = min(images.shape[0], gt_cmaps.shape[0])
                        images = images[:min_batch_size]
                        imgs2 = imgs2[:min_batch_size]
                        gt_cmaps = gt_cmaps[:min_batch_size]
                        print(f"Adjusted batch sizes to {min_batch_size}")
                    
                    new_images = images.clone()
                    new_gt_dens = gt_cmaps.clone()
                    
                    # 确保new_images和new_gt_dens的批次大小一致
                    if new_images.size(0) != new_gt_dens.size(0):
                        min_size = min(new_images.size(0), new_gt_dens.size(0))
                        new_images = new_images[:min_size]
                        new_gt_dens = new_gt_dens[:min_size]
                        print(f"Adjusted memory update batch size to {min_size}")
                    
                    # 不要在每个批次开始时将梯度置零，而是在累积达到一定数量后重置
                    loss = get_loss()
                    dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = model.forward_train(images, imgs2, gt_cmaps)
                    loss_den = compute_count_loss(loss, dmaps1, gt_datas, device=device, log_para=log_para) + compute_count_loss(loss, dmaps2, gt_datas, device=device, log_para=log_para)
                    loss_cls = F.binary_cross_entropy(cmaps1, gt_cmaps) + F.binary_cross_entropy(cmaps2, gt_cmaps)
                    loss_total = loss_den + 10 * loss_cls + 10 * loss_con # + loss_err 
                    
                    # Scale loss to adapt to gradient accumulation
                    loss_total = loss_total / args.gradient_accumulation_steps
                    loss_total.backward()
                    
                    # Accumulate gradients
                    accumulation_steps += 1
                    
                    # When accumulation reaches a certain number, update parameters
                    if accumulation_steps % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        accumulation_steps = 0
                    
                    epoch_loss += loss_total.item() * args.gradient_accumulation_steps  # Restore original loss size for recording
                    num_batches += 1
                    
                    # Memory replay step - use new sampling method based on dataset quotas
                    total_samples = sum(len(buffer) for buffer in dataset_memory_buffers)
                    if total_samples > 0 and dynamic_params['replay_batch_size'] > 0:
                        # 使用随机采样从内存中选择样本
                        replay_images, replay_gt_dens, replay_meta = random_sample_from_buffers(
                            dataset_buffers=dataset_memory_buffers,
                            replay_batch_size=dynamic_params['replay_batch_size'],
                            device=device,
                            max_batch_size=dynamic_params['max_samples']
                        )
                        
                        if replay_images.shape[0] > 0:
                            try:
                                # Ensure current batch and memory sample have the same size
                                current_img_shape = images.shape[2:]  # Get height and width
                                replay_img_shape = replay_images.shape[2:]
                                
                                # If sizes are different, adjust size of memory sample
                                if current_img_shape != replay_img_shape:
                                    replay_images = F.interpolate(replay_images, size=current_img_shape, mode='bilinear', align_corners=False)
                                
                                # Ensure density map also has the same size
                                current_den_shape = gt_cmaps.shape[2:] if len(gt_cmaps.shape) >= 3 else gt_cmaps.shape
                                replay_den_shape = replay_gt_dens.shape[2:] if len(replay_gt_dens.shape) >= 3 else replay_gt_dens.shape
                                
                                # If sizes are different, adjust size of density map of memory sample
                                if current_den_shape != replay_den_shape:
                                    if len(replay_gt_dens.shape) == 4:  # [B, C, H, W]
                                        replay_gt_dens = F.interpolate(replay_gt_dens, size=current_den_shape, mode='bilinear', align_corners=False)
                                    elif len(replay_gt_dens.shape) == 3:  # [B, H, W]
                                        replay_gt_dens = F.interpolate(replay_gt_dens.unsqueeze(1), size=current_den_shape, mode='bilinear', align_corners=False).squeeze(1)
                                
                                # 合并当前批次和回放样本前，先保存当前批次
                                current_images = images.clone()
                                current_gt_cmaps = gt_cmaps.clone()
                                
                                # 确保当前批次的images和gt_cmaps批次大小一致
                                if current_images.shape[0] != current_gt_cmaps.shape[0]:
                                    print(f"Warning: Current batch size mismatch - images: {current_images.shape[0]}, gt_cmaps: {current_gt_cmaps.shape[0]}")
                                    min_batch_size = min(current_images.shape[0], current_gt_cmaps.shape[0])
                                    current_images = current_images[:min_batch_size]
                                    current_gt_cmaps = current_gt_cmaps[:min_batch_size]
                                    print(f"Adjusted current batch sizes to {min_batch_size}")
                                
                                # 合并当前批次和回放样本
                                try:
                                    # 打印详细的形状信息以便调试
                                    print(f"Before merging - current_images: {current_images.shape}, current_gt_cmaps: {current_gt_cmaps.shape}")
                                    print(f"Before merging - replay_images: {replay_images.shape}, replay_gt_dens: {replay_gt_dens.shape}")
                                    
                                    images = torch.cat([current_images.to(device), replay_images], dim=0)
                                    gt_dens = torch.cat([current_gt_cmaps.to(device), replay_gt_dens], dim=0)
                                    
                                    print(f"After merging - images: {images.shape}, gt_dens: {gt_dens.shape}")
                                except RuntimeError as e:
                                    print(f"Error during merging: {e}")
                                    # 出错时使用原始批次
                                    images = current_images.to(device)
                                    gt_dens = current_gt_cmaps.to(device)
                                
                                # 添加行为克隆损失 - 仅对回放样本计算
                                if args.use_behavior_cloning and task_id > 0 and history_models:
                                    behavior_loss = 0.0
                                    
                                    # 对回放样本进行前向传播
                                    try:
                                        # 确保回放样本形状正确
                                        print(f"Behavior cloning - replay_images: {replay_images.shape}, replay_gt_dens: {replay_gt_dens.shape}")
                                        dmaps1_replay, dmaps2_replay, cmaps1_replay, cmaps2_replay, _, _, _ = model.forward_train(replay_images, replay_images, replay_gt_dens)
                                        
                                        # 对每个历史模型计算行为克隆损失
                                        for teacher_model in history_models:
                                            with torch.no_grad():
                                                teacher_model.eval()
                                                teacher_outputs = teacher_model.forward_train(replay_images, replay_images, replay_gt_dens)
                                            
                                            # 计算当前输出与教师输出之间的行为克隆损失 - 只对回放样本
                                            bc_loss = compute_behavior_cloning_loss(
                                                dmaps1_replay, teacher_outputs[0], temperature=args.bc_temperature
                                            )
                                            behavior_loss += bc_loss
                                        
                                        # 如果有多个历史模型，取平均值
                                        if len(history_models) > 0:
                                            behavior_loss /= len(history_models)
                                        
                                        # 添加到总损失中
                                        loss_total += args.bc_lambda * behavior_loss
                                        
                                        if batch_idx % 10 == 0:  # 每10个批次打印一次行为克隆损失
                                            print(f"Behavior cloning loss (replay samples only): {behavior_loss.item():.4f}")
                                    except RuntimeError as e:
                                        print(f"Error in behavior cloning: {e}")
                                        # 出错时不添加行为克隆损失
                                
                            except RuntimeError as e:
                                print(f"Error merging current batch and memory sample: {e}")
                                print(f"Current batch shape: images={images.shape}, gt_cmaps={gt_cmaps.shape}")
                                print(f"Memory sample shape: replay_images={replay_images.shape}, replay_gt_dens={replay_gt_dens.shape}")
                                # Don't merge, just use current batch
                                images = images.to(device)
                                gt_dens = gt_cmaps.to(device)
                        else:
                            images = images.to(device)
                            gt_dens = gt_cmaps.to(device)
                    else:
                        images = images.to(device)
                        gt_dens = gt_cmaps.to(device)
                    
                    # Update memory buffer - use new update function based on gradient contribution
                    for i in range(new_images.size(0)):
                        try:
                            sample_img = new_images[i].cpu()
                            sample_den = new_gt_dens[i].cpu()
                            
                            # 提取计数信息
                            count = 0  # 默认值
                            if isinstance(gt_datas, tuple) and len(gt_datas) >= 3:
                                # 如果gt_datas是元组，尝试获取计数
                                if len(gt_datas) >= 4 and isinstance(gt_datas[3], (torch.Tensor, int, float)):
                                    # 直接使用计数数据
                                    if isinstance(gt_datas[3], torch.Tensor) and i < gt_datas[3].size(0):
                                        count = gt_datas[3][i].item()
                                    elif not isinstance(gt_datas[3], torch.Tensor):
                                        count = gt_datas[3]
                                else:
                                    # 从密度图或其他数据计算计数
                                    try:
                                        dmaps = gt_datas[1]
                                        if isinstance(dmaps, torch.Tensor) and i < dmaps.size(0):
                                            count = dmaps[i].sum().item()
                                    except:
                                        count = 0  # 默认值
                            
                            sample_meta = {'count': count}
                            
                            # 使用简单的随机采样更新内存缓冲区
                            dataset_memory_counters = random_memory_update(
                                dataset_buffers=dataset_memory_buffers, 
                                sample=(sample_img, sample_den, sample_meta),
                                dataset_id=task_id,
                                memory_quotas=memory_quotas,
                                counters=dataset_memory_counters,
                                resize=True,
                                target_size=dynamic_params['resize_target']
                            )
                        except Exception as e:
                            print(f"Error updating memory for sample {i}: {e}")
                            continue
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    import traceback
                    traceback.print_exc()
                    if "CUDA" in str(e):
                        print("CUDA error, trying to clear memory and continue")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    continue
            
            # Ensure gradients from the last batch are applied
            if accumulation_steps > 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Record loss for each epoch
            if clearml_logger:
                clearml_logger.report_scalar(
                    title=f"Task_{task_id}_Loss",
                    series="Train",
                    value=epoch_loss / max(1, num_batches),
                    iteration=epoch
                )
            
            # Print size of each dataset's buffer
            print(f"Epoch {epoch+1} Memory buffer size: " + ", ".join([f"Task {i}: {len(buffer)}/{memory_quotas[i]}" for i, buffer in enumerate(dataset_memory_buffers)]))
            
            # Evaluate performance of current Task
            print(f"Val Task {task_id}...")
            val_dataset, collate_fn = tasks_val[task_id]
            
            # Create valid DataLoader, use collate_fn, disable pin_memory
            val_loader = DataLoader(val_dataset, **{**cfg['val_loader'], 'pin_memory': False})
            
            model.eval()
            mae_sum = 0
            mse_sum = 0
            val_count = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="evaling", leave=False):
                    try:
                        # Use trainer.val_step to evaluate
                        mae, metrics = trainer.val_step(model, batch)
                        mae_sum += mae
                        mse_sum += metrics['mse']
                        val_count += 1
                        
                        if val_count <= 2:  # Only print details of first two samples
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
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'task_id': task_id,
                        'mae': perf_current['mae'],
                        'rmse': perf_current['rmse']
                    }, f'checkpoints/kelong/task_{task_id}_best.pth')
                    print(f"Saved best model for task {task_id}, MAE: {perf_current['mae']:.2f}")
                
                print(f"  Epoch {epoch+1}/{args.epochs_per_task}, Task {task_id}, MAE = {perf_current['mae']:.2f}, RMSE = {perf_current['rmse']:.2f}")
                
                # Record validation metrics
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
        
        # After training for each task, evaluate performance of all tasks
        if task_id > 0:
            # Calculate impact on performance of previous tasks after training current task
            post_train_perf = {}
            for prev_task_id in range(task_id):
                val_dataset, val_collate_fn = tasks_val[prev_task_id]
                try:
                    post_train_perf[prev_task_id] = evaluate_performance(model, val_dataset, device, val_collate_fn)
                    print(f"Task {prev_task_id} performance after training: MAE={post_train_perf[prev_task_id]['mae']:.2f}, RMSE={post_train_perf[prev_task_id]['rmse']:.2f}")
                except Exception as e:
                    print(f"Error evaluating task {prev_task_id} after training: {e}")
                    post_train_perf[prev_task_id] = {'mae': float('inf'), 'rmse': float('inf')}
                    # No reward added on error
        
        # After training for each task, display best performance of that task
        print(f"\n=== Task {task_id} ({datasets_list[task_id]}) training completed ===")
        print(f"Best validation performance - MAE: {best_perf_each_task[task_id]['mae']:.2f}, RMSE: {best_perf_each_task[task_id]['rmse']:.2f}")
        
        # 保存当前任务的模型，用于后续任务的行为克隆
        if args.use_behavior_cloning:
            save_task_model(model, task_id)
            # 加载历史模型
            if task_id > 0:
                history_models = load_task_models(history_model_class, model_params, list(range(task_id)), device=device)
        
        # Print memory status after each task ends
        print_memory_stats(device, f"Task {task_id} ends")
    
    # Finally evaluate all Tasks
    print("\n=== All tasks are finally evaluated ===")
    
    # First display best validation performance for each task
    print("\n=== Best validation performance for each task ===")
    for task_id in range(len(tasks_train)):
        print(f"Task {task_id} ({datasets_list[task_id]}) - Best MAE: {best_perf_each_task[task_id]['mae']:.2f}, Best RMSE: {best_perf_each_task[task_id]['rmse']:.2f}")
    
    for task_id in range(len(tasks_train)):
        print(f"\nval Task {task_id} ({datasets_list[task_id]})")
        
        # Use test set, create valid DataLoader, disable pin_memory
        test_dataset, test_collate_fn = tasks_test[task_id]
        test_loader = DataLoader(test_dataset, **{**cfg['test_loader'], 'pin_memory': False})
        
        model.eval()
        mae_sum = 0
        mse_sum = 0
        test_count = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="evaluating", leave=False):
                try:
                    # Use trainer.test_step to evaluate
                    metrics = trainer.test_step(model, batch)
                    mae_sum += metrics['mae']
                    mse_sum += metrics['mse']
                    test_count += 1
                    
                    if test_count <= 2:  # Only print details of first two samples
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
            
            # Record test metrics
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
    
    # Calculate forgetting
    forgetting_vals = [
        {
            'mae': best_perf_each_task[t]['mae'] - final_perf_each_task[t]['mae'],
            'rmse': best_perf_each_task[t]['rmse'] - final_perf_each_task[t]['rmse']
        }
        for t in range(len(tasks_train))
    ]
    
    # Summarize results
    avg_final_mae = np.mean([perf['mae'] for perf in final_perf_each_task])
    avg_final_rmse = np.mean([perf['rmse'] for perf in final_perf_each_task])
    
    # Save final model
    os.makedirs('checkpoints/kelong', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_perf_each_task': final_perf_each_task,
        'best_perf_each_task': best_perf_each_task,
        'forgetting_vals': forgetting_vals,
        'avg_final_mae': avg_final_mae,
        'avg_final_rmse': avg_final_rmse
    }, 'checkpoints/kelong/final_model.pth')
    print("\nSaved final model to checkpoints/kelong/final_model.pth")
    
    # Record overall metrics
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
    
    # Record forgetting
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
    
    # Print memory status after training ends
    print_memory_stats(device, "Training ends")
    
    return {
        'final_perf_each_task': final_perf_each_task,
        'best_perf_each_task': best_perf_each_task,
        'forgetting_vals': forgetting_vals,
        'avg_final_mae': avg_final_mae,
        'avg_final_rmse': avg_final_rmse
    } 

# Helper function
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
    else:
        raise ValueError('Unknown model: {}'.format(name))
    
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

# 计算样本的梯度贡献
def compute_gradient_contribution(model, sample, criterion, device):
    """
    计算单个样本对模型的梯度贡献大小
    
    Args:
        model: 当前模型
        sample: 样本数据 (image, density map, metadata)
        criterion: 损失函数
        device: 计算设备
        
    Returns:
        gradient_magnitude: 梯度大小
    """
    model.train()  # 切换到训练模式
    
    # 分解样本数据
    image, gt_dmap, meta = sample
    image = image.unsqueeze(0).to(device)  # 添加batch维度
    
    # 确保梯度计算
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
    
    # 前向传播
    try:
        # 模型可能返回元组或单个张量
        model_output = model(image)
        
        # 处理模型输出可能是元组的情况
        if isinstance(model_output, tuple):
            # 假设第一个元素是密度图预测
            outputs = model_output[0]
        else:
            outputs = model_output
        
        # 确保gt_dmap尺寸与outputs相匹配
        gt_dmap = gt_dmap.unsqueeze(0).to(device)
        if gt_dmap.shape != outputs.shape:
            # 调整密度图尺寸
            gt_dmap = F.interpolate(gt_dmap, size=outputs.shape[2:], mode='bilinear', align_corners=False)
            # 保持总和一致
            if gt_dmap.sum() > 0:
                gt_dmap = gt_dmap * (gt_dmap.sum().item() / gt_dmap.sum().item())
        
        # 计算损失
        loss = criterion(outputs, gt_dmap)
        
        # 反向传播
        loss.backward()
        
        # 计算所有参数梯度的平方和
        total_gradient_magnitude = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_gradient_magnitude += torch.sum(param.grad.data ** 2).item()
        
        # 梯度大小的平方根
        gradient_magnitude = np.sqrt(total_gradient_magnitude)
    except Exception as e:
        print(f"计算梯度贡献时出错: {e}")
        gradient_magnitude = 0.0  # 出错时返回默认值
    
    # 清除梯度避免干扰
    model.zero_grad()
    
    return gradient_magnitude

# 基于梯度贡献更新内存缓冲区
def gradient_based_memory_update(model, criterion, dataset_buffers, sample, dataset_id, memory_quotas, counters, device, resize=True, target_size=(128, 128)):
    """
    基于梯度贡献大小更新内存缓冲区
    
    Args:
        model: 当前模型
        criterion: 损失函数
        dataset_buffers: 数据集内存缓冲区
        sample: 新样本
        dataset_id: 数据集ID
        memory_quotas: 每个数据集的内存配额
        counters: 每个数据集的计数器
        device: 计算设备
        resize: 是否调整大小
        target_size: 目标图像大小
        
    Returns:
        counters: 更新后的计数器
    """
    # 更新计数器
    counters[dataset_id] += 1
    
    # 当前数据集缓冲区和配额
    current_buffer = dataset_buffers[dataset_id]
    max_size = memory_quotas[dataset_id]
    
    # 处理样本，选择性调整大小
    try:
        if resize:
            img, den, meta = sample
            
            # 确保图像和密度图是张量
            if not isinstance(img, torch.Tensor) or not isinstance(den, torch.Tensor):
                print("警告：图像或密度图不是张量，跳过调整大小")
                return counters
            
            # 调整图像大小
            if len(img.shape) == 3:  # [C, H, W]
                img_small = F.interpolate(img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            else:
                print(f"警告：图像形状异常 {img.shape}，跳过调整大小")
                return counters
            
            # 调整密度图大小
            if len(den.shape) == 2:  # [H, W]
                den_small = F.interpolate(den.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            elif len(den.shape) == 3 and den.shape[0] == 1:  # [1, H, W]
                den_small = F.interpolate(den.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            else:
                print(f"警告：密度图形状异常 {den.shape}，跳过调整大小")
                return counters
            
            # 保持密度图总和不变（维持人数计数）
            if den.sum() > 0:
                scaling_factor = den.sum() / den_small.sum() if den_small.sum() > 0 else 1.0
                den_small = den_small * scaling_factor
            
            processed_sample = (img_small, den_small, meta)
        else:
            processed_sample = sample
        
        # 计算样本的梯度贡献
        gradient_magnitude = compute_gradient_contribution(model, processed_sample, criterion, device)
        
        # 如果缓冲区未满，直接添加
        if len(current_buffer) < max_size:
            current_buffer.append((processed_sample, gradient_magnitude))
        else:
            # 如果缓冲区已满，基于梯度大小替换
            
            # 找到梯度贡献最小的样本
            lowest_grad_idx = -1
            lowest_grad = float('inf')
            
            for i, (_, grad) in enumerate(current_buffer):
                if grad < lowest_grad:
                    lowest_grad = grad
                    lowest_grad_idx = i
            
            # 如果新样本梯度贡献更大，则替换最小的
            if gradient_magnitude > lowest_grad:
                current_buffer[lowest_grad_idx] = (processed_sample, gradient_magnitude)
            else:
                # 如果新样本梯度贡献不高，仍有低概率替换（保持一定探索性）
                if random.random() < 0.1:  # 10%的概率随机替换
                    replace_idx = random.randint(0, max_size - 1)
                    current_buffer[replace_idx] = (processed_sample, gradient_magnitude)
        
        return counters
    except Exception as e:
        print(f"更新内存缓冲区错误: {e}")
        # 出错时不更新缓冲区，但仍增加计数器
        return counters

# 计算行为克隆损失
def compute_behavior_cloning_loss(current_output, teacher_output, temperature=2.0):
    """
    计算行为克隆损失，使用知识蒸馏的方式约束当前模型的输出接近历史模型
    
    Args:
        current_output: 当前模型的输出
        teacher_output: 历史模型（教师模型）的输出
        temperature: 温度参数，控制软目标的平滑程度
        
    Returns:
        distillation_loss: 蒸馏损失
    """
    # 如果输出是元组，取第一个元素（假设是密度图）
    if isinstance(current_output, tuple):
        current_output = current_output[0]
    if isinstance(teacher_output, tuple):
        teacher_output = teacher_output[0]
    
    # 确保输出形状一致
    if current_output.shape != teacher_output.shape:
        teacher_output = F.interpolate(teacher_output, size=current_output.shape[2:], mode='bilinear', align_corners=False)
    
    # 计算KL散度损失
    log_pred = F.log_softmax(current_output / temperature, dim=1)
    soft_targets = F.softmax(teacher_output / temperature, dim=1)
    distillation_loss = F.kl_div(log_pred, soft_targets, reduction='batchmean') * (temperature ** 2)
    
    return distillation_loss

# 保存历史模型
def save_task_model(model, task_id, save_dir='checkpoints/history_models'):
    """
    保存特定任务训练完成后的模型
    
    Args:
        model: 当前模型
        task_id: 任务ID
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{save_dir}/task_{task_id}_model.pth')
    print(f"保存任务 {task_id} 的历史模型到 {save_dir}/task_{task_id}_model.pth")

# 加载历史模型
def load_task_models(model_class, model_params, task_ids, save_dir='checkpoints/history_models', device='cuda'):
    """
    加载多个历史任务的模型
    
    Args:
        model_class: 模型类
        model_params: 模型参数
        task_ids: 要加载的任务ID列表
        save_dir: 模型保存目录
        device: 设备
        
    Returns:
        models: 加载的历史模型列表
    """
    models = []
    for task_id in task_ids:
        model_path = f'{save_dir}/task_{task_id}_model.pth'
        if os.path.exists(model_path):
            model = model_class(**model_params).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()  # 设置为评估模式
            models.append(model)
            print(f"加载任务 {task_id} 的历史模型")
        else:
            print(f"找不到任务 {task_id} 的历史模型: {model_path}")
    
    return models

# 简单的随机采样更新函数
def random_memory_update(dataset_buffers, sample, dataset_id, memory_quotas, counters, resize=True, target_size=(128, 128)):
    """
    使用简单的随机采样策略更新内存缓冲区
    
    Args:
        dataset_buffers: 数据集内存缓冲区列表
        sample: 新样本 (img, den, meta)
        dataset_id: 数据集ID
        memory_quotas: 每个数据集的内存配额
        counters: 每个数据集的计数器
        resize: 是否调整图像大小
        target_size: 目标图像大小
        
    Returns:
        counters: 更新后的计数器
    """
    # 更新计数器
    counters[dataset_id] += 1
    
    # 当前数据集缓冲区和配额
    current_buffer = dataset_buffers[dataset_id]
    max_size = memory_quotas[dataset_id]
    
    # 处理样本，选择性调整大小
    try:
        if resize:
            img, den, meta = sample
            
            # 确保图像和密度图是张量
            if not isinstance(img, torch.Tensor) or not isinstance(den, torch.Tensor):
                print("警告：图像或密度图不是张量，跳过调整大小")
                return counters
            
            # 调整图像大小
            if len(img.shape) == 3:  # [C, H, W]
                img_small = F.interpolate(img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            else:
                print(f"警告：图像形状异常 {img.shape}，跳过调整大小")
                return counters
            
            # 调整密度图大小
            if len(den.shape) == 2:  # [H, W]
                den_small = F.interpolate(den.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            elif len(den.shape) == 3 and den.shape[0] == 1:  # [1, H, W]
                den_small = F.interpolate(den.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            else:
                print(f"警告：密度图形状异常 {den.shape}，跳过调整大小")
                return counters
            
            # 保持密度图总和不变（维持人数计数）
            if den.sum() > 0:
                scaling_factor = den.sum() / den_small.sum() if den_small.sum() > 0 else 1.0
                den_small = den_small * scaling_factor
            
            processed_sample = (img_small, den_small, meta)
        else:
            processed_sample = sample
        
        # 如果缓冲区未满，直接添加
        if len(current_buffer) < max_size:
            current_buffer.append((processed_sample, 1.0))  # 使用统一的价值1.0
        else:
            # 如果缓冲区已满，使用水库采样策略
            if random.random() < (max_size / counters[dataset_id]):
                # 随机选择一个样本替换
                replace_idx = random.randint(0, max_size - 1)
                current_buffer[replace_idx] = (processed_sample, 1.0)
        
        return counters
    except Exception as e:
        print(f"更新内存缓冲区错误: {e}")
        # 出错时不更新缓冲区，但仍增加计数器
        return counters

# 简单的随机采样函数
def random_sample_from_buffers(dataset_buffers, replay_batch_size, device, max_batch_size=5):
    """
    从多个数据集缓冲区中随机采样
    
    Args:
        dataset_buffers: 数据集缓冲区列表
        replay_batch_size: 采样数量
        device: 设备
        max_batch_size: 每次处理的最大批次大小
        
    Returns:
        sampled_imgs: 采样的图像
        sampled_gt_dens: 采样的密度图
        sampled_meta: 采样的元数据
    """
    if replay_batch_size <= 0:
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device), {}
    
    # 计算每个缓冲区的样本数
    buffer_sizes = [len(buffer) for buffer in dataset_buffers]
    total_samples = sum(buffer_sizes)
    
    if total_samples == 0:
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device), {}
    
    # 提取所有样本
    all_samples = []
    for buffer in dataset_buffers:
        for sample, _ in buffer:
            all_samples.append(sample)
    
    # 随机采样
    if len(all_samples) <= replay_batch_size:
        # 如果样本总数不足，全部使用
        chosen_indices = list(range(len(all_samples)))
    else:
        # 随机选择指定数量的样本
        chosen_indices = random.sample(range(len(all_samples)), replay_batch_size)
    
    # 处理采样结果
    sampled_imgs_list = []
    sampled_gt_dens_list = []
    sampled_meta_list = []
    
    # 获取第一个样本的尺寸信息，确保所有样本尺寸一致
    try:
        first_sample = all_samples[chosen_indices[0]]
        first_img, first_gt_den, _ = first_sample
        target_img_size = first_img.shape
        target_den_size = first_gt_den.shape
    except Exception as e:
        print(f"获取第一个样本尺寸信息时出错: {e}")
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device), {}
    
    # 分批处理采样结果，避免一次加载过多数据到GPU
    for i in range(0, len(chosen_indices), max_batch_size):
        batch_indices = chosen_indices[i:min(i+max_batch_size, len(chosen_indices))]
        batch_imgs = []
        batch_gt_dens = []
        batch_meta = []
        
        for idx in batch_indices:
            try:
                img, gt_den, meta = all_samples[idx]
                
                # 确保所有图像尺寸一致
                if img.shape != target_img_size:
                    if len(img.shape) == 3:  # [C, H, W]
                        img = F.interpolate(img.unsqueeze(0), size=target_img_size[1:], mode='bilinear', align_corners=False).squeeze(0)
                    else:
                        continue
                
                # 确保所有密度图尺寸一致
                if gt_den.shape != target_den_size:
                    if len(gt_den.shape) == 2:  # [H, W]
                        gt_den = F.interpolate(gt_den.unsqueeze(0).unsqueeze(0), size=target_den_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    elif len(gt_den.shape) == 3 and gt_den.shape[0] == 1:  # [1, H, W]
                        gt_den = F.interpolate(gt_den.unsqueeze(0), size=target_den_size[1:], mode='bilinear', align_corners=False).squeeze(0)
                    else:
                        continue
                    
                    # 保持密度图总和不变
                    original_sum = gt_den.sum().item()
                    if original_sum > 0:
                        gt_den = gt_den * (original_sum / gt_den.sum().item())
                
                batch_imgs.append(img.unsqueeze(0))
                batch_gt_dens.append(gt_den.unsqueeze(0))
                batch_meta.append(meta)
            except Exception:
                continue
        
        # 处理当前批次
        if batch_imgs:
            try:
                batch_imgs_tensor = torch.cat(batch_imgs, dim=0).to(device)
                batch_gt_dens_tensor = torch.cat(batch_gt_dens, dim=0).to(device)
                
                sampled_imgs_list.append(batch_imgs_tensor)
                sampled_gt_dens_list.append(batch_gt_dens_tensor)
                sampled_meta_list.extend(batch_meta)
            except RuntimeError:
                continue
            
            del batch_imgs, batch_gt_dens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 合并所有批次的结果
    if not sampled_imgs_list:
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device), {}
    
    try:
        # 检查所有批次的形状
        img_batch_shapes = [imgs.shape for imgs in sampled_imgs_list]
        den_batch_shapes = [dens.shape for dens in sampled_gt_dens_list]
        
        if len(set(str(s[1:]) for s in img_batch_shapes)) > 1:
            first_batch_shape = sampled_imgs_list[0].shape[1:]
            for i in range(1, len(sampled_imgs_list)):
                if sampled_imgs_list[i].shape[1:] != first_batch_shape:
                    sampled_imgs_list[i] = F.interpolate(sampled_imgs_list[i], size=first_batch_shape[1:], mode='bilinear', align_corners=False)
        
        if len(set(str(s[1:]) for s in den_batch_shapes)) > 1:
            first_batch_shape = sampled_gt_dens_list[0].shape[1:]
            for i in range(1, len(sampled_gt_dens_list)):
                if sampled_gt_dens_list[i].shape[1:] != first_batch_shape:
                    sampled_gt_dens_list[i] = F.interpolate(sampled_gt_dens_list[i], size=first_batch_shape[1:], mode='bilinear', align_corners=False)
        
        sampled_imgs = torch.cat(sampled_imgs_list, dim=0)
        sampled_gt_dens = torch.cat(sampled_gt_dens_list, dim=0)
    except RuntimeError:
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device), {}
    
    # 处理元数据
    try:
        if isinstance(sampled_meta_list[0], dict) and 'count' in sampled_meta_list[0]:
            count_values = []
            for meta in sampled_meta_list:
                if isinstance(meta['count'], torch.Tensor):
                    if meta['count'].numel() == 1:
                        count_values.append(meta['count'].item())
                    else:
                        count_values.append(meta['count'][0].item() if meta['count'].numel() > 0 else 0)
                else:
                    try:
                        count_values.append(float(meta['count']) if meta['count'] is not None else 0.0)
                    except (TypeError, ValueError):
                        count_values.append(0.0)
            sampled_meta = {'count': torch.tensor(count_values).to(device)}
        elif isinstance(sampled_meta_list[0], torch.Tensor):
            meta_tensors = []
            for meta in sampled_meta_list:
                if meta.numel() == 1:
                    meta_tensors.append(meta.item())
                else:
                    meta_tensors.append(meta[0].item() if meta.numel() > 0 else 0)
            sampled_meta = {'count': torch.tensor(meta_tensors).to(device)}
        else:
            try:
                sampled_meta = {'count': torch.tensor([float(m) if m is not None else 0.0 for m in sampled_meta_list]).to(device)}
            except (TypeError, ValueError):
                sampled_meta = {'count': torch.zeros(len(sampled_meta_list)).to(device)}
    except (TypeError, ValueError):
        sampled_meta = {'count': torch.zeros(len(sampled_meta_list)).to(device)}
    
    # 清理内存
    del sampled_imgs_list, sampled_gt_dens_list
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return sampled_imgs, sampled_gt_dens, sampled_meta

# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="fog,snow,stadium,street", 
                        help="Comma-separated list of datasets to use")
    parser.add_argument("--model_type", type=str, default="final", 
                        choices=["mem", "final"], help="Model type")
    parser.add_argument("--memory_size", type=int, default=200, 
                        help="Replay memory size")
    parser.add_argument("--replay_batch_size", type=int, default=16,
                        help="Replay batch size per step")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Training batch size")
    parser.add_argument("--epochs_per_task", type=int, default=5, 
                        help="Number of training epochs per Task")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--config", type=str, default="configs/jhu_domains_cl_config.yml", 
                        help="Config file path")
    parser.add_argument("--clearml_project", type=str, default="MPCount",
                        help="ClearML project name")
    parser.add_argument("--clearml_task", type=str, default="jhu_ContinualLearning_kelong",
                        help="ClearML task name")
    parser.add_argument("--use_clearml", action="store_true",
                        help="Whether to use ClearML for experiment tracking")
    parser.add_argument("--safe_mode", action="store_true",
                        help="Enable safe mode, use smaller batch size and more conservative memory settings")
    
    # Add gradient accumulation parameters
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps for gradient accumulation, reducing memory usage")
    
    # Add memory optimization options
    parser.add_argument("--memory_optimization", action="store_true",
                        help="Enable memory optimization mode, dynamically adjust memory size and sampling batch size")
    parser.add_argument("--memory_image_size", type=int, default=128,
                        help="Size of images in memory buffer")
    parser.add_argument("--max_samples_per_forward", type=int, default=5,
                        help="Maximum number of samples per forward pass")
    
    # Add memory quota related parameters
    parser.add_argument("--memory_quota_strategy", type=str, default="equal",
                        choices=["equal", "front_heavy", "back_heavy", "custom"],
                        help="Memory quota allocation strategy: equal-average allocation, front_heavy-more for previous tasks, back_heavy-more for later tasks, custom-user-defined")
    parser.add_argument("--custom_quotas", type=str, default="",
                        help="Custom memory quotas, comma-separated ratio values, e.g., '0.3,0.3,0.2,0.2', sum should be 1")
    
    # 使用随机采样策略
    parser.add_argument("--use_random_sampling", action="store_true", default=True,
                        help="使用简单的随机采样策略进行内存管理")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # 行为克隆相关参数
    parser.add_argument("--use_behavior_cloning", action="store_true",
                        help="使用行为克隆策略避免灾难性遗忘")
    parser.add_argument("--bc_lambda", type=float, default=1.0,
                        help="行为克隆损失的权重")
    parser.add_argument("--bc_temperature", type=float, default=2.0,
                        help="行为克隆中的温度参数")
    
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    # Read epochs_per_task from config file, if not provided use default value from command line arguments
    args.epochs_per_task = cfg.get('num_epochs', args.epochs_per_task) // len(cfg['train_dataset'])
    
    # Initialize memory quotas
    datasets_count = len(args.datasets.split(','))
    if args.memory_quota_strategy == "equal":
        memory_quotas = [args.memory_size // datasets_count] * datasets_count
    elif args.memory_quota_strategy == "front_heavy":
        # Previous tasks get more quotas, linear decrease
        total = datasets_count * (datasets_count + 1) // 2
        memory_quotas = [int(args.memory_size * (datasets_count - i) / total) for i in range(datasets_count)]
    elif args.memory_quota_strategy == "back_heavy":
        # Later tasks get more quotas, linear increase
        total = datasets_count * (datasets_count + 1) // 2
        memory_quotas = [int(args.memory_size * (i + 1) / total) for i in range(datasets_count)]
    elif args.memory_quota_strategy == "custom" and args.custom_quotas:
        # Use custom quotas provided by user
        try:
            quotas = [float(q) for q in args.custom_quotas.split(',')]
            if len(quotas) != datasets_count:
                print(f"Warning: Number of custom quotas ({len(quotas)}) does not match number of datasets ({datasets_count}), using average allocation")
                memory_quotas = [args.memory_size // datasets_count] * datasets_count
            else:
                # Normalize quotas and convert to integers
                total = sum(quotas)
                memory_quotas = [int(args.memory_size * q / total) for q in quotas]
        except ValueError:
            print("Warning: Incorrect format for custom quotas, using average allocation")
            memory_quotas = [args.memory_size // datasets_count] * datasets_count
    else:
        # Default use average allocation
        memory_quotas = [args.memory_size // datasets_count] * datasets_count
    
    # Ensure total quotas sum up to memory_size
    while sum(memory_quotas) < args.memory_size:
        memory_quotas[0] += 1
    while sum(memory_quotas) > args.memory_size:
        memory_quotas[-1] -= 1
    
    print(f"Memory quota allocation: {memory_quotas}")
    
    # Safe mode settings
    if args.safe_mode:
        print("Enable safe mode, reduce batch size")
        # Reduce batch_size to avoid memory issues
        if 'train_loader' in cfg and 'batch_size' in cfg['train_loader']:
            cfg['train_loader']['batch_size'] = min(4, cfg['train_loader']['batch_size'])
        if 'val_loader' in cfg and 'batch_size' in cfg['val_loader']:
            cfg['val_loader']['batch_size'] = 1 
        if 'test_loader' in cfg and 'batch_size' in cfg['test_loader']:
            cfg['test_loader']['batch_size'] = 1
        # Disable pin_memory
        if 'train_loader' in cfg:
            cfg['train_loader']['pin_memory'] = False
        if 'val_loader' in cfg:
            cfg['val_loader']['pin_memory'] = False
        if 'test_loader' in cfg:
            cfg['test_loader']['pin_memory'] = False
        # Reduce number of worker threads
        if 'train_loader' in cfg and 'num_workers' in cfg['train_loader']:
            cfg['train_loader']['num_workers'] = min(1, cfg['train_loader']['num_workers'])
        if 'val_loader' in cfg and 'num_workers' in cfg['val_loader']:
            cfg['val_loader']['num_workers'] = min(1, cfg['val_loader']['num_workers'])
        if 'test_loader' in cfg and 'num_workers' in cfg['test_loader']:
            cfg['test_loader']['num_workers'] = min(1, cfg['test_loader']['num_workers'])
    
    # Initialize ClearML
    task_name=join(args.clearml_task, args.model_type, args.datasets.replace(',', '_'))
    clearml_logger = CustomClearML('MPCount', task_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(vars(args))
    print("Using device:", device)
    
    # Set random seed
    master_seed = args.seed
    
    # Add calculated memory quotas to args
    args.memory_quotas = memory_quotas
    
    # 修改输出信息，不再引用use_ac参数
    print("使用简单的随机采样方法进行内存管理")
    
    # 打印行为克隆设置
    if args.use_behavior_cloning:
        print(f"启用行为克隆策略，权重={args.bc_lambda}，温度={args.bc_temperature}")
    
    # Use simple random sampling for training and evaluation
    results = train_and_evaluate_simple(cfg, args, device, master_seed)
    
    # Print final results
    print("\n==== Final Results ====")
    print(f"Average MAE: {results['avg_final_mae']:.2f}")
    print(f"Average RMSE: {results['avg_final_rmse']:.2f}")
    
    # Print performance of each task
    datasets_list = args.datasets.split(',')
    for i, (dataset, perf) in enumerate(zip(datasets_list, results['final_perf_each_task'])):
        print(f"Task {i} ({dataset}) - MAE: {perf['mae']:.2f}, RMSE: {perf['rmse']:.2f}")
    
    # Print forgetting
    print("\n==== Forgetting ====")
    for i, (dataset, forgetting) in enumerate(zip(datasets_list, results['forgetting_vals'])):
        print(f"Task {i} ({dataset}) - MAE Forgetting: {forgetting['mae']:.2f}, RMSE Forgetting: {forgetting['rmse']:.2f}")

if __name__ == "__main__":
    main() 