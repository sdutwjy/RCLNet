import abc
import logging
import os

import torch

LOGGER = logging.Logger("IncLearn", level="INFO")

logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import time
import copy
import random
import numpy as np
import yaml
import torch.nn.functional as F
from os.path import join
import gc  # 添加垃圾回收模块

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
from trainers.dgtrainer import DGTrainer
from tqdm import tqdm
from utils.misc import seed_worker, get_seeded_generator, seed_everything

# 导入PODNet相关模块
import math
import logging
from torch.nn import functional as F

# 配置日志
logger = logging.getLogger(__name__)

class IncrementalLearner(abc.ABC):
    """Base incremental learner.

    Methods are called in this order (& repeated for each new task):

    1. set_task_info
    2. before_task
    3. train_task
    4. after_task
    5. eval_task
    """

    def __init__(self, *args, **kwargs):
        self._network = None

    def set_task_info(self, task_info):
        self._task = task_info["task"]
        self._total_n_classes = task_info["total_n_classes"]
        self._task_size = task_info["increment"]
        self._n_train_data = task_info["n_train_data"]
        self._n_test_data = task_info["n_test_data"]
        self._n_tasks = task_info["max_task"]

    def before_task(self, train_loader, val_loader):
        LOGGER.info("Before task")
        self.eval()
        self._before_task(train_loader, val_loader)

    def train_task(self, train_loader, val_loader):
        LOGGER.info("train task")
        self.train()
        self._train_task(train_loader, val_loader)

    def after_task_intensive(self, inc_dataset):
        LOGGER.info("after task")
        self.eval()
        self._after_task_intensive(inc_dataset)

    def after_task(self, inc_dataset):
        LOGGER.info("after task")
        self.eval()
        self._after_task(inc_dataset)

    def eval_task(self, data_loader):
        LOGGER.info("eval task")
        self.eval()
        return self._eval_task(data_loader)

    def get_memory(self):
        return None

    def get_val_memory(self):
        return None

    def _before_task(self, data_loader, val_loader):
        pass

    def _train_task(self, train_loader, val_loader):
        raise NotImplementedError

    def _after_task_intensive(self, data_loader):
        pass

    def _after_task(self, data_loader):
        pass

    def _eval_task(self, data_loader):
        raise NotImplementedError

    def save_metadata(self, directory, run_id):
        pass

    def load_metadata(self, directory, run_id):
        pass

    @property
    def _new_task_index(self):
        return self._task * self._task_size

    @property
    def inc_dataset(self):
        return self.__inc_dataset

    @inc_dataset.setter
    def inc_dataset(self, inc_dataset):
        self.__inc_dataset = inc_dataset

    @property
    def network(self):
        return self._network

    def save_parameters(self, directory, run_id):
        path = os.path.join(directory, f"net_{run_id}_task_{self._task}.pth")
        logger.info(f"Saving model at {path}.")
        torch.save(self.network.state_dict(), path)

    def load_parameters(self, directory, run_id):
        path = os.path.join(directory, f"net_{run_id}_task_{self._task}.pth")
        if not os.path.exists(path):
            return

        logger.info(f"Loading model at {path}.")
        try:
            self.network.load_state_dict(torch.load(path))
        except Exception:
            logger.warning("Old method to save weights, it's deprecated!")
            self._network = torch.load(path)

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

class PODNet(IncrementalLearner):
    """Pooled Output Distillation Network.
    
    参考论文:
    * Small Task Incremental Learning
      Douillard et al. 2020
    """

    def __init__(self, args):
        super().__init__()
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        # 优化配置:
        self._batch_size = args["batch_size"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]
        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        # 记忆回放配置:
        self._memory_size = args["memory_size"]
        self._fixed_memory = args.get("fixed_memory", True)
        self._herding_selection = args.get("herding_selection", {"type": "random"})
        self._n_classes = 0
        self._last_results = None
        self._validation_percent = args.get("validation")

        # POD相关配置:
        self._pod_flat_config = args.get("pod_flat", {})
        self._pod_spatial_config = args.get("pod_spatial", {})

        self._nca_config = args.get("nca", {})
        self._softmax_ce = args.get("softmax_ce", False)

        self._perceptual_features = args.get("perceptual_features")
        self._perceptual_style = args.get("perceptual_style")

        self._groupwise_factors = args.get("groupwise_factors", {})
        self._groupwise_factors_bis = args.get("groupwise_factors_bis", {})

        self._class_weights_config = args.get("class_weights_config", {})

        self._evaluation_type = args.get("eval_type", "icarl")
        self._evaluation_config = args.get("evaluation_config", {})

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._gradcam_distil = args.get("gradcam_distil", {})

        # 初始化网络
        self._network = None  # 这里需要根据具体的模型结构进行实例化
        self._examplars = {}
        self._means = None
        self._old_model = None

        self._finetuning_config = args.get("finetuning_config")
        self._herding_indexes = []
        self._weight_generation = args.get("weight_generation")

        self._meta_transfer = args.get("meta_transfer", {})
        self._post_processing_type = None
        self._data_memory, self._targets_memory = None, None
        self._args = args
        self._args["_logs"] = {}
        self._task = 0  # 当前任务ID
        self._total_n_classes = 0  # 总类别数
        self._task_size = 0  # 当前任务类别数
        self._n_tasks = 0  # 总任务数
        
        # 记忆库
        self._memory_per_task = []  # 每个任务的记忆样本列表

    @property
    def _memory_per_class(self):
        """返回每个类别的样本数量。"""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes
        
    def set_task_info(self, task_info):
        """设置当前任务的信息"""
        self._task = task_info["task"]
        self._total_n_classes = task_info["total_n_classes"]
        self._task_size = task_info["increment"]
        self._n_tasks = task_info["max_task"]
        self._n_classes += self._task_size
        
    def set_network(self, network):
        """设置网络模型"""
        self._network = network
        
    def set_meta_transfer(self):
        """设置元传输学习参数"""
        if self._meta_transfer:
            logger.info("设置任务元传输")
            if self._task == 0:
                if hasattr(self._network, "apply_mtl"):
                    self._network.apply_mtl(False)
            elif self._task == 1:
                if self._meta_transfer["type"] != "none":
                    if hasattr(self._network, "apply_mtl"):
                        self._network.apply_mtl(True)

                if self._meta_transfer.get("mtl_bias"):
                    if hasattr(self._network, "apply_mtl_bias"):
                        self._network.apply_mtl_bias(True)
                elif self._meta_transfer.get("bias_on_weight"):
                    if hasattr(self._network, "apply_bias_on_weights"):
                        self._network.apply_bias_on_weights(True)

                if self._meta_transfer["freeze_convnet"]:
                    if hasattr(self._network, "freeze_convnet"):
                        self._network.freeze_convnet(
                            True,
                            bn_weights=self._meta_transfer.get("freeze_bn_weights"),
                            bn_stats=self._meta_transfer.get("freeze_bn_stats")
                        )
            elif self._meta_transfer["type"] != "none":
                if self._meta_transfer["type"] == "repeat" or (self._task == 2 and self._meta_transfer["type"] == "once"):
                    if hasattr(self._network, "fuse_mtl_weights"):
                        self._network.fuse_mtl_weights()
                    if hasattr(self._network, "reset_mtl_parameters"):
                        self._network.reset_mtl_parameters()

                    if self._meta_transfer["freeze_convnet"]:
                        if hasattr(self._network, "freeze_convnet"):
                            self._network.freeze_convnet(
                                True,
                                bn_weights=self._meta_transfer.get("freeze_bn_weights"),
                                bn_stats=self._meta_transfer.get("freeze_bn_stats")
                            )

    def _before_task(self, train_loader, val_loader):
        """任务开始前的准备工作"""
        logger.info(f"准备任务 {self._task}")
        
        # 设置优化器
        self._optimizer = self.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr, self._weight_decay
        )
        
        # 设置学习率调度器
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, self._scheduling, gamma=self._lr_decay
        )

    def _train_task(self, train_loader, val_loader):
        """训练当前任务"""
        if self._meta_transfer:
            logger.info("设置任务元传输")
            self.set_meta_transfer()

        for p in self._network.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))

        logger.debug(f"数据集大小: {len(train_loader.dataset)}.")

        if self._meta_transfer and self._meta_transfer.get("clip"):
            logger.info(f"裁剪MTL权重 ({self._meta_transfer.get('clip')}).")
            clipper = BoundClipper(*self._meta_transfer.get("clip"))
        else:
            clipper = None
            
        self._training_step(
            train_loader, val_loader, 0, self._n_epochs, record_bn=True, clipper=clipper
        )

        self._post_processing_type = None

        if self._finetuning_config and self._task != 0:
            logger.info("微调")
            if self._finetuning_config.get("scaling"):
                logger.info(
                    f"微调缩放系数为 {self._finetuning_config['scaling']}."
                )
                self._post_processing_type = self._finetuning_config["scaling"]

            # 这里简化为直接微调分类器
            if self._finetuning_config.get("tuning") == "all":
                parameters = self._network.parameters()
            elif self._finetuning_config.get("tuning") == "classifier":
                parameters = self._network.classifier.parameters() if hasattr(self._network, "classifier") else self._network.parameters()
            else:
                parameters = self._network.parameters()

            self._optimizer = self.get_optimizer(
                parameters, self._opt_name, self._finetuning_config.get("lr", self._lr * 0.1), self._weight_decay
            )
            self._scheduler = None
            self._training_step(
                train_loader,
                val_loader,
                self._n_epochs,
                self._n_epochs + self._finetuning_config.get("epochs", 10),
                record_bn=False
            )
            
    def _training_step(self, train_loader, val_loader, initial_epoch, nb_epochs, record_bn=True, clipper=None):
        """训练步骤"""
        best_epoch, best_acc = -1, -1.
        wait = 0

        for epoch in range(initial_epoch, nb_epochs):
            self._metrics = {}
            self._epoch_percent = epoch / (nb_epochs - initial_epoch)

            prog_bar = tqdm(
                train_loader,
                disable=self._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )
            
            self._network.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for i, input_dict in enumerate(prog_bar, start=1):
                # 处理输入数据
                if isinstance(input_dict, tuple):
                    # 适应人群计数的数据格式
                    images, imgs2, gt_datas = input_dict
                    inputs = images.to(self._device)
                    
                    # 获取密度图和目标计数
                    if isinstance(gt_datas, tuple) and len(gt_datas) >= 3:
                        gt_dmaps = gt_datas[1].to(self._device) if gt_datas[1] is not None else None
                        gt_counts = gt_datas[0].to(self._device) if isinstance(gt_datas[0], torch.Tensor) else None
                    else:
                        gt_dmaps = gt_datas.to(self._device) if gt_datas is not None else None
                        gt_counts = None
                    
                    # 设置记忆标志
                    memory_flags = torch.zeros(inputs.size(0)).bool().to(self._device)
                    
                    # 如果有记忆样本，添加到当前批次
                    if self._task > 0 and len(self._memory_per_task) > 0:
                        memory_inputs = []
                        memory_targets = []
                        memory_flags_list = []
                        
                        # 从每个任务的记忆库中随机采样
                        for task_id in range(self._task):
                            if task_id < len(self._memory_per_task) and len(self._memory_per_task[task_id]) > 0:
                                # 随机选择记忆样本
                                memory_indices = torch.randperm(len(self._memory_per_task[task_id]))[:min(5, len(self._memory_per_task[task_id]))]
                                for idx in memory_indices:
                                    mem_sample = self._memory_per_task[task_id][idx]
                                    if isinstance(mem_sample, tuple) and len(mem_sample) >= 2:
                                        mem_input, mem_target = mem_sample[0], mem_sample[1]
                                        memory_inputs.append(mem_input.unsqueeze(0))
                                        memory_targets.append(mem_target.unsqueeze(0) if mem_target.dim() < 4 else mem_target)
                                        memory_flags_list.append(torch.ones(1).bool())
                        
                        if memory_inputs:
                            # 将记忆样本与当前批次合并
                            try:
                                memory_inputs = torch.cat(memory_inputs, dim=0).to(self._device)
                                memory_targets = torch.cat(memory_targets, dim=0).to(self._device)
                                memory_flags_list = torch.cat(memory_flags_list, dim=0).to(self._device)
                                
                                # 确保尺寸一致
                                if memory_inputs.shape[2:] != inputs.shape[2:]:
                                    memory_inputs = F.interpolate(memory_inputs, size=inputs.shape[2:], mode='bilinear', align_corners=False)
                                
                                if memory_targets.shape != gt_dmaps.shape and gt_dmaps is not None:
                                    if memory_targets.dim() == 3:  # [B, H, W]
                                        memory_targets = F.interpolate(memory_targets.unsqueeze(1), size=gt_dmaps.shape[1:], mode='bilinear', align_corners=False).squeeze(1)
                                    elif memory_targets.dim() == 4:  # [B, C, H, W]
                                        memory_targets = F.interpolate(memory_targets, size=gt_dmaps.shape[2:], mode='bilinear', align_corners=False)
                                
                                # 合并当前批次和记忆样本
                                inputs = torch.cat([inputs, memory_inputs], dim=0)
                                if gt_dmaps is not None:
                                    gt_dmaps = torch.cat([gt_dmaps, memory_targets], dim=0)
                                memory_flags = torch.cat([memory_flags, memory_flags_list], dim=0)
                            except Exception as e:
                                logger.warning(f"合并记忆样本时出错: {e}")
                else:
                    # 原始PODNet的数据格式
                    inputs = input_dict["inputs"].to(self._device)
                    gt_dmaps = input_dict.get("targets", None)
                    if gt_dmaps is not None:
                        gt_dmaps = gt_dmaps.to(self._device)
                    gt_counts = None
                    memory_flags = input_dict.get("memory_flags", torch.zeros(inputs.size(0)).bool().to(self._device))

                self._optimizer.zero_grad()
                
                # 前向传播
                try:
                    # 获取模型的输入尺寸
                    target_size = (80, 80)  # 根据错误信息设置固定的目标尺寸，DGModel的输出层期望的尺寸是80x80
                    
                    # 统一调整输入图像和密度图的尺寸
                    if inputs.shape[2:] != target_size:
                        print(f"调整图像尺寸: {inputs.shape[2:]} -> {target_size}")
                        inputs = F.interpolate(inputs, size=target_size, mode='bilinear', align_corners=False)
                    if gt_dmaps is not None and gt_dmaps.shape[2:] != target_size:
                        print(f"调整密度图尺寸: {gt_dmaps.shape[2:]} -> {target_size}")
                        gt_dmaps = F.interpolate(gt_dmaps, size=target_size, mode='bilinear', align_corners=False)
                    
                    # 调整前打印尺寸信息
                    print(f"调整前: images={inputs.shape}, imgs2={imgs2.shape if imgs2 is not None else None}, gt_dmaps={gt_dmaps.shape if gt_dmaps is not None else None}")
                    
                    # 确保输入和目标尺寸一致
                    images, gt_dmaps = ensure_same_sizes(inputs, gt_dmaps, target_size=(80, 80), device=self._device)
                    imgs2, _ = ensure_same_sizes(imgs2, None, target_size=(80, 80), device=self._device)
                    
                    print(f"调整后: images={images.shape}, imgs2={imgs2.shape}, gt_dmaps={gt_dmaps.shape if gt_dmaps is not None else None}")
                    
                    # 调用模型
                    try:
                        # 模仿main_cl_jhu_ac.py中的方式
                        loss = get_loss()
                        dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = self._network.forward_train(images, imgs2, gt_dmaps)
                        # 计算损失
                        log_para = 1000  # 默认log_para值
                        gt_datas_tuple = (None, gt_dmaps, None)  # 创建与compute_count_loss兼容的格式
                        loss_den = F.mse_loss(dmaps1, gt_dmaps) + F.mse_loss(dmaps2, gt_dmaps)
                        loss_cls = F.binary_cross_entropy(cmaps1, gt_dmaps) + F.binary_cross_entropy(cmaps2, gt_dmaps)
                        loss_total = loss_den + 10 * loss_cls + 10 * loss_con  # 模仿main_cl_jhu_ac.py中的损失计算
                    except Exception as e:
                        print(f"前向传播或损失计算时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    if torch.isnan(loss_total):
                        logger.warning("损失为NaN，跳过此批次")
                        continue
                    
                    loss_total.backward()
                    self._optimizer.step()
                    
                    if clipper:
                        self._network.apply(clipper)
                    
                    epoch_loss += loss_total.item()
                    num_batches += 1
                except Exception as e:
                    print(f"处理批数据时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 更新进度条
            desc = f"任务{self._task+1}/{self._n_tasks}, E{epoch+1}/{nb_epochs}"
            if self._metrics:
                metrics_str = ", ".join(f"{k}: {v/num_batches:.3f}" for k, v in self._metrics.items())
                desc += f" => {metrics_str}"
            prog_bar.set_description(desc)

            if self._scheduler:
                self._scheduler.step(epoch)

            # 每个epoch后评估
            if self._eval_every_x_epochs and epoch != 0 and epoch % self._eval_every_x_epochs == 0 and val_loader is not None:
                self._network.eval()
                val_loss = self._validate(val_loader)
                logger.info(f"Epoch {epoch+1}/{nb_epochs}, 验证损失: {val_loss:.4f}")
                
                # 早停检查
                if self._early_stopping and self._early_stopping.get("patience") is not None:
                    if val_loss < best_acc or best_acc == -1:
                        best_epoch = epoch
                        best_acc = val_loss
                        wait = 0
                    else:
                        wait += 1
                    
                    if wait >= self._early_stopping["patience"]:
                        logger.info(f"早停! 最佳epoch: {best_epoch+1}, 最佳验证损失: {best_acc:.4f}")
                        break
                
                self._network.train()
            
            # 保存当前epoch的样本到记忆库
            if self._task == len(self._memory_per_task):
                self._memory_per_task.append([])
            
            # 每个epoch结束后更新记忆库
            if epoch == nb_epochs - 1:
                self._update_memory(train_loader)
                
    def _update_memory(self, dataloader):
        """更新记忆库"""
        if self._memory_size <= 0:
            return
            
        logger.info(f"更新记忆库 (当前任务: {self._task}).")
        
        # 记忆库已满，跳过
        if self._task < len(self._memory_per_task) and len(self._memory_per_task[self._task]) >= self._memory_size // self._n_tasks:
            logger.info(f"任务 {self._task} 的记忆库已满 ({len(self._memory_per_task[self._task])}/{self._memory_size // self._n_tasks}).")
            return
            
        # 确保当前任务的记忆库存在
        if self._task >= len(self._memory_per_task):
            self._memory_per_task.append([])
            
        # 随机选择样本存入记忆库
        dataset = dataloader.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
            
        # 计算要添加的样本数量
        remaining = self._memory_size // self._n_tasks - len(self._memory_per_task[self._task])
        if remaining <= 0:
            return
            
        # 随机选择索引
        indices = torch.randperm(len(dataset))[:min(remaining, len(dataset))]
        
        # 添加样本到记忆库
        added = 0
        for idx in indices:
            try:
                sample = dataset[idx]
                if sample is not None:
                    self._memory_per_task[self._task].append(sample)
                    added += 1
            except Exception as e:
                logger.warning(f"添加样本到记忆库时出错: {e}")
                
        logger.info(f"为任务 {self._task} 添加了 {added} 个样本到记忆库 (总计: {len(self._memory_per_task[self._task])}).")
        
    def _validate(self, dataloader):
        """验证模型性能"""
        self._network.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for input_dict in dataloader:
                try:
                    # 处理输入数据
                    if isinstance(input_dict, tuple):
                        # 适应人群计数的数据格式
                        images, imgs2, gt_datas = input_dict
                        inputs = images.to(self._device)
                        
                        # 获取密度图
                        if isinstance(gt_datas, tuple) and len(gt_datas) >= 3:
                            gt_dmaps = gt_datas[1].to(self._device) if gt_datas[1] is not None else None
                        else:
                            gt_dmaps = gt_datas.to(self._device) if gt_datas is not None else None
                    else:
                        inputs = input_dict["inputs"].to(self._device)
                        gt_dmaps = input_dict.get("targets", None)
                        if gt_dmaps is not None:
                            gt_dmaps = gt_dmaps.to(self._device)
                        imgs2 = inputs
                    
                    # 获取模型的输入尺寸
                    target_size = (80, 80)
                    
                    # 调整输入图像和密度图的尺寸
                    inputs, gt_dmaps = ensure_same_sizes(inputs, gt_dmaps, target_size=target_size, device=self._device)
                    imgs2, _ = ensure_same_sizes(imgs2, None, target_size=target_size, device=self._device)
                    
                    # 调用模型
                    dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = self._network.forward_train(inputs, imgs2, gt_dmaps)
                    
                    # 计算损失
                    loss_den = F.mse_loss(dmaps1, gt_dmaps) + F.mse_loss(dmaps2, gt_dmaps)
                    loss_cls = F.binary_cross_entropy(cmaps1, gt_dmaps) + F.binary_cross_entropy(cmaps2, gt_dmaps)
                    loss_total = loss_den + 10 * loss_cls + 10 * loss_con
                    
                    val_loss += loss_total.item()
                    num_batches += 1
                except Exception as e:
                    logger.warning(f"验证时出错: {e}")
                    continue
                    
        # 计算平均损失
        if num_batches > 0:
            val_loss /= num_batches
            
        return val_loss

def evaluate_performance(model, dataset, device, collate_fn=None):
    """评估模型性能"""
    model.eval()
    
    # 创建数据加载器，添加错误处理
    try:
        loader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            collate_fn=custom_collate_fn  # 使用自定义的collate函数
        )
    except Exception as e:
        print(f"创建评估数据加载器时出错: {e}")
        return {'mae': float('inf'), 'rmse': float('inf')}
        
    mae_sum = 0
    mse_sum = 0
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="评估", leave=False):
            try:
                # 处理不同的数据格式
                if batch is None:
                    continue
                    
                if isinstance(batch, tuple) and len(batch) >= 3:
                    img1, img2, gt_datas = batch[0], batch[1], batch[2:]
                    
                    # 确保数据是张量
                    if not isinstance(img1, torch.Tensor):
                        continue
                    
                    # 确保图像格式正确
                    if len(img1.shape) == 3:  # [C, H, W]
                        img1 = img1.unsqueeze(0)  # 添加批次维度
                    
                    img1 = img1.to(device)
                    
                    # 使用模型预测
                    try:
                        # 获取模型的输入尺寸
                        target_size = (80, 80)  # 根据错误信息设置固定的目标尺寸，DGModel的输出层期望的尺寸是80x80
                        
                        # 统一调整输入图像和密度图的尺寸
                        if img1.shape[2:] != target_size:
                            img1 = F.interpolate(img1, size=target_size, mode='bilinear', align_corners=False)
                        
                        if hasattr(model, 'predict'):
                            pred_count = model.predict(img1)
                        elif hasattr(model, 'forward_train'):
                            # 创建一个与输入图像相同尺寸的零密度图
                            dummy_gt = torch.zeros((img1.size(0), 1, target_size[0], target_size[1]), device=device)
                            outputs = model.forward_train(img1, img1, dummy_gt)
                            pred_dmap = outputs[0]
                            pred_count = pred_dmap.sum().item() / 1000  # 除以log_para
                        else:
                            outputs = model(img1)
                            if isinstance(outputs, tuple):
                                pred_dmap = outputs[0]
                            else:
                                pred_dmap = outputs
                            pred_count = pred_dmap.sum().item() / 1000
                    except Exception as e:
                        print(f"模型推理错误: {e}")
                        continue
                    
                    # 获取真实人数
                    gt_count = 0
                    if isinstance(gt_datas, tuple) and len(gt_datas) > 0:
                        if isinstance(gt_datas[0], torch.Tensor):
                            gt_count = gt_datas[0].shape[1] if len(gt_datas[0].shape) > 1 else gt_datas[0].sum().item()
                        elif len(gt_datas) > 1 and isinstance(gt_datas[1], torch.Tensor):  # 使用密度图
                            gt_count = gt_datas[1].sum().item()
                    elif isinstance(gt_datas, torch.Tensor):
                        gt_count = gt_datas.sum().item()
                else:
                    continue
                
                # 计算MAE和MSE
                batch_mae = abs(pred_count - gt_count)
                batch_mse = (pred_count - gt_count) ** 2
                
                mae_sum += batch_mae
                mse_sum += batch_mse
                sample_count += 1
                
            except Exception as e:
                print(f"评估样本时出错: {e}")
                continue
    
    if sample_count == 0:
        return {'mae': float('inf'), 'rmse': float('inf')}
        
    mae = mae_sum / sample_count
    rmse = math.sqrt(mse_sum / sample_count)
    
    return {'mae': mae, 'rmse': rmse}

def save_model(model, path):
    """保存模型"""
    try:
        torch.save(model.state_dict(), path)
        print(f"模型已保存到 {path}")
    except Exception as e:
        print(f"保存模型时出错: {e}")

def check_early_stopping(perf, early_stopping_config):
    """检查是否达到早停条件"""
    if not early_stopping_config:
        return False
    
    threshold = early_stopping_config.get("threshold")
    if threshold is not None and perf['mae'] < threshold:
        return True
    
    return False

def get_loss():
    """获取损失函数"""
    return nn.MSELoss()

def load_config(config_path):
    """加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def ensure_same_sizes(images, targets, target_size=(80, 80), device='cuda'):
    """确保输入图像和目标密度图的尺寸一致
    
    Args:
        images: 输入图像
        targets: 目标密度图
        target_size: 目标尺寸
        device: 设备
        
    Returns:
        调整后的图像和密度图
    """
    # 检查图像尺寸
    if images.shape[2:] != target_size:
        print(f"调整图像尺寸: {images.shape[2:]} -> {target_size}")
        images = F.interpolate(images, size=target_size, mode='bilinear', align_corners=False)
    
    # 如果目标为None，创建全零目标
    if targets is None:
        targets = torch.zeros((images.size(0), 1, target_size[0], target_size[1]), device=device)
    
    # 调整目标尺寸
    if targets.shape[2:] != target_size:
        print(f"调整目标尺寸: {targets.shape[2:]} -> {target_size}")
        targets = F.interpolate(targets, size=target_size, mode='bilinear', align_corners=False)
        
    return images, targets

def check_dataset_validity(dataset, num_samples=5):
    """
    检查数据集是否有效
    
    Args:
        dataset: 数据集
        num_samples: 检查的样本数量
        
    Returns:
        是否有效
    """
    print(f"检查数据集有效性，抽样 {num_samples} 个样本")
    valid_count = 0
    errors = []
    
    try:
        # 检查数据集长度
        if len(dataset) == 0:
            print("错误：数据集为空")
            return False
        
        # 检查随机抽取的样本
        indices = torch.randperm(len(dataset))[:min(num_samples, len(dataset))]
        for i, idx in enumerate(indices):
            try:
                sample = dataset[idx]
                if sample is None:
                    errors.append(f"样本 {idx} 为None")
                    continue
                    
                if not isinstance(sample, tuple):
                    errors.append(f"样本 {idx} 不是元组: {type(sample)}")
                    continue
                    
                if len(sample) < 3:  # 需要至少三个元素：图像1，图像2，目标数据
                    errors.append(f"样本 {idx} 元组长度不够: {len(sample)}")
                    continue
                    
                images, imgs2, *gt_data = sample
                
                if not isinstance(images, torch.Tensor):
                    errors.append(f"样本 {idx} 的图像不是张量: {type(images)}")
                    continue
                    
                if images.dim() < 3:  # 至少需要 [C, H, W]
                    errors.append(f"样本 {idx} 的图像维度不正确: {images.dim()}")
                    continue
                    
                valid_count += 1
            except Exception as e:
                errors.append(f"检查样本 {idx} 时出错: {str(e)}")
                
        # 打印错误信息
        for error in errors:
            print(f"警告: {error}")
            
        # 检查有效样本比例
        valid_ratio = valid_count / min(num_samples, len(dataset))
        print(f"数据集有效性检查: {valid_count}/{min(num_samples, len(dataset))} 个有效样本 ({valid_ratio:.0%})")
        
        return valid_ratio >= 0.5  # 如果至少一半的样本是有效的，则认为数据集有效
    except Exception as e:
        print(f"检查数据集有效性时出错: {e}")
        return False

def seed_worker(worker_id):
    """
    设置数据加载器的worker随机种子
    
    Args:
        worker_id: worker ID
    """
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

# 添加主函数和命令行参数解析
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="fog,snow,stadium,street", 
                        help="逗号分隔的数据集列表")
    parser.add_argument("--model_type", type=str, default="final", 
                        choices=["mem", "final"], help="模型类型")
    parser.add_argument("--memory_size", type=int, default=200, 
                        help="记忆库大小")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="训练批次大小")
    parser.add_argument("--epochs_per_task", type=int, default=5, 
                        help="每个任务的训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.0, 
                        help="权重衰减")
    parser.add_argument("--config", type=str, default="configs/jhu_domains_cl_config.yml", 
                        help="配置文件路径")
    parser.add_argument("--clearml_project", type=str, default="MPCount",
                        help="ClearML项目名称")
    parser.add_argument("--clearml_task", type=str, default="JHU_ContinualLearning_POD",
                        help="ClearML任务名称")
    parser.add_argument("--use_clearml", action="store_true",
                        help="是否使用ClearML进行实验跟踪")
    parser.add_argument("--safe_mode", action="store_true",
                        help="启用安全模式，使用较小的批次大小和更保守的内存设置")
    
    # PODNet特定参数
    parser.add_argument("--pod_flat_factor", type=float, default=1.0,
                        help="POD平面损失权重因子")
    parser.add_argument("--pod_spatial_factor", type=float, default=1.0,
                        help="POD空间损失权重因子")
    parser.add_argument("--save_every_x_epochs", type=int, default=5,
                        help="每隔多少个epoch保存一次模型")
    parser.add_argument("--early_stopping", type=float, default=None,
                        help="早停阈值，如果MAE低于此值则停止训练")
    
    # 记忆库相关参数
    parser.add_argument("--memory_quota_strategy", type=str, default="equal",
                        choices=["equal", "front_heavy", "back_heavy", "custom"],
                        help="记忆配额分配策略")
    parser.add_argument("--custom_quotas", type=str, default="",
                        help="自定义记忆配额，逗号分隔的比例值")
    
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # 初始化记忆配额
    datasets_count = len(args.datasets.split(','))
    if args.memory_quota_strategy == "equal":
        memory_quotas = [args.memory_size // datasets_count] * datasets_count
    elif args.memory_quota_strategy == "front_heavy":
        # 前面的任务获得更多配额
        total = datasets_count * (datasets_count + 1) // 2
        memory_quotas = [int(args.memory_size * (datasets_count - i) / total) for i in range(datasets_count)]
    elif args.memory_quota_strategy == "back_heavy":
        # 后面的任务获得更多配额
        total = datasets_count * (datasets_count + 1) // 2
        memory_quotas = [int(args.memory_size * (i + 1) / total) for i in range(datasets_count)]
    elif args.memory_quota_strategy == "custom" and args.custom_quotas:
        # 使用用户提供的自定义配额
        try:
            quotas = [float(q) for q in args.custom_quotas.split(',')]
            if len(quotas) != datasets_count:
                print(f"警告: 自定义配额数量 ({len(quotas)}) 与数据集数量 ({datasets_count}) 不匹配，使用平均分配")
                memory_quotas = [args.memory_size // datasets_count] * datasets_count
            else:
                # 归一化配额并转换为整数
                total = sum(quotas)
                memory_quotas = [int(args.memory_size * q / total) for q in quotas]
        except ValueError:
            print("警告: 自定义配额格式不正确，使用平均分配")
            memory_quotas = [args.memory_size // datasets_count] * datasets_count
    else:
        # 默认使用平均分配
        memory_quotas = [args.memory_size // datasets_count] * datasets_count
    
    # 确保配额总和等于memory_size
    while sum(memory_quotas) < args.memory_size:
        memory_quotas[0] += 1
    while sum(memory_quotas) > args.memory_size:
        memory_quotas[-1] -= 1
    
    args.memory_quotas = memory_quotas
    
    # 安全模式设置
    if args.safe_mode:
        print("启用安全模式，减小批次大小")
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
    
    # 初始化ClearML
    task_name = join(args.clearml_task, args.model_type, args.datasets.replace(',', '_'))
    if args.use_clearml:
        try:
            clearml_task = Task.init(project_name=args.clearml_project, task_name=task_name)
            clearml_task.connect(args)
        except Exception as e:
            print(f"初始化ClearML失败: {e}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(vars(args))
    print("使用设备:", device)
    
    # 设置随机种子
    master_seed = 42
    
    # 使用PODNet进行训练和评估
    results = train_and_evaluate_pod(cfg, args, device, master_seed)
    
    # 打印最终结果
    print("\n==== 最终结果 ====")
    print(f"平均MAE: {results['avg_final_mae']:.2f}")
    print(f"平均RMSE: {results['avg_final_rmse']:.2f}")
    
    # 打印每个任务的性能
    datasets_list = args.datasets.split(',')
    for i, (dataset, perf) in enumerate(zip(datasets_list, results['final_perf_each_task'])):
        print(f"任务 {i} ({dataset}) - MAE: {perf['mae']:.2f}, RMSE: {perf['rmse']:.2f}")
    
    # 打印遗忘度
    print("\n==== 遗忘度 ====")
    for i, (dataset, forgetting) in enumerate(zip(datasets_list, results['forgetting_vals'])):
        print(f"任务 {i} ({dataset}) - MAE遗忘: {forgetting['mae']:.2f}, RMSE遗忘: {forgetting['rmse']:.2f}")

def train_and_evaluate_pod(model, tasks_train, tasks_val, args, device, datasets_list=None):
    """
    使用PODNet方法训练和评估模型
    
    Args:
        model: 模型
        tasks_train: 训练任务列表
        tasks_val: 验证任务列表
        args: 参数
        device: 设备
        datasets_list: 数据集名称列表
        
    Returns:
        性能和遗忘度指标
    """
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        numpy.random.seed(args.seed)
        random.seed(args.seed)
        generator = torch.Generator()
        generator.manual_seed(args.seed)
    
    # 判断是否使用安全模式
    safe_mode = args.safe_mode if hasattr(args, 'safe_mode') else False
    
    # 初始化记忆库
    memory_per_task = []
    
    # 保存每个任务的最佳性能和最终性能
    best_perf_each_task = []
    final_perf_each_task = []
    
    # 为每个任务分配记忆配额
    memory_quota_per_task = []
    total_memory = args.memory_size
    num_tasks = len(tasks_train)
    
    if args.memory_strategy == 'equal':
        # 平均分配
        for _ in range(num_tasks):
            memory_quota_per_task.append(total_memory // num_tasks)
    elif args.memory_strategy == 'proportional':
        # 按任务数据量比例分配
        task_sizes = []
        for task_id in range(num_tasks):
            if task_id < len(tasks_train) and tasks_train[task_id][0] is not None:
                task_sizes.append(len(tasks_train[task_id][0]))
            else:
                task_sizes.append(0)
        
        total_size = sum(task_sizes)
        for size in task_sizes:
            if total_size > 0:
                memory_quota_per_task.append(int(size / total_size * total_memory))
            else:
                memory_quota_per_task.append(0)
    else:
        # 默认平均分配
        for _ in range(num_tasks):
            memory_quota_per_task.append(total_memory // num_tasks)
    
    # 打印设置信息
    print(f"记忆库总大小: {args.memory_size}")
    print(f"记忆分配策略: {args.memory_strategy}")
    print(f"每个任务的记忆配额: {memory_quota_per_task}")
    print(f"每个任务的训练epochs数: {args.epochs_per_task}")
    
    # 为每个任务训练
    for task_id in range(len(tasks_train)):
        print(f"\n=== 训练任务 {task_id} ({datasets_list[task_id] if datasets_list else task_id}) ===")
        
        # 检查任务是否有效
        if task_id >= len(tasks_train) or not tasks_train[task_id] or not tasks_train[task_id][0]:
            print(f"警告：任务 {task_id} 无效，跳过")
            continue
        
        dataset, collate_fn = tasks_train[task_id]
        
        # 检查数据集是否为空
        if len(dataset) == 0:
            print(f"警告：任务 {task_id} 的数据集为空，跳过")
            continue
        
        # 初始化性能记录
        best_perf_each_task.append({
            'mae': float('inf'),
            'rmse': float('inf')
        })
        
        final_perf_each_task.append({
            'mae': float('inf'),
            'rmse': float('inf')
        })
        
        # 验证数据集
        val_dataset = None
        if task_id < len(tasks_val) and tasks_val[task_id] and tasks_val[task_id][0]:
            val_dataset, _ = tasks_val[task_id]
        
        # 检查数据集有效性
        if not check_dataset_validity(dataset):
            print(f"警告：任务 {task_id} 的训练数据集无效，跳过")
            continue
            
        if val_dataset and not check_dataset_validity(val_dataset, num_samples=2):
            print(f"警告：任务 {task_id} 的验证数据集无效，将不使用验证")
            val_dataset = None
        
        # 创建数据加载器，添加错误处理
        try:
            # 使用custom_collate_fn而非数据集自带的collate_fn
            print(f"为任务 {task_id} 创建数据加载器")
            task_loader = DataLoader(
                dataset, 
                collate_fn=custom_collate_fn,  # 始终使用我们自定义的collate函数
                **{**args.train_loader_args, 'pin_memory': False},
                worker_init_fn=seed_worker if 'seed_worker' in globals() else None,
                generator=generator if 'generator' in locals() else None
            )
            # 打印数据集大小
            print(f"任务 {task_id} 数据集大小: {len(dataset)}")
            
            # 尝试获取一个批次，测试数据加载是否正常
            test_batch = next(iter(task_loader))
            if test_batch is None:
                print(f"警告：任务 {task_id} 测试批次为None，可能存在数据加载问题")
            else:
                print(f"任务 {task_id} 测试批次成功: {[type(item) for item in test_batch] if isinstance(test_batch, tuple) else type(test_batch)}")
        except Exception as e:
            print(f"为任务 {task_id} 创建数据加载器时出错: {e}")
            continue
            
        # 创建验证数据加载器
        val_loader = None
        if val_dataset:
            try:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=custom_collate_fn
                )
                print(f"为任务 {task_id} 创建验证数据加载器，大小: {len(val_dataset)}")
            except Exception as e:
                print(f"为任务 {task_id} 创建验证数据加载器时出错: {e}")
                val_loader = None
        
        # 初始化或重置优化器
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # 设置累积梯度步数
        accumulation_steps = args.accumulation_steps if hasattr(args, 'accumulation_steps') else 0
        
        # 训练循环
        for epoch in range(args.epochs_per_task):
            model.train()
            
            epoch_loss = 0.0
            num_batches = 0
            optimizer.zero_grad()
            
            # 使用tqdm创建进度条
            for batch_idx, batch in enumerate(tqdm(task_loader, desc=f"任务 {task_id} Epoch {epoch+1}")):
                try:
                    # 检查批次是否为None
                    if batch is None:
                        print(f"警告：批次 {batch_idx} 为None，跳过")
                        continue
                        
                    # 检查批次格式
                    if not isinstance(batch, tuple) or len(batch) < 3:
                        print(f"警告：批次 {batch_idx} 格式不正确: {type(batch)}，跳过")
                        continue
                    
                    # 处理输入数据
                    images, imgs2, gt_datas = batch
                    
                    # 检查图像是否为张量
                    if not isinstance(images, torch.Tensor):
                        print(f"警告：批次 {batch_idx} 的图像不是张量，类型={type(images)}，跳过")
                        continue
                        
                    # 移动数据到设备
                    images = images.to(device)
                    imgs2 = imgs2.to(device) if imgs2 is not None else images.to(device)
                    
                    # 处理密度图
                    if isinstance(gt_datas, tuple) and len(gt_datas) > 0:
                        gt_dmaps = gt_datas[1].to(device) if gt_datas[1] is not None else None
                    else:
                        gt_dmaps = gt_datas.to(device) if gt_datas is not None else None
                    
                    if safe_mode:
                        # 检查图像是否含有NaN
                        if torch.isnan(images).any():
                            print(f"警告：批次 {batch_idx} 的图像包含NaN，跳过")
                            continue
                            
                        # 检查密度图是否含有NaN
                        if gt_dmaps is not None and torch.isnan(gt_dmaps).any():
                            print(f"警告：批次 {batch_idx} 的密度图包含NaN，跳过")
                            continue
                    
                    # 正向传播
                    try:
                        # 调整前打印尺寸信息
                        print(f"调整前: images={images.shape}, imgs2={imgs2.shape}, gt_dmaps={gt_dmaps.shape if gt_dmaps is not None else None}")
                        
                        # 确保输入和目标尺寸一致
                        images, gt_dmaps = ensure_same_sizes(images, gt_dmaps, target_size=(80, 80), device=device)
                        imgs2, _ = ensure_same_sizes(imgs2, None, target_size=(80, 80), device=device)
                        
                        print(f"调整后: images={images.shape}, imgs2={imgs2.shape}, gt_dmaps={gt_dmaps.shape if gt_dmaps is not None else None}")
                        
                        # 调用模型
                        try:
                            # 模仿main_cl_jhu_ac.py中的方式
                            loss = get_loss()
                            dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = model.forward_train(images, imgs2, gt_dmaps)
                            # 计算损失
                            log_para = 1000  # 默认log_para值
                            gt_datas_tuple = (None, gt_dmaps, None)  # 创建与compute_count_loss兼容的格式
                            loss_den = F.mse_loss(dmaps1, gt_dmaps) + F.mse_loss(dmaps2, gt_dmaps)
                            loss_cls = F.binary_cross_entropy(cmaps1, gt_dmaps) + F.binary_cross_entropy(cmaps2, gt_dmaps)
                            loss_total = loss_den + 10 * loss_cls + 10 * loss_con  # 模仿main_cl_jhu_ac.py中的损失计算
                        except Exception as e:
                            print(f"前向传播或损失计算时出错: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                        
                        if torch.isnan(loss_total):
                            print("警告：损失为NaN，跳过此批次")
                            continue
                        
                        # 使用梯度累积
                        if accumulation_steps > 0:
                            loss_total = loss_total / accumulation_steps
                            loss_total.backward()
                            
                            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(task_loader):
                                optimizer.step()
                                optimizer.zero_grad()
                        else:
                            loss_total.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                        
                        epoch_loss += loss_total.item()
                        num_batches += 1
                    except Exception as e:
                        print(f"处理批数据时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                except Exception as e:
                    print(f"处理批次时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 计算平均损失
            if num_batches > 0:
                epoch_loss /= num_batches
                # 更新最佳性能
                best_perf_each_task[task_id] = {
                    'mae': min(best_perf_each_task[task_id]['mae'], epoch_loss),
                    'rmse': min(best_perf_each_task[task_id]['rmse'], math.sqrt(epoch_loss))
                }
                
                # 更新最终性能
                final_perf_each_task[task_id] = {
                    'mae': epoch_loss,
                    'rmse': math.sqrt(epoch_loss)
                }
                
                # 打印训练结果
                print(f"任务 {task_id} Epoch {epoch+1} 损失: {epoch_loss:.4f}")
            else:
                print(f"警告：任务 {task_id} Epoch {epoch+1} 没有成功处理任何批次")
            
            # 评估在验证集上的性能
            if val_loader is not None:
                val_perf = evaluate_performance(model, val_dataset, device)
                print(f"任务 {task_id} Epoch {epoch+1} 验证 MAE: {val_perf['mae']:.2f}, RMSE: {val_perf['rmse']:.2f}")
                
                # 更新最佳性能
                best_perf_each_task[task_id] = {
                    'mae': min(best_perf_each_task[task_id]['mae'], val_perf['mae']),
                    'rmse': min(best_perf_each_task[task_id]['rmse'], val_perf['rmse'])
                }
                
                # 更新最终性能
                final_perf_each_task[task_id] = {
                    'mae': val_perf['mae'],
                    'rmse': val_perf['rmse']
                }
                
            # 保存模型
            if hasattr(args, 'save_every_x_epochs') and epoch % args.save_every_x_epochs == 0:
                try:
                    os.makedirs("checkpoints/pod", exist_ok=True)
                    save_model(model, f"checkpoints/pod/task_{task_id}_epoch_{epoch+1}.pth")
                except Exception as e:
                    print(f"保存模型时出错: {e}")
            
            # 检查早停
            if num_batches > 0 and hasattr(args, 'early_stopping') and check_early_stopping(best_perf_each_task[task_id], args.early_stopping):
                print(f"任务 {task_id} 达到早停条件，停止训练")
                break
                
            # 随机选择样本添加到记忆库
            if num_batches > 0 and epoch == args.epochs_per_task - 1:  # 最后一个epoch
                try:
                    # 初始化当前任务的记忆库
                    if task_id >= len(memory_per_task):
                        memory_per_task.append([])
                        
                    # 确定要选择的样本数
                    num_samples_to_select = min(memory_quota_per_task[task_id], len(dataset))
                    
                    # 随机选择样本
                    indices = torch.randperm(len(dataset))[:num_samples_to_select]
                    
                    # 添加到记忆库
                    for idx in indices:
                        try:
                            sample = dataset[idx]
                            if sample is not None:
                                memory_per_task[task_id].append(sample)
                        except Exception as e:
                            print(f"添加样本到记忆库时出错: {e}")
                            
                    print(f"为任务 {task_id} 添加了 {len(memory_per_task[task_id])} 个样本到记忆库")
                except Exception as e:
                    print(f"更新记忆库时出错: {e}")
        
        # 打印任务结束后的性能
        if num_batches > 0:
            print(f"任务 {task_id} 最佳性能: MAE={best_perf_each_task[task_id]['mae']:.2f}, RMSE={best_perf_each_task[task_id]['rmse']:.2f}")
            print(f"任务 {task_id} 最终性能: MAE={final_perf_each_task[task_id]['mae']:.2f}, RMSE={final_perf_each_task[task_id]['rmse']:.2f}")
        else:
            print(f"警告：任务 {task_id} 没有成功训练")

    # 计算平均性能
    avg_final_mae = sum(perf['mae'] for perf in final_perf_each_task) / len(final_perf_each_task) if final_perf_each_task else float('inf')
    avg_final_rmse = sum(perf['rmse'] for perf in final_perf_each_task) / len(final_perf_each_task) if final_perf_each_task else float('inf')

    # 计算遗忘度
    forgetting_vals = []
    for task_id in range(len(tasks_train)):
        if task_id < len(best_perf_each_task) and task_id < len(final_perf_each_task):
            mae_forgetting = best_perf_each_task[task_id]['mae'] - final_perf_each_task[task_id]['mae']
            rmse_forgetting = best_perf_each_task[task_id]['rmse'] - final_perf_each_task[task_id]['rmse']
            forgetting_vals.append({
                'mae': mae_forgetting,
                'rmse': rmse_forgetting
            })

    # 返回结果
    return {
        'final_perf_each_task': final_perf_each_task,
        'best_perf_each_task': best_perf_each_task,
        'forgetting_vals': forgetting_vals,
        'avg_final_mae': avg_final_mae,
        'avg_final_rmse': avg_final_rmse
    }

if __name__ == "__main__":
    main()