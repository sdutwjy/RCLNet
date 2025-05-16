import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

def create_teacher_model(model):
    """
    创建教师模型（原始模型的深拷贝）
    
    参数:
        model: 当前模型
        
    返回:
        教师模型的深拷贝
    """
    teacher_model = copy.deepcopy(model)
    # 冻结教师模型的参数
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()  # 设置为评估模式
    return teacher_model

def pod_feature_distillation(current_features, teacher_features, collapse_channels="spatial", normalize=True):
    """
    实现PODNet中的特征蒸馀方法
    
    参数:
        current_features: 当前模型的特征图
        teacher_features: 教师模型的特征图
        collapse_channels: 如何聚合通道信息 ["spatial", "channels", "width", "height", "gap"]
        normalize: 是否归一化特征
        
    返回:
        特征蒸馀损失
    """
    assert current_features.shape == teacher_features.shape, f"特征形状不匹配: {current_features.shape} vs {teacher_features.shape}"
    
    # 计算特征的平方值
    current_features = torch.pow(current_features, 2)
    teacher_features = torch.pow(teacher_features, 2)
    
    if collapse_channels == "channels":
        # 按通道聚合（b, n, w, h）-> (b, w*h)
        current_features = current_features.sum(dim=1).view(current_features.shape[0], -1)
        teacher_features = teacher_features.sum(dim=1).view(teacher_features.shape[0], -1)
    elif collapse_channels == "width":
        # 按宽度聚合 (b, n, w, h) -> (b, n*h)
        current_features = current_features.sum(dim=2).view(current_features.shape[0], -1)
        teacher_features = teacher_features.sum(dim=2).view(teacher_features.shape[0], -1)
    elif collapse_channels == "height":
        # 按高度聚合 (b, n, w, h) -> (b, n*w)
        current_features = current_features.sum(dim=3).view(current_features.shape[0], -1)
        teacher_features = teacher_features.sum(dim=3).view(teacher_features.shape[0], -1)
    elif collapse_channels == "gap":
        # 全局平均池化 (b, n, w, h) -> (b, n)
        current_features = F.adaptive_avg_pool2d(current_features, (1, 1))[..., 0, 0]
        teacher_features = F.adaptive_avg_pool2d(teacher_features, (1, 1))[..., 0, 0]
    elif collapse_channels == "spatial":
        # 空间聚合（保留宽度和高度信息）
        current_h = current_features.sum(dim=3).view(current_features.shape[0], -1)  # (b, n*w)
        teacher_h = teacher_features.sum(dim=3).view(teacher_features.shape[0], -1)
        current_w = current_features.sum(dim=2).view(current_features.shape[0], -1)  # (b, n*h)
        teacher_w = teacher_features.sum(dim=2).view(teacher_features.shape[0], -1)
        current_features = torch.cat([current_h, current_w], dim=-1)
        teacher_features = torch.cat([teacher_h, teacher_w], dim=-1)
    else:
        raise ValueError(f"未知的聚合方法: {collapse_channels}")
    
    if normalize:
        current_features = F.normalize(current_features, dim=1, p=2)
        teacher_features = F.normalize(teacher_features, dim=1, p=2)
    
    # 计算Frobenius范数距离
    distillation_loss = torch.mean(torch.frobenius_norm(current_features - teacher_features, dim=-1))
    
    return distillation_loss

def perceptual_feature_distillation(current_features, teacher_features, factor=1.0):
    """
    感知特征重建蒸馀方法
    
    参数:
        current_features: 当前模型的特征图
        teacher_features: 教师模型的特征图
        factor: 缩放因子
        
    返回:
        感知特征蒸馀损失
    """
    bs, c, w, h = current_features.shape
    
    # 将特征展平为 (b, c*w*h)
    current_features = current_features.view(bs, -1)
    teacher_features = teacher_features.view(bs, -1)
    
    # 归一化特征
    current_features = F.normalize(current_features, p=2, dim=-1)
    teacher_features = F.normalize(teacher_features, p=2, dim=-1)
    
    # 计算成对距离的平方
    layer_loss = (F.pairwise_distance(current_features, teacher_features, p=2)**2) / (c * w * h)
    
    return factor * torch.mean(layer_loss)

def perceptual_style_distillation(current_features, teacher_features, factor=1.0):
    """
    感知风格重建蒸馀方法（类似于风格转移中的Gram矩阵）
    
    参数:
        current_features: 当前模型的特征图
        teacher_features: 教师模型的特征图
        factor: 缩放因子
        
    返回:
        风格蒸馀损失
    """
    bs, c, w, h = current_features.shape
    
    current_features = current_features.view(bs, c, w * h)
    teacher_features = teacher_features.view(bs, c, w * h)
    
    # 计算Gram矩阵
    gram_current = torch.bmm(current_features, current_features.transpose(2, 1)) / (c * w * h)
    gram_teacher = torch.bmm(teacher_features, teacher_features.transpose(2, 1)) / (c * w * h)
    
    # 计算Frobenius范数的平方
    layer_loss = torch.frobenius_norm(gram_current - gram_teacher, dim=(1, 2))**2
    
    return factor * torch.mean(layer_loss)

def embeddings_similarity(current_features, teacher_features):
    """
    基于嵌入相似性的蒸馀方法（UCIR中使用）
    
    参数:
        current_features: 当前模型的特征
        teacher_features: 教师模型的特征
        
    返回:
        嵌入相似性损失
    """
    return torch.mean(torch.frobenius_norm(
        F.normalize(current_features, dim=1, p=2) - 
        F.normalize(teacher_features, dim=1, p=2), 
        dim=1
    ))

def combine_losses(current_task_loss, distillation_losses, alpha=0.5, scheduled_factor=False, n_classes=None, task_size=None):
    """
    组合当前任务损失和蒸馀损失
    
    参数:
        current_task_loss: 当前任务的分类/回归损失
        distillation_losses: 蒸馀损失字典 (例如: {'pod': pod_loss, 'perceptual': perceptual_loss})
        alpha: 权重系数
        scheduled_factor: 是否使用调度因子（随着任务增加调整权重）
        n_classes: 当前类别总数
        task_size: 每个任务的类别数
        
    返回:
        组合损失和损失指标字典
    """
    total_distill_loss = 0.0
    metrics = {}
    
    for name, loss in distillation_losses.items():
        # 如果使用调度因子，随着类别增加动态调整权重
        if scheduled_factor and n_classes is not None and task_size is not None:
            factor = alpha * math.sqrt(n_classes / task_size)
        else:
            factor = alpha
            
        weighted_loss = factor * loss
        total_distill_loss += weighted_loss
        metrics[name] = loss.item()
    
    # 组合当前任务损失和蒸馀损失
    total_loss = current_task_loss + total_distill_loss
    
    return total_loss, metrics 