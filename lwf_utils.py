import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def distillation_loss(current_outputs, teacher_outputs, temperature=2.0):
    """
    计算知识蒸馏损失（KL散度）
    
    参数:
        current_outputs: 当前模型的输出
        teacher_outputs: 教师模型的输出
        temperature: 控制软标签平滑度的温度参数
    
    返回:
        蒸馏损失
    """
    # 对于密度图预测，我们可以直接使用MSE损失
    if isinstance(current_outputs, torch.Tensor) and isinstance(teacher_outputs, torch.Tensor):
        if current_outputs.size() == teacher_outputs.size():
            # 普通的密度图情况 - 直接用MSE
            return F.mse_loss(current_outputs, teacher_outputs)
    
    # 对于分类输出，使用KL散度
    if isinstance(current_outputs, (list, tuple)) and isinstance(teacher_outputs, (list, tuple)):
        distill_loss = 0
        for y, teacher_y in zip(current_outputs, teacher_outputs):
            if isinstance(y, torch.Tensor) and isinstance(teacher_y, torch.Tensor):
                if len(y.size()) > 1 and y.size(1) > 1:  # 分类情况
                    # 转换为概率分布
                    soft_y = F.log_softmax(y / temperature, dim=1)
                    soft_teacher = F.softmax(teacher_y / temperature, dim=1)
                    # 计算KL散度
                    curr_loss = F.kl_div(soft_y, soft_teacher, reduction='batchmean') * (temperature ** 2)
                    distill_loss += curr_loss
                else:  # 回归情况
                    curr_loss = F.mse_loss(y, teacher_y)
                    distill_loss += curr_loss
        return distill_loss
    
    # 默认情况
    return torch.tensor(0.0, device=current_outputs.device if isinstance(current_outputs, torch.Tensor) else 'cpu')

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

def combine_losses(classification_loss, distill_loss, alpha=0.5):
    """
    组合分类损失和蒸馏损失
    
    参数:
        classification_loss: 当前任务的分类损失
        distill_loss: 知识蒸馏损失
        alpha: 权重系数，控制两种损失的比例
        
    返回:
        组合损失
    """
    return classification_loss + alpha * distill_loss 