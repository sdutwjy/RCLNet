import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class SamplingActor(nn.Module):
    """
    Actor网络：根据样本特征输出采样概率
    """
    def __init__(self, feature_dim=256, hidden_dim=128):
        super(SamplingActor, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出0-1之间的概率值
        )
        
    def forward(self, features):
        """
        输入：样本特征
        输出：采样概率
        """
        x = self.feature_extractor(features)
        probs = self.policy_head(x)
        return probs
    
    def get_action_probs(self, features):
        """
        获取采样动作的概率分布
        """
        with torch.no_grad():
            probs = self.forward(features)
        return probs
    
    def sample_action(self, features):
        """
        根据概率分布采样动作
        """
        probs = self.get_action_probs(features)
        m = Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action)

class SamplingCritic(nn.Module):
    """
    Critic网络：评估样本对训练的价值
    """
    def __init__(self, feature_dim=256, hidden_dim=128):
        super(SamplingCritic, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值头
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, features):
        """
        输入：样本特征
        输出：样本价值估计
        """
        x = self.feature_extractor(features)
        value = self.value_head(x)
        return value

class MemoryFeatureExtractor(nn.Module):
    """
    从记忆样本中提取特征
    """
    def __init__(self, input_channels=3, feature_dim=256):
        super(MemoryFeatureExtractor, self).__init__()
        self.feature_dim = feature_dim
        
        # 使用简化的CNN提取图像特征，增大步长减少内存使用
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=4, padding=2),  # 增大步长和核大小
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 添加池化层进一步减少特征图大小
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化到1x1
        )
        
        # 简化的密度图特征提取
        self.density_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=4, padding=2),  # 减少通道数，增大步长
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 元数据特征提取（如count值）
        self.meta_encoder = nn.Sequential(
            nn.Linear(1, 16),  # 减少特征维度
            nn.ReLU()
        )
        
        # 融合所有特征
        self.fusion = nn.Sequential(
            nn.Linear(64 + 16 + 16, feature_dim),  # 调整输入维度
            nn.ReLU()
        )
        
    def forward(self, img, density_map, meta_data):
        """
        提取样本特征
        
        Args:
            img: 图像张量 [B, C, H, W]
            density_map: 密度图 [B, 1, H, W]
            meta_data: 元数据 [B, 1] (例如count值)
            
        Returns:
            特征向量 [B, feature_dim]
        """
        # 提取图像特征
        img_features = self.cnn(img).flatten(1)
        
        # 提取密度图特征
        density_features = self.density_encoder(density_map).flatten(1)
        
        # 提取元数据特征
        meta_features = self.meta_encoder(meta_data)
        
        # 融合特征
        combined = torch.cat([img_features, density_features, meta_features], dim=1)
        features = self.fusion(combined)
        
        return features

class ActorCriticMemorySampler:
    """
    Actor-Critic框架的记忆采样器
    """
    def __init__(self, 
                 feature_dim=256, 
                 hidden_dim=128, 
                 actor_lr=1e-4, 
                 critic_lr=1e-4,
                 gamma=0.99,
                 device='cuda'):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.device = device
        
        # 特征提取器
        self.feature_extractor = MemoryFeatureExtractor(
            input_channels=3, 
            feature_dim=feature_dim
        ).to(device)
        
        # Actor和Critic网络
        self.actor = SamplingActor(feature_dim, hidden_dim).to(device)
        self.critic = SamplingCritic(feature_dim, hidden_dim).to(device)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.actor.parameters()), 
            lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.critic.parameters()), 
            lr=critic_lr
        )
        
        # 记录训练数据
        self.saved_log_probs = []
        self.rewards = []
        self.values = []
        
    def extract_features_batch(self, memory_buffer):
        """
        从记忆缓冲区批量提取特征
        """
        features_list = []
        
        for sample in memory_buffer:
            img, density_map, meta = sample
            
            # 确保数据格式正确
            if isinstance(img, torch.Tensor):
                img = img.unsqueeze(0).to(self.device)  # 添加批次维度
            
            if isinstance(density_map, torch.Tensor):
                density_map = density_map.unsqueeze(0).to(self.device)
            
            # 从meta中提取count值
            if isinstance(meta, dict) and 'count' in meta:
                count = meta['count']
                if isinstance(count, torch.Tensor):
                    count = count.unsqueeze(0).to(self.device)
                else:
                    count = torch.tensor([[float(count)]]).to(self.device)
            else:
                count = torch.tensor([[0.0]]).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                feature = self.feature_extractor(img, density_map, count)
                features_list.append(feature)
        
        if features_list:
            return torch.cat(features_list, dim=0)
        else:
            return torch.tensor([]).to(self.device)
    
    def get_sampling_probs(self, memory_buffer):
        """
        获取记忆缓冲区中所有样本的采样概率
        """
        if not memory_buffer:
            return None
        
        # 分批处理记忆缓冲区
        batch_size = 10  # 每批处理10个样本
        all_probs = []
        
        for i in range(0, len(memory_buffer), batch_size):
            # 获取当前批次
            batch = memory_buffer[i:min(i+batch_size, len(memory_buffer))]
            
            # 提取特征
            features_list = []
            for sample in batch:
                img, density_map, meta = sample
                
                # 确保数据格式正确
                if isinstance(img, torch.Tensor):
                    img = img.unsqueeze(0).to(self.device)  # 添加批次维度
                
                if isinstance(density_map, torch.Tensor):
                    density_map = density_map.unsqueeze(0).to(self.device)
                
                # 从meta中提取count值
                try:
                    if isinstance(meta, dict) and 'count' in meta:
                        count = meta['count']
                        if isinstance(count, torch.Tensor):
                            count = count.unsqueeze(0).to(self.device)
                        else:
                            count = torch.tensor([[float(count)]]).to(self.device)
                    else:
                        count = torch.tensor([[0.0]]).to(self.device)
                except (ValueError, TypeError) as e:
                    # 如果无法转换为float，使用默认值
                    print(f"警告: 无法处理count值: {e}, 使用默认值0.0")
                    count = torch.tensor([[0.0]]).to(self.device)
                
                # 提取特征
                with torch.no_grad():
                    feature = self.feature_extractor(img, density_map, count)
                    features_list.append(feature)
                    
                # 立即清理不需要的GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if features_list:
                features = torch.cat(features_list, dim=0)
                with torch.no_grad():
                    probs = self.actor(features)
                all_probs.append(probs.cpu().numpy())
            
            # 批次处理完成后清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 合并所有批次的结果
        if all_probs:
            return np.concatenate(all_probs).flatten()
        else:
            return None
    
    def get_sample_value(self, sample):
        """
        获取单个样本的价值估计
        
        Args:
            sample: (img, density_map, meta) 三元组
            
        Returns:
            样本价值 (标量)
        """
        try:
            img, density_map, meta = sample
            
            # 确保数据格式正确
            if isinstance(img, torch.Tensor):
                img = img.unsqueeze(0).to(self.device)  # 添加批次维度
            
            if isinstance(density_map, torch.Tensor):
                density_map = density_map.unsqueeze(0).to(self.device)
            
            # 从meta中提取count值
            if isinstance(meta, dict) and 'count' in meta:
                count = meta['count']
                if isinstance(count, torch.Tensor):
                    count = count.unsqueeze(0).to(self.device)
                else:
                    count = torch.tensor([[float(count)]]).to(self.device)
            else:
                count = torch.tensor([[0.0]]).to(self.device)
            
            # 提取特征并获取价值
            with torch.no_grad():
                feature = self.feature_extractor(img, density_map, count)
                value = self.critic(feature)
                return value.item()
        except Exception as e:
            print(f"获取样本价值时出错: {e}")
            return 0.0  # 默认价值
    
    def update_stats(self, dataset_id, sample_idx, value):
        """
        更新样本使用统计信息
        
        Args:
            dataset_id: 数据集ID
            sample_idx: 样本索引
            value: 样本价值
        """
        # 记录样本使用情况，可用于后续分析
        # 当前版本简单实现，不做具体操作
        pass
    
    def select_action(self, memory_buffer, idx):
        """
        为特定样本选择动作（是否采样）
        """
        if idx >= len(memory_buffer):
            return None, None
        
        sample = memory_buffer[idx]
        img, density_map, meta = sample
        
        # 确保数据格式正确
        if isinstance(img, torch.Tensor):
            img = img.unsqueeze(0).to(self.device)
        
        if isinstance(density_map, torch.Tensor):
            density_map = density_map.unsqueeze(0).to(self.device)
        
        # 从meta中提取count值，添加错误处理
        try:
            if isinstance(meta, dict) and 'count' in meta:
                count = meta['count']
                if isinstance(count, torch.Tensor):
                    count = count.unsqueeze(0).to(self.device)
                elif count is None:
                    # 处理None值
                    count = torch.tensor([[0.0]]).to(self.device)
                else:
                    try:
                        # 尝试转换为float
                        count = torch.tensor([[float(count)]]).to(self.device)
                    except (ValueError, TypeError):
                        # 如果无法转换，使用默认值
                        print(f"警告: 无法将count值 '{count}' 转换为float，使用默认值0.0")
                        count = torch.tensor([[0.0]]).to(self.device)
            else:
                count = torch.tensor([[0.0]]).to(self.device)
        except Exception as e:
            print(f"处理count值时出错: {e}，使用默认值0.0")
            count = torch.tensor([[0.0]]).to(self.device)
        
        # 提取特征
        feature = self.feature_extractor(img, density_map, count)
        
        # 获取动作概率和价值
        prob = self.actor(feature)
        value = self.critic(feature)
        
        # 记录概率和价值
        self.saved_log_probs.append(prob.log())
        self.values.append(value)
        
        return prob.item(), value.item()
    
    def update_policy(self, final_reward):
        """
        使用策略梯度更新Actor和Critic网络
        
        Args:
            final_reward: 最终奖励（例如MAE改善程度）
        """
        # 计算折扣奖励
        rewards = []
        R = final_reward
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards, device=self.device)
        
        # 标准化奖励
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # 计算损失
        policy_loss = []
        value_loss = []
        for log_prob, value, reward in zip(self.saved_log_probs, self.values, rewards):
            advantage = reward - value.item()
            policy_loss.append(-log_prob * advantage)  # 策略梯度
            value_loss.append(F.mse_loss(value, torch.tensor([reward], device=self.device)))
        
        # 优化Actor
        self.actor_optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # 优化Critic
        self.critic_optimizer.zero_grad()
        value_loss = torch.cat(value_loss).sum()
        value_loss.backward()
        self.critic_optimizer.step()
        
        # 清除记录
        self.saved_log_probs = []
        self.rewards = []
        self.values = []
        
    def add_reward(self, reward):
        """
        添加奖励
        """
        self.rewards.append(reward)
    
    def save_model(self, path):
        """
        保存模型
        """
        torch.save({
            'feature_extractor': self.feature_extractor.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)
    
    def load_model(self, path):
        """
        加载模型
        """
        checkpoint = torch.load(path)
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer']) 