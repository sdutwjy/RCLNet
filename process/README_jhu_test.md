# JHU域数据集测试集创建与使用指南

本文档说明如何从JHU域数据集的训练集中创建测试集，并解释如何在持续学习框架中使用这些测试集。

## 1. 测试集创建

我们提供了两个脚本来从各个域（fog、snow、stadium、street）的训练集中随机抽取图片作为测试集：
- `create_jhu_test.py`：基础脚本，创建测试集
- `create_test_dataset.py`：增强脚本，创建并验证测试集

### 使用方法

```bash
# 使用默认参数运行（从每个域的训练集中移动30-40张图片作为测试集）
python create_jhu_test.py

# 在移动前创建备份
python create_jhu_test.py --backup

# 只处理特定域
python create_jhu_test.py --domain fog

# 自定义参数
python create_jhu_test.py --test_ratio 0.25 --min_samples 25 --max_samples 35 --seed 123
```

### 参数说明

- `--jhu_root`: JHU重组数据集的根目录 (默认: `/scratch/jianyong/MPCount/data/jhu_reorganized`)
- `--test_ratio`: 从训练集中抽取作为测试集的比例 (默认: 0.3)
- `--min_samples`: 每个域最少抽取的样本数 (默认: 30)
- `--max_samples`: 每个域最多抽取的样本数 (默认: 40)
- `--seed`: 随机种子，确保结果可重现 (默认: 42)
- `--domain`: 只处理指定域，如fog、snow、stadium、street (默认处理所有域)
- `--backup`: 在移动文件前创建数据备份，备份保存在`/scratch/jianyong/MPCount/data/jhu_backup`

### 增强版脚本额外参数

`create_test_dataset.py`增强版脚本提供以下额外参数：

- `--skip_create`: 跳过创建步骤，仅验证现有测试集

### 运行结果

脚本执行后会：

1. 如果指定了`--backup`，会在移动前创建训练集的备份
2. 在每个域目录下创建`test`子目录
3. 将选定的图片文件(`.jpg`)及其对应的密度图文件(`.npy`)从`train`目录**移动**到`test`目录
4. 在`/home/jianyong/exp/MPCount/jhu_domains/`下生成测试集列表文件：
   - `jhu_fog_test.txt`
   - `jhu_snow_test.txt`
   - `jhu_stadium_test.txt`
   - `jhu_street_test.txt`
5. 对于增强版脚本，会验证创建的测试集是否可以正确加载

## 2. 在持续学习框架中使用测试集

要在`main_cl_jhu_domains.py`中使用新创建的测试集，需要对代码进行一些修改。

### 修改create_jhu_domain_tasks函数

在`main_cl_jhu_domains.py`中，修改`create_jhu_domain_tasks`函数，使其加载测试集而不是使用验证集作为测试集：

```python
def create_jhu_domain_tasks(cfg):
    """创建JHU域任务，每个域（street, stadium, snow, fog）视为一个子任务"""
    tasks_train = []
    tasks_val = []
    tasks_test = []
    
    # 获取JHU域数据集的根路径
    jhu_root = cfg.get('jhu_root', '/scratch/jianyong/MPCount/data/jhu_reorganized')
    
    # 所有域名称
    domains = ['street', 'stadium', 'snow', 'fog']
    domain_datasets = cfg['jhu_domains']
    
    # 从配置获取数据集参数
    base_params = domain_datasets['params'].copy() if 'params' in domain_datasets else {}
    
    for domain in domains:
        # 为每个域创建特定的参数
        domain_params = base_params.copy()
        domain_params['domain'] = domain
        domain_params['root'] = jhu_root  # 直接使用根目录，不再额外拼接域名
        
        # 创建训练集
        train_dataset, collate = get_dataset(
            domain_datasets['name'],
            {**domain_params, 'split': 'train'},
            method='train'
        )
        
        # 创建验证集
        val_dataset, _ = get_dataset(
            domain_datasets['name'],
            {**domain_params, 'split': 'val'},
            method='val'
        )
        
        # 使用新创建的测试集，而不是使用验证集作为测试集
        test_dataset, _ = get_dataset(
            domain_datasets['name'],
            {**domain_params, 'split': 'test'},  # 改为'test'
            method='test'
        )
        
        # 包装为IndexedDataset
        indexed_train = IndexedDataset(train_dataset)
        indexed_val = IndexedDataset(val_dataset)
        indexed_test = IndexedDataset(test_dataset)
        
        tasks_train.append((indexed_train, collate))
        tasks_val.append((indexed_val, collate))
        tasks_test.append((indexed_test, collate))
        
        print(f"已创建{domain}域数据集 - train:{len(indexed_train)}, val:{len(indexed_val)}, test:{len(indexed_test)}")
    
    return tasks_train, tasks_val, tasks_test
```

### 修改JHUDomainDataset和JHUDomainClsDataset类

需要确保数据集类支持`test`分割。检查这些类中的`__init__`方法，确认它能够处理`split='test'`的情况：

```python
def __init__(self, root="/scratch/jianyong/MPCount/data/jhu_reorganized",
             domain="street", split="train", scale=0.5, downsample=4,
             augmentation=True, method='train', **kwargs):
    # ...
    
    # 确保split参数可以处理'test'
    assert split in ['train', 'val', 'test'], f"不支持的分割类型: {split}"
    
    # 其余的初始化代码...
```

这些修改后，`main_cl_jhu_domains.py`将能够使用新创建的测试集进行更准确的性能评估。

## 3. 注意事项

1. **重要：这个脚本会将选定的图片和密度图从train目录移动到test目录，这个操作不可逆！强烈建议使用`--backup`参数在移动前创建备份**
2. 测试集的路径格式与原始的train和val文本文件保持一致，这样可以兼容现有代码
3. 测试集中的图片是从训练集中选择的，所以在使用时要注意避免数据泄露（即不要用测试集的图片进行训练）

## 4. 常见问题

**Q: 为什么从训练集中抽取测试集而不是从验证集中抽取？**

A: 训练集通常比验证集大，从中抽取更能确保测试集的多样性。同时，这样做可以保持验证集的完整性，用于模型选择。

**Q: 是否需要重新生成密度图？**

A: 不需要。脚本会移动原始的密度图文件（.npy文件），这些文件已经预先生成好了。

**Q: 如果要进行多次实验，每次是否都需要重新创建测试集？**

A: 不需要。通过设置固定的随机种子（--seed参数），可以确保每次生成相同的测试集，便于结果的可重复性。

**Q: 如何恢复原始数据结构？**

A: 如果使用了`--backup`参数，可以从备份目录（`/scratch/jianyong/MPCount/data/jhu_backup`）中恢复原始数据。 