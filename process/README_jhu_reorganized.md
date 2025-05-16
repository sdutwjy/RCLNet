# JHU数据集域重组工具

这个工具用于按照提供的域文件列表重新组织JHU数据集，将图像按域（street、stadium、snow、fog）和数据集分割（train、val）整理到新的目录结构中。同时处理与图像对应的密度图文件（例如：0002.jpg -> 0002_dmap.npy）。

## 功能

- 支持将图像按不同域和分割（训练集/验证集）重新组织
- 支持通过符号链接或复制方式重组数据集
- 自动处理原始数据集中不存在的图像
- 自动处理与每张图像对应的密度图文件（_dmap.npy格式）

## 使用方法

### 基本用法

```bash
# 默认使用 jhu_domains 目录中的域文件，输出到 jhu_reorganized 目录（复制文件）
python reorganize_jhu_dataset.py

# 使用符号链接而不是复制文件（节省存储空间）
python reorganize_jhu_dataset.py --use_symlinks

# 指定域文件目录和输出目录
python reorganize_jhu_dataset.py --domains_dir /path/to/domains --output_dir /path/to/output

# 不包含密度图文件（只处理图像文件）
python reorganize_jhu_dataset.py --without_dmap
```

### 参数说明

- `--domains_dir`: 包含域文件的目录路径（默认：jhu_domains）
- `--output_dir`: 重组数据集的输出目录（默认：jhu_reorganized）
- `--use_symlinks`: 使用符号链接而不是复制文件（默认：不使用，即复制文件）
- `--without_dmap`: 不包含密度图文件（默认：包含密度图文件）

## 输出目录结构

脚本执行后将生成以下目录结构：

```
jhu_reorganized/
├── street/
│   ├── train/
│   │   ├── 0001.jpg
│   │   ├── 0001_dmap.npy
│   │   ├── 0002.jpg
│   │   ├── 0002_dmap.npy
│   │   └── ...
│   └── val/
│       ├── 1001.jpg
│       ├── 1001_dmap.npy
│       └── ...
├── stadium/
│   ├── train/
│   └── val/
├── snow/
│   ├── train/
│   └── val/
└── fog/
    ├── train/
    └── val/
```

## 密度图处理

脚本会自动寻找与每张图像对应的密度图文件。密度图文件的命名规则为：
- 例如图像路径为 `/path/to/0001.jpg`
- 对应的密度图应为 `/path/to/0001_dmap.npy`

如果密度图文件不存在，脚本会输出警告信息并继续处理其他文件。

## 注意事项

- 如果原始图像或密度图路径不存在，脚本会输出警告信息但继续处理其他文件
- 如果输出目录中已存在同名文件，将被覆盖
- 使用符号链接模式时，如果原始图像或密度图被删除或移动，符号链接将失效 