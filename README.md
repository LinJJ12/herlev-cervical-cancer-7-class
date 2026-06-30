# 宫颈涂片细胞分类 - CNN模型

## 项目结构

```
实训任务3/
├── cervical_cancer_cnn.py             - 原始CNN模型代码
├── cervical_cancer_cnn_improved.py    - 改进版
├── cervical_cancer_cnn_final.py       - 最终优化版（需下载预训练权重）
├── cervical_cancer_cnn_no_pretrain.py - 【推荐！无需预训练权重】
├── requirements.txt                    - Python依赖包列表
├── 实训报告.md                         - 实训报告文档
├── README.md                           - 本文件
├── 实训任务3数据集/                    - 数据集文件夹
│   ├── normal_superficiel/             - 类别1：浅表鳞状上皮
│   ├── normal_intermediate/            - 类别2：中度鳞状上皮
│   ├── normal_columnar/                - 类别3：柱状上皮
│   ├── light_dysplastic/               - 类别4：轻度发育不良
│   ├── moderate_dysplastic/            - 类别5：中度发育不良
│   ├── severe_dysplastic/              - 类别6：重度发育不良
│   └── carcinoma_in_situ/              - 类别7：原位癌
└── 实训任务3数据集.zip                 - 数据集压缩包
```

## 快速开始

### 1. 激活conda环境
```bash
conda activate crawler
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行【无需预训练版】（强烈推荐！）
```bash
python cervical_cancer_cnn_no_pretrain.py
```

### 4. 运行改进版
```bash
python cervical_cancer_cnn_improved.py
```

### 5. 运行原始模型
```bash
python cervical_cancer_cnn.py
```

## ⚠️ 网络问题说明

您遇到的错误是**无法下载预训练权重**，这是网络连接问题导致的。

### 解决方案：

#### ✅ **推荐方案：使用无需预训练版本**
直接运行 `cervical_cancer_cnn_no_pretrain.py` - 这个版本：
- 自定义ResNet架构
- 从头训练，无需下载任何东西
- 保持所有优化策略

#### 🔧 备选方案：手动下载预训练权重
如果您确实想用预训练版本，可以：
1. 手动从 https://download.pytorch.org/models/resnet50-0676ba61.pth 下载
2. 放到 `C:\Users\someo/.cache\torch\hub\checkpoints\` 目录
3. 然后运行 `cervical_cancer_cnn_final.py`

## 无需预训练版特点

`cervical_cancer_cnn_no_pretrain.py` 核心优化：

### 1. **自定义ResNet架构**
- 类似ResNet的残差连接
- 深度：stem + 4层residual blocks
- 表达能力强，适合医学图像

### 2. **针对重度发育不良的特殊优化**
```python
# 给重度发育不良更高的损失权重
class_weights[severe_idx] *= 1.5
# 采样时给予2倍权重
weights[severe_idx] *= 2.0
```

### 3. **更强的数据增强**
- 随机裁剪 (RandomCrop)
- 透视变换 (RandomPerspective)
- 更强的色彩抖动
- 更大的旋转角度 (45度)
- 更激进的仿射变换

### 4. **更好的优化器和学习率策略**
- AdamW优化器（带权重衰减）
- ReduceLROnPlateau（自适应学习率）
- 梯度裁剪防止梯度爆炸
- Label Smoothing（标签平滑）
- He权重初始化

### 5. **训练更多轮数**
- 120轮训练
- 更大的训练集比例（75%）
- 过采样（样本数×2）

## 输出结果

### 无需预训练版
- `best_cervical_model_no_pretrain.pth` - 最佳模型权重
- `training_history_no_pretrain.png` - 训练曲线图
- `confusion_matrix_no_pretrain.png` - 混淆矩阵

## 性能预期

| 版本 | 七分类准确率 | 二分类准确率 | 说明 |
|------|-------------|-------------|------|
| 原始 | ~38% | ~84% | 简单CNN |
| 改进版 | ~57% | ~84% | ResNet18 |
| **无需预训练版** | **>70%** | **>90%** | **自定义ResNet + 优化** |

## 预期改进

运行无需预训练版后应该看到：
- ✅ 重度发育不良F1大幅提升
- ✅ 整体准确率显著提高
- ✅ 各类别都有均衡的表现
- ✅ 二分类准确率超过90%
- ✅ 无需任何网络下载！
