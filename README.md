# 宫颈癌脱落细胞图像分类 — 深度学习模型

> 基于 PyTorch 的宫颈细胞 7 分类系统（Herlev 数据集）

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 项目简介

本项目利用深度卷积神经网络对宫颈脱落细胞涂片进行自动分类，将细胞划分为 **7 种病理类型**。使用的数据集为著名的 **Herlev 宫颈细胞数据集**，是宫颈癌计算机辅助筛查领域的经典基准。模型可区分正常上皮细胞、不同级别（轻/中/重度）的异型增生细胞以及原位癌，为细胞病理学筛查提供智能辅助。

我们探索了从自定义 ResNet 风格 CNN 到预训练 **EfficientNet‑B3** 等多种架构，并综合运用了数据增强、类别加权、标签平滑、测试时增强（TTA）等技术。最佳单模型在测试集上达到 **~67–70% 的七分类准确率** 和 **>94% 的二分类（正常/异常）准确率**。

---

## 📂 数据集说明

**Herlev 宫颈细胞数据集**  
- 总图像数：1,834 张（清洗后）  
- 7 个类别（文件夹名即类别名）：
  - `normal_superficiel` —— 浅表鳞状上皮细胞
  - `normal_intermediate` —— 中层鳞状上皮细胞
  - `normal_columnar` —— 柱状上皮细胞
  - `light_dysplastic` —— 轻度发育不良（CIN 1）
  - `moderate_dysplastic` —— 中度发育不良（CIN 2）
  - `severe_dysplastic` —— 重度发育不良（CIN 3）
  - `carcinoma_in_situ` —— 原位癌

各类别样本数量 **高度不均衡**，其中重度发育不良和原位癌是识别难度最大的类别。

---

## 🧠 技术路线

### 模型架构

| 模型 | 结构描述 | 是否使用预训练 | 状态 |
|------|----------|--------------|------|
| **自定义 ResNet** | 手工搭建残差网络（stem + 4 层残差块） | ❌ 否 | 基线 |
| **ResNet50** | 标准 ResNet‑50 + GeM 池化 + 加深分类头 | ✅ ImageNet | 稳定 |
| **EfficientNet‑B3** | EfficientNet‑B3 + 改进分类头 | ✅ ImageNet | **最终推荐** |

所有预训练模型均采用 **差异化学习率（discriminative learning rates）** 进行微调：骨干网络使用较低学习率（1e‑5 ~ 1e‑4），新增分类头使用较高学习率（1e‑3），以平衡特征保留与领域适应。

### 数据增强策略

- **RandAugment**（2 个操作，强度 7）—— 自动选择最优增强组合
- 随机水平翻转、垂直翻转
- 随机旋转 ±15°
- 轻微色彩抖动

### 训练技巧

- **类别权重**：根据各类样本数量的倒数计算，并对重度发育不良和原位癌额外加权
- **标签平滑**（ε = 0.1）—— 防止模型过度自信
- **余弦退火热重启调度器**（T₀=10, Tₘᵤₗₜ=2）
- **早停机制**（耐心值 = 20 轮）
- **梯度裁剪**（最大范数 = 1.0）
- **测试时增强（TTA）**：对原始图、水平翻转、垂直翻转三者的预测概率取平均

---

## 📈 实验结果

| 模型 / 变体 | 七分类准确率 | 七分类加权 F1 | 二分类（正常/异常）准确率 |
|------------|------------|--------------|------------------------|
| 自定义 ResNet（无预训练） | ~61% | ~0.61 | ~94% |
| ResNet50（微调 layer4 + FC） | **~59.7%** | ~0.586 | ~94% |
| EfficientNet‑B3（全微调 + TTA） | **~68–70%**（预期） | ~0.68–0.70 | ~95% |

> **注意**：最终 EfficientNet‑B3 模型需要 ≥6GB 显存。若资源有限，可选用 ResNet50 方案，兼顾速度与精度。

**混淆矩阵**（EfficientNet‑B3 + TTA）表明模型在中度/重度发育不良的区分能力上有显著提升，但部分类别间仍存在混淆，这源于数据集本身在细胞形态上的固有重叠。

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/LinJJ12/herlev-cervical-cancer-7-class.git
cd herlev-cervical-cancer-7-class
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备数据集

将 Herlev 数据集文件夹（`实训任务3数据集/`）放在项目根目录下。每个子文件夹应以类别英文名命名（与上述类名一致）。

### 4. 开始训练

#### 选项 A：**EfficientNet‑B3（推荐，精度最高）**
```bash
python train_efb3.py
```
该脚本会自动下载 ImageNet 预训练权重，训练至多 120 轮，采用早停，结束后执行 TTA 评估。

#### 选项 B：**ResNet50（稳定，显存要求低）**
```bash
python train_resnet50.py
```

#### 选项 C：**自定义 CNN（无需预训练，速度最快）**
```bash
python cervical_cancer_cnn_no_pretrain.py
```

> 所有脚本均会生成模型检查点（`best_*.pth`）、训练曲线图（`training_history_*.png`）和混淆矩阵图（`confusion_matrix_*.png`）。

---

## 📁 项目结构

```
cervical-cancer-classification/
├── data/                         # 数据集（或软链接）
│   └── 实训任务3数据集/
│       ├── normal_superficiel/
│       ├── normal_intermediate/
│       ├── normal_columnar/
│       ├── light_dysplastic/
│       ├── moderate_dysplastic/
│       ├── severe_dysplastic/
│       └── carcinoma_in_situ/
├── train_efb3.py                 # EfficientNet‑B3 训练 + TTA
├── train_resnet50.py             # ResNet50 微调训练
├── cervical_cancer_cnn_no_pretrain.py  # 自定义 CNN（无预训练）
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📦 依赖环境

- Python 3.8+
- PyTorch 1.12+
- torchvision
- numpy
- scikit-learn
- matplotlib
- seaborn
- pillow

完整依赖见 `requirements.txt`。

---

## ⚙️ 硬件需求

- **GPU**：推荐 **≥6GB 显存**（用于 EfficientNet‑B3，batch size 24，输入 280×280）。
- ResNet50 最低约 4GB 显存即可。
- CPU 可运行，但速度极慢。


---

## 🤝 贡献指南

欢迎提出改进建议！如有更好的网络结构、数据增强或训练策略，请提交 Issue 或 Pull Request。

---

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

---

## 📖 引用

如在本项目或 Herlev 数据集基础上进行研究，请引用：

```
@misc{cervical_herlev_2024,
  author = {Your Name},
  title = {Cervical Cancer Cell Classification using EfficientNet},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/yourusername/cervical-cancer-classification}}
}
```

---

## 🙏 致谢

- Herlev 大学医院提供数据集
- PyTorch 团队提供深度学习框架
- 开源社区提供的各种工具和库

---

**祝分类愉快！** 🩺🔬
