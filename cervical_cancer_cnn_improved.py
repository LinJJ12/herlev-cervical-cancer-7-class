import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径配置
DATA_DIR = '实训任务3数据集'

# 类别映射
CLASS_NAMES = {
    'normal_superficiel': 0,
    'normal_intermediate': 1,
    'normal_columnar': 2,
    'light_dysplastic': 3,
    'moderate_dysplastic': 4,
    'severe_dysplastic': 5,
    'carcinoma_in_situ': 6
}

CLASS_NAMES_REV = {v: k for k, v in CLASS_NAMES.items()}

# 类别中文名称
CLASS_NAMES_CN = {
    'normal_superficiel': '浅表鳞状上皮',
    'normal_intermediate': '中度鳞状上皮',
    'normal_columnar': '柱状上皮',
    'light_dysplastic': '轻度发育不良',
    'moderate_dysplastic': '中度发育不良',
    'severe_dysplastic': '重度发育不良',
    'carcinoma_in_situ': '原位癌'
}


class CervicalDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_data():
    print("正在加载数据集...")
    
    image_paths = []
    labels = []
    
    for class_name, class_idx in CLASS_NAMES.items():
        class_dir = os.path.join(DATA_DIR, class_name)
        if os.path.exists(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.bmp', '.jpg', '.png')):
                    image_paths.append(os.path.join(class_dir, file))
                    labels.append(class_idx)
    
    print(f"总共加载 {len(image_paths)} 张图像")
    
    # 统计各类别样本数
    class_counts = {}
    for label in labels:
        class_name = CLASS_NAMES_REV[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\n各类别样本分布:")
    for class_name, count in class_counts.items():
        print(f"  {CLASS_NAMES_CN[class_name]} ({class_name}): {count}")
    
    return image_paths, labels, class_counts


def get_model(num_classes=7):
    # 使用ResNet18作为预训练模型
    model = models.resnet18(pretrained=True)
    
    # 冻结前面的层
    for param in model.parameters():
        param.requires_grad = False
    
    # 替换最后的全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device='cuda'):
    model.to(device)
    best_val_acc = 0.0
    best_val_f1 = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"\n开始训练，共 {num_epochs} 轮...")
    
    for epoch in range(num_epochs):
        # 解冻部分层进行微调（从第20轮开始）
        if epoch == 20:
            print("\n解冻部分卷积层进行微调...")
            for param in model.layer4.parameters():
                param.requires_grad = True
        
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = correct_val / total_val
        
        # 计算验证集的F1分数
        _, _, f1, _ = precision_recall_fscore_support(all_val_labels, all_val_preds, average='weighted')
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {f1:.4f}')
        
        # 保存最佳模型（基于F1分数）
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': f1
            }, 'best_cervical_model_improved.pth')
            print(f'  → 保存最佳模型 (Val F1: {best_val_f1:.4f}, Acc: {best_val_acc:.4f})')
        
        # 更新学习率
        scheduler.step()
    
    print(f'\n训练完成！最佳验证 F1: {best_val_f1:.4f}, Acc: {best_val_acc:.4f}')
    return model, train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, test_loader, device='cuda'):
    checkpoint = torch.load('best_cervical_model_improved.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n开始评估模型...")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 详细分类报告
    print("\n" + "="*80)
    print("【七分类详细评估指标】")
    print("="*80)
    report = classification_report(all_labels, all_preds, 
                                   target_names=[CLASS_NAMES_CN[name] for name in CLASS_NAMES.keys()],
                                   digits=4,
                                   zero_division=0)
    print(report)
    
    # 计算每类的详细指标
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, zero_division=0)
    print("\n【各类别详细指标】")
    print(f"{'类别':<20} {'精确率(Precision)':<15} {'召回率(Recall)':<15} {'F1分数':<10} {'样本数':<10}")
    print("-"*80)
    for i, class_name in enumerate(CLASS_NAMES.keys()):
        print(f"{CLASS_NAMES_CN[class_name]:<20} {precision[i]:<15.4f} {recall[i]:<15.4f} {f1[i]:<10.4f} {support[i]:<10}")
    
    # 二分类评估（正常/异常）
    binary_preds = [1 if p >= 3 else 0 for p in all_preds]
    binary_labels = [1 if l >= 3 else 0 for l in all_labels]
    
    print("\n" + "="*80)
    print("【二分类（正常/异常）评估指标】")
    print("="*80)
    binary_report = classification_report(binary_labels, binary_preds, 
                                          target_names=['正常', '异常'],
                                          digits=4,
                                          zero_division=0)
    print(binary_report)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(14, 12))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[CLASS_NAMES_CN[name] for name in CLASS_NAMES.keys()], 
                yticklabels=[CLASS_NAMES_CN[name] for name in CLASS_NAMES.keys()],
                annot_kws={'size': 12})
    plt.title('混淆矩阵 - 七分类', fontsize=16, fontweight='bold')
    plt.xlabel('预测类别', fontsize=14)
    plt.ylabel('真实类别', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix_improved.png', dpi=300, bbox_inches='tight')
    print("\n混淆矩阵已保存为 confusion_matrix_improved.png")
    
    return all_preds, all_labels, all_probs


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss曲线
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2, color='#1f77b4')
    axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2, color='#ff7f0e')
    axes[0, 0].set_title('训练/验证 Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[0, 1].plot(train_accs, label='Train Acc', linewidth=2, color='#1f77b4')
    axes[0, 1].plot(val_accs, label='Val Acc', linewidth=2, color='#ff7f0e')
    axes[0, 1].set_title('训练/验证 准确率', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 放大的Loss曲线（后半段）
    mid = len(train_losses) // 2
    axes[1, 0].plot(range(mid, len(train_losses)), train_losses[mid:], label='Train Loss', linewidth=2, color='#1f77b4')
    axes[1, 0].plot(range(mid, len(val_losses)), val_losses[mid:], label='Val Loss', linewidth=2, color='#ff7f0e')
    axes[1, 0].set_title('训练/验证 Loss (后半段)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 放大的准确率曲线（后半段）
    axes[1, 1].plot(range(mid, len(train_accs)), train_accs[mid:], label='Train Acc', linewidth=2, color='#1f77b4')
    axes[1, 1].plot(range(mid, len(val_accs)), val_accs[mid:], label='Val Acc', linewidth=2, color='#ff7f0e')
    axes[1, 1].set_title('训练/验证 准确率 (后半段)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy', fontsize=12)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_improved.png', dpi=300, bbox_inches='tight')
    print("训练历史曲线已保存为 training_history_improved.png")


def main():
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    image_paths, labels, class_counts = load_data()
    
    # 计算类别权重用于处理不平衡
    total_samples = len(labels)
    class_weights = torch.tensor([total_samples / class_counts[name] for name in CLASS_NAMES.keys()], 
                                 dtype=torch.float32)
    print(f"\n类别权重: {class_weights}")
    
    # 数据划分 (训练集 70%, 验证集 15%, 测试集 15%)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(train_paths)}")
    print(f"  验证集: {len(val_paths)}")
    print(f"  测试集: {len(test_paths)}")
    
    # 更强的数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = CervicalDataset(train_paths, train_labels, train_transform)
    val_dataset = CervicalDataset(val_paths, val_labels, val_test_transform)
    test_dataset = CervicalDataset(test_paths, test_labels, val_test_transform)
    
    # 使用WeightedRandomSampler处理类别不平衡
    train_labels_np = np.array(train_labels)
    class_sample_counts = np.array([np.sum(train_labels_np == i) for i in range(7)])
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    samples_weights = weights[train_labels_np]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 创建模型（使用ResNet18迁移学习）
    model = get_model(num_classes=7)
    print("\n模型结构:")
    print(model)
    
    # 损失函数（使用类别权重）
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # 优化器和学习率调度器
    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr': 0.001},
        {'params': model.layer4.parameters(), 'lr': 0.0001}  # 这部分一开始是冻结的
    ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # 训练模型
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=50, device=device
    )
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # 在测试集上评估
    evaluate_model(model, test_loader, device=device)
    
    print("\n" + "="*80)
    print("任务完成！改进版本已运行完毕。")
    print("="*80)


if __name__ == "__main__":
    main()
