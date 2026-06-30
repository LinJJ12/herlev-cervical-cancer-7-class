import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

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

# 正常/异常二分类映射
BINARY_LABELS = {0: '正常', 1: '正常', 2: '正常', 3: '异常', 4: '异常', 5: '异常', 6: '异常'}


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
        print(f"  {class_name}: {count}")
    
    return image_paths, labels


class CervicalCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CervicalCNN, self).__init__()
        
        self.features = nn.Sequential(
            # 第一层
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第四层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cuda'):
    model.to(device)
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"\n开始训练，共 {num_epochs} 轮...")
    
    for epoch in range(num_epochs):
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
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_cervical_model.pth')
            print(f'  → 保存最佳模型 (Val Acc: {best_val_acc:.4f})')
    
    print(f'\n训练完成！最佳验证准确率: {best_val_acc:.4f}')
    return model, train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, test_loader, device='cuda'):
    model.load_state_dict(torch.load('best_cervical_model.pth'))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("\n开始评估模型...")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 分类报告
    print("\n=== 七分类详细分类报告 ===")
    print(classification_report(all_labels, all_preds, target_names=list(CLASS_NAMES.keys())))
    
    # 二分类评估（正常/异常）
    binary_preds = [1 if p >= 3 else 0 for p in all_preds]
    binary_labels = [1 if l >= 3 else 0 for l in all_labels]
    
    print("\n=== 二分类（正常/异常）分类报告 ===")
    print(classification_report(binary_labels, binary_preds, target_names=['正常', '异常']))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES.keys(), 
                yticklabels=CLASS_NAMES.keys())
    plt.title('混淆矩阵 - 七分类')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("混淆矩阵已保存为 confusion_matrix.png")
    
    return all_preds, all_labels


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('训练/验证 Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('训练/验证 准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("训练历史曲线已保存为 training_history.png")


def main():
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    image_paths, labels = load_data()
    
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
    
    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集和数据加载器
    train_dataset = CervicalDataset(train_paths, train_labels, train_transform)
    val_dataset = CervicalDataset(val_paths, val_labels, val_test_transform)
    test_dataset = CervicalDataset(test_paths, test_labels, val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 创建模型
    model = CervicalCNN(num_classes=7)
    print("\n模型结构:")
    print(model)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 训练模型
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=25, device=device
    )
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # 在测试集上评估
    evaluate_model(model, test_loader, device=device)
    
    print("\n任务完成！")


if __name__ == "__main__":
    main()
