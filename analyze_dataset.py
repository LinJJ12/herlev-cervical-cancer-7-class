import os
import numpy as np
from collections import Counter
from PIL import Image

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

CLASS_NAMES_CN = {
    'normal_superficiel': '浅表鳞状上皮',
    'normal_intermediate': '中度鳞状上皮',
    'normal_columnar': '柱状上皮',
    'light_dysplastic': '轻度发育不良',
    'moderate_dysplastic': '中度发育不良',
    'severe_dysplastic': '重度发育不良',
    'carcinoma_in_situ': '原位癌'
}

def load_dataset_info():
    """加载数据集的基本信息"""
    print("=" * 80)
    print("【数据集统计分析】")
    print("=" * 80)
    
    all_image_paths = []
    all_labels = []
    class_info = {}
    
    # 统计每个类别的信息
    for class_name, class_idx in CLASS_NAMES.items():
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"警告: 类别 {class_name} 的目录不存在！")
            continue
        
        # 获取该类别的所有图片
        image_files = []
        for f in os.listdir(class_dir):
            if f.lower().endswith(('.bmp', '.jpg', '.png')):
                image_files.append(f)
        
        # 统计图片尺寸
        image_sizes = []
        for img_file in image_files[:10]:  # 只统计前10张图片的尺寸
            try:
                img_path = os.path.join(class_dir, img_file)
                with Image.open(img_path) as img:
                    image_sizes.append(img.size)
            except Exception as e:
                pass
        
        # 保存信息
        class_info[class_name] = {
            'count': len(image_files),
            'image_sizes': image_sizes,
            'sample_paths': [os.path.join(class_dir, f) for f in image_files[:5]]  # 只保存前5个样本路径
        }
        
        all_image_paths.extend([os.path.join(class_dir, f) for f in image_files])
        all_labels.extend([class_idx] * len(image_files))
    
    return all_image_paths, all_labels, class_info

def print_class_distribution(class_info):
    """打印类别分布统计"""
    print("\n" + "=" * 80)
    print("【1. 类别样本分布统计】")
    print("=" * 80)
    
    total_samples = sum(info['count'] for info in class_info.values())
    print(f"总样本数: {total_samples}")
    print()
    
    # 按顺序打印各类别信息
    for class_name, class_idx in CLASS_NAMES.items():
        info = class_info[class_name]
        count = info['count']
        percentage = (count / total_samples) * 100
        
        print(f"{CLASS_NAMES_CN[class_name]} ({class_name}):")
        print(f"  样本数: {count:4d} ({percentage:5.2f}%)")
        
        if info['image_sizes']:
            sizes = info['image_sizes']
            unique_sizes = Counter(sizes)
            if len(unique_sizes) > 1:
                print(f"  图像尺寸: 多个尺寸 {sizes[:3]}...")
            else:
                print(f"  图像尺寸: {sizes[0]}")
        print()
    
    # 计算类别不平衡指标
    counts = [info['count'] for info in class_info.values()]
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count
    
    print(f"最大/最小样本数比: {imbalance_ratio:.2f}:1")
    print(f"最大样本类别: {CLASS_NAMES_CN[CLASS_NAMES_REV[counts.index(max_count)]]} ({max_count})")
    print(f"最小样本类别: {CLASS_NAMES_CN[CLASS_NAMES_REV[counts.index(min_count)]]} ({min_count})")
    
    # 正常 vs 异常统计
    normal_samples = sum(info['count'] for name, info in class_info.items() if CLASS_NAMES[name] < 3)
    abnormal_samples = total_samples - normal_samples
    normal_ratio = (normal_samples / total_samples) * 100
    abnormal_ratio = (abnormal_samples / total_samples) * 100
    
    print("\n正常 vs 异常样本统计:")
    print(f"  正常样本: {normal_samples:4d} ({normal_ratio:5.2f}%)")
    print(f"  异常样本: {abnormal_samples:4d} ({abnormal_ratio:5.2f}%)")
    print(f"  正常/异常比: 1:{abnormal_samples/normal_samples:.2f}")
    
    return counts

CLASS_NAMES_REV = {v: k for k, v in CLASS_NAMES.items()}

def analyze_imbalance(counts):
    """分析类别不平衡情况"""
    print("\n" + "=" * 80)
    print("【2. 类别不平衡分析】")
    print("=" * 80)
    
    total = sum(counts)
    mean_count = total / len(counts)
    
    print(f"平均每类样本数: {mean_count:.1f}")
    print()
    
    # 计算各类别相对于平均值的偏差
    for i, (class_name, class_idx) in enumerate(CLASS_NAMES.items()):
        count = counts[i]
        deviation = ((count - mean_count) / mean_count) * 100
        
        status = "  (正常)"
        if count < mean_count * 0.5:
            status = "  ⚠️  (欠采样)"
        elif count > mean_count * 2:
            status = "  ⚠️  (过采样)"
        
        print(f"{CLASS_NAMES_CN[class_name]}:")
        print(f"  样本数: {count:4d}, 偏差: {deviation:+6.1f}% {status}")
        print()

def suggest_optimizations(counts, class_info):
    """基于统计给出优化建议"""
    print("\n" + "=" * 80)
    print("【3. 优化建议】")
    print("=" * 80)
    
    total = sum(counts)
    mean_count = total / len(counts)
    
    # 1. 损失函数建议
    print("\n【损失函数优化】")
    print("-" * 50)
    
    max_count = max(counts)
    class_weights = [max_count / c for c in counts]
    
    print("建议1: 使用类别加权的交叉熵损失")
    print(f"  建议权重: {[f'w{i}={w:.2f}' for i, w in enumerate(class_weights)]}")
    print()
    
    print("建议2: 考虑使用 Focal Loss")
    print("  原因: 类别不平衡 + 可能存在难易样本差异")
    print("  建议参数: gamma=2.0, alpha=类别权重")
    print()
    
    # 2. 采样策略建议
    print("\n【采样策略优化】")
    print("-" * 50)
    
    under_represented = []
    over_represented = []
    
    for i, (class_name, class_idx) in enumerate(CLASS_NAMES.items()):
        count = counts[i]
        if count < mean_count * 0.5:
            under_represented.append((CLASS_NAMES_CN[class_name], count))
        elif count > mean_count * 2:
            over_represented.append((CLASS_NAMES_CN[class_name], count))
    
    if under_represented:
        print(f"建议1: 对以下类别进行过采样 (目标达到均值的 {mean_count:.0f} 样本):")
        for name, count in under_represented:
            print(f"  - {name}: 当前 {count} → 目标 {int(mean_count)}")
    print()
    
    if over_represented:
        print(f"建议2: 对以下类别可以考虑轻微欠采样:")
        for name, count in over_represented:
            print(f"  - {name}: 当前 {count} → 可降至 {int(mean_count * 1.5)}")
    print()
    
    print("建议3: 使用 WeightedRandomSampler")
    print(f"  采样权重 (反比例): {[f'w{i}={1/c:.4f}' for i, c in enumerate(counts)]}")
    print()
    
    # 3. 数据增强建议
    print("\n【数据增强优化】")
    print("-" * 50)
    print("建议: 对样本较少的类别使用更强的数据增强")
    print("  - 随机裁剪、翻转、旋转")
    print("  - 颜色抖动、亮度调整")
    print("  - 透视变换、弹性形变")
    print()
    
    # 4. 评估建议
    print("\n【评估策略优化】")
    print("-" * 50)
    print("建议1: 使用分层抽样确保训练/验证/测试集分布一致")
    print("建议2: 重点关注 F1-score 和 Recall，而非仅看准确率")
    print("建议3: 对少数类类别（如重度发育不良）单独评估")
    print("建议4: 考虑使用混淆矩阵分析误分类模式")
    print()
    
    # 5. 快速验证建议
    print("\n【快速验证方案】")
    print("-" * 50)
    print("方案A: 仅调整损失函数权重（最低风险）")
    print("方案B: 使用 WeightedRandomSampler + 数据增强")
    print("方案C: 尝试 Focal Loss + 更强的采样策略")
    print()

def main():
    """主函数"""
    try:
        # 加载并分析数据
        all_image_paths, all_labels, class_info = load_dataset_info()
        
        # 打印分布
        counts = print_class_distribution(class_info)
        
        # 分析不平衡
        analyze_imbalance(counts)
        
        # 给出建议
        suggest_optimizations(counts, class_info)
        
        print("\n" + "=" * 80)
        print("分析完成！以上信息可用于指导后续优化。")
        print("=" * 80)
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
