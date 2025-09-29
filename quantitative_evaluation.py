#!/usr/bin/env python3
"""
详细量化评估脚本
提供全面的模型性能分析和诊断
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys
import argparse
import json

# 添加项目路径
sys.path.append('.')

# 直接导入模型和数据加载功能
from cdanet.models import CDAnet
from cdanet.data import RBDataModule

DATASET_TO_MODEL = torch.tensor([1, 0, 2, 3], dtype=torch.long)
MODEL_TO_DATASET = torch.tensor([1, 0, 2, 3], dtype=torch.long)

def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """从检查点加载模型"""
    print(f"从检查点加载模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        model_config = {
            'in_channels': 4,
            'feature_channels': 32,
            'mlp_hidden_dims': [32, 32],
            'activation': 'softplus',
            'coord_dim': 3,
            'output_dim': 4,
            'igres': (4, 64, 64),
            'unet_nf': 16,
            'unet_mf': 256,
            'imnet_nf': 32
        }

    # 创建模型
    model = CDAnet(
        in_channels=model_config['in_channels'],
        feature_channels=model_config['feature_channels'],
        mlp_hidden_dims=model_config['mlp_hidden_dims'],
        activation=model_config['activation'],
        coord_dim=model_config['coord_dim'],
        output_dim=model_config['output_dim'],
        igres=tuple(model_config['igres']),
        unet_nf=model_config['unet_nf'],
        unet_mf=model_config['unet_mf'],
        imnet_nf=model_config.get('imnet_nf', None)
    )

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'unet_state_dict' in checkpoint and 'imnet_state_dict' in checkpoint:
        model.feature_extractor.load_state_dict(checkpoint['unet_state_dict'])
        model.implicit_net.load_state_dict(checkpoint['imnet_state_dict'])
    else:
        raise KeyError('Checkpoint missing model weights: expected model_state_dict or unet/imnet state dicts')
    model.to(device)
    model.eval()

    print(f"✅ 模型加载成功，参数: {sum(p.numel() for p in model.parameters()):,}")
    return model

def setup_simple_data_module(data_dir, Ra_numbers=[1e5]):
    """简化的数据模块设置"""
    print(f"设置数据模块: {data_dir}")

    data_module = RBDataModule(
        data_dir=data_dir,
        batch_size=1,
        normalize=True,
        num_workers=0
    )
    data_module.setup(Ra_numbers=Ra_numbers)

    print(f"✅ 数据模块设置完成")
    return data_module

def comprehensive_evaluation(model, data_module, device='cuda', max_batches=5):
    """全面的模型评估"""

    print("🔬 开始详细量化评估...")
    print("=" * 60)

    model.eval()
    all_metrics = {}
    all_predictions = []
    all_targets = []

    # 测试不同Ra数
    Ra_numbers = [1e5]

    dataset_to_model = DATASET_TO_MODEL.to(device)
    model_to_dataset = MODEL_TO_DATASET.to(device)

    for Ra in Ra_numbers:
        print(f"\n📊 评估 Ra = {Ra:.0e}")

        test_loader = data_module.get_dataloader(Ra, 'test')

        batch_metrics = {
            'mse': [], 'mae': [], 'r2': [],
            'pred_range': [], 'target_range': [],
            'pred_std': [], 'target_std': [],
            'correlation': []
        }

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= max_batches:  # 限制批次数避免过长
                    break

                print(f"  处理批次 {i+1}...")

                low_res = batch['low_res'].to(device)
                coords = batch['coords'].to(device)
                targets = batch['targets'].to(device)

                # 分块推理
                batch_size, num_coords, coord_dim = coords.shape
                chunk_size = 4096
                predictions_list = []

                low_res_reordered = low_res.index_select(1, dataset_to_model)

                for j in range(0, num_coords, chunk_size):
                    end_idx = min(j + chunk_size, num_coords)
                    coord_chunk = coords[:, j:end_idx, :]

                    pred_chunk = model(low_res_reordered, coord_chunk)
                    pred_chunk = pred_chunk.index_select(-1, model_to_dataset)
                    predictions_list.append(pred_chunk.cpu())

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                predictions = torch.cat(predictions_list, dim=1).to(device)

                # 反归一化
                if data_module.normalizer is not None:
                    predictions_cpu = predictions.cpu()
                    targets_cpu = targets.cpu()

                    pred_denorm = data_module.normalizer.denormalize(
                        predictions_cpu.view(-1, 4)).view(predictions_cpu.shape)
                    target_denorm = data_module.normalizer.denormalize(
                        targets_cpu.view(-1, 4)).view(targets_cpu.shape)
                else:
                    pred_denorm = predictions.cpu()
                    target_denorm = targets.cpu()

                # 计算各个物理量的指标
                for field_idx, field_name in enumerate(['T', 'p', 'u', 'v']):
                    pred_field = pred_denorm[0, :, field_idx].numpy()
                    target_field = target_denorm[0, :, field_idx].numpy()

                    # 基础指标
                    mse = mean_squared_error(target_field, pred_field)
                    mae = mean_absolute_error(target_field, pred_field)

                    # 相关系数
                    correlation = np.corrcoef(target_field, pred_field)[0, 1]

                    # R²
                    r2 = r2_score(target_field, pred_field)

                    # 范围和变异性
                    pred_range = pred_field.max() - pred_field.min()
                    target_range = target_field.max() - target_field.min()

                    batch_metrics['mse'].append(mse)
                    batch_metrics['mae'].append(mae)
                    batch_metrics['r2'].append(r2)
                    batch_metrics['correlation'].append(correlation)
                    batch_metrics['pred_range'].append(pred_range)
                    batch_metrics['target_range'].append(target_range)
                    batch_metrics['pred_std'].append(np.std(pred_field))
                    batch_metrics['target_std'].append(np.std(target_field))

                # 保存用于详细分析
                all_predictions.append(pred_denorm[0].numpy())
                all_targets.append(target_denorm[0].numpy())

        all_metrics[Ra] = batch_metrics

        # 打印统计结果
        print(f"  📈 统计结果:")
        print(f"    MSE: {np.mean(batch_metrics['mse']):.6f} ± {np.std(batch_metrics['mse']):.6f}")
        print(f"    MAE: {np.mean(batch_metrics['mae']):.6f} ± {np.std(batch_metrics['mae']):.6f}")
        print(f"    R²:  {np.mean(batch_metrics['r2']):.4f} ± {np.std(batch_metrics['r2']):.4f}")
        print(f"    相关系数: {np.mean(batch_metrics['correlation']):.4f} ± {np.std(batch_metrics['correlation']):.4f}")
        print(f"    预测范围/真实范围: {np.mean(batch_metrics['pred_range'])/np.mean(batch_metrics['target_range']):.4f}")
        print(f"    预测标准差/真实标准差: {np.mean(batch_metrics['pred_std'])/np.mean(batch_metrics['target_std']):.4f}")

    return all_metrics, all_predictions, all_targets

def create_detailed_plots(all_predictions, all_targets, save_dir='./detailed_analysis'):
    """创建详细分析图表"""

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n🎨 创建详细分析图表...")

    # 合并所有数据
    all_pred = np.concatenate(all_predictions, axis=0)  # [N_total, 4]
    all_true = np.concatenate(all_targets, axis=0)

    field_names = ['Temperature', 'Pressure', 'U-velocity', 'V-velocity']

    # 1. 散点图 - 预测vs真实值
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Prediction vs Ground Truth', fontsize=16)

    for i, (ax, field_name) in enumerate(zip(axes.flat, field_names)):
        pred_vals = all_pred[:, i]
        true_vals = all_true[:, i]

        # 随机采样避免过多点
        n_points = min(10000, len(pred_vals))
        idx = np.random.choice(len(pred_vals), n_points, replace=False)

        ax.scatter(true_vals[idx], pred_vals[idx], alpha=0.5, s=1)

        # 完美预测线
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')

        ax.set_xlabel(f'True {field_name}')
        ax.set_ylabel(f'Predicted {field_name}')
        ax.set_title(f'{field_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        r2 = r2_score(true_vals, pred_vals)
        correlation = np.corrcoef(true_vals, pred_vals)[0, 1]
        ax.text(0.05, 0.95, f'R²={r2:.3f}\nCorr={correlation:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{save_dir}/prediction_vs_truth.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 误差分布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Error Distributions', fontsize=16)

    for i, (ax, field_name) in enumerate(zip(axes.flat, field_names)):
        errors = all_pred[:, i] - all_true[:, i]

        ax.hist(errors, bins=50, alpha=0.7, density=True)
        ax.axvline(0, color='red', linestyle='--', label='Perfect')
        ax.axvline(np.mean(errors), color='green', linestyle='-',
                   label=f'Mean={np.mean(errors):.4f}')

        ax.set_xlabel(f'Error ({field_name})')
        ax.set_ylabel('Density')
        ax.set_title(f'{field_name} Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        ax.text(0.05, 0.95, f'Std={np.std(errors):.4f}\nMAE={np.mean(np.abs(errors)):.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 场的统计对比
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Field Statistics Comparison', fontsize=16)

    stats_to_plot = ['mean', 'std', 'min', 'max']

    for i, (ax, stat_name) in enumerate(zip(axes.flat, stats_to_plot)):
        true_stats = []
        pred_stats = []

        for j in range(4):  # 4个物理场
            if stat_name == 'mean':
                true_stat = np.mean(all_true[:, j])
                pred_stat = np.mean(all_pred[:, j])
            elif stat_name == 'std':
                true_stat = np.std(all_true[:, j])
                pred_stat = np.std(all_pred[:, j])
            elif stat_name == 'min':
                true_stat = np.min(all_true[:, j])
                pred_stat = np.min(all_pred[:, j])
            else:  # max
                true_stat = np.max(all_true[:, j])
                pred_stat = np.max(all_pred[:, j])

            true_stats.append(true_stat)
            pred_stats.append(pred_stat)

        x = np.arange(len(field_names))
        width = 0.35

        ax.bar(x - width/2, true_stats, width, label='Ground Truth', alpha=0.7)
        ax.bar(x + width/2, pred_stats, width, label='Prediction', alpha=0.7)

        ax.set_xlabel('Field')
        ax.set_ylabel(stat_name.capitalize())
        ax.set_title(f'{stat_name.capitalize()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(['T', 'p', 'u', 'v'])
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/field_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 详细分析图表已保存到 {save_dir}/")

def diagnose_model_issues(all_predictions, all_targets):
    """诊断模型问题"""

    print(f"\n🔍 模型问题诊断...")
    print("-" * 40)

    all_pred = np.concatenate(all_predictions, axis=0)
    all_true = np.concatenate(all_targets, axis=0)

    field_names = ['Temperature', 'Pressure', 'U-velocity', 'V-velocity']

    for i, field_name in enumerate(field_names):
        pred_vals = all_pred[:, i]
        true_vals = all_true[:, i]

        print(f"\n📊 {field_name}:")
        print(f"  真实值范围: [{true_vals.min():.4f}, {true_vals.max():.4f}]")
        print(f"  预测值范围: [{pred_vals.min():.4f}, {pred_vals.max():.4f}]")
        print(f"  范围比率: {(pred_vals.max()-pred_vals.min())/(true_vals.max()-true_vals.min()):.4f}")
        print(f"  变异系数比: {(np.std(pred_vals)/np.mean(np.abs(pred_vals)))/(np.std(true_vals)/np.mean(np.abs(true_vals))):.4f}")

        # 检查问题
        issues = []

        if (pred_vals.max() - pred_vals.min()) < 0.1 * (true_vals.max() - true_vals.min()):
            issues.append("⚠️  预测范围过窄")

        if np.std(pred_vals) < 0.1 * np.std(true_vals):
            issues.append("⚠️  预测变异性不足")

        if abs(np.mean(pred_vals) - np.mean(true_vals)) > 0.1 * np.std(true_vals):
            issues.append("⚠️  预测均值偏差过大")

        if np.corrcoef(true_vals, pred_vals)[0, 1] < 0.5:
            issues.append("⚠️  相关性低")

        if len(issues) == 0:
            issues.append("✅ 看起来正常")

        for issue in issues:
            print(f"  {issue}")

def main():
    parser = argparse.ArgumentParser(description='详细量化评估')
    parser.add_argument('--checkpoint', required=True, help='模型检查点路径')
    parser.add_argument('--data_dir', default='./rb_data_numerical', help='数据目录')
    parser.add_argument('--max_batches', type=int, default=5, help='每个Ra评估的最大批次数')

    args = parser.parse_args()

    print("🚀 CDAnet详细量化评估")
    print("=" * 60)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型和数据
    try:
        model = load_model_from_checkpoint(args.checkpoint, device)
        data_module = setup_simple_data_module(args.data_dir, [1e5])

        # 详细评估
        all_metrics, all_predictions, all_targets = comprehensive_evaluation(
            model, data_module, device, max_batches=args.max_batches)

        # 创建详细图表
        create_detailed_plots(all_predictions, all_targets)

        # 诊断问题
        diagnose_model_issues(all_predictions, all_targets)

        # 保存指标快照
        os.makedirs('./detailed_analysis', exist_ok=True)
        summary = {}
        for Ra, metrics in all_metrics.items():
            summary[str(Ra)] = {}
            for key, values in metrics.items():
                if len(values):
                    summary[str(Ra)][key] = float(np.mean(values))

        with open('./detailed_analysis/metrics_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n🎉 详细评估完成!")
        print(f"📁 结果保存在: ./detailed_analysis/")

    except Exception as e:
        print(f"❌ 评估过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
