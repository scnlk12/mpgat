"""
模型诊断工具 - 深入分析当前模型的误差，找出优化方向

功能:
1. 识别模型的主要弱点（哪些类型的样本预测不好）
2. 分析误差的根本原因
3. 生成针对性的优化建议
4. 可视化高误差样本的预测情况
"""

import argparse
import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import datetime

import utils
import data_prepare
from model import GMAN
from utils import cal_lape
from utils.metrics import RMSE_MAE_MAPE


class ModelDiagnostics:
    """模型诊断器"""

    def __init__(self, model, test_loader, scaler, device, output_dir='./model_diagnosis'):
        self.model = model
        self.test_loader = test_loader
        self.scaler = scaler
        self.device = device
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'sample_details'), exist_ok=True)

        # 存储数据
        self.y_true_all = []
        self.y_pred_all = []
        self.errors_all = []
        self.x_all = []  # 输入历史数据

    @torch.no_grad()
    def collect_predictions(self):
        """收集所有预测结果"""
        print("正在收集预测数据...")
        self.model.eval()

        for batch in self.test_loader:
            batch.to_tensor(self.device)
            x_batch = batch['x']
            y_batch = batch['y']

            TE = x_batch[:, :, :, 1:]

            # 预测
            out_batch = self.model(x_batch, TE)
            out_batch = self.scaler.inverse_transform(out_batch)
            y_batch = self.scaler.inverse_transform(y_batch[:, :, :, 0])
            x_batch_inverse = self.scaler.inverse_transform(x_batch[:, :, :, 0])

            # 转为numpy
            out_batch = out_batch.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            x_batch_inverse = x_batch_inverse.cpu().numpy()

            # 保存
            for i in range(out_batch.shape[0]):
                self.y_true_all.append(y_batch[i])
                self.y_pred_all.append(out_batch[i])
                self.x_all.append(x_batch_inverse[i])

        # 转为numpy数组
        self.y_true_all = np.array(self.y_true_all)  # (samples, timesteps, nodes)
        self.y_pred_all = np.array(self.y_pred_all)
        self.x_all = np.array(self.x_all)  # 历史数据
        self.errors_all = np.abs(self.y_pred_all - self.y_true_all)

        print(f"收集完成: {len(self.y_true_all)} 个样本")
        print(f"数据形状: {self.y_true_all.shape}")

    def analyze_error_by_flow_magnitude(self):
        """按流量大小分析误差"""
        print("\n" + "="*80)
        print("按流量大小分析误差")
        print("="*80)

        # 定义流量区间
        bins = [0, 5, 10, 20, 30, 50, 100, float('inf')]
        labels = ['0-5', '5-10', '10-20', '20-30', '30-50', '50-100', '100+']

        true_flat = self.y_true_all.flatten()
        pred_flat = self.y_pred_all.flatten()
        error_flat = self.errors_all.flatten()

        results = []
        print(f"\n{'流量区间':<12} {'样本数':<12} {'占比':<10} {'平均MAE':<12} {'平均MAPE':<12} {'RMSE':<12}")
        print("-" * 80)

        for i in range(len(bins) - 1):
            mask = (true_flat >= bins[i]) & (true_flat < bins[i+1])
            count = mask.sum()

            if count > 0:
                mae = error_flat[mask].mean()
                mape = (error_flat[mask] / (true_flat[mask] + 1e-5)).mean() * 100
                rmse = np.sqrt(np.mean((pred_flat[mask] - true_flat[mask]) ** 2))
                ratio = count / len(true_flat) * 100

                results.append({
                    'range': labels[i],
                    'count': count,
                    'ratio': ratio,
                    'mae': mae,
                    'mape': mape,
                    'rmse': rmse
                })

                print(f"{labels[i]:<12} {count:<12} {ratio:<10.2f}% {mae:<12.3f} {mape:<12.2f}% {rmse:<12.3f}")

        # 可视化
        self._plot_error_by_flow(results)

        # 诊断结论
        print(f"\n📊 诊断结论:")
        sorted_by_mae = sorted(results, key=lambda x: x['mae'], reverse=True)
        worst = sorted_by_mae[0]
        print(f"  ⚠️  误差最大的流量区间: {worst['range']} (MAE={worst['mae']:.3f})")
        print(f"     占总样本的 {worst['ratio']:.1f}%")

        if worst['range'] in ['0-5', '5-10']:
            print(f"\n💡 优化建议:")
            print(f"  1. 低流量预测不准，考虑:")
            print(f"     - 使用加权损失函数，提高低流量样本的权重")
            print(f"     - 对低流量样本进行过采样")
            print(f"     - 数据预处理：对数变换 log(x+1)")
        elif worst['range'] in ['50-100', '100+']:
            print(f"\n💡 优化建议:")
            print(f"  1. 高流量预测不准，考虑:")
            print(f"     - 使用更鲁棒的损失函数 (Huber Loss)")
            print(f"     - 检查是否有数据异常值")
            print(f"     - 增加高流量样本的数据增强")

        return results

    def analyze_error_by_temporal_pattern(self):
        """按时间模式分析误差"""
        print("\n" + "="*80)
        print("按时间模式分析误差")
        print("="*80)

        num_samples = self.y_true_all.shape[0]
        num_timesteps = self.y_true_all.shape[1]

        # 分析每个时间步的误差
        timestep_stats = []
        for t in range(num_timesteps):
            mae_t = self.errors_all[:, t, :].mean()
            rmse_t = np.sqrt(np.mean((self.y_pred_all[:, t, :] - self.y_true_all[:, t, :]) ** 2))

            # 分析预测偏差
            pred_mean = self.y_pred_all[:, t, :].mean()
            true_mean = self.y_true_all[:, t, :].mean()
            bias = pred_mean - true_mean
            bias_ratio = bias / (true_mean + 1e-5) * 100

            timestep_stats.append({
                'timestep': t + 1,
                'mae': mae_t,
                'rmse': rmse_t,
                'bias': bias,
                'bias_ratio': bias_ratio,
                'true_mean': true_mean,
                'pred_mean': pred_mean
            })

        # 显示
        print(f"\n{'时间步':<8} {'MAE':<10} {'RMSE':<10} {'偏差':<12} {'偏差率':<12}")
        print("-" * 60)
        for stat in timestep_stats:
            print(f"{stat['timestep']:<8} {stat['mae']:<10.3f} {stat['rmse']:<10.3f} "
                  f"{stat['bias']:<12.3f} {stat['bias_ratio']:<12.2f}%")

        # 诊断
        print(f"\n📊 诊断结论:")

        # 检查误差是否递增
        first_3 = np.mean([s['mae'] for s in timestep_stats[:3]])
        last_3 = np.mean([s['mae'] for s in timestep_stats[-3:]])
        growth_rate = (last_3 - first_3) / first_3 * 100

        print(f"  前3步平均MAE: {first_3:.3f}")
        print(f"  后3步平均MAE: {last_3:.3f}")
        print(f"  误差增长率: {growth_rate:+.1f}%")

        if growth_rate > 30:
            print(f"\n  ⚠️  长期预测能力较弱!")
            print(f"\n💡 优化建议:")
            print(f"  1. 增强时间建模能力:")
            print(f"     - 增加时间注意力层的深度 (L参数)")
            print(f"     - 使用更大的时间注意力窗口")
            print(f"  2. 使用时间步加权损失:")
            print(f"     - 对后面的时间步赋予更高权重")
            print(f"     - 参考 utils/metrics.py 中的 temporal_weighted_loss")
            print(f"  3. 增加历史窗口:")
            print(f"     - 当前P={timestep_stats[0]['timestep']-1}，可尝试增大到18或24")

        # 检查系统性偏差
        avg_bias_ratio = np.mean([abs(s['bias_ratio']) for s in timestep_stats])
        if avg_bias_ratio > 10:
            bias_direction = "过预测" if np.mean([s['bias']) for s in timestep_stats]) > 0 else "欠预测"
            print(f"\n  ⚠️  存在系统性{bias_direction}! (平均偏差率: {avg_bias_ratio:.1f}%)")
            print(f"\n💡 优化建议:")
            print(f"  1. 检查数据归一化方法")
            print(f"  2. 尝试不同的损失函数")
            print(f"  3. 添加偏差校正层")

        # 可视化
        self._plot_temporal_error_pattern(timestep_stats)

        return timestep_stats

    def analyze_error_by_node_type(self):
        """按节点类型分析误差"""
        print("\n" + "="*80)
        print("按节点特征分析误差")
        print("="*80)

        num_nodes = self.y_true_all.shape[2]

        # 分析每个节点
        node_stats = []
        for n in range(num_nodes):
            mae_n = self.errors_all[:, :, n].mean()
            rmse_n = np.sqrt(np.mean((self.y_pred_all[:, :, n] - self.y_true_all[:, :, n]) ** 2))
            true_mean = self.y_true_all[:, :, n].mean()
            true_std = self.y_true_all[:, :, n].std()
            pred_std = self.y_pred_all[:, :, n].std()

            # 流量变异系数
            cv = true_std / (true_mean + 1e-5)

            node_stats.append({
                'node': n,
                'mae': mae_n,
                'rmse': rmse_n,
                'true_mean': true_mean,
                'true_std': true_std,
                'pred_std': pred_std,
                'cv': cv
            })

        df_nodes = pd.DataFrame(node_stats)

        # 按MAE分组
        df_nodes['error_level'] = pd.cut(df_nodes['mae'],
                                          bins=[0, df_nodes['mae'].quantile(0.5),
                                                df_nodes['mae'].quantile(0.8),
                                                float('inf')],
                                          labels=['Low Error', 'Medium Error', 'High Error'])

        # 分析高误差节点的特征
        high_error_nodes = df_nodes[df_nodes['error_level'] == 'High Error']

        print(f"\n高误差节点 (Top 20%):")
        print(f"  节点数: {len(high_error_nodes)}")
        print(f"  平均MAE: {high_error_nodes['mae'].mean():.3f}")
        print(f"  平均流量: {high_error_nodes['true_mean'].mean():.2f}")
        print(f"  平均变异系数: {high_error_nodes['cv'].mean():.3f}")

        # 显示最差的10个节点
        worst_nodes = df_nodes.nlargest(10, 'mae')
        print(f"\nMAE最高的10个节点:")
        print(f"{'节点':<8} {'MAE':<10} {'平均流量':<12} {'流量std':<12} {'变异系数':<12}")
        print("-" * 60)
        for _, row in worst_nodes.iterrows():
            print(f"{int(row['node']):<8} {row['mae']:<10.3f} {row['true_mean']:<12.2f} "
                  f"{row['true_std']:<12.2f} {row['cv']:<12.3f}")

        # 诊断
        print(f"\n📊 诊断结论:")

        # 检查高误差节点的特征
        high_cv = high_error_nodes['cv'].mean()
        low_error_nodes = df_nodes[df_nodes['error_level'] == 'Low Error']
        low_cv = low_error_nodes['cv'].mean()

        print(f"  高误差节点平均变异系数: {high_cv:.3f}")
        print(f"  低误差节点平均变异系数: {low_cv:.3f}")

        if high_cv > low_cv * 1.5:
            print(f"\n  ⚠️  高误差节点的流量波动更大!")
            print(f"\n💡 优化建议:")
            print(f"  1. 增强对波动的建模能力:")
            print(f"     - 使用更深的网络")
            print(f"     - 增加注意力头数K")
            print(f"  2. 对高波动节点使用不同的策略:")
            print(f"     - 节点级别的注意力权重")
            print(f"     - 自适应的正则化")

        # 保存节点分析结果
        csv_path = os.path.join(self.output_dir, 'node_analysis.csv')
        df_nodes.to_csv(csv_path, index=False)
        print(f"\n  已保存节点分析结果: node_analysis.csv")

        # 可视化
        self._plot_node_analysis(df_nodes)

        return df_nodes

    def visualize_worst_samples(self, top_k=10):
        """可视化最差的样本"""
        print("\n" + "="*80)
        print(f"可视化最差的 {top_k} 个样本")
        print("="*80)

        # 计算每个样本的平均误差
        sample_mae = self.errors_all.mean(axis=(1, 2))
        worst_indices = np.argsort(sample_mae)[-top_k:][::-1]

        for rank, idx in enumerate(worst_indices, 1):
            mae = sample_mae[idx]
            y_true = self.y_true_all[idx]  # (timesteps, nodes)
            y_pred = self.y_pred_all[idx]
            x_hist = self.x_all[idx]  # 历史数据

            # 找出误差最大的节点
            node_mae = self.errors_all[idx].mean(axis=0)  # (nodes,)
            top_error_node = np.argmax(node_mae)

            print(f"\n样本 #{rank} (索引={idx}, MAE={mae:.3f})")
            print(f"  误差最大的节点: Node {top_error_node} (MAE={node_mae[top_error_node]:.3f})")

            # 可视化该节点的预测
            self._plot_sample_prediction(
                idx, top_error_node, x_hist[:, top_error_node],
                y_true[:, top_error_node], y_pred[:, top_error_node],
                mae, rank
            )

    def _plot_sample_prediction(self, sample_idx, node_idx, x_hist, y_true, y_pred, mae, rank):
        """绘制单个样本的预测曲线"""
        fig, ax = plt.subplots(figsize=(12, 6))

        hist_len = len(x_hist)
        pred_len = len(y_true)

        # 时间轴
        hist_time = list(range(-hist_len, 0))
        pred_time = list(range(0, pred_len))

        # 绘制历史
        ax.plot(hist_time, x_hist, 'o-', color='gray', label='Historical', linewidth=2, markersize=5)

        # 绘制真实值和预测值
        ax.plot(pred_time, y_true, 'o-', color='green', label='True', linewidth=2, markersize=6)
        ax.plot(pred_time, y_pred, 's--', color='red', label='Predicted', linewidth=2, markersize=6)

        # 标注误差
        errors = np.abs(y_pred - y_true)
        for t, err in enumerate(errors):
            if err > mae * 1.5:  # 标注超过平均误差1.5倍的点
                ax.annotate(f'{err:.1f}',
                           xy=(pred_time[t], y_pred[t]),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           fontsize=8,
                           color='red')

        ax.axvline(x=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Flow Value', fontsize=12)
        ax.set_title(f'Worst Sample #{rank} (Sample {sample_idx}, Node {node_idx}, MAE={mae:.3f})',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'worst_sample_{rank}_idx{sample_idx}_node{node_idx}.png'
        plt.savefig(os.path.join(self.output_dir, 'sample_details', filename), dpi=300)
        plt.close()

        print(f"    已保存: sample_details/{filename}")

    def _plot_error_by_flow(self, results):
        """绘制按流量分组的误差"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ranges = [r['range'] for r in results]
        maes = [r['mae'] for r in results]
        ratios = [r['ratio'] for r in results]

        # MAE柱状图
        ax1.bar(ranges, maes, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Flow Range', fontsize=12)
        ax1.set_ylabel('MAE', fontsize=12)
        ax1.set_title('MAE by Flow Range', fontsize=13)
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 样本分布饼图
        ax2.pie(ratios, labels=ranges, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Sample Distribution by Flow Range', fontsize=13)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'error_by_flow.png'), dpi=300)
        plt.close()
        print(f"  已保存: plots/error_by_flow.png")

    def _plot_temporal_error_pattern(self, timestep_stats):
        """绘制时间步误差模式"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        timesteps = [s['timestep'] for s in timestep_stats]
        maes = [s['mae'] for s in timestep_stats]
        biases = [s['bias'] for s in timestep_stats]

        # MAE曲线
        axes[0].plot(timesteps, maes, 'o-', linewidth=2, markersize=6, color='steelblue')
        axes[0].set_xlabel('Prediction Timestep', fontsize=12)
        axes[0].set_ylabel('MAE', fontsize=12)
        axes[0].set_title('MAE by Prediction Timestep', fontsize=13)
        axes[0].grid(True, alpha=0.3)

        # 偏差曲线
        axes[1].plot(timesteps, biases, 'o-', linewidth=2, markersize=6, color='coral')
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Prediction Timestep', fontsize=12)
        axes[1].set_ylabel('Bias (Pred - True)', fontsize=12)
        axes[1].set_title('Prediction Bias by Timestep', fontsize=13)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'temporal_error_pattern.png'), dpi=300)
        plt.close()
        print(f"  已保存: plots/temporal_error_pattern.png")

    def _plot_node_analysis(self, df_nodes):
        """绘制节点分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # MAE分布
        axes[0, 0].hist(df_nodes['mae'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('MAE', fontsize=11)
        axes[0, 0].set_ylabel('Number of Nodes', fontsize=11)
        axes[0, 0].set_title('Distribution of Node MAE', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)

        # MAE vs 平均流量
        axes[0, 1].scatter(df_nodes['true_mean'], df_nodes['mae'], alpha=0.5, s=30)
        axes[0, 1].set_xlabel('Average Flow', fontsize=11)
        axes[0, 1].set_ylabel('MAE', fontsize=11)
        axes[0, 1].set_title('MAE vs Average Flow', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)

        # MAE vs 变异系数
        axes[1, 0].scatter(df_nodes['cv'], df_nodes['mae'], alpha=0.5, s=30, color='coral')
        axes[1, 0].set_xlabel('Coefficient of Variation', fontsize=11)
        axes[1, 0].set_ylabel('MAE', fontsize=11)
        axes[1, 0].set_title('MAE vs Flow Variability', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)

        # Top20高误差节点
        top20 = df_nodes.nlargest(20, 'mae')
        axes[1, 1].barh(range(len(top20)), top20['mae'].values)
        axes[1, 1].set_yticks(range(len(top20)))
        axes[1, 1].set_yticklabels([f"Node {int(n)}" for n in top20['node'].values], fontsize=8)
        axes[1, 1].set_xlabel('MAE', fontsize=11)
        axes[1, 1].set_title('Top 20 Nodes by MAE', fontsize=12)
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'node_analysis.png'), dpi=300)
        plt.close()
        print(f"  已保存: plots/node_analysis.png")

    def generate_optimization_report(self):
        """生成优化建议报告"""
        print("\n" + "="*80)
        print("生成优化建议报告")
        print("="*80)

        report_path = os.path.join(self.output_dir, 'optimization_suggestions.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("模型优化建议报告\n")
            f.write(f"生成时间: {datetime.datetime.now()}\n")
            f.write("="*80 + "\n\n")

            f.write("基于误差分析，以下是针对性的优化建议:\n\n")

            f.write("📌 优先级1: 立即尝试\n")
            f.write("-" * 40 + "\n")
            f.write("1. 调整损失函数\n")
            f.write("   - 如果低流量误差大: 使用加权MAE或Focal Loss\n")
            f.write("   - 如果有异常值: 使用Huber Loss\n")
            f.write("   - 如果长期预测差: 使用时间步加权损失\n")
            f.write("   参考: utils/metrics.py 中的损失函数\n\n")

            f.write("2. 数据预处理\n")
            f.write("   - 检查数据归一化方法 (StandardScaler vs MinMaxScaler)\n")
            f.write("   - 对低流量数据: 尝试 log(x+1) 变换\n")
            f.write("   - 检查是否有数据异常值需要处理\n\n")

            f.write("📌 优先级2: 调整超参数\n")
            f.write("-" * 40 + "\n")
            f.write("1. 如果长期预测能力弱:\n")
            f.write("   - 增加历史窗口 P (12 -> 18 或 24)\n")
            f.write("   - 增加STAtt层数 L (2 -> 3)\n")
            f.write("   - 增加注意力头数 K (8 -> 12)\n\n")

            f.write("2. 如果特定节点误差大:\n")
            f.write("   - 检查图结构是否合理\n")
            f.write("   - 调整空间注意力机制\n")
            f.write("   - 考虑添加节点特征\n\n")

            f.write("📌 优先级3: 训练策略\n")
            f.write("-" * 40 + "\n")
            f.write("1. 学习率调整\n")
            f.write("   - 降低初始学习率 (0.001 -> 0.0005)\n")
            f.write("   - 使用warmup策略\n")
            f.write("   - 尝试Cosine Annealing\n\n")

            f.write("2. 正则化\n")
            f.write("   - 添加weight decay\n")
            f.write("   - 添加Dropout (0.1-0.3)\n")
            f.write("   - Label smoothing\n\n")

            f.write("3. 数据增强\n")
            f.write("   - 对高误差样本过采样\n")
            f.write("   - 时间窗口滑动采样\n\n")

            f.write("="*80 + "\n")
            f.write("详细分析请查看其他输出文件\n")
            f.write("="*80 + "\n")

        print(f"  已保存优化建议: optimization_suggestions.txt")

    def run_full_diagnosis(self):
        """运行完整诊断"""
        print("\n" + "="*80)
        print("开始模型诊断")
        print("="*80)

        # 1. 收集数据
        self.collect_predictions()

        # 2. 整体统计
        rmse, mae, mape = RMSE_MAE_MAPE(self.y_true_all, self.y_pred_all)
        print(f"\n整体性能:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MAPE: {mape:.4f}%")

        # 3. 按流量大小分析
        flow_results = self.analyze_error_by_flow_magnitude()

        # 4. 按时间模式分析
        temporal_results = self.analyze_error_by_temporal_pattern()

        # 5. 按节点分析
        node_results = self.analyze_error_by_node_type()

        # 6. 可视化最差样本
        self.visualize_worst_samples(top_k=10)

        # 7. 生成优化建议
        self.generate_optimization_report()

        print(f"\n" + "="*80)
        print(f"诊断完成! 所有结果保存在: {self.output_dir}")
        print(f"  - plots/                  可视化图表")
        print(f"  - sample_details/         高误差样本详情")
        print(f"  - node_analysis.csv       节点分析数据")
        print(f"  - optimization_suggestions.txt  优化建议")
        print("="*80)


def load_model_and_data(args):
    """加载模型和数据"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("加载数据...")
    _, _, test_loader, scaler = data_prepare.get_dataloaders(args, log=None)

    # 数据集信息
    dataset_name = args.traffic_file.split('/')[-1].replace('.npz', '')
    dataset_dir = '/'.join(args.traffic_file.split('/')[:-1])
    csv_file = os.path.join(dataset_dir, f'{dataset_name}.csv')
    txt_file = os.path.join(dataset_dir, f'{dataset_name}.txt')

    # 读取节点
    temp_nodes = set()
    with open(csv_file, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            temp_nodes.add(int(row[0]))
            temp_nodes.add(int(row[1]))

    if os.path.exists(txt_file):
        with open(txt_file, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}
        num_nodes = len(id_dict)
    else:
        sorted_nodes = sorted(list(temp_nodes))
        id_dict = {node_id: idx for idx, node_id in enumerate(sorted_nodes)}
        num_nodes = len(sorted_nodes)

    # 邻接矩阵
    adj_mx = np.zeros((num_nodes, num_nodes), dtype=float)
    with open(csv_file, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if i in id_dict and j in id_dict:
                idx_i = id_dict[i]
                idx_j = id_dict[j]
                adj_mx[idx_i][idx_j] = 1
                adj_mx[idx_j][idx_i] = 1

    lap_mx, LAP = cal_lape(adj_mx)
    lap_mx = lap_mx.to(device)

    # 加载模型
    print("加载模型...")
    model = GMAN(args.input_dim, args.P, args.Q, args.T, args.L, args.K, args.d, lap_mx, LAP)
    model = model.to(device)

    if args.model_path:
        print(f"从 {args.model_path} 加载模型权重...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    return model, test_loader, scaler, device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模型诊断工具')

    parser.add_argument('--time_slot', type=int, default=5)
    parser.add_argument('--P', type=int, default=12)
    parser.add_argument('--Q', type=int, default=12)
    parser.add_argument('--L', type=int, default=2)
    parser.add_argument('--T', type=int, default=288)
    parser.add_argument('--embed_dim', type=int, default=1)
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--d', type=int, default=8)
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--traffic_file', default='data/PEMS03/PEMS03.npz')
    parser.add_argument('--model_path', required=True, help='模型路径')
    parser.add_argument('--output_dir', default='./model_diagnosis', help='输出目录')

    args = parser.parse_args()

    # 加载并诊断
    model, test_loader, scaler, device = load_model_and_data(args)
    diagnostics = ModelDiagnostics(model, test_loader, scaler, device, args.output_dir)
    diagnostics.run_full_diagnosis()

    print("\n诊断完成!")