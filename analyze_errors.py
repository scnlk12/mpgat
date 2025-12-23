"""
误差分析工具 - 用于分析测试集中预测误差较大的样本

功能:
1. 逐样本分析预测误差
2. 识别高误差样本(时间段、节点)
3. 按时间步、节点、时间段统计误差分布
4. 生成详细的可视化报告
"""

import argparse
import time
import datetime
import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

import utils
import data_prepare
from model import GMAN
from utils import cal_lape
from utils.metrics import RMSE_MAE_MAPE


class ErrorAnalyzer:
    """误差分析器"""

    def __init__(self, model, test_loader, scaler, device, output_dir='./error_analysis'):
        self.model = model
        self.test_loader = test_loader
        self.scaler = scaler
        self.device = device
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

        # 存储预测结果和误差
        self.y_true_all = []
        self.y_pred_all = []
        self.errors_all = []
        self.sample_indices = []

    @torch.no_grad()
    def collect_predictions(self):
        """收集所有预测结果和真实值"""
        print("正在收集预测结果...")
        self.model.eval()

        sample_idx = 0
        for batch in self.test_loader:
            batch.to_tensor(self.device)
            x_batch = batch['x']
            y_batch = batch['y']

            TE = x_batch[:, :, :, 1:]

            # 预测
            out_batch = self.model(x_batch, TE)
            out_batch = self.scaler.inverse_transform(out_batch)
            y_batch = self.scaler.inverse_transform(y_batch[:, :, :, 0])

            # 转换为numpy
            out_batch = out_batch.cpu().numpy()
            y_batch = y_batch.cpu().numpy()

            # 保存
            batch_size = out_batch.shape[0]
            for i in range(batch_size):
                self.y_true_all.append(y_batch[i])
                self.y_pred_all.append(out_batch[i])
                self.sample_indices.append(sample_idx)
                sample_idx += 1

        # 转换为numpy数组
        self.y_true_all = np.array(self.y_true_all)  # (samples, out_steps, num_nodes)
        self.y_pred_all = np.array(self.y_pred_all)  # (samples, out_steps, num_nodes)

        # 计算误差
        self.errors_all = np.abs(self.y_pred_all - self.y_true_all)
        self.mape_all = np.abs((self.y_pred_all - self.y_true_all) / (self.y_true_all + 1e-5)) * 100

        print(f"收集完成: {len(self.y_true_all)} 个样本")
        print(f"数据形状: {self.y_true_all.shape}")

    def analyze_overall_statistics(self):
        """分析整体统计信息"""
        print("\n" + "="*80)
        print("整体统计分析")
        print("="*80)

        # 计算整体指标
        rmse, mae, mape = RMSE_MAE_MAPE(self.y_true_all, self.y_pred_all)

        print(f"整体性能:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MAPE: {mape:.4f}%")

        # 误差分布统计
        print(f"\n误差分布:")
        print(f"  最小误差: {self.errors_all.min():.4f}")
        print(f"  25%分位: {np.percentile(self.errors_all, 25):.4f}")
        print(f"  中位数:   {np.median(self.errors_all):.4f}")
        print(f"  75%分位: {np.percentile(self.errors_all, 75):.4f}")
        print(f"  90%分位: {np.percentile(self.errors_all, 90):.4f}")
        print(f"  95%分位: {np.percentile(self.errors_all, 95):.4f}")
        print(f"  99%分位: {np.percentile(self.errors_all, 99):.4f}")
        print(f"  最大误差: {self.errors_all.max():.4f}")

        # 零流量分析
        zero_mask = self.y_true_all < 1e-3
        zero_ratio = zero_mask.sum() / self.y_true_all.size
        print(f"\n零流量分析:")
        print(f"  零流量比例: {zero_ratio*100:.2f}%")
        print(f"  零流量MAE: {self.errors_all[zero_mask].mean():.4f}")
        print(f"  非零流量MAE: {self.errors_all[~zero_mask].mean():.4f}")

        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'error_stats': {
                'min': float(self.errors_all.min()),
                'p25': float(np.percentile(self.errors_all, 25)),
                'median': float(np.median(self.errors_all)),
                'p75': float(np.percentile(self.errors_all, 75)),
                'p90': float(np.percentile(self.errors_all, 90)),
                'p95': float(np.percentile(self.errors_all, 95)),
                'p99': float(np.percentile(self.errors_all, 99)),
                'max': float(self.errors_all.max()),
            },
            'zero_flow': {
                'ratio': float(zero_ratio),
                'mae': float(self.errors_all[zero_mask].mean()),
                'non_zero_mae': float(self.errors_all[~zero_mask].mean()),
            }
        }

    def analyze_timestep_errors(self):
        """分析不同时间步的误差"""
        print("\n" + "="*80)
        print("时间步误差分析")
        print("="*80)

        num_timesteps = self.y_true_all.shape[1]
        timestep_errors = []

        for t in range(num_timesteps):
            mae_t = np.mean(self.errors_all[:, t, :])
            rmse_t = np.sqrt(np.mean((self.y_pred_all[:, t, :] - self.y_true_all[:, t, :]) ** 2))
            mape_t = np.mean(self.mape_all[:, t, :])

            timestep_errors.append({
                'timestep': t + 1,
                'mae': mae_t,
                'rmse': rmse_t,
                'mape': mape_t
            })

            print(f"时间步 {t+1:2d}: MAE={mae_t:6.3f}, RMSE={rmse_t:6.3f}, MAPE={mape_t:6.2f}%")

        # 可视化时间步误差
        self._plot_timestep_errors(timestep_errors)

        return timestep_errors

    def analyze_node_errors(self, top_k=20):
        """分析不同节点的误差"""
        print("\n" + "="*80)
        print(f"节点误差分析 (Top {top_k})")
        print("="*80)

        num_nodes = self.y_true_all.shape[2]
        node_errors = []

        for n in range(num_nodes):
            mae_n = np.mean(self.errors_all[:, :, n])
            rmse_n = np.sqrt(np.mean((self.y_pred_all[:, :, n] - self.y_true_all[:, :, n]) ** 2))
            mape_n = np.mean(self.mape_all[:, :, n])
            avg_flow = np.mean(self.y_true_all[:, :, n])

            node_errors.append({
                'node': n,
                'mae': mae_n,
                'rmse': rmse_n,
                'mape': mape_n,
                'avg_flow': avg_flow
            })

        # 按MAE排序
        node_errors_sorted = sorted(node_errors, key=lambda x: x['mae'], reverse=True)

        print(f"\nMAE最高的 {top_k} 个节点:")
        print(f"{'节点ID':<8} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'平均流量':<10}")
        print("-" * 60)
        for i, item in enumerate(node_errors_sorted[:top_k], 1):
            print(f"{item['node']:<8} {item['mae']:<10.3f} {item['rmse']:<10.3f} "
                  f"{item['mape']:<10.2f} {item['avg_flow']:<10.2f}")

        # 可视化节点误差分布
        self._plot_node_errors(node_errors, top_k)

        return node_errors

    def find_high_error_samples(self, threshold_percentile=90):
        """找出高误差样本"""
        print("\n" + "="*80)
        print(f"高误差样本分析 (>{threshold_percentile}%分位)")
        print("="*80)

        # 计算每个样本的平均误差
        sample_mae = np.mean(self.errors_all, axis=(1, 2))
        threshold = np.percentile(sample_mae, threshold_percentile)

        high_error_indices = np.where(sample_mae > threshold)[0]

        print(f"阈值: {threshold:.3f}")
        print(f"高误差样本数: {len(high_error_indices)} / {len(sample_mae)}")
        print(f"比例: {len(high_error_indices) / len(sample_mae) * 100:.2f}%")

        # 分析高误差样本的特征
        high_error_samples = []
        for idx in high_error_indices[:50]:  # 只显示前50个
            sample_error = sample_mae[idx]
            sample_true = self.y_true_all[idx]
            sample_pred = self.y_pred_all[idx]

            # 找出该样本中误差最大的时间步和节点
            error_matrix = self.errors_all[idx]
            max_error_pos = np.unravel_index(error_matrix.argmax(), error_matrix.shape)
            max_error_timestep = max_error_pos[0]
            max_error_node = max_error_pos[1]
            max_error_value = error_matrix[max_error_pos]

            high_error_samples.append({
                'sample_idx': int(idx),
                'avg_mae': float(sample_error),
                'max_error': float(max_error_value),
                'max_error_timestep': int(max_error_timestep + 1),
                'max_error_node': int(max_error_node),
                'avg_true_value': float(sample_true.mean()),
                'avg_pred_value': float(sample_pred.mean()),
            })

        # 显示前10个
        print(f"\n误差最大的10个样本:")
        print(f"{'样本ID':<10} {'平均MAE':<12} {'最大误差':<12} {'位置(步,节点)':<18} {'真实均值':<12} {'预测均值':<12}")
        print("-" * 90)
        for i, sample in enumerate(high_error_samples[:10], 1):
            print(f"{sample['sample_idx']:<10} {sample['avg_mae']:<12.3f} "
                  f"{sample['max_error']:<12.3f} "
                  f"({sample['max_error_timestep']},{sample['max_error_node']}){' ':<8} "
                  f"{sample['avg_true_value']:<12.3f} {sample['avg_pred_value']:<12.3f}")

        # 保存详细列表
        self._save_high_error_samples(high_error_samples)

        return high_error_samples

    def analyze_error_patterns(self):
        """分析误差模式"""
        print("\n" + "="*80)
        print("误差模式分析")
        print("="*80)

        # 1. 误差与真实值的关系
        true_flat = self.y_true_all.flatten()
        error_flat = self.errors_all.flatten()

        # 按真实值分段统计误差
        bins = [0, 10, 20, 30, 50, 100, float('inf')]
        bin_labels = ['0-10', '10-20', '20-30', '30-50', '50-100', '100+']

        print("\n按真实流量值分段的MAE:")
        print(f"{'流量范围':<15} {'样本数':<15} {'平均MAE':<15} {'平均MAPE':<15}")
        print("-" * 60)

        for i in range(len(bins) - 1):
            mask = (true_flat >= bins[i]) & (true_flat < bins[i+1])
            if mask.sum() > 0:
                avg_mae = error_flat[mask].mean()
                avg_mape = (error_flat[mask] / (true_flat[mask] + 1e-5)).mean() * 100
                print(f"{bin_labels[i]:<15} {mask.sum():<15} {avg_mae:<15.3f} {avg_mape:<15.2f}%")

        # 2. 过预测 vs 欠预测
        over_pred = self.y_pred_all > self.y_true_all
        over_pred_ratio = over_pred.sum() / over_pred.size

        print(f"\n预测偏向分析:")
        print(f"  过预测比例: {over_pred_ratio*100:.2f}%")
        print(f"  欠预测比例: {(1-over_pred_ratio)*100:.2f}%")
        print(f"  过预测平均误差: {self.errors_all[over_pred].mean():.3f}")
        print(f"  欠预测平均误差: {self.errors_all[~over_pred].mean():.3f}")

    def _plot_timestep_errors(self, timestep_errors):
        """绘制时间步误差图"""
        timesteps = [e['timestep'] for e in timestep_errors]
        maes = [e['mae'] for e in timestep_errors]
        rmses = [e['rmse'] for e in timestep_errors]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # MAE
        ax1.plot(timesteps, maes, marker='o', linewidth=2, markersize=6)
        ax1.set_xlabel('Prediction Timestep', fontsize=12)
        ax1.set_ylabel('MAE', fontsize=12)
        ax1.set_title('MAE by Prediction Timestep', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # RMSE
        ax2.plot(timesteps, rmses, marker='s', linewidth=2, markersize=6, color='orange')
        ax2.set_xlabel('Prediction Timestep', fontsize=12)
        ax2.set_ylabel('RMSE', fontsize=12)
        ax2.set_title('RMSE by Prediction Timestep', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'timestep_errors.png'), dpi=300)
        plt.close()
        print(f"  已保存: timestep_errors.png")

    def _plot_node_errors(self, node_errors, top_k):
        """绘制节点误差图"""
        # 提取数据
        maes = [e['mae'] for e in node_errors]

        # 1. MAE分布直方图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 直方图
        axes[0, 0].hist(maes, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('MAE', fontsize=11)
        axes[0, 0].set_ylabel('Number of Nodes', fontsize=11)
        axes[0, 0].set_title('Distribution of Node MAE', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)

        # Top K节点柱状图
        sorted_errors = sorted(node_errors, key=lambda x: x['mae'], reverse=True)[:top_k]
        nodes = [e['node'] for e in sorted_errors]
        node_maes = [e['mae'] for e in sorted_errors]

        axes[0, 1].barh(range(len(nodes)), node_maes)
        axes[0, 1].set_yticks(range(len(nodes)))
        axes[0, 1].set_yticklabels([f"Node {n}" for n in nodes], fontsize=8)
        axes[0, 1].set_xlabel('MAE', fontsize=11)
        axes[0, 1].set_title(f'Top {top_k} Nodes by MAE', fontsize=12)
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(True, alpha=0.3, axis='x')

        # MAE vs 平均流量散点图
        avg_flows = [e['avg_flow'] for e in node_errors]
        axes[1, 0].scatter(avg_flows, maes, alpha=0.5, s=30)
        axes[1, 0].set_xlabel('Average Flow', fontsize=11)
        axes[1, 0].set_ylabel('MAE', fontsize=11)
        axes[1, 0].set_title('MAE vs Average Flow', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)

        # 节点MAE排序曲线
        sorted_maes = sorted(maes, reverse=True)
        axes[1, 1].plot(range(len(sorted_maes)), sorted_maes, linewidth=2)
        axes[1, 1].set_xlabel('Node Rank', fontsize=11)
        axes[1, 1].set_ylabel('MAE', fontsize=11)
        axes[1, 1].set_title('Sorted Node MAE Curve', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'node_errors.png'), dpi=300)
        plt.close()
        print(f"  已保存: node_errors.png")

    def _save_high_error_samples(self, high_error_samples):
        """保存高误差样本详情"""
        df = pd.DataFrame(high_error_samples)
        csv_path = os.path.join(self.output_dir, 'high_error_samples.csv')
        df.to_csv(csv_path, index=False)
        print(f"  已保存高误差样本列表: high_error_samples.csv")

    def generate_report(self):
        """生成完整的分析报告"""
        print("\n" + "="*80)
        print("开始生成误差分析报告")
        print("="*80)

        # 1. 收集预测
        self.collect_predictions()

        # 2. 整体统计
        overall_stats = self.analyze_overall_statistics()

        # 3. 时间步分析
        timestep_errors = self.analyze_timestep_errors()

        # 4. 节点分析
        node_errors = self.analyze_node_errors(top_k=20)

        # 5. 高误差样本
        high_error_samples = self.find_high_error_samples(threshold_percentile=90)

        # 6. 误差模式
        self.analyze_error_patterns()

        # 7. 保存完整报告
        report_path = os.path.join(self.output_dir, 'error_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("误差分析报告\n")
            f.write(f"生成时间: {datetime.datetime.now()}\n")
            f.write("="*80 + "\n\n")

            f.write(f"数据集大小: {len(self.y_true_all)} 个样本\n")
            f.write(f"数据形状: {self.y_true_all.shape}\n\n")

            f.write(f"整体性能:\n")
            f.write(f"  RMSE: {overall_stats['rmse']:.4f}\n")
            f.write(f"  MAE:  {overall_stats['mae']:.4f}\n")
            f.write(f"  MAPE: {overall_stats['mape']:.4f}%\n\n")

            f.write(f"误差分布:\n")
            for k, v in overall_stats['error_stats'].items():
                f.write(f"  {k}: {v:.4f}\n")

        print(f"\n分析完成! 报告已保存至: {self.output_dir}")
        print(f"  - 完整报告: error_analysis_report.txt")
        print(f"  - 高误差样本: high_error_samples.csv")
        print(f"  - 可视化图表: plots/")


def load_model_and_data(args):
    """加载模型和数据"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载数据
    print("加载数据...")
    _, _, test_loader, scaler = data_prepare.get_dataloaders(args, log=None)

    # 自动推断数据集信息
    dataset_name = args.traffic_file.split('/')[-1].replace('.npz', '')
    dataset_dir = '/'.join(args.traffic_file.split('/')[:-1])
    csv_file = os.path.join(dataset_dir, f'{dataset_name}.csv')
    txt_file = os.path.join(dataset_dir, f'{dataset_name}.txt')

    # 读取节点信息
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

    # 构建邻接矩阵
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
    else:
        print("警告: 未指定模型路径,使用未训练的模型!")

    return model, test_loader, scaler, device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='误差分析工具')

    # 数据参数
    parser.add_argument('--time_slot', type=int, default=5, help='时间步长(分钟)')
    parser.add_argument('--P', type=int, default=12, help='历史步数')
    parser.add_argument('--Q', type=int, default=12, help='预测步数')
    parser.add_argument('--L', type=int, default=2, help='STAtt块数量')
    parser.add_argument('--T', type=int, default=288, help='一天的时间步数')
    parser.add_argument('--embed_dim', type=int, default=1)
    parser.add_argument('--K', type=int, default=8, help='注意力头数')
    parser.add_argument('--input_dim', type=int, default=3, help='输入维度')
    parser.add_argument('--d', type=int, default=8, help='每个注意力头的维度')
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=16)

    # 文件路径
    parser.add_argument('--traffic_file', default='data/PEMS03/PEMS03.npz', help='交通数据文件')
    parser.add_argument('--model_path', required=True, help='训练好的模型路径')
    parser.add_argument('--output_dir', default='./error_analysis', help='输出目录')

    args = parser.parse_args()

    # 加载模型和数据
    model, test_loader, scaler, device = load_model_and_data(args)

    # 创建分析器并生成报告
    analyzer = ErrorAnalyzer(model, test_loader, scaler, device, args.output_dir)
    analyzer.generate_report()

    print("\n分析完成!")