"""
误差热力图可视化工具 - 生成时空误差热力图

功能:
1. 生成时间-节点误差热力图
2. 可视化误差在时空上的分布
3. 识别高误差的时空区域
4. 对比两个模型的误差热力图
"""

import argparse
import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import utils
import data_prepare
from model import GMAN
from utils import cal_lape


class ErrorHeatmapVisualizer:
    """误差热力图可视化器"""

    def __init__(self, model, test_loader, scaler, device, output_dir='./error_heatmaps'):
        self.model = model
        self.test_loader = test_loader
        self.scaler = scaler
        self.device = device
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    @torch.no_grad()
    def generate_error_matrices(self):
        """生成误差矩阵"""
        print("生成误差矩阵...")
        self.model.eval()

        all_errors = []

        for batch in self.test_loader:
            batch.to_tensor(self.device)
            x_batch = batch['x']
            y_batch = batch['y']

            TE = x_batch[:, :, :, 1:]

            # 预测
            out_batch = self.model(x_batch, TE)
            out_batch = self.scaler.inverse_transform(out_batch)
            y_batch = self.scaler.inverse_transform(y_batch[:, :, :, 0])

            # 计算误差
            errors = torch.abs(out_batch - y_batch).cpu().numpy()
            all_errors.append(errors)

        # 合并所有batch
        all_errors = np.concatenate(all_errors, axis=0)  # (samples, timesteps, nodes)

        print(f"误差矩阵形状: {all_errors.shape}")

        return all_errors

    def plot_timestep_node_heatmap(self, errors, sample_idx=None, top_k_nodes=50):
        """绘制时间步-节点热力图"""
        if sample_idx is not None:
            # 单个样本
            error_matrix = errors[sample_idx]  # (timesteps, nodes)
            title = f'Error Heatmap - Sample {sample_idx}'
            filename = f'heatmap_sample_{sample_idx}.png'
        else:
            # 平均误差
            error_matrix = np.mean(errors, axis=0)  # (timesteps, nodes)
            title = 'Average Error Heatmap - All Samples'
            filename = 'heatmap_average.png'

        # 选择误差最大的top_k个节点
        node_avg_errors = np.mean(error_matrix, axis=0)
        top_nodes = np.argsort(node_avg_errors)[-top_k_nodes:]
        error_matrix_subset = error_matrix[:, top_nodes]

        # 绘制热力图
        plt.figure(figsize=(16, 8))
        sns.heatmap(
            error_matrix_subset.T,
            cmap='YlOrRd',
            cbar_kws={'label': 'Absolute Error'},
            xticklabels=range(1, error_matrix.shape[0] + 1),
            yticklabels=[f'Node {n}' for n in top_nodes]
        )
        plt.xlabel('Prediction Timestep', fontsize=12)
        plt.ylabel(f'Top {top_k_nodes} Nodes (by avg error)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

        print(f"  已保存热力图: {filename}")

    def plot_spatial_error_distribution(self, errors, timestep=None):
        """绘制空间误差分布"""
        if timestep is not None:
            # 特定时间步
            error_vec = np.mean(errors[:, timestep, :], axis=0)  # (nodes,)
            title = f'Spatial Error Distribution - Timestep {timestep + 1}'
            filename = f'spatial_distribution_t{timestep + 1}.png'
        else:
            # 所有时间步平均
            error_vec = np.mean(errors, axis=(0, 1))  # (nodes,)
            title = 'Spatial Error Distribution - All Timesteps'
            filename = 'spatial_distribution_all.png'

        # 绘制柱状图
        plt.figure(figsize=(14, 6))
        plt.bar(range(len(error_vec)), error_vec, alpha=0.7)
        plt.xlabel('Node ID', fontsize=12)
        plt.ylabel('Average Absolute Error', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

        print(f"  已保存空间分布图: {filename}")

    def plot_temporal_error_distribution(self, errors, node=None):
        """绘制时间误差分布"""
        if node is not None:
            # 特定节点
            error_vec = np.mean(errors[:, :, node], axis=0)  # (timesteps,)
            title = f'Temporal Error Distribution - Node {node}'
            filename = f'temporal_distribution_node{node}.png'
        else:
            # 所有节点平均
            error_vec = np.mean(errors, axis=(0, 2))  # (timesteps,)
            title = 'Temporal Error Distribution - All Nodes'
            filename = 'temporal_distribution_all.png'

        # 绘制折线图
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(error_vec) + 1), error_vec, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Prediction Timestep', fontsize=12)
        plt.ylabel('Average Absolute Error', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

        print(f"  已保存时间分布图: {filename}")

    def plot_error_statistics_heatmap(self, errors):
        """绘制误差统计热力图(每个时间步-节点的统计信息)"""
        num_timesteps = errors.shape[1]
        num_nodes = errors.shape[2]

        # 计算每个时间步-节点的统计信息
        mean_errors = np.mean(errors, axis=0)  # (timesteps, nodes)
        std_errors = np.std(errors, axis=0)
        max_errors = np.max(errors, axis=0)
        p95_errors = np.percentile(errors, 95, axis=0)

        # 绘制4个子图
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))

        # 选择误差最大的50个节点用于展示
        node_avg = np.mean(mean_errors, axis=0)
        top_nodes = np.argsort(node_avg)[-50:]

        # 平均误差
        sns.heatmap(mean_errors[:, top_nodes].T, cmap='YlOrRd', ax=axes[0, 0],
                    cbar_kws={'label': 'Mean Error'}, xticklabels=range(1, num_timesteps + 1),
                    yticklabels=[f'N{n}' for n in top_nodes])
        axes[0, 0].set_title('Mean Error', fontsize=12)
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('Top 50 Nodes')

        # 标准差
        sns.heatmap(std_errors[:, top_nodes].T, cmap='Blues', ax=axes[0, 1],
                    cbar_kws={'label': 'Std Error'}, xticklabels=range(1, num_timesteps + 1),
                    yticklabels=[f'N{n}' for n in top_nodes])
        axes[0, 1].set_title('Standard Deviation', fontsize=12)
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('Top 50 Nodes')

        # 最大误差
        sns.heatmap(max_errors[:, top_nodes].T, cmap='Reds', ax=axes[1, 0],
                    cbar_kws={'label': 'Max Error'}, xticklabels=range(1, num_timesteps + 1),
                    yticklabels=[f'N{n}' for n in top_nodes])
        axes[1, 0].set_title('Maximum Error', fontsize=12)
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('Top 50 Nodes')

        # 95分位误差
        sns.heatmap(p95_errors[:, top_nodes].T, cmap='OrRd', ax=axes[1, 1],
                    cbar_kws={'label': 'P95 Error'}, xticklabels=range(1, num_timesteps + 1),
                    yticklabels=[f'N{n}' for n in top_nodes])
        axes[1, 1].set_title('95th Percentile Error', fontsize=12)
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].set_ylabel('Top 50 Nodes')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'error_statistics_heatmap.png'), dpi=300)
        plt.close()

        print(f"  已保存误差统计热力图: error_statistics_heatmap.png")

    def generate_all_visualizations(self, top_k_samples=5, top_k_nodes=50):
        """生成所有可视化"""
        print("\n" + "="*80)
        print("生成误差热力图可视化")
        print("="*80)

        # 1. 生成误差矩阵
        errors = self.generate_error_matrices()

        # 2. 平均误差热力图
        print("\n生成平均误差热力图...")
        self.plot_timestep_node_heatmap(errors, sample_idx=None, top_k_nodes=top_k_nodes)

        # 3. 高误差样本的热力图
        print(f"\n生成Top {top_k_samples}高误差样本的热力图...")
        sample_avg_errors = np.mean(errors, axis=(1, 2))
        top_error_samples = np.argsort(sample_avg_errors)[-top_k_samples:]

        for i, sample_idx in enumerate(top_error_samples, 1):
            print(f"  生成样本 {sample_idx} (第{i}高误差)...")
            self.plot_timestep_node_heatmap(errors, sample_idx=sample_idx, top_k_nodes=30)

        # 4. 空间误差分布
        print("\n生成空间误差分布...")
        self.plot_spatial_error_distribution(errors, timestep=None)
        # 特定时间步
        for t in [0, 5, 11]:  # 第1步、第6步、第12步
            self.plot_spatial_error_distribution(errors, timestep=t)

        # 5. 时间误差分布
        print("\n生成时间误差分布...")
        self.plot_temporal_error_distribution(errors, node=None)
        # 找出误差最大的3个节点
        node_avg_errors = np.mean(errors, axis=(0, 1))
        top_error_nodes = np.argsort(node_avg_errors)[-3:]
        for node in top_error_nodes:
            self.plot_temporal_error_distribution(errors, node=node)

        # 6. 误差统计热力图
        print("\n生成误差统计热力图...")
        self.plot_error_statistics_heatmap(errors)

        print(f"\n可视化完成! 结果保存在: {self.output_dir}")


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
    parser = argparse.ArgumentParser(description='误差热力图可视化工具')

    # 数据参数
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

    # 文件路径
    parser.add_argument('--traffic_file', default='data/PEMS03/PEMS03.npz')
    parser.add_argument('--model_path', required=True, help='训练好的模型路径')
    parser.add_argument('--output_dir', default='./error_heatmaps', help='输出目录')

    # 可视化参数
    parser.add_argument('--top_k_samples', type=int, default=5, help='可视化前K个高误差样本')
    parser.add_argument('--top_k_nodes', type=int, default=50, help='热力图显示前K个高误差节点')

    args = parser.parse_args()

    # 加载模型和数据
    model, test_loader, scaler, device = load_model_and_data(args)

    # 创建可视化器并生成图表
    visualizer = ErrorHeatmapVisualizer(model, test_loader, scaler, device, args.output_dir)
    visualizer.generate_all_visualizations(args.top_k_samples, args.top_k_nodes)

    print("\n可视化完成!")