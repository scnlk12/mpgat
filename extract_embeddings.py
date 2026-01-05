"""
提取 MPGAT 模型的 embedding 用于迁移学习

这个脚本会：
1. 加载训练好的模型
2. 在测试集上提取 end_conv1 后的 embedding [B, Q, N, 256]
3. 保存 embedding 和对应的标签用于迁移学习
"""

import torch
import numpy as np
import yaml
import os
from tqdm import tqdm
from model.model import GMAN
from utils.data_prepare import data_loader
from utils.utils import compute_laplacian_matrix


def extract_embeddings(config_path='config.yaml', save_dir='embeddings'):
    """
    提取模型的 embedding

    Args:
        config_path: 配置文件路径
        save_dir: 保存 embedding 的目录
    """
    # 读取配置
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'])

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    print("加载数据...")
    train_dataloader, val_dataloader, test_dataloader, scaler = data_loader(
        dataset=config['data']['dataset'],
        batch_size=config['data']['batch_size'],
        test_batch_size=config['data']['test_batch_size'],
    )

    # 计算拉普拉斯矩阵
    print("计算图结构...")
    laplacian_pe, laplacian_matrix = compute_laplacian_matrix(
        f"data/{config['data']['dataset']}/{config['data']['dataset']}.csv",
        config['data']['lag'],
    )
    laplacian_matrix = torch.FloatTensor(laplacian_matrix).to(device)

    # 加载模型
    print("加载模型...")
    model = GMAN(
        model_dim=config['model']['model_dim'],
        P=config['data']['lag'],
        Q=config['data']['horizon'],
        T=config['data']['num_of_hours'],
        L=config['model']['L'],
        K=config['model']['K'],
        d=config['model']['d'],
        lap_mx=laplacian_pe,
        LAP=laplacian_matrix,
        num_node=config['data']['num_nodes'],
        embed_dim=config['model']['embed_dim'],
        skip_dim=config['model']['skip_dim'],
    ).to(device)

    # 加载训练好的权重
    checkpoint_path = f"checkpoints/{config['data']['dataset']}/best_model.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到模型文件: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型加载成功，来自 epoch {checkpoint.get('epoch', 'unknown')}")

    # 提取 embedding
    print("提取 embeddings...")

    embeddings_list = []
    labels_list = []
    inputs_list = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="处理测试集"):
            x_batch = batch.feature.to(device)  # [B, P, N, 3]
            y_batch = batch.label.to(device)    # [B, Q, N, 3]

            # 提取时间编码
            TE = x_batch[:, :, :, 1:]  # [B, P, N, 2]

            # 提取 embedding (return_embedding=True)
            embedding = model(x_batch, TE, return_embedding=True)
            # embedding 形状: [B, Q, N, 256]

            embeddings_list.append(embedding.cpu().numpy())
            labels_list.append(y_batch[:, :, :, 0].cpu().numpy())  # 只保存流量值
            inputs_list.append(x_batch.cpu().numpy())

    # 合并所有批次
    embeddings = np.concatenate(embeddings_list, axis=0)  # [num_samples, Q, N, 256]
    labels = np.concatenate(labels_list, axis=0)          # [num_samples, Q, N]
    inputs = np.concatenate(inputs_list, axis=0)          # [num_samples, P, N, 3]

    print(f"\nEmbedding 形状: {embeddings.shape}")
    print(f"Labels 形状: {labels.shape}")
    print(f"Inputs 形状: {inputs.shape}")

    # 保存
    dataset_name = config['data']['dataset']
    save_path = os.path.join(save_dir, f"{dataset_name}_embeddings.npz")

    np.savez_compressed(
        save_path,
        embeddings=embeddings,
        labels=labels,
        inputs=inputs,
        config=config,
        mean=scaler.mean_,
        std=scaler.scale_,
    )

    print(f"\n✓ Embeddings 已保存到: {save_path}")
    print(f"  - embeddings: {embeddings.shape} (源模型的高层特征)")
    print(f"  - labels: {labels.shape} (真实流量值)")
    print(f"  - inputs: {inputs.shape} (原始输入)")
    print(f"  - mean & std: 用于反归一化的统计量")

    return save_path


def load_embeddings(embedding_path):
    """
    加载保存的 embeddings

    Args:
        embedding_path: embedding 文件路径

    Returns:
        dict: 包含 embeddings, labels, inputs 等
    """
    data = np.load(embedding_path, allow_pickle=True)

    print(f"加载 embeddings 从: {embedding_path}")
    print(f"  - embeddings: {data['embeddings'].shape}")
    print(f"  - labels: {data['labels'].shape}")
    print(f"  - inputs: {data['inputs'].shape}")

    return {
        'embeddings': data['embeddings'],
        'labels': data['labels'],
        'inputs': data['inputs'],
        'config': data['config'].item() if 'config' in data else None,
        'mean': data['mean'] if 'mean' in data else None,
        'std': data['std'] if 'std' in data else None,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='提取 MPGAT 模型的 embeddings')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--save_dir', type=str, default='embeddings', help='保存目录')

    args = parser.parse_args()

    # 提取 embeddings
    save_path = extract_embeddings(args.config, args.save_dir)

    # 验证加载
    print("\n验证加载...")
    data = load_embeddings(save_path)
    print("\n✓ 提取完成！可用于迁移学习。")