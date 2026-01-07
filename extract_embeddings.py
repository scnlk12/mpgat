"""
提取 MPGAT 模型的 embedding 用于迁移学习

这个脚本会：
1. 加载训练好的模型
2. 提取指定数据集（train/val/test/all）的 embedding [B, Q, N, 256]
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


def extract_embeddings(config_path='config.yaml', save_dir='embeddings',
                      checkpoint_path=None, split='all', save_path=None):
    """
    提取模型的 embedding

    Args:
        config_path: 配置文件路径
        save_dir: 保存 embedding 的目录
        checkpoint_path: 模型权重文件路径，如果为 None 则使用默认路径
        split: 提取哪个数据集 ('train', 'val', 'test', 'all')
        save_path: 完整的保存路径，如果指定则忽略 save_dir
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
    if checkpoint_path is None:
        checkpoint_path = f"checkpoints/{config['data']['dataset']}/best_model.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到模型文件: {checkpoint_path}")

    print(f"从 {checkpoint_path} 加载模型权重...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型加载成功，来自 epoch {checkpoint.get('epoch', 'unknown')}")

    # 确定要处理的数据加载器
    if split == 'train':
        dataloaders = [('train', train_dataloader)]
    elif split == 'val':
        dataloaders = [('val', val_dataloader)]
    elif split == 'test':
        dataloaders = [('test', test_dataloader)]
    elif split == 'all':
        dataloaders = [
            ('train', train_dataloader),
            ('val', val_dataloader),
            ('test', test_dataloader)
        ]
    else:
        raise ValueError(f"无效的 split 参数: {split}，必须是 'train', 'val', 'test' 或 'all'")

    # 提取 embedding
    print(f"提取 {split} 数据集的 embeddings...")

    all_embeddings = {}
    all_labels = {}
    all_inputs = {}

    with torch.no_grad():
        for split_name, dataloader in dataloaders:
            embeddings_list = []
            labels_list = []
            inputs_list = []

            for batch in tqdm(dataloader, desc=f"处理 {split_name} 集"):
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

            # 合并当前分割的所有批次
            all_embeddings[split_name] = np.concatenate(embeddings_list, axis=0)
            all_labels[split_name] = np.concatenate(labels_list, axis=0)
            all_inputs[split_name] = np.concatenate(inputs_list, axis=0)

            print(f"  {split_name}: embeddings={all_embeddings[split_name].shape}, "
                  f"labels={all_labels[split_name].shape}")

    # 如果是单个分割，直接使用；如果是 all，合并所有数据
    if split == 'all':
        embeddings = np.concatenate([all_embeddings['train'],
                                    all_embeddings['val'],
                                    all_embeddings['test']], axis=0)
        labels = np.concatenate([all_labels['train'],
                                all_labels['val'],
                                all_labels['test']], axis=0)
        inputs = np.concatenate([all_inputs['train'],
                                all_inputs['val'],
                                all_inputs['test']], axis=0)
    else:
        embeddings = all_embeddings[split]
        labels = all_labels[split]
        inputs = all_inputs[split]

    print(f"\nEmbedding 形状: {embeddings.shape}")
    print(f"Labels 形状: {labels.shape}")
    print(f"Inputs 形状: {inputs.shape}")

    # 保存
    dataset_name = config['data']['dataset']
    if save_path is None:
        # 生成默认文件名，包含 split 信息
        filename = f"{dataset_name}_{split}_embeddings.npz"
        save_path = os.path.join(save_dir, filename)

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    # 保存数据，如果是 all，额外保存每个分割的信息
    save_dict = {
        'embeddings': embeddings,
        'labels': labels,
        'inputs': inputs,
        'config': config,
        'mean': scaler.mean_,
        'std': scaler.scale_,
        'split': split,
    }

    # 如果提取了所有数据，也保存分割信息（用于后续可能需要单独访问）
    if split == 'all':
        save_dict.update({
            'train_size': all_embeddings['train'].shape[0],
            'val_size': all_embeddings['val'].shape[0],
            'test_size': all_embeddings['test'].shape[0],
        })

    np.savez_compressed(save_path, **save_dict)

    # 计算文件大小
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)

    print(f"\n✓ Embeddings 已保存到: {save_path}")
    print(f"  文件大小: {file_size_mb:.2f} MB")
    print(f"  - embeddings: {embeddings.shape} (源模型的高层特征)")
    print(f"  - labels: {labels.shape} (真实流量值)")
    print(f"  - inputs: {inputs.shape} (原始输入)")
    print(f"  - split: {split}")
    if split == 'all':
        print(f"  - 数据分布: train={save_dict['train_size']}, "
              f"val={save_dict['val_size']}, test={save_dict['test_size']}")
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

    parser = argparse.ArgumentParser(
        description='提取 MPGAT 模型的 embeddings 用于迁移学习',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 提取全量数据集（默认）
  python extract_embeddings.py --split all

  # 只提取测试集
  python extract_embeddings.py --split test

  # 指定模型路径和保存路径
  python extract_embeddings.py --checkpoint checkpoints/PEMS08/best_model.pth --save_path ./my_embeddings.npz

  # 使用自定义配置文件
  python extract_embeddings.py --config my_config.yaml --split all
        """
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--save_dir', type=str, default='embeddings',
                       help='保存目录 (默认: embeddings/)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型权重文件路径 (默认: checkpoints/{dataset}/best_model.pth)')
    parser.add_argument('--split', type=str, default='all',
                       choices=['train', 'val', 'test', 'all'],
                       help='提取哪个数据集 (默认: all)')
    parser.add_argument('--save_path', type=str, default=None,
                       help='完整的保存文件路径，如果指定则忽略 --save_dir')

    args = parser.parse_args()

    # 提取 embeddings
    print("="*60)
    print("MPGAT Embedding 提取工具")
    print("="*60)
    save_path = extract_embeddings(
        config_path=args.config,
        save_dir=args.save_dir,
        checkpoint_path=args.checkpoint,
        split=args.split,
        save_path=args.save_path
    )

    # 验证加载
    print("\n" + "="*60)
    print("验证加载...")
    print("="*60)
    data = load_embeddings(save_path)
    print("\n✓ 提取完成！可用于迁移学习。")