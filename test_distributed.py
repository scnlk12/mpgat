"""
分布式训练脚本 - 支持多GPU并发训练
"""

# 标准库
import argparse
import copy
import csv
import datetime
import os
import random
import time
from functools import partial

# 第三方库
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import yaml

# 本地模块
from utils import data_prepare
import model
import utils
from utils.metrics import (RMSE_MAE_MAPE, masked_mae_torch, masked_huber_loss,
                          masked_mae_torch_weighted, masked_huber_loss_weighted)
from model import GMAN
from utils import cal_lape, print_model_parameters
from utils.config_loader import load_config, save_config, validate_config


def setup_distributed(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(model, loader, rank, data_scaler):
    """预测"""
    if isinstance(model, DDP):
        model = model.module
    model.eval()

    y = []
    out = []

    for batch in loader:
        x_batch, y_batch = batch
        x_batch = x_batch.to(f'cuda:{rank}')
        y_batch = y_batch.to(f'cuda:{rank}')
        TE = x_batch[:, :, :, 1:]

        out_batch = model(x_batch, TE)
        out_batch = data_scaler.inverse_transform(out_batch)
        y_batch = y_batch[:, :, :, 0]
        # y_batch = data_scaler.inverse_transform(y_batch[:, :, :, 0])

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    y_pred = np.concatenate(out, axis=0)
    y_true = np.concatenate(y, axis=0)

    return y_true, y_pred


@torch.no_grad()
def test_model(model, testset_loader, rank, data_scaler, log=None):
    """测试模型"""
    # if rank != 0:
    #     return

    model.eval()
    print("--------- Test ---------")
    if log:
        utils.log_string(log, "--------- Test ---------")

    start = time.time()
    y_true, y_pred = predict(model, testset_loader, rank, data_scaler)
    end = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = f"All Steps RMSE = {rmse_all:.5f}, MAE = {mae_all:.5f}, MAPE = {mape_all:.5f}\n"

    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += f"Step {i+1:2d} RMSE = {rmse:.5f}, MAE = {mae:.5f}, MAPE = {mape:.5f}\n"

    print(out_str)
    if log:
        utils.log_string(log, out_str)
    print(f"Inference time: {end - start:.2f} s")
    if log:
        utils.log_string(log, f"Inference time: {end - start:.2f} s")


def main_worker(rank, world_size, config):
    """每个GPU上运行的主函数"""

    # 设置分布式环境
    if world_size > 1:
        setup_distributed(rank, world_size)

    # 设置随机种子
    setup_seed(config.misc.seed + rank)

    # 设置设备
    device = torch.device(f'cuda:{rank}')

    # 只在主进程打印和记录日志
    # if rank == 0:
    print(f"Using {world_size} GPU(s) for training")
    print(f"Config:\n{yaml.dump(config.to_dict(), default_flow_style=False)}")

    # 创建日志
    log = None
    writer = None
    # if rank == 0:
    os.makedirs(config.logging.log_dir, exist_ok=True)
    log_file = os.path.join(
        config.logging.log_dir,
        f"log_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"
    )
    log = open(log_file, 'w')
    utils.log_string(log, str(config.to_dict()))

    writer = SummaryWriter(log_dir=config.logging.tensorboard_dir)

    # 加载数据
    # if rank == 0:
    print('Loading data...')

    # 转换config为argparse.Namespace以兼容现有代码
    class Args:
        pass
    args = Args()
    args.traffic_file = config.data.traffic_file
    args.P = config.model.P
    args.Q = config.model.Q
    args.batch_size = config.data.batch_size
    args.train_ratio = config.data.train_ratio
    args.val_ratio = config.data.val_ratio
    args.test_ratio = config.data.test_ratio

    train_loader, val_loader, test_loader, data_scaler = data_prepare.get_dataloaders(
        args, log=log, world_size=world_size, rank=rank
    )

    # if rank == 0:
    print('Data loaded!')

    # 加载图结构
    dataset_name = config.data.traffic_file.split('/')[-1].replace('.npz', '')
    dataset_dir = '/'.join(config.data.traffic_file.split('/')[:-1])
    txt_file = os.path.join(dataset_dir, f'{dataset_name}.txt')
    csv_file = os.path.join(dataset_dir, f'{dataset_name}.csv')

    # if rank == 0:
    print(f"Using dataset: {dataset_name}")
    print(f"Graph structure file: {csv_file}")

    # 读取图结构
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

    # if rank == 0:
    print(f"Number of nodes: {num_nodes}")

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
                # 无权矩阵
                adj_mx[idx_i][idx_j] = 1
                adj_mx[idx_j][idx_i] = 1

    # if rank == 0:
    print(f"Graph loaded: {num_nodes} nodes, {int(np.sum(adj_mx > 0) / 2)} edges")

    # 计算拉普拉斯矩阵
    lap_mx, LAP = cal_lape(adj_mx, config.model.lape_dim)
    lap_mx = lap_mx.to(device)
    # Convert sparse matrix to dense tensor
    LAP = torch.from_numpy(LAP.toarray()).float().to(device)

    # 创建模型
    gman_model = GMAN(
        config.model.input_dim, config.model.P, config.model.Q, config.model.T,
        config.model.L, config.model.K, config.model.d, lap_mx, LAP,
        num_nodes, config.model.embed_dim, config.model.skip_dim
    )
    gman_model = gman_model.to(device)

    # 分布式包装
    if world_size > 1:
        # 禁用find_unused_parameters以避免NCCL同步死锁
        gman_model = DDP(gman_model, device_ids=[rank], find_unused_parameters=False)

    gman_model.load_state_dict(torch.load("saved_models/GMAN-PEMS08-2025-12-30-22-37-40.pt"))

    # 模型保存路径
    save_path = None
    # if rank == 0:
    os.makedirs(config.saving.model_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_path = os.path.join(config.saving.model_dir, f"GMAN-{dataset_name}-{now}.pt")

    # 测试
    test_model(gman_model, test_loader, rank, data_scaler, log)

    # 清理
    # if rank == 0:
    if writer:
        writer.close()
    if log:
        log.close()

    if world_size > 1:
        cleanup_distributed()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MP-STGAT Distributed Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='使用的GPU ID，如 "0,1,2"')
    parser.add_argument('--traffic_file', type=str, default=None,
                       help='数据集路径（覆盖配置文件）')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 命令行参数覆盖
    if args.gpu_ids is not None:
        config.gpu.device_ids = [int(x) for x in args.gpu_ids.split(',')]
    if args.traffic_file is not None:
        config.data.traffic_file = args.traffic_file

    # 验证配置
    try:
        validate_config(config)
        print("✓ 配置验证通过")
    except ValueError as e:
        print(f"✗ {e}")
        return

    # 确定使用的GPU
    gpu_ids = config.gpu.device_ids
    world_size = len(gpu_ids)

    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Using GPUs: {gpu_ids}")

    if world_size == 1:
        # 单GPU训练
        main_worker(gpu_ids[0], 1, config)
    else:
        # 多GPU训练
        import torch.multiprocessing as mp
        mp.spawn(main_worker, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
