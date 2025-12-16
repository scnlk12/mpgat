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
from utils.metrics import RMSE_MAE_MAPE, masked_mae_torch, masked_huber_loss
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


def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,
    criterion,
    config,
    data_scaler,
    writer=None,
    log=None,
    save=None,
    rank=0,
    amp_scaler=None,  # 用于混合精度训练
):
    """训练函数"""
    wait = 0
    min_val_loss = np.inf
    train_loss_list = []
    val_loss_list = []

    max_epochs = config.training.max_epoch
    early_stop = config.training.patience
    clip_grad = config.training.clip_grad
    verbose = config.logging.verbose
    use_amp = config.misc.get('use_amp', False)

    for epoch in range(max_epochs):
        # 分布式训练时需要设置epoch
        if hasattr(trainset_loader.sampler, 'set_epoch'):
            trainset_loader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, trainset_loader, optimizer, criterion,
            clip_grad, use_amp, amp_scaler, rank, data_scaler
        )
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, valset_loader, criterion, rank, data_scaler)
        val_loss_list.append(val_loss)

        # 学习率调度 - 适配不同类型的scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau需要传入验证损失
                scheduler.step(val_loss)
            else:
                # 其他scheduler直接调用
                scheduler.step()

        # 只在主进程记录
        if rank == 0:
            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)

            if (epoch + 1) % verbose == 0:
                print(f"{datetime.datetime.now()} Epoch {epoch + 1:4d} | "
                      f"Train Loss = {train_loss:.5f} | Val Loss = {val_loss:.5f}")

                if log is not None:
                    str_out = (f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}\t"
                              f"Epoch {epoch + 1}\tTrain Loss = {train_loss:.5f}, "
                              f"Val Loss = {val_loss:.5f}")
                    utils.log_string(log, str_out)

        # 早停逻辑
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            # 保存最佳模型（只在主进程）
            if rank == 0:
                if isinstance(model, DDP):
                    best_state_dict = copy.deepcopy(model.module.state_dict())
                else:
                    best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= early_stop:
                if rank == 0:
                    print(f"Early stop at epoch: {epoch + 1}")
                break

    # 加载最佳模型
    if rank == 0:
        if isinstance(model, DDP):
            model.module.load_state_dict(best_state_dict)
        else:
            model.load_state_dict(best_state_dict)

        # 评估
        train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader, rank, data_scaler))
        val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader, rank, data_scaler))

        out_str = f"Early stopping at epoch: {epoch + 1}\n"
        out_str += f"Best at epoch {best_epoch + 1}:\n"
        out_str += f"Train Loss = {train_loss_list[best_epoch]:.5f}\n"
        out_str += f"Train RMSE = {train_rmse:.5f}, MAE = {train_mae:.5f}, MAPE = {train_mape:.5f}\n"
        out_str += f"Val Loss = {val_loss_list[best_epoch]:.5f}\n"
        out_str += f"Val RMSE = {val_rmse:.5f}, MAE = {val_mae:.5f}, MAPE = {val_mape:.5f}"

        print(out_str)
        if log is not None:
            utils.log_string(log, out_str)

        if save:
            torch.save(best_state_dict, save)

    return model


def train_one_epoch(model, trainset_loader, optimizer, criterion,
                    clip_grad, use_amp, amp_scaler, rank, data_scaler):
    """训练一个epoch"""
    model.train()
    batch_loss_list = []

    # 添加进度条 (仅主进程)
    if rank == 0:
        from tqdm import tqdm
        trainset_loader = tqdm(trainset_loader, desc="Training", leave=False)

    for batch in trainset_loader:
        batch.to_tensor(f'cuda:{rank}')

        x_batch = batch['x']
        y_batch = batch['y']
        TE = x_batch[:, :, :, 1:]

        optimizer.zero_grad()

        # 混合精度训练
        if use_amp and amp_scaler is not None:
            with torch.amp.autocast('cuda'):  # 使用新的API,避免FutureWarning
                out_batch = model(x_batch, TE)
                out_batch = data_scaler.inverse_transform(out_batch)
                y_batch_inv = data_scaler.inverse_transform(y_batch[:, :, :, 0])
                loss = criterion(out_batch, y_batch_inv)

            amp_scaler.scale(loss).backward()
            if clip_grad:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            out_batch = model(x_batch, TE)
            out_batch = data_scaler.inverse_transform(out_batch)
            y_batch_inv = data_scaler.inverse_transform(y_batch[:, :, :, 0])
            loss = criterion(out_batch, y_batch_inv)

            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        batch_loss_list.append(loss.item())

    epoch_loss = np.mean(batch_loss_list)
    return epoch_loss


@torch.no_grad()
def eval_model(model, valset_loader, criterion, rank, data_scaler):
    """评估模型"""
    model.eval()
    batch_loss_list = []

    for batch in valset_loader:
        batch.to_tensor(f'cuda:{rank}')

        x_batch = batch['x']
        y_batch = batch['y']
        TE = x_batch[:, :, :, 1:]

        out_batch = model(x_batch, TE)
        out_batch = data_scaler.inverse_transform(out_batch)
        y_batch = data_scaler.inverse_transform(y_batch[:, :, :, 0])
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)


@torch.no_grad()
def predict(model, loader, rank, data_scaler):
    """预测"""
    if isinstance(model, DDP):
        model = model.module
    model.eval()

    y = []
    out = []

    for batch in loader:
        batch.to_tensor(f'cuda:{rank}')
        x_batch = batch['x']
        y_batch = batch['y']
        TE = x_batch[:, :, :, 1:]

        out_batch = model(x_batch, TE)
        out_batch = data_scaler.inverse_transform(out_batch)
        y_batch = data_scaler.inverse_transform(y_batch[:, :, :, 0])

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()
    y = np.vstack(y).squeeze()

    return y, out


@torch.no_grad()
def test_model(model, testset_loader, rank, data_scaler, log=None):
    """测试模型"""
    if rank != 0:
        return

    model.eval()
    print("--------- Test ---------")
    if log:
        utils.log_string(log, "--------- Test ---------")

    start = time.time()
    y_true, y_pred = predict(model, testset_loader, rank, data_scaler)
    end = time.time()

    # 调试信息: 打印数据范围
    print(f"Data Statistics:")
    print(f"  y_true - min: {y_true.min():.2f}, max: {y_true.max():.2f}, mean: {y_true.mean():.2f}")
    print(f"  y_pred - min: {y_pred.min():.2f}, max: {y_pred.max():.2f}, mean: {y_pred.mean():.2f}")

    # MAPE异常分析: 检查不同流量区间的误差
    print(f"\nMAPE Analysis by Flow Range:")
    print(f"  Samples with y_true < 1.0: {(y_true < 1.0).sum()} ({(y_true < 1.0).mean()*100:.1f}%)")
    print(f"  Samples with y_true < 5.0: {(y_true < 5.0).sum()} ({(y_true < 5.0).mean()*100:.1f}%)")
    print(f"  Samples with y_true < 10.0: {(y_true < 10.0).sum()} ({(y_true < 10.0).mean()*100:.1f}%)")

    # 分别计算不同流量区间的MAPE
    mask_small = (y_true >= 0.1) & (y_true < 5.0)
    mask_medium = (y_true >= 5.0) & (y_true < 20.0)
    mask_large = (y_true >= 20.0)

    if mask_small.sum() > 0:
        mape_small = np.mean(np.abs((y_pred[mask_small] - y_true[mask_small]) / y_true[mask_small])) * 100
        mae_small = np.mean(np.abs(y_pred[mask_small] - y_true[mask_small]))
        print(f"  Small flow (0.1-5.0): MAPE={mape_small:.2f}%, MAE={mae_small:.3f}, count={mask_small.sum()}")

    if mask_medium.sum() > 0:
        mape_medium = np.mean(np.abs((y_pred[mask_medium] - y_true[mask_medium]) / y_true[mask_medium])) * 100
        mae_medium = np.mean(np.abs(y_pred[mask_medium] - y_true[mask_medium]))
        print(f"  Medium flow (5.0-20.0): MAPE={mape_medium:.2f}%, MAE={mae_medium:.3f}, count={mask_medium.sum()}")

    if mask_large.sum() > 0:
        mape_large = np.mean(np.abs((y_pred[mask_large] - y_true[mask_large]) / y_true[mask_large])) * 100
        mae_large = np.mean(np.abs(y_pred[mask_large] - y_true[mask_large]))
        print(f"  Large flow (>=20.0): MAPE={mape_large:.2f}%, MAE={mae_large:.3f}, count={mask_large.sum()}")

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
    if rank == 0:
        print(f"Using {world_size} GPU(s) for training")
        print(f"Config:\n{yaml.dump(config.to_dict(), default_flow_style=False)}")

    # 创建日志
    log = None
    writer = None
    if rank == 0:
        os.makedirs(config.logging.log_dir, exist_ok=True)
        log_file = os.path.join(
            config.logging.log_dir,
            f"log_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"
        )
        log = open(log_file, 'w')
        utils.log_string(log, str(config.to_dict()))

        writer = SummaryWriter(log_dir=config.logging.tensorboard_dir)

    # 加载数据
    if rank == 0:
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

    if rank == 0:
        print('Data loaded!')

    # 加载图结构
    dataset_name = config.data.traffic_file.split('/')[-1].replace('.npz', '')
    dataset_dir = '/'.join(config.data.traffic_file.split('/')[:-1])
    txt_file = os.path.join(dataset_dir, f'{dataset_name}.txt')
    csv_file = os.path.join(dataset_dir, f'{dataset_name}.csv')

    if rank == 0:
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

    if rank == 0:
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
                adj_mx[idx_i][idx_j] = 1
                adj_mx[idx_j][idx_i] = 1

    if rank == 0:
        print(f"Graph loaded: {num_nodes} nodes, {int(np.sum(adj_mx > 0) / 2)} edges")

    # 计算拉普拉斯矩阵
    lap_mx, LAP = cal_lape(adj_mx)
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

    # 参数初始化
    for p in gman_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    if rank == 0:
        print_model_parameters(gman_model, only_num=False)

    # 分布式包装
    if world_size > 1:
        gman_model = DDP(gman_model, device_ids=[rank], find_unused_parameters=True)

    # 优化器 - 添加权重衰减以减少过拟合
    optimizer = torch.optim.Adam(
        params=gman_model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.get('weight_decay', 1e-4)  # L2正则化
    )

    # 学习率调度 - 使用CosineAnnealing实现平滑衰减
    lr_scheduler = None
    if config.training.get('use_cosine_annealing', True):
        # Cosine Annealing: 学习率从初始值平滑降到最小值
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config.training.max_epoch,  # 周期长度
            eta_min=1e-6  # 最小学习率
        )
    elif config.training.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=config.training.step_size,
            gamma=config.training.gamma
        )

    # 损失函数
    if config.training.loss_func == 'mae':
        criterion = torch.nn.L1Loss()
    elif config.training.loss_func == 'mse':
        criterion = torch.nn.MSELoss()
    elif config.training.loss_func == 'masked_mae':
        criterion = partial(masked_mae_torch, null_val=0)
    elif config.training.loss_func == 'huber':
        # Huber Loss: 对离群点更鲁棒,有助于降低RMSE
        # delta=1.0: 对于归一化后的数据更合适
        criterion = partial(masked_huber_loss, null_val=0, delta=1.0)
    else:
        raise ValueError(f"Unknown loss function: {config.training.loss_func}")

    # 混合精度训练
    amp_scaler = None
    if config.misc.get('use_amp', False):
        amp_scaler = torch.cuda.amp.GradScaler()

    # 模型保存路径
    save_path = None
    if rank == 0:
        os.makedirs(config.saving.model_dir, exist_ok=True)
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_path = os.path.join(config.saving.model_dir, f"GMAN-{dataset_name}-{now}.pt")

    # 训练
    gman_model = train(
        gman_model, train_loader, val_loader, optimizer, lr_scheduler,
        criterion, config, data_scaler, writer, log, save_path, rank, amp_scaler
    )

    if rank == 0:
        print(f"Saved Model: {save_path}")
        if log:
            utils.log_string(log, f"Saved Model: {save_path}")

    # 测试
    test_model(gman_model, test_loader, rank, data_scaler, log)

    # 清理
    if rank == 0:
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
        # 设置启动方法为spawn,兼容screen/tmux等后台运行环境
        mp.set_start_method('spawn', force=True)
        mp.spawn(main_worker, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()