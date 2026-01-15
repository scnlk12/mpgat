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

    # 【新增】定义课程学习的切换点
    cl_milestones = [40, 70]

    for epoch in range(max_epochs):
        # 分布式训练时需要设置epoch
        if hasattr(trainset_loader.sampler, 'set_epoch'):
            trainset_loader.sampler.set_epoch(epoch)
        
        # 【新增】检查是否到达课程切换点，如果是，重置早停逻辑
        if epoch in cl_milestones:
            if rank == 0:
                print(f"\n[Curriculum Info] Epoch {epoch}: Difficulty Level Increased! Resetting Early Stopping...")
                if log is not None:
                    utils.log_string(log, f"[Curriculum Info] Epoch {epoch}: Resetting Early Stopping...")
            wait = 0
            min_val_loss = np.inf
            # 可选：如果在切换难度时验证损失剧增，这里重置为无穷大可以让模型重新寻找最优解

        train_loss = train_one_epoch(
            model, trainset_loader, optimizer, criterion,
            clip_grad, use_amp, amp_scaler, rank, data_scaler, epoch=epoch
        )
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, valset_loader, criterion, rank, data_scaler, epoch=epoch)
        val_loss_list.append(val_loss)

        # 学习率调度 - 适配不同类型的scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau需要传入验证损失
                scheduler.step(val_loss)
            else:
                # 其他scheduler直接调用
                scheduler.step()

        # 定期清理显存缓存 (每10个epoch)
        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
            # if rank == 0:
            print(f"[Epoch {epoch + 1}] GPU cache cleared")

        # 只在主进程记录
        # if rank == 0:
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
            writer.flush()  # 立即写入磁盘,避免内存累积

        if (epoch + 1) % verbose == 0:
            print(f"{datetime.datetime.now()} Epoch {epoch + 1:4d} | "
                    f"Train Loss = {train_loss:.5f} | Val Loss = {val_loss:.5f} | "
                    f"")

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
            # if rank == 0:
            if isinstance(model, DDP):
                best_state_dict = copy.deepcopy(model.module.state_dict())
            else:
                best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= early_stop:
                # if rank == 0:
                print(f"Early stop at epoch: {epoch + 1}")
                break

    # 加载最佳模型
    # if rank == 0:
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
                    clip_grad, use_amp, amp_scaler, rank, data_scaler, epoch): # 1. 新增 epoch 参数
    """训练一个epoch"""
    model.train()
    batch_loss_list = []

    # ==========================================
    # 2. 定义课程学习策略 (Curriculum Learning)
    # ==========================================
    if epoch < 40:
        pred_steps = 3   # 前 20 个 epoch，只学前 15 分钟 (3步)
    elif epoch < 70:
        pred_steps = 6   # 中间 20 个 epoch，学前 30 分钟 (6步)
    else:
        pred_steps = 12  # 之后学习全部 60 分钟 (12步)

    # 添加进度条 (仅主进程)
    # if rank == 0:
    from tqdm import tqdm
    trainset_loader = tqdm(trainset_loader, desc="Training", leave=False)
    for batch in trainset_loader:
        # 现在的 batch 是一个列表 [x, y]
        x_batch, y_batch = batch
        # 手动搬运到 GPU
        x_batch = x_batch.to(f'cuda:{rank}', non_blocking=True)
        y_batch = y_batch.to(f'cuda:{rank}', non_blocking=True)

        TE = x_batch[:, :, :, 1:]

        optimizer.zero_grad()

        # 定义差分损失的权重 (可以尝试 1.0 ~ 5.0)
        lambda_diff = 2.0
        lambda_step1 = 0.5  # 强迫起点精准 (解决 Step 1 误差)

        # 混合精度训练
        if use_amp and amp_scaler is not None:
            with torch.amp.autocast('cuda'):  # 使用新的API,避免FutureWarning
                out_batch = model(x_batch, TE)
                # out_batch = data_scaler.inverse_transform(out_batch)
                y_batch_inv = y_batch[:, :, :, 0]
                # y_batch_inv = data_scaler.inverse_transform(y_batch[:, :, :, 0])

                # loss = criterion(out_batch, y_batch_inv)
                # ==========================================
                # 3. 应用切片：只计算前 pred_steps 的 Loss
                # ==========================================
                loss_pred = criterion(out_batch[:, :pred_steps, :], y_batch_inv[:, :pred_steps, :])

                # --- 2. Step 1 强化 Loss (Anchor Accuracy) ---
                # 单独把第1步拎出来再算一遍，强迫模型打好地基
                loss_step1 = criterion(out_batch[:, 0:1, :], y_batch_inv[:, 0:1, :])
                
                # 3. 差分/趋势 Loss (First-order Difference)
                if pred_steps > 1:
                    # 计算相邻时间步的差值 (Delta)
                    # Pred[t+1] - Pred[t]
                    pred_diff = out_batch[:, 1:pred_steps, :] - out_batch[:, :pred_steps-1, :]
                    # True[t+1] - True[t]
                    true_diff = y_batch_inv[:, 1:pred_steps, :] - y_batch_inv[:, :pred_steps-1, :]
                    
                    # 这里的 criterion 通常是 Masked MAE
                    loss_diff = criterion(pred_diff, true_diff)
                    
                    # 【核心修改】总 Loss 组合
                    loss = loss_pred + lambda_diff * loss_diff + lambda_step1 * loss_step1
                else:
                    # 如果只预测1步，没法算差分，但可以加重 Step 1
                    loss = loss_pred + lambda_step1 * loss_step1
                # --- 修改结束 ---

            amp_scaler.scale(loss).backward()
            if clip_grad:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            out_batch = model(x_batch, TE)
            # out_batch = data_scaler.inverse_transform(out_batch)
            y_batch_inv = y_batch[:, :, :, 0]
            # y_batch_inv = data_scaler.inverse_transform(y_batch[:, :, :, 0])

            # loss = criterion(out_batch, y_batch_inv)
            # ==========================================
            # 3. 应用切片：只计算前 pred_steps 的 Loss
            # ==========================================
            loss_pred = criterion(out_batch[:, :pred_steps, :], y_batch_inv[:, :pred_steps, :])

            # --- 2. Step 1 强化 Loss ---
            loss_step1 = criterion(out_batch[:, 0:1, :], y_batch_inv[:, 0:1, :])

            # 2. 差分/趋势 Loss
            if pred_steps > 1:
                # 计算相邻时间步的差值
                pred_diff = out_batch[:, 1:pred_steps, :] - out_batch[:, :pred_steps-1, :]
                true_diff = y_batch_inv[:, 1:pred_steps, :] - y_batch_inv[:, :pred_steps-1, :]
                
                # 约束变化趋势
                loss_diff = criterion(pred_diff, true_diff)
                
                # 【核心修改】总 Loss 组合
                loss = loss_pred + lambda_diff * loss_diff + lambda_step1 * loss_step1
            else:
                loss = loss_pred + lambda_step1 * loss_step1

            # --- 修改结束 ---

            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        batch_loss_list.append(loss.item())

    epoch_loss = np.mean(batch_loss_list)
    return epoch_loss


@torch.no_grad()
def eval_model(model, valset_loader, criterion, rank, data_scaler, epoch):
    """评估模型"""
    model.eval()
    batch_loss_list = []

    # 保持和 train 一致的课程逻辑
    if epoch < 40:
        pred_steps = 3
    elif epoch < 70:
        pred_steps = 6
    else:
        pred_steps = 12

    for batch in valset_loader:
        x_batch, y_batch = batch
        x_batch = x_batch.to(f'cuda:{rank}', non_blocking=True)
        y_batch = y_batch.to(f'cuda:{rank}', non_blocking=True)
        TE = x_batch[:, :, :, 1:]

        out_batch = model(x_batch, TE)
        # out_batch = data_scaler.inverse_transform(out_batch)
        y_batch = y_batch[:, :, :, 0]
        # y_batch = data_scaler.inverse_transform(y_batch[:, :, :, 0])
        
        # loss = criterion(out_batch, y_batch)
        # 【关键】只计算当前难度的 Loss
        loss = criterion(out_batch[:, :pred_steps, :], y_batch[:, :pred_steps, :])
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
    # 【修正 1】初始化为无穷大，而不是 0！
    # 0 代表距离最近，inf 代表没有连接
    dist_mx = np.full((num_nodes, num_nodes), np.inf, dtype=float)
    # 【修正 2】对角线距离设为 0 (自己到自己)
    for k in range(num_nodes):
        dist_mx[k, k] = 0
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
                 # 填入真实物理距离
                dist_mx[idx_i][idx_j] = distance
                dist_mx[idx_j][idx_i] = distance
    
    # 3. 计算标准差 (Sigma)
    # 只取有限的距离值计算标准差
    distances = dist_mx[~np.isinf(dist_mx)]
    std = distances.std()
    
    # 4. 应用高斯核公式: W_ij = exp( - (dist_ij^2) / (std^2) )
    # 距离越小，结果越接近 1；距离越大，结果越接近 0
    adj_mx = np.exp(-np.square(dist_mx / std))

    # 3. 【修改点】应用 k-NN 策略 (保留概率最大的 k 个邻居)
    # PEMS08 节点数 170，k 取 10 或 20 比较合适
    k = 10 
    
    # 对每一行(每个节点)，找到权重最大的 k 个值的索引
    # argsort 从小到大排，取最后 k 个
    topk_indices = np.argsort(adj_mx, axis=1)[:, -k:]
    
    # 创建一个新的稀疏矩阵
    new_adj = np.zeros_like(adj_mx)
    for i in range(adj_mx.shape[0]):
        new_adj[i, topk_indices[i]] = adj_mx[i, topk_indices[i]]
        
    # 重新归一化或直接使用 new_adj
    adj_mx = new_adj
    
    # 确保对称性 (可选，GMAN 其实支持有向图，但无向图更稳)
    adj_mx = np.maximum(adj_mx, adj_mx.T)

    # 5. 阈值过滤 (Sparsification)
    # 权重太小的边视为噪声，强制置为 0，保持矩阵稀疏性
    # 0.1 是一个经验值，表示必须至少保留 10% 的相关性
    # adj_mx[adj_mx < 0.1] = 0

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

    # 参数初始化
    for p in gman_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    # if rank == 0:
    print_model_parameters(gman_model, only_num=False)

    # 分布式包装
    if world_size > 1:
        # 禁用find_unused_parameters以避免NCCL同步死锁
        gman_model = DDP(gman_model, device_ids=[rank], find_unused_parameters=False)

    # 优化器 (添加weight_decay支持L2正则化)
    weight_decay = config.training.get('weight_decay', 0.0)
    optimizer = torch.optim.AdamW(
        params=gman_model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=weight_decay
    )

    # 学习率调度 - 使用CosineAnnealing实现平滑衰减
    lr_scheduler = None
    warmup_epochs = config.training.get('warmup_epochs', 0)

    if config.training.get('use_cosine_annealing', True):
        # Cosine Annealing: 学习率从初始值平滑降到最小值
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config.training.max_epoch,  # 周期长度
            eta_min=1e-6  # 最小学习率
        )

        # 添加Warmup
        if warmup_epochs > 0:
            # from torch.optim.lr_scheduler import LinearLR, SequentialLR
            # warmup_scheduler = LinearLR(
            #     optimizer=optimizer,
            #     start_factor=0.1,  # 从10%学习率开始
            #     end_factor=1.0,    # 逐步增加到100%
            #     total_iters=warmup_epochs
            # )
            # lr_scheduler = SequentialLR(
            #     optimizer=optimizer,
            #     schedulers=[warmup_scheduler, base_scheduler],
            #     milestones=[warmup_epochs]
            # )
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.training.learning_rate,
                epochs=config.training.max_epoch,
                steps_per_epoch=len(train_loader),
                pct_start=warmup_epochs / config.training.max_epoch, # 前10%时间用来热身
                div_factor=10.0,
                final_div_factor=100.0
            )
        else:
            lr_scheduler = base_scheduler

    elif config.training.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=config.training.step_size,
            gamma=config.training.gamma
        )

    # 损失函数
    use_temporal_weighting = config.training.get('use_temporal_weighting', False)
    weight_scheme = config.training.get('temporal_weight_scheme', 'progressive')

    # if rank == 0:
    if use_temporal_weighting:
        print(f"Using temporal weighted loss with scheme: {weight_scheme}")
    else:
        print(f"Using standard loss: {config.training.loss_func}")

    if config.training.loss_func == 'mae':
        criterion = torch.nn.L1Loss()
    elif config.training.loss_func == 'mse':
        criterion = torch.nn.MSELoss()
    elif config.training.loss_func == 'masked_mae':
        if use_temporal_weighting:
            # 时间步加权的Masked MAE
            criterion = partial(masked_mae_torch_weighted, null_val=-1, weight_scheme=weight_scheme)
            # if rank == 0:
            print(f"  -> Using masked_mae_torch_weighted")
        else:
            # 标准Masked MAE
            criterion = partial(masked_mae_torch, null_val=-1)
    elif config.training.loss_func == 'huber':
        if use_temporal_weighting:
            # 时间步加权的Huber Loss
            criterion = partial(masked_huber_loss_weighted, null_val=-1, delta=5.0, weight_scheme=weight_scheme)
            # if rank == 0:
            print(f"  -> Using masked_huber_loss_weighted")
        else:
            # 标准Huber Loss
            # criterion = partial(masked_huber_loss, null_val=-1, delta=1.0)
            criterion = nn.HuberLoss(delta=1.0)
    else:
        raise ValueError(f"Unknown loss function: {config.training.loss_func}")

    # 混合精度训练
    amp_scaler = None
    if config.misc.get('use_amp', False):
        amp_scaler = torch.cuda.amp.GradScaler()

    # 模型保存路径
    save_path = None
    # if rank == 0:
    os.makedirs(config.saving.model_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_path = os.path.join(config.saving.model_dir, f"GMAN-{dataset_name}-{now}.pt")

    # 训练
    gman_model = train(
        gman_model, train_loader, val_loader, optimizer, lr_scheduler,
        criterion, config, data_scaler, writer, log, save_path, rank, amp_scaler
    )

    # if rank == 0:
    print(f"Saved Model: {save_path}")
    if log:
        utils.log_string(log, f"Saved Model: {save_path}")

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
