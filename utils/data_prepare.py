# 标准库
import copy
import os

# 第三方库
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

# 本地模块
from .batch import Batch
from .list_data import ListDataset
from .utils import StandardScaler, log_string
import torch.utils.data as torch_data

# ! X shape: (B, T, N, C)

def seq2instance(data, P, Q):
    num_step, num_nodes, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape=(num_sample, P, num_nodes, dims))
    y = np.zeros(shape=(num_sample, Q, num_nodes, dims))
    for i in range(num_sample):
        x[i] = data[i: i + P]
        y[i] = data[i + P: i + P + Q]
    return x, y


def get_dataloaders(args, log, world_size=1, rank=0):
    # data [num_steps, N]
    data = np.load(args.traffic_file)["data"][:, :, 0].astype(np.float32)
    # data [num_steps, N, F]
    data = np.expand_dims(data, axis=-1)

    L, N, F = data.shape
    # feature_list 1 × T × N × D
    feature_list = [data]

    # numerical time_in_day 12 * 24 = 288
    # time_ind 1 × 288
    time_ind = [i % 288 for i in range(data.shape[0])]
    time_ind = np.array(time_ind)
    time_in_day = np.tile(time_ind, [1, N, 1]).transpose((2, 1, 0))
    # 需要进行归一化
    time_in_day_norm = time_in_day / 288  # 归一化到 [0, 1)

    # time_in_day 16992 × 307 × 1
    feature_list.append(time_in_day_norm)

    # numerical day_in_week
    day_in_week = [(i // 288) % 7 for i in range(data.shape[0])]
    day_in_week = np.array(day_in_week)
    day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((2, 1, 0))
    # 归一化
    day_in_week_norm = day_in_week / 7.0 # 归一化到 [0, 1)
    # day_in_week 16992 × 307 × 1
    # feature_list 3 × T × N × D
    feature_list.append(day_in_week_norm)

    # generate_data
    # data 16992 × 307 × 3
    data = np.concatenate(feature_list, axis=-1)
    x, y = seq2instance(data, args.P, args.Q)

    # split train/val/test
    num_step = data.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    # X, Y 
    # train = data[: train_steps]
    # val = data[train_steps: train_steps + val_steps]
    # test = data[-test_steps:]

    # trainX B × P × N × F
    # trainY B × Q × N × F
    trainX, trainY = x[0: train_steps], y[0: train_steps]
    valX, valY = x[train_steps: train_steps + val_steps], y[train_steps: train_steps + val_steps]
    testX, testY = x[-test_steps:], y[-test_steps:]

    # print('trainX: %s\ttrainY: %s' % (trainX.shape, trainY.shape))
    # print('valX: %s\tvalY: %s' % (valX.shape, valY.shape))
    # print('testX: %s\ttestY: %s' % (testX.shape, testY.shape))

    log_string(log, "train\t" + "x: " + str(trainX.shape) + ", y: " + str(trainX.shape))
    log_string(log, "eval\t" + "x: " + str(valX.shape) + ", y: " + str(valY.shape))
    log_string(log, "test\t" + "x: " + str(testX.shape) + ", y: " + str(testX.shape))

    # normalization
    # TODO 更多归一化方法
    scaler = StandardScaler(mean=trainX[..., 0].mean(), std=trainX[..., 0].std())

    trainX[..., 0] = scaler.transform(trainX[..., 0])
    trainY[..., 0] = scaler.transform(trainY[..., 0])
    valX[..., 0] = scaler.transform(valX[..., 0])
    valY[..., 0] = scaler.transform(valY[..., 0])
    testX[..., 0] = scaler.transform(testX[..., 0])
    # testY[..., 0] = scaler.transform(testY[..., 0])

    # 转换为 Tensor
    train_dataset = torch_data.TensorDataset(torch.FloatTensor(trainX), torch.FloatTensor(trainY))
    eval_dataset = torch_data.TensorDataset(torch.FloatTensor(valX), torch.FloatTensor(valY))
    test_dataset = torch_data.TensorDataset(torch.FloatTensor(testX), torch.FloatTensor(testY))

    # Create distributed samplers for multi-GPU training
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42
        )
        val_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    # Create DataLoaders with optimized settings
    # Use 2 workers per dataloader for async data loading
    num_workers = 2
    trainset_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        collate_fn=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    valset_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    testset_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        shuffle=False,
        collate_fn=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return trainset_loader, valset_loader, testset_loader, scaler
