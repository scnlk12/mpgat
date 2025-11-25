import torch
import numpy as np
import os
from utils import StandardScaler, log_string

from list_data import ListDataset

from batch import Batch

import copy


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


def get_dataloaders(args, log):
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

    # time_in_day 16992 × 307 × 1
    feature_list.append(time_in_day)

    # numerical day_in_week
    day_in_week = [(i // 288) % 7 for i in range(data.shape[0])]
    day_in_week = np.array(day_in_week)
    day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((2, 1, 0))
    # day_in_week 16992 × 307 × 1
    # feature_list 3 × T × N × D
    feature_list.append(day_in_week)

    # generate_data
    # data 16992 × 307 × 3
    data = np.concatenate(feature_list, axis=-1)
    x, y = seq2instance(data, args.P, args.Q)

    # split train/val/test
    # TODO 缺少padding处理
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

    print('trainX: %s\ttrainY: %s' % (trainX.shape, trainY.shape))
    print('valX: %s\tvalY: %s' % (valX.shape, valY.shape))
    print('testX: %s\ttestY: %s' % (testX.shape, testY.shape))

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
    testY[..., 0] = scaler.transform(testY[..., 0])

    train_data = list(zip(trainX, trainY))
    eval_data = list(zip(valX, valY))
    test_data = list(zip(testX, testY))

    num_padding = (args.batch_size - (len(train_data) % args.batch_size)) % args.batch_size
    data_padding = np.repeat(train_data[-1:], num_padding, axis=0)
    train_data = np.concatenate([train_data, data_padding], axis=0)
    num_padding = (args.batch_size - (len(eval_data) % args.batch_size)) % args.batch_size
    data_padding = np.repeat(eval_data[-1:], num_padding, axis=0)
    eval_data = np.concatenate([eval_data, data_padding], axis=0)
    num_padding = (args.batch_size - (len(test_data) % args.batch_size)) % args.batch_size
    data_padding = np.repeat(test_data[-1:], num_padding, axis=0)
    test_data = np.concatenate([test_data, data_padding], axis=0)

    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)

    # trainset = torch.utils.data.TensorDataset(
    #     torch.FloatTensor(trainX), torch.FloatTensor(trainY)
    # )
    # valset = torch.utils.data.TensorDataset(
    #     torch.FloatTensor(valX), torch.FloatTensor(valY)
    # )
    # testset = torch.utils.data.TensorDataset(
    #     torch.FloatTensor(testX), torch.FloatTensor(testY)
    # )

    feature_name = {'x': 'float', 'y': 'float'}

    def collator(indices):
        batch = Batch(feature_name, pad_item=None, pad_max_len=None)
        for item in indices:
            batch.append(copy.deepcopy(item))
        batch.padding()
        return batch

    trainset_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator,
    )
    valset_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
    )
    testset_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
    )

    return trainset_loader, valset_loader, testset_loader, scaler
