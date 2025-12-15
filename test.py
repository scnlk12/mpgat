# import pandas as pd
# import numpy as np
# import torch


# if __name__ == "__main__":
#     # df = pd.read_hdf("PeMS\pems-bay.h5")
#     # Traffic = df.values
#     # Time = df.index
#     # print(Time.shape)
#     # dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
#     # print(dayofweek.shape)

#     # timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
#     #             // 300
#     # print(timeofday)
#     # timeofday = np.reshape(timeofday, newshape = (-1, 1))
#     # print(timeofday.shape)
#     # print(timeofday)

#     # Time = np.concatenate((dayofweek, timeofday), axis=-1)
#     # print(Time.shape)


#     # pems = np.load("PeMS\PEMS04.npz")
#     # data = pems['data'][:, :, 0]
#     # print(data.shape)

#     # import torch
#     # test1 = torch.ones(5, 24, 7)
#     # test2 = torch.ones(5, 24, 288)
#     # test3 = torch.cat((test1, test2), dim=-1)
#     # print(test3.shape)
#     # test4 = test3.unsqueeze(dim=2)
#     # print(test4.shape)

    

#     pt = torch.load("C:/Users/12062/Downloads/GMAN-pytorch/PeMS/saved_models/GMAN-PeMS04-2024-03-01-09-11-10.pt")
#     print(pt)
    

import torch
import argparse
import utils
import os
import time
import numpy as np
from utils import cal_lape, data_prepare
from model import GMAN
from utils.metrics import RMSE_MAE_MAPE

@torch.no_grad()
def predict(model, loader):
    model.eval()
    y = []
    out = []

    for batch in loader:
        batch.to_tensor(DEVICE)
        x_batch = batch['x']
        y_batch = batch['y']

        # TE = torch.cat((x_batch[:, :, :, 1:], y_batch[:, :, :, 1:]), dim=1)
        TE = x_batch[:, :, :, 1:]

        # out_batch = model(torch.unsqueeze(x_batch[:, :, :, 0], -1), TE)
        out_batch = model(x_batch, TE)
        out_batch = SCALER.inverse_transform(out_batch)
        y_batch = SCALER.inverse_transform(y_batch[:, :, :, 0])

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out

@torch.no_grad()
def test_model(model, testset_loader, log=None):
    model.eval()
    print("--------- Test ---------")

    start = time.time()
    y_true, y_pred = predict(model, testset_loader)
    end = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print(out_str)
    print("Inference time: %.2f s" % (end - start))

if __name__ == "__main__":
    # df = pd.read_hdf("PeMS\pems-bay.h5")
    # Traffic = df.values
    # Time = df.index
    # print(Time.shape)
    # dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
    # print(dayofweek.shape)

    # timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
    #             // 300
    # print(timeofday)
    # timeofday = np.reshape(timeofday, newshape = (-1, 1))
    # print(timeofday.shape)
    # print(timeofday)

    # Time = np.concatenate((dayofweek, timeofday), axis=-1)
    # print(Time.shape)


    # pems = np.load("PeMS\PEMS04.npz")
    # data = pems['data'][:, :, 0]
    # print(data.shape)

    # import torch
    # test1 = torch.ones(5, 24, 7)
    # test2 = torch.ones(5, 24, 288)
    # test3 = torch.cat((test1, test2), dim=-1)
    # print(test3.shape)
    # test4 = test3.unsqueeze(dim=2)
    # print(test4.shape)

    parser = argparse.ArgumentParser()
    parser.add_argument('--time_slot', type = int, default = 5,
                    help = 'a time step is 5 mins')
    parser.add_argument('--P', type = int, default = 12,
                    help = 'history steps')
    parser.add_argument('--Q', type = int, default = 12,
                        help = 'prediction steps')
    parser.add_argument('--L', type = int, default = 1,
                        help = 'number of STAtt Blocks')
    parser.add_argument('--T', type = int, default = 288)
    parser.add_argument('--embed_dim', type = int, default = 1)
    parser.add_argument('--K', type = int, default = 8,
                        help = 'number of attention heads')
    parser.add_argument('--input_dim', type = int, default = 3)
    parser.add_argument('--d', type = int, default = 8,
                        help = 'dims of each head attention outputs')
    parser.add_argument('--train_ratio', type = float, default = 0.6,
                        help = 'training set [default : 0.7]')
    parser.add_argument('--val_ratio', type = float, default = 0.2,
                        help = 'validation set [default : 0.1]')
    parser.add_argument('--test_ratio', type = float, default = 0.2,
                        help = 'testing set [default : 0.2]')
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'batch size')
    parser.add_argument('--max_epoch', type = int, default = 1000,
                        help = 'epoch to run')
    parser.add_argument('--patience', type = int, default = 10,
                        help = 'patience for early stop')
    parser.add_argument('--learning_rate', type=float, default = 0.0001,
                        help = 'initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default = 5,
                        help = 'decay epoch')
    parser.add_argument('--traffic_file', default = 'data/PEMS04.npz',
                        help = 'traffic file')
    parser.add_argument('--SE_file', default = 'data/SE.txt',
                        help = 'spatial emebdding file')
    parser.add_argument('--model_file', default = 'data/GMAN(PeMS)',
                        help = 'save the model to disk')
    parser.add_argument('--log_file', default = 'data/log(PeMS)',
                        help = 'log file')
    parser.add_argument('--loss_func', default = 'mae',
                        help = 'loss function')
    parser.add_argument('--lr_decay', default=True)
    parser.add_argument('--lr_decay_rate', default='0.3')
    parser.add_argument('--lr_decay_step', default='5,20,40,70')
    
    args = parser.parse_args()

    start = time.time()

    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu")
    # DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    current_directory = os.getcwd()
    print("当前工作目录是：", current_directory)

    log = open(args.log_file, 'w')
    utils.log_string(log, str(args)[10 : -1])

    # load data
    utils.log_string(log, 'loading data...')
    train_loader, val_loader, test_loader, SCALER = data_prepare.get_dataloaders(args, log=log)
    utils.log_string(log, 'data loaded!')

    # lap_mx
    weight_adj_epsilon = 0.1
    adj_mx = np.zeros((307, 307), dtype=float)

    # with open('data/PEMS04.csv', 'r') as f:
    #     f.readline()  # 略过表头那一行
    #     reader = csv.reader(f)
    #     for row in reader:
    #         if len(row) != 3:
    #             continue
    #         i, j, distance = int(row[0]), int(row[1]), float(row[2])

    #         adj_mx[i][j] = distance

    with open('data/Adj.txt', 'r') as f:
        data = f.readlines()
        for i in data:
                i = i.strip('\n')
                arr = i.split('\t')
                res =float(arr[2])
                # res = float(arr[2])
                # print(res, end='')
                adj_mx[int(arr[0])][int(arr[1])] = res
    
    # 使用Gussian计算节点之间距离
    # distances = adj_mx[~np.isinf(adj_mx)].flatten()
    # std = distances.std()
    # adj_mx = np.exp(-np.square(adj_mx / std))

    # adj_mx[adj_mx < weight_adj_epsilon] = 0

    lap_mx, LAP = cal_lape(adj_mx)
    lap_mx = lap_mx.to(torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))

    # init model
    model = GMAN(args.input_dim, args.P, args.Q, args.T, args.L, args.K, args.d, lap_mx, LAP)
    model = model.to(DEVICE)

    # model = torch.load("../saved_models/GMAN-PeMS04-2024-03-01-09-11-10.pt")
    # model.eval()
    # model.load_state_dict(torch.load("../saved_models/GMAN-PeMS04-2024-08-08-11-05-23.pt"))
    # model.load_state_dict(torch.load("../saved_models/GMAN-PeMS04-2024-08-02-15-21-23.pt"))
    # model.load_state_dict(torch.load("model_cache/model_pems04_epoch23.tar")['model_state_dict'])
    model.load_state_dict(torch.load("model_cache/model_2025-03-07-11-44-23_pems04_epoch31.tar")['model_state_dict'])


    test_model(model, testset_loader=test_loader)





