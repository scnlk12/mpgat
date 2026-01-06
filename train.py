import argparse
import time, datetime
from functools import partial

import torch
import torch.nn as nn
import numpy as np

import utils, model, data_prepare
from model import GMAN
from utils import print_model_parameters, cal_lape
from metrics import RMSE_MAE_MAPE, masked_mae_torch

import copy
import matplotlib.pyplot as plt

import os
# For pems04
import csv

import random

from torch.utils.tensorboard import SummaryWriter

def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,

    criterion,
    clip_grad=0,
    max_epochs=200,
    early_stop=20,
    verbose=1,
    plot=False,
    log=None,
    save=None,
):
    model = model.to(DEVICE)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=log
        )
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)

        # ---------------------------
        # 📌✨ 在这里加入 TensorBoard
        # ---------------------------
        writer.add_scalar("Loss/train", train_loss, global_step)
        writer.add_scalar("Loss/val", val_loss, global_step)
        global_step += 1
        # ---------------------------

        if (epoch + 1) % verbose == 0:
            print(datetime.datetime.now(), "Epoch", epoch + 1," \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss)
            
            str_out = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "\tEpoch" + str(epoch + 1) + " \tTrain Loss = " + str(train_loss) + ", Val Loss = " + str(val_loss)
            
            utils.log_string(log, str_out)
            # config = dict()
            # config['model_state_dict'] = model.state_dict()
            # config['optimizer_state_dict'] = optimizer.state_dict()
            # config['epoch'] = epoch
            # save_model_with_epoch(epoch, log, config)

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= early_stop:
                print(f"early stop at epoch: %04d" % epoch)
                # utils.log_string(log, "early stop at epoch: " + epoch)
                break

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))

    out_str = f"Early stopping at epoch: {epoch+1}\n"
    out_str += f"Best at epoch {best_epoch+1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print(out_str)
    utils.log_string(log, out_str)

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if save:
        torch.save(best_state_dict, save)
    return model

def save_model_with_epoch(epoch, log, config):
    model_cache_path = './model_cache'
    if not os.path.exists(model_cache_path):
        os.makedirs(model_cache_path)

    model_path = model_cache_path + '/' + 'model' + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_' + 'pems03' + '_epoch%d.tar' % epoch
    torch.save(config, model_path)
    # self._logger.info("Saved model at {}".format(epoch))
    utils.log_string(log, "Saved model at {}".format(epoch))
    return model_path


@torch.no_grad()
def predict(model, loader, return_embeddings=False):
    model.eval()
    y = []
    out = []
    embeddings = [] if return_embeddings else None

    for batch in loader:
        batch.to_tensor(DEVICE)
        x_batch = batch['x']
        y_batch = batch['y']

        # TE = torch.cat((x_batch[:, :, :, 1:], y_batch[:, :, :, 1:]), dim=1)
        TE = x_batch[:, :, :, 1:]

        # 提取 embedding（如果需要）
        if return_embeddings:
            embedding_batch = model(x_batch, TE, return_embedding=True)
            embeddings.append(embedding_batch.cpu().numpy())

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

    if return_embeddings:
        embeddings = np.vstack(embeddings)  # (samples, Q, num_nodes, 256)
        return y, out, embeddings

    return y, out

@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    for batch in valset_loader:
        batch.to_tensor(DEVICE)

        x_batch = batch['x']
        y_batch = batch['y']

        # TE = torch.cat((x_batch[:, :, :, 1:], y_batch[:, :, :, 1:]), dim=1)
        TE = x_batch[:, :, :, 1:]

        # out_batch = model(torch.unsqueeze(x_batch[:, :, :, 0], -1), TE)
        out_batch = model(x_batch, TE)
        out_batch = SCALER.inverse_transform(out_batch)
        y_batch = SCALER.inverse_transform(y_batch[:, :, :, 0])
        loss = criterion(out_batch, y_batch)
        # loss = criterion(y_batch, out_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)

def train_one_epoch(
    model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=None
):
    
    model.train()
    batch_loss_list = []
    for batch in trainset_loader:
        batch.to_tensor(DEVICE)

        x_batch = batch['x']
        y_batch = batch['y']

        # TE (B, T, N, 2) 
        # x_batch (B, P, N, F) y_batch (B, Q, N, F)
        # TE (B, P, N, 2)
        # TE = torch.cat((x_batch[:, :, :, 1:], y_batch[:, :, :, 1:]), dim=1)
        TE = x_batch[:, :, :, 1:]

        # out_batch = model(torch.unsqueeze(x_batch[:, :, :, 0], -1), TE)
        out_batch = model(x_batch, TE)
        out_batch = SCALER.inverse_transform(out_batch)
        y_batch = SCALER.inverse_transform(y_batch[:, :, :, 0])
        
        loss = criterion(out_batch, y_batch)
        # loss = criterion(y_batch, out_batch)
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss

@torch.no_grad()
def test_model(model, testset_loader, log=None):
    model.eval()
    print("--------- Test ---------")
    utils.log_string(log, "--------- Test ---------")

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
    utils.log_string(log, out_str)
    print("Inference time: %.2f s" % (end - start))
    utils.log_string(log, "Inference time: " + str(end - start))

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--time_slot', type = int, default = 5,
                    help = 'a time step is 5 mins')
    parser.add_argument('--P', type = int, default = 12,
                    help = 'history steps')
    parser.add_argument('--Q', type = int, default = 12,
                        help = 'prediction steps')
    parser.add_argument('--L', type = int, default = 2,
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
    parser.add_argument('--batch_size', type = int, default = 16,
                        help = 'batch size')
    parser.add_argument('--max_epoch', type = int, default = 1000,
                        help = 'epoch to run')
    parser.add_argument('--patience', type = int, default = 50,
                        help = 'patience for early stop')
    parser.add_argument('--learning_rate', type=float, default = 0.001,
                        help = 'initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default = 30,
                        help = 'decay epoch')
    parser.add_argument('--traffic_file', default = 'data/PEMS03/PEMS03.npz',
                        help = 'traffic file')
    parser.add_argument('--SE_file', default = 'data/SE.txt',
                        help = 'spatial emebdding file')
    parser.add_argument('--model_file', default = 'data/GMAN(PeMS)',
                        help = 'save the model to disk')
    parser.add_argument('--log_file', default = 'data/log(PeMS)' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                        help = 'log file')
    parser.add_argument('--loss_func', default = 'masked_mae',
                        help = 'loss function')
    parser.add_argument('--lr_decay', default=True)
    parser.add_argument('--lr_decay_rate', default='0.3')
    parser.add_argument('--lr_decay_step', default='5,20,40,70')
    # 随机种子 
    setup_seed(42)
    # tensorboard
    writer = SummaryWriter(log_dir="./runLog/mpgat") 
    global_step = 0
    
    args = parser.parse_args()

    start = time.time()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu")
    # DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # current_directory = os.getcwd()
    # print("当前工作目录是：", current_directory)

    log = open(args.log_file, 'w')
    utils.log_string(log, str(args)[10 : -1])

    # load data
    utils.log_string(log, 'loading data...')
    train_loader, val_loader, test_loader, SCALER = data_prepare.get_dataloaders(args, log=log)
    utils.log_string(log, 'data loaded!')

    # 自动推断数据集名称和路径
    dataset_name = args.traffic_file.split('/')[-1].replace('.npz', '')
    dataset_dir = '/'.join(args.traffic_file.split('/')[:-1])
    txt_file = os.path.join(dataset_dir, f'{dataset_name}.txt')
    csv_file = os.path.join(dataset_dir, f'{dataset_name}.csv')

    print(f"Using dataset: {dataset_name}")
    print(f"Graph structure file: {csv_file}")
    utils.log_string(log, f'Dataset: {dataset_name}')
    utils.log_string(log, f'CSV file: {csv_file}')

    # 首先读取CSV获取节点数
    temp_nodes = set()
    with open(csv_file, 'r') as f:
        f.readline()  # 跳过表头
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            temp_nodes.add(int(row[0]))
            temp_nodes.add(int(row[1]))

    # 如果有txt文件，使用id映射；否则直接使用节点索引
    if os.path.exists(txt_file):
        print(f"Found node ID mapping file: {txt_file}")
        utils.log_string(log, f'Using node mapping from: {txt_file}')
        with open(txt_file, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}
        num_nodes = len(id_dict)
    else:
        print(f"No .txt file found, using direct node indexing")
        utils.log_string(log, 'No .txt file found, using direct node indexing')
        # 创建从原始ID到连续索引的映射
        sorted_nodes = sorted(list(temp_nodes))
        id_dict = {node_id: idx for idx, node_id in enumerate(sorted_nodes)}
        num_nodes = len(sorted_nodes)

    print(f"Number of nodes: {num_nodes}")
    utils.log_string(log, f'Number of nodes: {num_nodes}')

    # 初始化邻接矩阵
    adj_mx = np.zeros((num_nodes, num_nodes), dtype=float)

    # 读取图结构CSV
    with open(csv_file, 'r') as f:
        f.readline()  # 略过表头那一行
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])

            # 检查i和j是否在id_dict中（防止KeyError）
            if i in id_dict and j in id_dict:
                idx_i = id_dict[i]  # 获取i对应的新索引
                idx_j = id_dict[j]  # 获取j对应的新索引
                adj_mx[idx_i][idx_j] = 1  # 填充邻接矩阵
                adj_mx[idx_j][idx_i] = 1  # 如果是无向图，对称填充

    print(f"Graph loaded: {num_nodes} nodes, {np.sum(adj_mx > 0) / 2} edges")
    utils.log_string(log, f'Graph: {num_nodes} nodes, {int(np.sum(adj_mx > 0) / 2)} edges')

    lap_mx, LAP = cal_lape(adj_mx)
    lap_mx = lap_mx.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # init model
    model = GMAN(args.input_dim, args.P, args.Q, args.T, args.L, args.K, args.d, lap_mx, LAP)
    model = model.to(DEVICE)
    for p in model.parameters():
        if p.dim() > 1:
            # 初始化参数设置方法 保证输入和输出方差相同
            nn.init.xavier_uniform_(p)
        else:
            # 均匀分布初始化
            nn.init.uniform_(p)
    print_model_parameters(model, only_num=False)

    # model saving path
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_path = f"./saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"GMAN-{dataset_name}-{now}.pt")

    #init loss function, optimizer
    if args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(DEVICE)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(DEVICE)
    elif args.loss_func == 'masked_mae':
        loss = partial(masked_mae_torch, null_val=0)
    else:
        raise ValueError
    
    # weight_decay 权值衰减
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, eps=1.0e-8,
    #                          weight_decay=True, amsgrad=False)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)


    #learning rate decay
    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                       step_size=100,
                                                       gamma=0.9)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)
        
    # criterion = nn.HuberLoss().to(DEVICE)
    # criterion = nn.L1Loss().to(DEVICE)  # 定义损失函数
    
    # train
    # try:
        model = train(
            model,
            train_loader,
            val_loader,
            optimizer,
            lr_scheduler,
            loss,
            clip_grad=5,
            max_epochs=400,
            early_stop=args.patience,
            verbose=1,
            log=log,
            save=save,
        )

        print(f"Saved Model: {save}")
        utils.log_string(log, "Saved Model:" + save)

        # 自动保存 embeddings 用于迁移学习
        print("\n" + "="*60)
        print("Extracting embeddings for transfer learning...")
        utils.log_string(log, "Extracting embeddings for transfer learning...")

        embedding_dir = "./embeddings"
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)

        # 在测试集上提取 embeddings
        y_test, pred_test, embeddings_test = predict(model, test_loader, return_embeddings=True)

        # 保存 embeddings
        embedding_save_path = os.path.join(embedding_dir, f"{dataset_name}_embeddings.npz")
        np.savez_compressed(
            embedding_save_path,
            embeddings=embeddings_test,  # [N, Q, num_nodes, 256]
            labels=y_test,               # [N, Q, num_nodes]
            predictions=pred_test,       # [N, Q, num_nodes]
            mean=SCALER.mean,
            std=SCALER.std,
            dataset=dataset_name,
        )

        print(f"✓ Embeddings saved to: {embedding_save_path}")
        print(f"  - embeddings: {embeddings_test.shape}")
        print(f"  - labels: {y_test.shape}")
        print(f"  - predictions: {pred_test.shape}")
        utils.log_string(log, f"Embeddings saved: {embedding_save_path}")
        utils.log_string(log, f"  embeddings: {embeddings_test.shape}, labels: {y_test.shape}")
        print("="*60 + "\n")

        test_model(model, test_loader, log=log)
    # except Exception as e:
    #     print(f"Error: {e}")
    #     utils.log_string(log, "Error: " + str(e))

