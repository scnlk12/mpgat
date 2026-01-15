import torch
import numpy as np
from utils import data_prepare
from utils.metrics import RMSE_MAE_MAPE

# 伪造一个 args 类，复用你的 data_prepare
class Args:
    traffic_file = 'data/PEMS08/PEMS08.npz' # 确保路径对
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    batch_size = 32
    P = 12
    Q = 12
    T = 288

args = Args()
print("Loading Data...")
_, _, test_loader, scaler = data_prepare.get_dataloaders(args, log=None)

y_trues = []
y_preds_naive = []

print("Running Naive Baseline Test...")
for batch in test_loader:
    # batch 是 [x, y]
    x, y = batch
    
    # x shape: (B, P, N, 3) -> 归一化过的
    # y shape: (B, Q, N, 3) -> 归一化过的 (取决于你最新的 data_prepare)
    
    # 1. 获取“最近时刻”的流量 (作为预测值)
    # 取输入的最后一个时间步
    latest_x = x[:, -1, :, 0] # (B, N)
    
    # 我们假设未来12步都等于当前这一步 (最笨的预测)
    # 扩展到 (B, Q, N)
    naive_pred = latest_x.unsqueeze(1).expand(-1, args.Q, -1)
    
    # 2. 获取真实值
    y_true = y[:, :, :, 0]
    
    # 3. 反归一化 (还原为真实车流量)
    # 转 numpy
    naive_pred_np = naive_pred.numpy()
    y_true_np = y_true.numpy()
    
    # 反归一化
    naive_pred_real = scaler.inverse_transform(naive_pred_np)
    # y_true_real = scaler.inverse_transform(y_true_np)
    y_true_real = y_true_np
    
    y_trues.append(y_true_real)
    y_preds_naive.append(naive_pred_real)

# 拼接
y_true_all = np.concatenate(y_trues, axis=0)
y_pred_all = np.concatenate(y_preds_naive, axis=0)

# 计算指标
print("\n========== Naive Baseline (Persistence) ==========")
# 计算 Step 1 的指标
rmse, mae, mape = RMSE_MAE_MAPE(y_true_all[:, 0, :], y_pred_all[:, 0, :])
print(f"Step 1 | RMSE = {rmse:.4f}, MAE = {mae:.4f}, MAPE = {mape:.4f}")

# 计算 Step 12 的指标
rmse, mae, mape = RMSE_MAE_MAPE(y_true_all[:, 11, :], y_pred_all[:, 11, :])
print(f"Step 12| RMSE = {rmse:.4f}, MAE = {mae:.4f}, MAPE = {mape:.4f}")