import numpy as np
import torch


def MSE(y_true, y_pred, null_val=0):
    """
    计算均方误差(Mean Square Error)

    Args:
        y_true: 真实值
        y_pred: 预测值
        null_val: 需要mask的值(默认为0)

    Returns:
        MSE值
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype(np.float32)

        # 计算平方误差
        mse = np.square(y_pred - y_true)
        mse = mse * mask
        mse = np.nan_to_num(mse)

        # 只对有效值求平均(不进行mask归一化)
        valid_count = np.sum(mask)
        if valid_count > 0:
            return np.sum(mse) / valid_count
        else:
            return 0.0


def masked_mae_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_huber_loss(preds, labels, null_val=np.nan, mask_val=np.nan, delta=1.0):
    """
    使用PyTorch内置的smooth_l1_loss计算Huber损失

    Args:
        preds: 预测值
        labels: 真实值
        null_val: 需要mask的值
        mask_val: 最小阈值mask
        delta: Huber损失的delta参数(beta),控制二次损失和线性损失的转换点

    Returns:
        Huber损失值
    """
    labels_copy = labels.clone()
    labels_copy[torch.abs(labels_copy) < 1e-4] = 0

    # 创建mask
    if np.isnan(null_val):
        mask = ~torch.isnan(labels_copy)
    else:
        mask = labels_copy.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels_copy.ge(mask_val)

    mask = mask.float()
    mask_sum = torch.sum(mask)

    # 避免除零
    if mask_sum == 0:
        return torch.tensor(0.0, device=preds.device)

    # 使用PyTorch内置实现(更稳定,支持FP16)
    loss = torch.nn.functional.smooth_l1_loss(
        preds, labels_copy,
        reduction='none',
        beta=delta  # delta=1.0更适合归一化数据
    )

    # 应用mask并求平均
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sum(loss) / mask_sum


def RMSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def MAE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae


def MAPE(y_true, y_pred, null_val=0, epsilon=1e-3):
    """
    计算平均绝对百分比误差(Mean Absolute Percentage Error)

    Args:
        y_true: 真实值
        y_pred: 预测值
        null_val: 需要mask的值
        epsilon: 最小阈值,过滤小于此值的真实值以避免除零错误

    Returns:
        MAPE值(百分比,已乘以100)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # 创建mask: 过滤null值和小于epsilon的值
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)

        # 过滤绝对值小于epsilon的值以避免除零和数值不稳定
        mask = mask & (np.abs(y_true) >= epsilon)
        mask = mask.astype("float32")

        # 计算百分比误差
        mape = np.abs(np.divide(y_pred - y_true, y_true))
        mape = mape * mask

        # 只对有效值求平均(不进行mask归一化)
        mape = np.nan_to_num(mape)
        valid_count = np.sum(mask)

        if valid_count > 0:
            # 返回百分比值(0-100)
            return np.sum(mape) / valid_count * 100
        else:
            return 0.0
    

def RMSE_MAE_MAPE(y_true, y_pred):
    return (
        RMSE(y_true, y_pred),
        MAE(y_true, y_pred),
        MAPE(y_true, y_pred),
    )


def MSE_RMSE_MAE_MAPE(y_true, y_pred):
    return (
        MSE(y_true, y_pred),
        RMSE(y_true, y_pred),
        MAE(y_true, y_pred),
        MAPE(y_true, y_pred),
    )
