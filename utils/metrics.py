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
        mask /= np.mean(mask)  # mask归一化,与masked_mae_torch一致

        # 计算平方误差
        mse = np.square(y_pred - y_true)
        mse = np.nan_to_num(mse * mask)
        mse = np.mean(mse)
        return mse


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


def RMSE(y_true, y_pred, null_val=0):
    """
    计算均方根误差(Root Mean Square Error)

    Args:
        y_true: 真实值
        y_pred: 预测值
        null_val: 需要mask的值(默认为0)

    Returns:
        RMSE值
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)  # mask归一化,与masked_mae_torch一致

        # 计算平方误差
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def MAE(y_true, y_pred, null_val=0):
    """
    计算平均绝对误差(Mean Absolute Error)

    Args:
        y_true: 真实值
        y_pred: 预测值
        null_val: 需要mask的值(默认为0)

    Returns:
        MAE值
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)  # mask归一化,与masked_mae_torch一致

        # 计算绝对误差
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae


def MAPE(y_true, y_pred, null_val=0, epsilon=5.0):
    """
    计算平均绝对百分比误差(Mean Absolute Percentage Error)

    Args:
        y_true: 真实值
        y_pred: 预测值
        null_val: 需要mask的值
        epsilon: 最小阈值,过滤小于此值的真实值以避免除零错误
                (对于交通流量数据,设置为5.0以过滤小流量值,关注主要流量时段)

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
        # 对于交通流量预测,过滤掉流量<0.1的样本是合理的
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
