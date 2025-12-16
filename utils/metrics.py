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
    # 删除了 labels[torch.abs(labels) < 1e-4] = 0 这行代码
    # 原因: 1e-4阈值在原始数据上没有意义,且可能误处理小值
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
    # 删除了 labels_copy[torch.abs(labels_copy) < 1e-4] = 0 这行代码
    # 原因: 1e-4阈值在原始数据上没有意义,且可能误处理小值

    # 创建mask
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)

    mask = mask.float()
    mask_sum = torch.sum(mask)

    # 避免除零
    if mask_sum == 0:
        return torch.tensor(0.0, device=preds.device)

    # 使用PyTorch内置实现(更稳定,支持FP16)
    loss = torch.nn.functional.smooth_l1_loss(
        preds, labels,
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


def RMSE_MAE_MAPE_with_zero_stats(y_true, y_pred, epsilon=1e-3):
    """
    计算RMSE, MAE, MAPE，并额外统计零值的预测情况

    Args:
        y_true: 真实值
        y_pred: 预测值
        epsilon: 判定为零的阈值

    Returns:
        dict: 包含整体指标和零值/非零值分离统计
    """
    # 整体指标
    overall_rmse, overall_mae, overall_mape = RMSE_MAE_MAPE(y_true, y_pred)

    # 零值mask
    zero_mask = np.abs(y_true) < epsilon
    non_zero_mask = ~zero_mask

    # 零值位置的统计
    zero_count = zero_mask.sum()
    if zero_count > 0:
        zero_mae = np.abs(y_pred[zero_mask] - y_true[zero_mask]).mean()
        zero_rmse = np.sqrt(np.square(y_pred[zero_mask] - y_true[zero_mask]).mean())
    else:
        zero_mae = 0.0
        zero_rmse = 0.0

    # 非零值位置的统计
    non_zero_count = non_zero_mask.sum()
    if non_zero_count > 0:
        non_zero_mae = np.abs(y_pred[non_zero_mask] - y_true[non_zero_mask]).mean()
        non_zero_rmse = np.sqrt(np.square(y_pred[non_zero_mask] - y_true[non_zero_mask]).mean())
    else:
        non_zero_mae = 0.0
        non_zero_rmse = 0.0

    return {
        'overall': {
            'RMSE': overall_rmse,
            'MAE': overall_mae,
            'MAPE': overall_mape
        },
        'zero_flow': {
            'MAE': zero_mae,
            'RMSE': zero_rmse,
            'count': int(zero_count),
            'ratio': float(zero_count / y_true.size)
        },
        'non_zero': {
            'MAE': non_zero_mae,
            'RMSE': non_zero_rmse,
            'count': int(non_zero_count),
            'ratio': float(non_zero_count / y_true.size)
        }
    }


def MSE_RMSE_MAE_MAPE(y_true, y_pred):
    return (
        MSE(y_true, y_pred),
        RMSE(y_true, y_pred),
        MAE(y_true, y_pred),
        MAPE(y_true, y_pred),
    )
