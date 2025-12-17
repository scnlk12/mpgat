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


def temporal_weighted_loss(preds, labels, loss_func, weight_scheme='progressive', null_val=np.nan, **kwargs):
    """
    时间步加权损失函数 - 对未来步赋予更高权重

    Args:
        preds: 预测值 [B, T, N] 或 [B, T, N, F]
        labels: 真实值 [B, T, N] 或 [B, T, N, F]
        loss_func: 基础损失函数 (masked_mae_torch 或 masked_huber_loss)
        weight_scheme: 权重方案
            - 'progressive': 渐进式 (前4步:1.0, 中4步:1.2, 后4步:1.5)
            - 'linear': 线性递增 (从1.0到2.0)
            - 'exponential': 指数递增 (1.0, 1.1, 1.2, ..., 1.5)
            - 'custom': 自定义权重 (通过kwargs['weights']传入)
        null_val: mask值
        **kwargs: 传递给基础损失函数的其他参数

    Returns:
        加权后的损失值
    """
    # preds: [B, T, N] 或 [B, T, N, F]
    if preds.dim() == 3:
        B, T, N = preds.shape
    else:
        B, T, N, F = preds.shape

    # 生成时间步权重
    if weight_scheme == 'progressive':
        # 渐进式：前4步1.0, 中4步1.2, 后4步1.5
        weights = torch.ones(T, device=preds.device)
        third = T // 3
        weights[:third] = 1.0
        weights[third:2*third] = 1.2
        weights[2*third:] = 1.5
    elif weight_scheme == 'linear':
        # 线性递增：从1.0到2.0
        weights = torch.linspace(1.0, 2.0, T, device=preds.device)
    elif weight_scheme == 'exponential':
        # 指数递增：1.0 到 1.5
        weights = torch.linspace(0, 1, T, device=preds.device)
        weights = 1.0 + 0.5 * weights  # [1.0, ..., 1.5]
    elif weight_scheme == 'custom':
        # 自定义权重
        weights = kwargs.get('weights', torch.ones(T, device=preds.device))
        if isinstance(weights, (list, tuple)):
            weights = torch.tensor(weights, device=preds.device, dtype=preds.dtype)
    else:
        raise ValueError(f"Unknown weight_scheme: {weight_scheme}")

    # 计算每个时间步的损失
    step_losses = []
    for t in range(T):
        if preds.dim() == 3:
            pred_t = preds[:, t, :]  # [B, N]
            label_t = labels[:, t, :]  # [B, N]
        else:
            pred_t = preds[:, t, :, :]  # [B, N, F]
            label_t = labels[:, t, :, :]  # [B, N, F]

        # 计算该步的损失
        loss_t = loss_func(pred_t, label_t, null_val=null_val, **kwargs)
        step_losses.append(loss_t * weights[t])

    # 加权平均
    total_loss = torch.stack(step_losses).sum() / weights.sum()

    return total_loss


def masked_mae_torch_weighted(preds, labels, null_val=np.nan, mask_val=np.nan, weight_scheme='progressive'):
    """
    带时间步加权的Masked MAE

    Args:
        preds: 预测值 [B, T, N]
        labels: 真实值 [B, T, N]
        null_val: mask值
        mask_val: 最小阈值mask
        weight_scheme: 权重方案 ('progressive', 'linear', 'exponential')

    Returns:
        加权MAE损失
    """
    return temporal_weighted_loss(
        preds, labels,
        loss_func=lambda p, l, **kw: masked_mae_torch(p, l, null_val=kw.get('null_val', np.nan), mask_val=kw.get('mask_val', np.nan)),
        weight_scheme=weight_scheme,
        null_val=null_val,
        mask_val=mask_val
    )


def masked_huber_loss_weighted(preds, labels, null_val=np.nan, mask_val=np.nan, delta=1.0, weight_scheme='progressive'):
    """
    带时间步加权的Masked Huber Loss

    Args:
        preds: 预测值 [B, T, N]
        labels: 真实值 [B, T, N]
        null_val: mask值
        mask_val: 最小阈值mask
        delta: Huber损失的delta参数
        weight_scheme: 权重方案 ('progressive', 'linear', 'exponential')

    Returns:
        加权Huber损失
    """
    return temporal_weighted_loss(
        preds, labels,
        loss_func=lambda p, l, **kw: masked_huber_loss(p, l, null_val=kw.get('null_val', np.nan),
                                                        mask_val=kw.get('mask_val', np.nan),
                                                        delta=kw.get('delta', 1.0)),
        weight_scheme=weight_scheme,
        null_val=null_val,
        mask_val=mask_val,
        delta=delta
    )


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
