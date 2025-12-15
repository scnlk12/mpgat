import os
from typing import Any, Dict

import yaml


class Config:
    """配置类，支持点访问和字典访问"""
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def to_dict(self):
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str = 'config.yaml') -> Config:
    """
    从YAML文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        Config对象
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def merge_args_with_config(config: Config, args: argparse.Namespace) -> Config:
    """
    将命令行参数合并到配置中（命令行参数优先级更高）

    Args:
        config: 配置对象
        args: 命令行参数

    Returns:
        合并后的配置对象
    """
    # 将args转换为字典
    args_dict = vars(args)

    # 映射关系：命令行参数名 -> 配置路径
    mappings = {
        'traffic_file': ('data', 'traffic_file'),
        'batch_size': ('data', 'batch_size'),
        'learning_rate': ('training', 'learning_rate'),
        'max_epoch': ('training', 'max_epoch'),
        'patience': ('training', 'patience'),
        'P': ('model', 'P'),
        'Q': ('model', 'Q'),
        'L': ('model', 'L'),
        'K': ('model', 'K'),
        'd': ('model', 'd'),
        'gpu_ids': ('gpu', 'device_ids'),
    }

    # 合并参数
    for arg_name, config_path in mappings.items():
        if arg_name in args_dict and args_dict[arg_name] is not None:
            if len(config_path) == 2:
                section, key = config_path
                config[section][key] = args_dict[arg_name]
            else:
                config[config_path[0]] = args_dict[arg_name]

    return config


def validate_config(config: Config) -> None:
    """
    验证配置参数的合法性

    Args:
        config: 配置对象

    Raises:
        ValueError: 配置参数不合法时抛出异常
    """
    errors = []

    # 验证数据配置
    if hasattr(config, 'data'):
        data_config = config.data

        # 验证traffic_file存在
        if hasattr(data_config, 'traffic_file'):
            if not os.path.exists(data_config.traffic_file):
                errors.append(f"数据文件不存在: {data_config.traffic_file}")
        else:
            errors.append("缺少必需配置: data.traffic_file")

        # 验证batch_size
        if hasattr(data_config, 'batch_size'):
            if not isinstance(data_config.batch_size, int) or data_config.batch_size <= 0:
                errors.append(f"batch_size必须是正整数, 当前值: {data_config.batch_size}")
        else:
            errors.append("缺少必需配置: data.batch_size")

        # 验证数据集划分比例
        if hasattr(data_config, 'train_ratio') and hasattr(data_config, 'val_ratio') and hasattr(data_config, 'test_ratio'):
            total_ratio = data_config.train_ratio + data_config.val_ratio + data_config.test_ratio
            if abs(total_ratio - 1.0) > 1e-6:
                errors.append(f"train_ratio + val_ratio + test_ratio 必须等于1.0, 当前和为: {total_ratio}")

            for ratio_name in ['train_ratio', 'val_ratio', 'test_ratio']:
                ratio_value = getattr(data_config, ratio_name)
                if not (0 < ratio_value < 1):
                    errors.append(f"{ratio_name}必须在(0,1)区间内, 当前值: {ratio_value}")
    else:
        errors.append("缺少必需配置段: data")

    # 验证模型配置
    if hasattr(config, 'model'):
        model_config = config.model

        # 验证必需参数
        required_model_params = ['input_dim', 'P', 'Q', 'T', 'L', 'K', 'd', 'embed_dim', 'skip_dim']
        for param in required_model_params:
            if not hasattr(model_config, param):
                errors.append(f"缺少必需配置: model.{param}")
            else:
                value = getattr(model_config, param)
                if not isinstance(value, int) or value <= 0:
                    errors.append(f"model.{param}必须是正整数, 当前值: {value}")
    else:
        errors.append("缺少必需配置段: model")

    # 验证训练配置
    if hasattr(config, 'training'):
        training_config = config.training

        # 验证learning_rate
        if hasattr(training_config, 'learning_rate'):
            if not (0 < training_config.learning_rate < 1):
                errors.append(f"learning_rate必须在(0,1)区间内, 当前值: {training_config.learning_rate}")
        else:
            errors.append("缺少必需配置: training.learning_rate")

        # 验证max_epoch
        if hasattr(training_config, 'max_epoch'):
            if not isinstance(training_config.max_epoch, int) or training_config.max_epoch <= 0:
                errors.append(f"max_epoch必须是正整数, 当前值: {training_config.max_epoch}")
        else:
            errors.append("缺少必需配置: training.max_epoch")

        # 验证patience
        if hasattr(training_config, 'patience'):
            if not isinstance(training_config.patience, int) or training_config.patience <= 0:
                errors.append(f"patience必须是正整数, 当前值: {training_config.patience}")

        # 验证clip_grad
        if hasattr(training_config, 'clip_grad'):
            if training_config.clip_grad is not None and training_config.clip_grad <= 0:
                errors.append(f"clip_grad必须为None或正数, 当前值: {training_config.clip_grad}")

        # 验证loss_func
        if hasattr(training_config, 'loss_func'):
            valid_loss_funcs = ['mae', 'mse', 'masked_mae']
            if training_config.loss_func not in valid_loss_funcs:
                errors.append(f"loss_func必须是{valid_loss_funcs}之一, 当前值: {training_config.loss_func}")
    else:
        errors.append("缺少必需配置段: training")

    # 验证GPU配置
    if hasattr(config, 'gpu'):
        gpu_config = config.gpu
        if hasattr(gpu_config, 'device_ids'):
            if not isinstance(gpu_config.device_ids, list):
                errors.append(f"gpu.device_ids必须是列表, 当前类型: {type(gpu_config.device_ids)}")
            elif len(gpu_config.device_ids) == 0:
                errors.append("gpu.device_ids不能为空列表")
            else:
                for gpu_id in gpu_config.device_ids:
                    if not isinstance(gpu_id, int) or gpu_id < 0:
                        errors.append(f"gpu.device_ids中的ID必须是非负整数, 发现: {gpu_id}")

    # 验证日志配置
    if hasattr(config, 'logging'):
        logging_config = config.logging

        # 验证verbose
        if hasattr(logging_config, 'verbose'):
            if not isinstance(logging_config.verbose, int) or logging_config.verbose <= 0:
                errors.append(f"logging.verbose必须是正整数, 当前值: {logging_config.verbose}")

    # 如果有错误,抛出异常
    if errors:
        error_message = "配置验证失败:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ValueError(error_message)


def save_config(config: Config, save_path: str):
    """
    保存配置到YAML文件

    Args:
        config: 配置对象
        save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)