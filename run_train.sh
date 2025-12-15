#!/bin/bash

# MP-STGAT 训练启动脚本

# ==================== 配置区域 ====================

# 使用配置文件
CONFIG_FILE="config.yaml"

# 数据集选择（可选：PEMS03, PEMS04, PEMS07, PEMS08）
DATASET="PEMS03"

# GPU设置：指定使用哪些GPU（逗号分隔，如 "0,1,2" 表示使用前3张GPU）
GPU_IDS="0"

# ==================== 执行区域 ====================

echo "=========================================="
echo "MP-STGAT Training Script"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Dataset: $DATASET"
echo "GPU IDs: $GPU_IDS"
echo "=========================================="

# 执行训练
python train_distributed.py \
    --config $CONFIG_FILE \
    --gpu_ids $GPU_IDS \
    --traffic_file data/$DATASET/$DATASET.npz

echo "=========================================="
echo "Training completed!"
echo "=========================================="