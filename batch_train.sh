#!/bin/bash

# 批量训练脚本 - 在所有PEMS数据集上训练

echo "=========================================="
echo "MP-STGAT Batch Training on All Datasets"
echo "=========================================="

# 配置
CONFIG="configs/all_datasets.yaml"
GPU_IDS="0,1,2,3"  # 修改为你的GPU配置

# 数据集列表
DATASETS=("PEMS03" "PEMS04" "PEMS07" "PEMS08")

# 循环训练
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training on $dataset..."
    echo "=========================================="

    python train_distributed.py \
        --config $CONFIG \
        --gpu_ids $GPU_IDS \
        --traffic_file data/$dataset/$dataset.npz

    if [ $? -eq 0 ]; then
        echo "✓ $dataset training completed successfully"
    else
        echo "✗ $dataset training failed"
    fi
done

echo ""
echo "=========================================="
echo "All training tasks completed!"
echo "=========================================="