#!/bin/bash

# 并行实验脚本 - 在不同GPU上同时运行不同数据集的实验
# 适用于有4张或更多GPU的服务器

echo "=========================================="
echo "MP-STGAT Parallel Experiments"
echo "在不同GPU上并行运行多个实验"
echo "=========================================="

# 配置文件
CONFIG="config.yaml"

# 创建日志目录
mkdir -p parallel_logs

# 后台运行函数
run_experiment() {
    local gpu_id=$1
    local dataset=$2
    local log_file="parallel_logs/${dataset}_gpu${gpu_id}.log"

    echo "启动实验: $dataset on GPU $gpu_id"

    CUDA_VISIBLE_DEVICES=$gpu_id python train_distributed.py \
        --config $CONFIG \
        --gpu_ids 0 \
        --traffic_file data/$dataset/$dataset.npz \
        > $log_file 2>&1 &

    echo "  日志文件: $log_file"
    echo "  进程ID: $!"
}

# ==================== 并行实验配置 ====================

# 实验1: PEMS03 on GPU 0
run_experiment 0 "PEMS03"
sleep 2  # 避免同时启动造成资源竞争

# 实验2: PEMS04 on GPU 1
run_experiment 1 "PEMS04"
sleep 2

# 实验3: PEMS07 on GPU 2
run_experiment 2 "PEMS07"
sleep 2

# 实验4: PEMS08 on GPU 3
run_experiment 3 "PEMS08"

echo ""
echo "=========================================="
echo "所有实验已在后台启动！"
echo "=========================================="
echo ""
echo "查看实验状态："
echo "  watch -n 1 nvidia-smi"
echo ""
echo "查看日志："
echo "  tail -f parallel_logs/PEMS03_gpu0.log"
echo "  tail -f parallel_logs/PEMS04_gpu1.log"
echo "  tail -f parallel_logs/PEMS07_gpu2.log"
echo "  tail -f parallel_logs/PEMS08_gpu3.log"
echo ""
echo "等待所有实验完成："
echo "  wait"
echo ""

# 等待所有后台任务完成
wait

echo "=========================================="
echo "所有实验已完成！"
echo "=========================================="