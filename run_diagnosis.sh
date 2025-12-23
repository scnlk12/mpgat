#!/bin/bash

# 模型诊断快速启动脚本
# 使用方法: bash run_diagnosis.sh

echo "=========================================="
echo "      模型诊断工具"
echo "=========================================="
echo ""

# ============ 配置区域 - 请修改这里 ============

# 数据集
DATASET="PEMS03"
TRAFFIC_FILE="data/PEMS03/PEMS03.npz"

# 模型路径 - 修改为你的模型文件
MODEL_PATH="./saved_models/GMAN-PEMS03-xxx.pt"

# 模型参数
P=12
Q=12
L=2
K=8
D=8
INPUT_DIM=3
BATCH_SIZE=16

# 输出目录
OUTPUT_DIR="./diagnosis_$(date +%Y%m%d_%H%M%S)"

# ============================================

echo "配置信息:"
echo "  数据集: $DATASET"
echo "  模型文件: $MODEL_PATH"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "请修改脚本中的 MODEL_PATH 变量"
    exit 1
fi

# 检查数据文件
if [ ! -f "$TRAFFIC_FILE" ]; then
    echo "错误: 数据文件不存在: $TRAFFIC_FILE"
    echo "请修改脚本中的 TRAFFIC_FILE 变量"
    exit 1
fi

echo "开始诊断..."
echo ""

# 运行诊断
python diagnose_model.py \
    --traffic_file "$TRAFFIC_FILE" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --P $P --Q $Q --L $L --K $K --d $D \
    --input_dim $INPUT_DIM \
    --batch_size $BATCH_SIZE

if [ $? -ne 0 ]; then
    echo ""
    echo "错误: 诊断失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "          诊断完成!"
echo "=========================================="
echo ""
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "目录结构:"
echo "  $OUTPUT_DIR/"
echo "  ├── plots/                        可视化图表"
echo "  │   ├── error_by_flow.png         按流量区间的误差"
echo "  │   ├── temporal_error_pattern.png 时间步误差模式"
echo "  │   └── node_analysis.png         节点误差分析"
echo "  ├── sample_details/               最差样本详情"
echo "  │   ├── worst_sample_1_*.png      样本1的预测曲线"
echo "  │   └── ...                       (共10个)"
echo "  ├── node_analysis.csv             节点分析数据"
echo "  └── optimization_suggestions.txt  优化建议"
echo ""
echo "下一步:"
echo "  1. 查看可视化图表: open $OUTPUT_DIR/plots/*.png"
echo "  2. 查看优化建议: cat $OUTPUT_DIR/optimization_suggestions.txt"
echo "  3. 查看最差样本: open $OUTPUT_DIR/sample_details/*.png"
echo "  4. 根据诊断结果调整模型配置和训练策略"
echo ""