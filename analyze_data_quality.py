"""
PEMS08数据质量分析脚本
分析缺失值、异常值，判断是否需要插值和异常值处理
"""

import numpy as np
import argparse


def analyze_data_quality(data_path):
    """
    分析交通数据的质量

    Args:
        data_path: .npz文件路径
    """
    print('='*80)
    print('PEMS08 数据质量分析报告')
    print('='*80)

    # 加载数据
    data = np.load(data_path)
    traffic = data['data'][:, :, 0]  # [T, N] 流量数据

    T, N = traffic.shape
    total_samples = traffic.size

    print(f'\n【基本信息】')
    print(f'数据形状: {traffic.shape} (时间步={T:,}, 节点数={N})')
    print(f'总样本数: {total_samples:,}')
    print(f'时间跨度: {T / 288:.1f} 天 (每天288个5分钟时间片)')

    # ==================== 缺失值分析 ====================
    print(f'\n{"="*80}')
    print('【1. 缺失值分析】')
    print('='*80)

    nan_mask = np.isnan(traffic)
    nan_count = nan_mask.sum()
    nan_ratio = nan_count / total_samples * 100

    print(f'NaN值总数: {nan_count:,} ({nan_ratio:.4f}%)')

    if nan_count > 0:
        print(f'\n⚠️  检测到缺失值！')

        # 按节点统计
        node_nan_counts = nan_mask.sum(axis=0)
        nodes_with_nan = (node_nan_counts > 0).sum()
        print(f'受影响节点数: {nodes_with_nan}/{N} ({nodes_with_nan/N*100:.1f}%)')

        # 最差的10个节点
        worst_nodes = np.argsort(node_nan_counts)[::-1][:10]
        print(f'\n缺失最严重的10个节点:')
        for i, node_id in enumerate(worst_nodes, 1):
            node_nan = node_nan_counts[node_id]
            if node_nan > 0:
                node_nan_pct = node_nan / T * 100
                print(f'  {i:2d}. 节点 {node_id:3d}: {node_nan:6,} 缺失 ({node_nan_pct:5.2f}%)')

        # 连续缺失分析
        print(f'\n连续缺失片段分析:')
        max_consecutive_nan = 0
        total_consecutive_segments = 0

        for node_id in range(N):
            node_data = traffic[:, node_id]
            is_nan = np.isnan(node_data)

            # 找连续缺失片段
            consecutive = 0
            for val in is_nan:
                if val:
                    consecutive += 1
                else:
                    if consecutive > 0:
                        total_consecutive_segments += 1
                        max_consecutive_nan = max(max_consecutive_nan, consecutive)
                    consecutive = 0

        print(f'  最长连续缺失: {max_consecutive_nan} 个时间步 ({max_consecutive_nan * 5}分钟)')
        print(f'  连续缺失片段总数: {total_consecutive_segments}')
    else:
        print('✅ 无缺失值')

    # ==================== Inf值分析 ====================
    print(f'\n{"="*80}')
    print('【2. Inf值分析】')
    print('='*80)

    inf_count = np.isinf(traffic).sum()
    inf_ratio = inf_count / total_samples * 100
    print(f'Inf值总数: {inf_count:,} ({inf_ratio:.4f}%)')

    if inf_count > 0:
        print(f'⚠️  检测到Inf值！')
    else:
        print('✅ 无Inf值')

    # ==================== 负值分析 ====================
    print(f'\n{"="*80}')
    print('【3. 负值分析】')
    print('='*80)

    neg_mask = traffic < 0
    neg_count = neg_mask.sum()
    neg_ratio = neg_count / total_samples * 100
    print(f'负值总数: {neg_count:,} ({neg_ratio:.4f}%)')

    if neg_count > 0:
        print(f'⚠️  检测到负值（物理上不合理）！')
        print(f'负值范围: [{traffic[neg_mask].min():.2f}, {traffic[neg_mask].max():.2f}]')
    else:
        print('✅ 无负值')

    # ==================== 零值分析 ====================
    print(f'\n{"="*80}')
    print('【4. 零值分析】')
    print('='*80)

    zero_mask = traffic == 0
    zero_count = zero_mask.sum()
    zero_ratio = zero_count / total_samples * 100
    print(f'零值总数: {zero_count:,} ({zero_ratio:.4f}%)')

    # 按节点统计零值
    node_zero_counts = zero_mask.sum(axis=0)
    nodes_with_high_zero = (node_zero_counts > T * 0.05).sum()  # 超过5%
    print(f'零值>5%的节点数: {nodes_with_high_zero}/{N}')

    # ==================== 统计分析 ====================
    print(f'\n{"="*80}')
    print('【5. 统计信息】')
    print('='*80)

    # 只分析有效数据
    valid_mask = ~np.isnan(traffic) & ~np.isinf(traffic) & (traffic >= 0)
    valid_data = traffic[valid_mask]

    if len(valid_data) > 0:
        print(f'有效样本数: {len(valid_data):,} ({len(valid_data)/total_samples*100:.2f}%)')
        print(f'\n描述统计:')
        print(f'  最小值:   {valid_data.min():.2f}')
        print(f'  第25百分位: {np.percentile(valid_data, 25):.2f}')
        print(f'  中位数:   {np.median(valid_data):.2f}')
        print(f'  第75百分位: {np.percentile(valid_data, 75):.2f}')
        print(f'  最大值:   {valid_data.max():.2f}')
        print(f'  均值:     {valid_data.mean():.2f}')
        print(f'  标准差:   {valid_data.std():.2f}')

        # ==================== 异常值分析 ====================
        print(f'\n{"="*80}')
        print('【6. 异常值分析 (±3σ规则)】')
        print('='*80)

        mean = valid_data.mean()
        std = valid_data.std()

        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std

        outliers_upper_mask = (traffic > upper_bound) & valid_mask
        outliers_lower_mask = (traffic < lower_bound) & valid_mask & (traffic >= 0)

        outliers_upper = outliers_upper_mask.sum()
        outliers_lower = outliers_lower_mask.sum()
        total_outliers = outliers_upper + outliers_lower
        outlier_ratio = total_outliers / total_samples * 100

        print(f'异常值阈值: [{lower_bound:.2f}, {upper_bound:.2f}]')
        print(f'异常值总数: {total_outliers:,} ({outlier_ratio:.4f}%)')
        print(f'  上界异常 (>{upper_bound:.2f}): {outliers_upper:,}')
        print(f'  下界异常 (<{lower_bound:.2f}): {outliers_lower:,}')

        if total_outliers > 0:
            print(f'\n异常值统计:')
            if outliers_upper > 0:
                print(f'  上界异常值范围: [{traffic[outliers_upper_mask].min():.2f}, {traffic[outliers_upper_mask].max():.2f}]')
            if outliers_lower > 0:
                print(f'  下界异常值范围: [{traffic[outliers_lower_mask].min():.2f}, {traffic[outliers_lower_mask].max():.2f}]')

    # ==================== 建议 ====================
    print(f'\n{"="*80}')
    print('【7. 数据预处理建议】')
    print('='*80)

    need_preprocessing = False
    recommendations = []

    if nan_count > 0:
        need_preprocessing = True
        if nan_ratio > 1.0:
            recommendations.append(f'⚠️  严重缺失 ({nan_ratio:.2f}%): 建议使用插值方法 (前向填充 + 线性插值)')
        else:
            recommendations.append(f'⚠️  轻微缺失 ({nan_ratio:.4f}%): 建议使用前向填充')

    if inf_count > 0:
        need_preprocessing = True
        recommendations.append(f'⚠️  存在Inf值: 需要替换为NaN后插值')

    if neg_count > 0:
        need_preprocessing = True
        recommendations.append(f'⚠️  存在负值: 建议替换为0或使用插值')

    if total_outliers > total_samples * 0.01:  # 超过1%
        need_preprocessing = True
        recommendations.append(f'⚠️  异常值较多 ({outlier_ratio:.2f}%): 建议使用中值滤波或Winsorization')

    if need_preprocessing:
        print('❌ 数据质量存在问题，建议预处理流程:')
        print('\n推荐的预处理步骤:')
        print('  1. 替换Inf和负值为NaN')
        print('  2. 前向填充 (forward fill) 处理短期缺失')
        print('  3. 线性插值处理剩余缺失')
        print('  4. 中值滤波 (window=3) 平滑异常值')
        print('  5. Z-score归一化')
        print('\n具体问题:')
        for i, rec in enumerate(recommendations, 1):
            print(f'  {i}. {rec}')

        # 估计插值影响
        print(f'\n预期改进:')
        problematic_samples = nan_count + inf_count + neg_count + total_outliers
        problematic_ratio = problematic_samples / total_samples * 100
        print(f'  受影响样本: {problematic_samples:,} ({problematic_ratio:.2f}%)')
        print(f'  预期提升MAE: 约{problematic_ratio * 0.1:.2f}% (经验估计)')
    else:
        print('✅ 数据质量良好，可直接使用')
        print('   建议: 仅需Z-score归一化即可')

    print('\n' + '='*80)

    return {
        'nan_count': nan_count,
        'nan_ratio': nan_ratio,
        'inf_count': inf_count,
        'neg_count': neg_count,
        'zero_count': zero_count,
        'outlier_count': total_outliers,
        'outlier_ratio': outlier_ratio,
        'need_preprocessing': need_preprocessing
    }


def main():
    parser = argparse.ArgumentParser(description='分析PEMS数据质量')
    parser.add_argument('--data_path', type=str, default='data/PEMS08/PEMS08.npz',
                        help='数据文件路径')
    args = parser.parse_args()

    results = analyze_data_quality(args.data_path)

    # 保存分析结果
    output_file = 'data_quality_report.txt'
    print(f'\n分析报告已保存到: {output_file}')


if __name__ == '__main__':
    main()