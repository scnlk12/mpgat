"""深入分析零值节点的特征"""
import numpy as np
import sys

def analyze_zero_patterns(npz_file, node_id=128):
    """分析特定节点的零值模式"""
    print(f"\n{'='*60}")
    print(f"分析节点 {node_id} 的零值模式")
    print(f"{'='*60}")

    data = np.load(npz_file)
    traffic_data = data['data'][:, :, 0]

    node_data = traffic_data[:, node_id]

    print(f"\n节点{node_id}统计:")
    print(f"总时间步: {len(node_data)}")
    print(f"零值数量: {(node_data == 0).sum()}")
    print(f"零值比例: {(node_data == 0).sum() / len(node_data) * 100:.2f}%")
    print(f"非零值数量: {(node_data != 0).sum()}")
    print(f"非零值比例: {(node_data != 0).sum() / len(node_data) * 100:.2f}%")

    # 非零值的统计
    non_zero = node_data[node_data != 0]
    if len(non_zero) > 0:
        print(f"\n非零值统计:")
        print(f"均值: {non_zero.mean():.2f}")
        print(f"标准差: {non_zero.std():.2f}")
        print(f"最小值: {non_zero.min():.2f}")
        print(f"最大值: {non_zero.max():.2f}")
        print(f"中位数: {np.median(non_zero):.2f}")

    # 分析零值的分布模式
    print(f"\n=== 零值分布模式 ===")
    is_zero = (node_data == 0)

    # 找连续零值段
    zero_segments = []
    in_segment = False
    segment_start = 0

    for i, val in enumerate(is_zero):
        if val and not in_segment:
            in_segment = True
            segment_start = i
        elif not val and in_segment:
            in_segment = False
            zero_segments.append((segment_start, i - 1, i - segment_start))

    if in_segment:
        zero_segments.append((segment_start, len(is_zero) - 1, len(is_zero) - segment_start))

    if zero_segments:
        print(f"连续零值段数量: {len(zero_segments)}")
        print(f"最长连续零值: {max(seg[2] for seg in zero_segments)} 个时间步")
        print(f"平均连续长度: {np.mean([seg[2] for seg in zero_segments]):.2f}")

        # 显示最长的5个零值段
        print(f"\n最长的5个连续零值段:")
        sorted_segments = sorted(zero_segments, key=lambda x: x[2], reverse=True)[:5]
        for start, end, length in sorted_segments:
            # 每天288个时间步（5分钟间隔）
            days = length / 288
            hours = (length % 288) / 12
            print(f"  位置 {start}-{end}: {length}步 ({days:.1f}天 {hours:.1f}小时)")

    # 分析零值的时间规律
    print(f"\n=== 零值时间规律 ===")
    zero_indices = np.where(is_zero)[0]

    # 日内分布（288个时间槽）
    time_of_day = zero_indices % 288
    if len(time_of_day) > 0:
        print(f"一天内零值最多的时段:")
        time_hist, _ = np.histogram(time_of_day, bins=24)
        peak_hour = np.argmax(time_hist)
        print(f"  峰值小时: {peak_hour}:00-{peak_hour+1}:00 ({time_hist[peak_hour]}次)")

    # 周内分布
    day_of_week = (zero_indices // 288) % 7
    if len(day_of_week) > 0:
        print(f"一周内零值分布:")
        day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        for i in range(7):
            count = (day_of_week == i).sum()
            print(f"  {day_names[i]}: {count}次")

def compare_normal_vs_problematic_nodes(npz_file):
    """比较正常节点和问题节点"""
    print(f"\n{'='*60}")
    print(f"比较正常节点vs问题节点")
    print(f"{'='*60}")

    data = np.load(npz_file)
    traffic_data = data['data'][:, :, 0]

    # 找零值比例最高和最低的节点
    zero_ratios = [(node_data == 0).sum() / len(node_data)
                   for node_data in traffic_data.T]

    worst_node = np.argmax(zero_ratios)
    best_node = np.argmin(zero_ratios)

    print(f"\n问题最严重节点 (节点{worst_node}):")
    print(f"  零值比例: {zero_ratios[worst_node] * 100:.2f}%")
    print(f"  非零值均值: {traffic_data[:, worst_node][traffic_data[:, worst_node] != 0].mean():.2f}")

    print(f"\n正常节点 (节点{best_node}):")
    print(f"  零值比例: {zero_ratios[best_node] * 100:.2f}%")
    print(f"  非零值均值: {traffic_data[:, best_node][traffic_data[:, best_node] != 0].mean():.2f}")

    # 统计零值比例的分布
    print(f"\n=== 所有节点零值比例分布 ===")
    zero_ratios_pct = np.array(zero_ratios) * 100
    print(f"中位数: {np.median(zero_ratios_pct):.2f}%")
    print(f"平均值: {np.mean(zero_ratios_pct):.2f}%")
    print(f"最大值: {np.max(zero_ratios_pct):.2f}%")

    # 统计有问题的节点数
    problematic_counts = {
        '>1%': (zero_ratios_pct > 1).sum(),
        '>5%': (zero_ratios_pct > 5).sum(),
        '>10%': (zero_ratios_pct > 10).sum(),
        '>20%': (zero_ratios_pct > 20).sum(),
    }

    print(f"\n零值比例阈值统计:")
    for threshold, count in problematic_counts.items():
        print(f"  {threshold}: {count}个节点 ({count/len(zero_ratios)*100:.1f}%)")

if __name__ == '__main__':
    # 分析PEMS04节点128
    print("\n" + "="*60)
    print("PEMS04 数据集分析")
    print("="*60)
    analyze_zero_patterns('data/PEMS04/PEMS04.npz', node_id=128)
    compare_normal_vs_problematic_nodes('data/PEMS04/PEMS04.npz')

    # 分析PEMS08节点155
    print("\n\n" + "="*60)
    print("PEMS08 数据集分析")
    print("="*60)
    data08 = np.load('data/PEMS08/PEMS08.npz')
    if data08['data'].shape[1] > 155:
        analyze_zero_patterns('data/PEMS08/PEMS08.npz', node_id=155)
    compare_normal_vs_problematic_nodes('data/PEMS08/PEMS08.npz')