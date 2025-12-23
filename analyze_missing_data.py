"""分析数据集中的缺失值"""
import numpy as np
import sys

def analyze_dataset(npz_file):
    print(f"\n{'='*60}")
    print(f"分析数据集: {npz_file}")
    print(f"{'='*60}")

    data = np.load(npz_file)
    traffic_data = data['data'][:, :, 0]

    print(f'\n数据形状: {traffic_data.shape}')
    print(f'数据类型: {traffic_data.dtype}')

    print('\n=== 缺失值统计 ===')
    nan_count = np.isnan(traffic_data).sum()
    inf_count = np.isinf(traffic_data).sum()
    zero_count = (traffic_data == 0).sum()
    total_size = traffic_data.size

    print(f'NaN数量: {nan_count} ({nan_count/total_size*100:.4f}%)')
    print(f'Inf数量: {inf_count} ({inf_count/total_size*100:.4f}%)')
    print(f'零值数量: {zero_count} ({zero_count/total_size*100:.4f}%)')
    print(f'总数据点: {total_size}')

    print('\n=== 数据范围 ===')
    print(f'最小值: {np.nanmin(traffic_data):.4f}')
    print(f'最大值: {np.nanmax(traffic_data):.4f}')
    print(f'均值: {np.nanmean(traffic_data):.4f}')
    print(f'标准差: {np.nanstd(traffic_data):.4f}')
    print(f'中位数: {np.nanmedian(traffic_data):.4f}')

    # 分析每个节点的缺失值
    print('\n=== 节点级缺失值分析 ===')
    num_nodes = traffic_data.shape[1]
    nodes_with_missing = 0
    max_missing_node = -1
    max_missing_pct = 0

    for i in range(num_nodes):
        node_data = traffic_data[:, i]
        nan_count = np.isnan(node_data).sum()
        zero_count = (node_data == 0).sum()
        total = len(node_data)

        if nan_count > 0 or zero_count > total * 0.01:  # 超过1%的零值
            nodes_with_missing += 1
            if i < 10:  # 只打印前10个
                print(f'节点{i}: NaN={nan_count}({nan_count/total*100:.2f}%), '
                      f'Zero={zero_count}({zero_count/total*100:.2f}%)')

        missing_pct = (nan_count + zero_count) / total * 100
        if missing_pct > max_missing_pct:
            max_missing_pct = missing_pct
            max_missing_node = i

    print(f'\n有缺失/异常值的节点数: {nodes_with_missing}/{num_nodes}')
    print(f'最严重节点: 节点{max_missing_node}, 缺失率={max_missing_pct:.2f}%')

    # 分析连续缺失
    print('\n=== 时间序列连续性分析 ===')
    for i in range(min(3, num_nodes)):
        node_data = traffic_data[:, i]
        # 找到最长的连续NaN序列
        is_nan = np.isnan(node_data)
        if is_nan.any():
            # 找连续True的最长序列
            consecutive = np.diff(np.where(np.concatenate(([is_nan[0]],
                                           is_nan[:-1] != is_nan[1:],
                                           [True])))[0])[::2]
            if len(consecutive) > 0:
                max_consecutive = consecutive.max()
                print(f'节点{i}: 最长连续NaN={max_consecutive}个时间步')

if __name__ == '__main__':
    datasets = [
        'data/PEMS04/PEMS04.npz',
        'data/PEMS08/PEMS08.npz',
    ]

    for dataset in datasets:
        try:
            analyze_dataset(dataset)
        except FileNotFoundError:
            print(f"文件未找到: {dataset}")
        except Exception as e:
            print(f"分析 {dataset} 时出错: {e}")