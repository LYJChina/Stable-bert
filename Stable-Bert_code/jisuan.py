import json
import statistics
import argparse
import os

def compute_mean_std(data):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'weighted_f1': [],
        'macro_f1': []
    }

    for run in data.values():
        for key in metrics.keys():
            if key in run:
                metrics[key].append(run[key])
            else:
                raise KeyError(f"指标 '{key}' 在运行数据中不存在。")
    stats = {}
    for key, values in metrics.items():
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[key] = {'Mean': mean, 'Std Dev': std_dev}

    return stats

def format_mean_std(stats):
    summary_metrics = []
    for metric, values in stats.items():
        mean = values['Mean'] * 100  
        std = values['Std Dev'] * 100
        summary = f"{mean:.2f}±{std:.2f}"
        summary_metrics.append(summary)


    summary_string = " ".join(summary_metrics)
    return summary_string

def add_statistics_to_json(original_json, output_json):
    with open(original_json, 'r') as f:
        data = json.load(f)

    stats = compute_mean_std(data)

    summary_string = format_mean_std(stats)

    data['summary'] = summary_string

    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"统计数据已添加并保存到 '{output_json}'")
    print(f"Summary: {summary_string}")

def main():
    parser = argparse.ArgumentParser(description="计算 JSON 文件中各指标的均值和标准差，并将其以 '均值±标准差' 格式添加回 JSON 文件。")
    parser.add_argument('--input_json', type=str, required=True, help='输入的原始 JSON 文件路径')
    parser.add_argument('--output_json', type=str, required=False, help='输出的修改后 JSON 文件路径 (默认在原文件名后加 "_with_summary")')

    args = parser.parse_args()

    input_json = args.input_json

    if not os.path.isfile(input_json):
        print(f"错误: 输入文件 '{input_json}' 不存在。")
        return

    if args.output_json:
        output_json = args.output_json
    else:
        base, ext = os.path.splitext(input_json)
        output_json = f"{base}_with_summary{ext}"

    add_statistics_to_json(input_json, output_json)

if __name__ == "__main__":
    main()
