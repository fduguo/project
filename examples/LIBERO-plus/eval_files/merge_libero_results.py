import json
import os
import argparse
import glob
import csv
import re
'''
python examples/LIBERO-plus/eval_files/merge_libero_results.py

'''


def extract_success_rate_from_log(log_path):
    """从 .log 文件中提取 success_rate，返回浮点数，若未找到则返回 None"""
    if not os.path.exists(log_path):
        print(f"警告：日志文件不存在 - {log_path}")
        return None
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 匹配类似 "eval/libero_long/success_rate 0.46844" 或 "success_rate = 0.46844" 等模式
    # 更通用的匹配：查找 success_rate 后跟数字
    pattern = r'success_rate\s*[:=]?\s*([0-9.]+)'
    matches = re.findall(pattern, content)
    if matches:
        # 取最后一个出现的值（通常汇总在末尾）
        return float(matches[-1])
    else:
        print(f"警告：在 {log_path} 中未找到 success_rate")
        return None

def merge_results(log_dir):
    # 1. 处理 JSON 文件，获得各条件（Camera, Robot, ...）和总体（Total）的成功率
    pattern = os.path.join(log_dir, "libero_*.json")
    task_files = glob.glob(pattern)

    if not task_files:
        print(f"错误：在目录 {log_dir} 中没有找到 libero_*.json 文件")
        return

    overall_results = {"overall": {"total_count": 0, "success_count": 0}}

    for file_path in task_files:
        with open(file_path, 'r') as f:
            results = json.load(f)
        for task_name, stats in results.items():
            overall_results["overall"]["total_count"] += stats["total_count"]
            overall_results["overall"]["success_count"] += stats["success_count"]
            if task_name not in overall_results:
                overall_results[task_name] = stats.copy()
            else:
                overall_results[task_name]["total_count"] += stats["total_count"]
                overall_results[task_name]["success_count"] += stats["success_count"]

    for category in overall_results:
        total = overall_results[category]["total_count"]
        success = overall_results[category]["success_count"]
        overall_results[category]["success_rate"] = success / total if total > 0 else 0.0

    # 保存 JSON 汇总文件
    output_path = os.path.join(log_dir, "overall_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2)
    print(f"JSON 汇总完成！结果已保存至：{output_path}")
    print(f"总任务数：{overall_results['overall']['total_count']}")
    print(f"总成功数：{overall_results['overall']['success_count']}")
    print(f"总成功率：{overall_results['overall']['success_rate']:.2%}")

    # 2. 从 .log 文件中提取四个任务套件的成功率
    log_mapping = {
        "Spatial": "libero_spatial.log",
        "Object": "libero_object.log",
        "Goal": "libero_goal.log",
        "Long": "libero_10.log"
    }
    suite_rates = {}
    for suite_name, log_filename in log_mapping.items():
        log_path = os.path.join(log_dir, log_filename)
        rate = extract_success_rate_from_log(log_path)
        if rate is not None:
            suite_rates[suite_name] = rate
        else:
            suite_rates[suite_name] = 0.0  # 缺失时填充 0
    # 计算平均值（只对成功提取的套件求平均，若全部缺失则 avg=0）
    valid_rates = [r for r in suite_rates.values() if r > 0 or True]  # 排除 None 情况上面已处理
    avg_rate = sum(suite_rates.values()) / len(suite_rates) if suite_rates else 0.0

    # 3. 从 overall_results 中提取条件列的成功率（按需要的顺序）
    column_mapping = [
        ("Camera Viewpoints", "Camera"),
        ("Robot Initial States", "Robot"),
        ("Language Instructions", "Language"),
        ("Light Conditions", "Light"),
        ("Background Textures", "Background"),
        ("Sensor Noise", "Noise"),
        ("Objects Layout", "Layout"),
        ("overall", "Total")
    ]
    condition_rates = []
    for json_key, _ in column_mapping:
        if json_key in overall_results:
            condition_rates.append(overall_results[json_key]["success_rate"])
        else:
            condition_rates.append(0.0)

    # 4. 构建完整的表头和数据行
    headers = ["Spatial", "Object", "Goal", "Long", "avg"] + [col for _, col in column_mapping]
    row = [
        suite_rates["Spatial"],
        suite_rates["Object"],
        suite_rates["Goal"],
        suite_rates["Long"],
        avg_rate
    ] + condition_rates

    # 5. 保存 CSV 文件（保留 4 位小数）
    csv_path = os.path.join(log_dir, "results_summary.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(headers)
        writer.writerow([f"{x:.4f}" for x in row])
    print(f"CSV 表格已保存至：{csv_path}")

    # 6. 同时打印表格，方便直接复制
    print("\n表格内容如下（可直接复制）：")
    print("\t".join(headers))
    print("\t".join([f"{x:.4f}" for x in row]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 LIBERO 评测结果 JSON 文件并生成完整 CSV 表格")
    parser.add_argument("log_dir", help="包含 libero_*.json 和 libero_*.log 文件的目录路径")
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        print(f"错误：目录不存在 - {args.log_dir}")
        exit(1)

    merge_results(args.log_dir)
