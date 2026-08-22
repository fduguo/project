import json
import os
import argparse
import glob
import csv
import re


def extract_success_rate_from_log(log_path):
    """从 .log 文件中提取 success_rate，返回浮点数，若未找到则返回 None"""
    if not os.path.exists(log_path):
        print(f"警告：日志文件不存在 - {log_path}")
        return None
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = r'success_rate\s*[:=]?\s*([0-9.]+)'
    matches = re.findall(pattern, content)
    if matches:
        return float(matches[-1])
    else:
        print(f"警告：在 {log_path} 中未找到 success_rate")
        return None


def merge_results(log_dir):
    # 1. 读取每个 suite 的 JSON 结果文件
    pattern = os.path.join(log_dir, "libero_*.json")
    task_files = glob.glob(pattern)

    if not task_files:
        print(f"错误：在目录 {log_dir} 中没有找到 libero_*.json 文件")
        return

    suite_results = {}
    for file_path in task_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        suite_name = data.get("task_suite", os.path.basename(file_path).replace(".json", ""))
        if suite_name == "libero_10":
            display_name = "Long"
        else:
            display_name = suite_name.replace("libero_", "").capitalize()
        suite_results[display_name] = {
            "suite": suite_name,
            "total_tasks": data["total_tasks"],
            "total_episodes": data["total_episodes"],
            "total_successes": data["total_successes"],
            "success_rate": data["success_rate"],
        }
        print(f"  {display_name}: {data['total_successes']}/{data['total_episodes']} = {data['success_rate']:.4f}")

    # 2. 汇总所有 suite 的总体成功率
    overall_episodes = sum(r["total_episodes"] for r in suite_results.values())
    overall_successes = sum(r["total_successes"] for r in suite_results.values())
    overall_rate = overall_successes / overall_episodes if overall_episodes > 0 else 0.0

    # 3. 同时从 .log 文件中提取成功率进行交叉验证
    log_mapping = {
        "Spatial": "libero_spatial.log",
        "Object": "libero_object.log",
        "Goal": "libero_goal.log",
        "Long": "libero_10.log"
    }
    suite_rates_from_log = {}
    for suite_name, log_filename in log_mapping.items():
        log_path = os.path.join(log_dir, log_filename)
        rate = extract_success_rate_from_log(log_path)
        if rate is not None:
            suite_rates_from_log[suite_name] = rate
        else:
            suite_rates_from_log[suite_name] = 0.0

    # 4. 构建 CSV
    # 列顺序: Spatial, Object, Goal, Long, avg
    suite_order = ["Spatial", "Object", "Goal", "Long"]
    suite_rates_json = {
        s: suite_results[s]["success_rate"] if s in suite_results else 0.0
        for s in suite_order
    }
    avg_json = sum(suite_rates_json.values()) / len(suite_rates_json) if suite_rates_json else 0.0

    headers = ["Spatial", "Object", "Goal", "Long", "avg", "Spatial(log)", "Object(log)", "Goal(log)", "Long(log)", "avg(log)"]
    row = (
        [f"{suite_rates_json[s]:.4f}" for s in suite_order]
        + [f"{avg_json:.4f}"]
        + [f"{suite_rates_from_log[s]:.4f}" for s in suite_order]
        + [f"{sum(suite_rates_from_log.values()) / len(suite_rates_from_log):.4f}"]
    )

    # 5. 保存 JSON 汇总
    summary = {
        "suites": {s: suite_results[s] for s in suite_results},
        "overall_total_episodes": overall_episodes,
        "overall_total_successes": overall_successes,
        "overall_success_rate": overall_rate,
    }
    output_path = os.path.join(log_dir, "overall_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON 汇总已保存至：{output_path}")
    print(f"总 episodes: {overall_episodes}, 总成功: {overall_successes}, 总成功率: {overall_rate:.4f}")

    # 6. 保存 CSV
    csv_path = os.path.join(log_dir, "results_summary.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(headers)
        writer.writerow(row)
    print(f"CSV 已保存至：{csv_path}")

    # 7. 打印表格
    print("\n表格内容（可直接复制）：")
    print("\t".join(headers))
    print("\t".join(row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 LIBERO 评测结果并生成 CSV 表格")
    parser.add_argument("log_dir", help="包含 libero_*.json 和 libero_*.log 文件的目录路径")
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        print(f"错误：目录不存在 - {args.log_dir}")
        exit(1)

    merge_results(args.log_dir)
