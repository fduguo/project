#!/usr/bin/env python3
"""Summarize the longest successful LIBERO rollout steps from eval logs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


EPISODE_END_RE = re.compile(
    r"\[EPISODE_END\]\s+"
    r"task_id=(?P<task_id>\d+),\s+"
    r"episode=(?P<episode>\d+)/(?P<episodes_per_task>\d+),\s+"
    r"steps_used=(?P<steps_used>\d+),\s+"
    r"max_steps=(?P<max_steps>\d+),\s+"
    r"warmup_steps=(?P<warmup_steps>\d+),\s+"
    r"total_env_steps=(?P<total_env_steps>\d+),\s+"
    r"success=(?P<success>YES|NO)"
)
TASK_SUITE_RE = re.compile(r"Task suite:\s*(?P<suite>[A-Za-z0-9_+-]+)")
ARGS_TASK_SUITE_RE = re.compile(r'"task_suite_name"\s*:\s*"(?P<suite>[^"]+)"')
TASK_RE = re.compile(r"\[TASK\]\s+(?P<task_num>\d+)/(?P<total_tasks>\d+):\s*(?P<description>.*)")


def find_log_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]

    globber = path.rglob if recursive else path.glob
    files = sorted(
        file
        for file in globber("libero_*.log")
        if file.is_file() and not file.name.startswith("policy_server_")
    )
    if files:
        return files

    return sorted(
        file
        for file in globber("*.log")
        if file.is_file() and not file.name.startswith("policy_server_")
    )


def parse_log_file(path: Path) -> dict[str, Any]:
    suite: str | None = None
    task_descriptions: dict[int, str] = {}
    total_episodes = 0
    success_count = 0
    failure_count = 0
    max_success: dict[str, Any] | None = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if suite is None:
                suite_match = TASK_SUITE_RE.search(line) or ARGS_TASK_SUITE_RE.search(line)
                if suite_match:
                    suite = suite_match.group("suite")

            task_match = TASK_RE.search(line)
            if task_match:
                task_id = int(task_match.group("task_num")) - 1
                task_descriptions[task_id] = task_match.group("description").strip()
                continue

            episode_match = EPISODE_END_RE.search(line)
            if not episode_match:
                continue

            values = episode_match.groupdict()
            record = {
                "task_id": int(values["task_id"]),
                "episode": int(values["episode"]),
                "episodes_per_task": int(values["episodes_per_task"]),
                "steps_used": int(values["steps_used"]),
                "max_steps": int(values["max_steps"]),
                "warmup_steps": int(values["warmup_steps"]),
                "total_env_steps": int(values["total_env_steps"]),
                "success": values["success"] == "YES",
                "line": line_no,
                "log_file": str(path),
            }
            description = task_descriptions.get(record["task_id"])
            if description:
                record["task_description"] = description

            total_episodes += 1
            if record["success"]:
                success_count += 1
                if max_success is None or record["steps_used"] > max_success["steps_used"]:
                    max_success = record
            else:
                failure_count += 1

    if suite is None:
        suite = path.stem

    return {
        "suite": suite,
        "log_file": str(path),
        "total_episodes": total_episodes,
        "success_count": success_count,
        "failure_count": failure_count,
        "max_success": max_success,
    }


def merge_by_suite(file_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suites: dict[str, dict[str, Any]] = {}

    for summary in file_summaries:
        suite = summary["suite"]
        merged = suites.setdefault(
            suite,
            {
                "suite": suite,
                "log_files": [],
                "total_episodes": 0,
                "success_count": 0,
                "failure_count": 0,
                "max_success": None,
            },
        )
        merged["log_files"].append(summary["log_file"])
        merged["total_episodes"] += summary["total_episodes"]
        merged["success_count"] += summary["success_count"]
        merged["failure_count"] += summary["failure_count"]

        candidate = summary["max_success"]
        current = merged["max_success"]
        if candidate is not None and (current is None or candidate["steps_used"] > current["steps_used"]):
            merged["max_success"] = candidate

    return sorted(suites.values(), key=lambda item: item["suite"])


def build_payload(input_path: Path, log_files: list[Path], suites: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input": str(input_path),
        "log_files": [str(path) for path in log_files],
        "suites": suites,
    }


def format_source(path_text: str | None, root: Path) -> str:
    if not path_text:
        return "-"
    path = Path(path_text)
    try:
        return str(path.relative_to(root)) if root.is_dir() else path.name
    except ValueError:
        return path.name


def format_table(suites: list[dict[str, Any]], root: Path) -> str:
    headers = [
        "suite",
        "max_success_steps",
        "successes/episodes",
        "success_rate",
        "task_id",
        "episode",
        "source",
    ]
    rows: list[list[str]] = []

    for summary in suites:
        total = summary["total_episodes"]
        successes = summary["success_count"]
        rate = successes / total if total else 0.0
        record = summary["max_success"]

        if record is None:
            max_steps = task_id = episode = source = "-"
        else:
            max_steps = str(record["steps_used"])
            task_id = str(record["task_id"])
            episode = f'{record["episode"]}/{record["episodes_per_task"]}'
            source = f'{format_source(record["log_file"], root)}:{record["line"]}'

        rows.append(
            [
                summary["suite"],
                max_steps,
                f"{successes}/{total}",
                f"{rate:.4f}",
                task_id,
                episode,
                source,
            ]
        )

    if not rows:
        return "No LIBERO eval episodes found."

    widths = [
        max(len(headers[col_idx]), *(len(row[col_idx]) for row in rows))
        for col_idx in range(len(headers))
    ]
    lines = [
        "  ".join(header.ljust(widths[col_idx]) for col_idx, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[col_idx]) for col_idx, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse LIBERO eval logs and report the maximum steps_used among "
            "successful episodes for each task suite."
        )
    )
    parser.add_argument("path", help="Log directory or a single .log file.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search log directories recursively.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        help="Optional output file. Defaults to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.path).expanduser()
    if not input_path.exists():
        print(f"error: path does not exist: {input_path}", file=sys.stderr)
        return 1

    log_files = find_log_files(input_path, recursive=args.recursive)
    if not log_files:
        print(f"error: no .log files found under: {input_path}", file=sys.stderr)
        return 1

    file_summaries = [parse_log_file(path) for path in log_files]
    suites = merge_by_suite(file_summaries)
    payload = build_payload(input_path, log_files, suites)

    if args.format == "json":
        output = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        output = format_table(suites, input_path)

    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
