#!/usr/bin/env python3
"""Replace machine-specific path prefixes in downloaded checkpoint configs.

This script is intended for checkpoint directories copied from the training
server to an inference/eval server. It edits config-like files in place and
creates timestamped backups by default.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


DEFAULT_OLD_PREFIX = "/inspire/ssd/project/advanced-machine-learning/yanjunchi-24040/youjunqi/zyc"
DEFAULT_NEW_PREFIX = "/mnt/nas/gezuhao/zhouyuchen"
DEFAULT_CONFIG_NAMES = {
    "config.yaml",
    "config.full.yaml",
    "config.yml",
    "config.json",
    "dataset_statistics.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace old absolute path prefix with local prefix in checkpoint config files."
    )
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Checkpoint run directory, e.g. /mnt/nas/.../playground/checkpoint/ki_depth_gated_layer_6_9",
    )
    parser.add_argument(
        "--old-prefix",
        default=DEFAULT_OLD_PREFIX,
        help=f"Path prefix used on the training server. Default: {DEFAULT_OLD_PREFIX}",
    )
    parser.add_argument(
        "--new-prefix",
        default=DEFAULT_NEW_PREFIX,
        help=f"Path prefix used on this inference server. Default: {DEFAULT_NEW_PREFIX}",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also scan nested directories. By default only files directly under checkpoint_dir are edited.",
    )
    parser.add_argument(
        "--extra-file",
        action="append",
        default=[],
        help="Additional config file path to edit. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files and replacement counts without modifying them.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak_<timestamp> backups before modifying files.",
    )
    return parser.parse_args()


def iter_config_files(checkpoint_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        candidates = [p for p in checkpoint_dir.rglob("*") if p.is_file() and p.name in DEFAULT_CONFIG_NAMES]
    else:
        candidates = [checkpoint_dir / name for name in DEFAULT_CONFIG_NAMES]
        candidates = [p for p in candidates if p.is_file()]
    return sorted(set(candidates))


def replace_in_file(path: Path, old_prefix: str, new_prefix: str, *, dry_run: bool, backup_suffix: str | None) -> int:
    data = path.read_text(encoding="utf-8")
    count = data.count(old_prefix)
    if count == 0:
        return 0
    if dry_run:
        return count

    if backup_suffix is not None:
        backup_path = path.with_name(path.name + backup_suffix)
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
    path.write_text(data.replace(old_prefix, new_prefix), encoding="utf-8")
    return count


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"checkpoint_dir is not a directory: {checkpoint_dir}")
    if not args.old_prefix:
        raise SystemExit("--old-prefix must not be empty")

    files = iter_config_files(checkpoint_dir, args.recursive)
    files.extend(Path(p).expanduser().resolve() for p in args.extra_file)
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"No config files found under {checkpoint_dir}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_suffix = None if args.no_backup else f".bak_{stamp}"
    total = 0

    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"old_prefix:     {args.old_prefix}")
    print(f"new_prefix:     {args.new_prefix}")
    print(f"dry_run:        {args.dry_run}")
    print(f"backup_suffix:  {backup_suffix or '<disabled>'}")
    print()

    for path in files:
        if not path.is_file():
            print(f"SKIP missing file: {path}")
            continue
        count = replace_in_file(
            path,
            args.old_prefix,
            args.new_prefix,
            dry_run=args.dry_run,
            backup_suffix=backup_suffix,
        )
        total += count
        print(f"{'WOULD_EDIT' if args.dry_run and count else 'EDITED' if count else 'UNCHANGED'} {count:4d} {path}")

    print()
    print(f"total_replacements: {total}")
    if args.dry_run:
        print("Dry run only. Re-run without --dry-run to modify files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
