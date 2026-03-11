#!/usr/bin/env python3
"""Export fixed train/val/test splits for multitask experiments."""

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mp_data_pipeline.config import DB_PATH
from mp_data_pipeline.ml.splits import (
    build_chemsys_ood_split,
    build_complexity_ood_split,
    build_iid_split,
    load_material_meta,
    save_split,
)


def update_manifest(manifest_path: Path, rows: list[tuple[str, str, str, Path, str]]) -> None:
    """Rewrite split manifest table with current rows."""
    lines = [
        "# Splits Manifest",
        "",
        "记录所有训练/验证/测试划分文件的路径、规则和版本。",
        "",
        "| 日期 | Split 名称 | 规则 | 文件路径 | 备注 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        d, name, rule, path, note = row
        rel = path.relative_to(PROJECT_ROOT)
        lines.append(f"| {d} | {name} | {rule} | `{rel}` | {note} |")
    manifest_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed data splits")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="ASE DB path")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits",
        help="Directory to store split JSON files",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for IID/grouped splits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_material_meta(args.db)

    iid = build_iid_split(meta, seed=args.seed)
    chemsys = build_chemsys_ood_split(meta, seed=args.seed)
    complexity = build_complexity_ood_split(meta)

    iid_path = args.out_dir / "split_iid_seed42.json"
    chemsys_path = args.out_dir / "split_chemsys_ood_seed42.json"
    complexity_path = args.out_dir / "split_complexity_ood.json"

    save_split(iid, iid_path, split_name="iid", seed=args.seed)
    save_split(chemsys, chemsys_path, split_name="chemsys_ood", seed=args.seed)
    save_split(complexity, complexity_path, split_name="complexity_ood", seed=None)

    today = date.today().isoformat()
    manifest_rows = [
        (today, "iid", "random 80/10/10 by mp_id", iid_path, "seed=42"),
        (
            today,
            "chemsys_ood",
            "group split by derived chemsys, ~70/15/15",
            chemsys_path,
            "seed=42",
        ),
        (
            today,
            "complexity_ood",
            "train<=4 elements, val=5, test>=6",
            complexity_path,
            "deterministic",
        ),
    ]
    update_manifest(PROJECT_ROOT / "reports" / "plans" / "splits_manifest.md", manifest_rows)

    for name, split in [("iid", iid), ("chemsys_ood", chemsys), ("complexity_ood", complexity)]:
        print(name, {k: len(v) for k, v in split.items()})


if __name__ == "__main__":
    main()
