# Splits Manifest

记录所有训练/验证/测试划分文件的路径、规则和版本。

| 日期 | Split 名称 | 规则 | 文件路径 | 备注 |
|---|---|---|---|---|
| 2026-03-03 | iid | random 80/10/10 by mp_id | `data/splits/split_iid_seed42.json` | seed=42 |
| 2026-03-03 | chemsys_ood | group split by derived chemsys, ~70/15/15 | `data/splits/split_chemsys_ood_seed42.json` | seed=42 |
| 2026-03-03 | complexity_ood | train<=4 elements, val=5, test>=6 | `data/splits/split_complexity_ood.json` | deterministic |
