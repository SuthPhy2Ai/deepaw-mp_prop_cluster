#!/usr/bin/env python3
"""Generate a boss-readable one-page executive summary in Markdown and PDF."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
SOURCE_CSV = REPORTS_DIR / "full_method_comparison_stagea_stageb_branches.csv"
MASTER_REPORT = REPORTS_DIR / "MASTER_COMPLETE_REPORT_STAGEA_STAGEB_STAGEC_ZERO.md"

OUT_MD = REPORTS_DIR / "EXEC_SUMMARY_ONE_PAGE_STAGEA_STAGEB_STAGEC_ZERO.md"
OUT_HTML = REPORTS_DIR / "EXEC_SUMMARY_ONE_PAGE_STAGEA_STAGEB_STAGEC_ZERO.html"
OUT_PDF = REPORTS_DIR / "EXEC_SUMMARY_ONE_PAGE_STAGEA_STAGEB_STAGEC_ZERO.pdf"

METHODS = [
    "Stage A",
    "Stage A PyG",
    "v1",
    "v2",
    "v3",
    "v4",
    "zero",
    "Stage C h1",
    "Stage C h2",
    "Stage C hybrid",
]
STAGE_A_SYSTEM_LABEL = "Stage A体系（Stage A + Stage A PyG）"
TASK_ORDER = [
    "energy_per_atom",
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
    "cbm",
    "vbm",
    "efermi",
    "is_metal",
    "is_stable",
    "volume",
    "density",
    "bulk_modulus_vrh",
    "shear_modulus_vrh",
    "homogeneous_poisson",
    "universal_anisotropy",
]
FAMILIES = {
    "热力学": ["energy_per_atom", "formation_energy_per_atom", "energy_above_hull"],
    "电子结构": ["band_gap", "cbm", "vbm", "efermi", "is_metal"],
    "稳定性": ["is_stable"],
    "结构": ["volume", "density"],
    "弹性": ["bulk_modulus_vrh", "shear_modulus_vrh", "homogeneous_poisson", "universal_anisotropy"],
}


def fmt_num(value: float | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def best_rows(df: pd.DataFrame, split: str) -> pd.DataFrame:
    cur = df[(df["split"] == split) & df["method"].isin(METHODS)].copy()
    cur = cur[cur["score_value"].notna()]
    cur = cur.sort_values(["task", "score_value"], ascending=[True, False])
    return cur.groupby("task", as_index=False).first()


def win_counts(best_df: pd.DataFrame) -> dict[str, int]:
    counts = {m: 0 for m in METHODS}
    for method, count in best_df["method"].value_counts().items():
        counts[method] = int(count)
    return counts


def build_route_table() -> list[tuple[str, str, str]]:
    return [
        ("优先推进", "zero", "15/15 已覆盖，热力学/稳定性/结构整体最强。"),
        ("保留主线", "Stage A", "原始多任务主线，cbm/efermi/test band_gap 仍强。"),
        ("保留参考", "Stage A PyG", "band_gap/vbm/is_metal 有局部提升，但未改写全局格局。"),
        ("保留补位", "v2 / v4", "稀疏弹性任务仍需任务特化路线。"),
        ("观察", "Stage C hybrid", "只在 val universal_anisotropy 给出局部最优信号。"),
        ("暂停", "v1 / v3 / Stage C h1/h2", "当前没有形成主线竞争力。"),
    ]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def build_outputs() -> tuple[str, str]:
    df = pd.read_csv(SOURCE_CSV)
    val_best = best_rows(df, "val")
    test_best = best_rows(df, "test")
    val_wins = win_counts(val_best)
    test_wins = win_counts(test_best)

    stage_a_family_val = val_wins["Stage A"] + val_wins["Stage A PyG"]
    stage_a_family_test = test_wins["Stage A"] + test_wins["Stage A PyG"]

    family_rows = []
    for family_name, tasks in FAMILIES.items():
        val_slice = val_best[val_best["task"].isin(tasks)]
        test_slice = test_best[test_best["task"].isin(tasks)]
        val_leaders = ", ".join(f"{row.task}->{row.method}" for row in val_slice.itertuples())
        test_leaders = ", ".join(f"{row.task}->{row.method}" for row in test_slice.itertuples())
        family_rows.append([family_name, val_leaders, test_leaders])

    win_rows = [
        ["zero", str(val_wins["zero"]), str(test_wins["zero"]), "全局最强单方法"],
        [STAGE_A_SYSTEM_LABEL, str(stage_a_family_val), str(stage_a_family_test), "电子结构簇主导家族"],
        ["v2", str(val_wins["v2"]), str(test_wins["v2"]), "弹性共享路线补位"],
        ["v4", str(val_wins["v4"]), str(test_wins["v4"]), "弹性单任务补位"],
        ["Stage C hybrid", str(val_wins["Stage C hybrid"]), str(test_wins["Stage C hybrid"]), "局部信号"],
    ]

    hardest_rows = []
    hardest_val = val_best.sort_values("score_value", ascending=True).head(4)
    hardest_test = test_best.sort_values("score_value", ascending=True).head(4)
    for row in hardest_val.itertuples():
        hardest_rows.append(["VAL", row.task, row.method, fmt_num(row.score_value)])
    for row in hardest_test.itertuples():
        hardest_rows.append(["TEST", row.task, row.method, fmt_num(row.score_value)])

    route_rows = [[a, b, c] for a, b, c in build_route_table()]

    md_lines = [
        "# 老板看的一页执行摘要",
        "",
        "范围：`Stage A / Stage A PyG / Stage B v1-v4 / Stage C h1-h2-hybrid / zero`，统一口径基于任务级 `R2/ACC`，不以 `best_val_loss` 作为跨方法结论依据。",
        "",
        "## 一句话结论",
        "",
        f"- 当前最优决策不是押注单一路线，而是 `zero + Stage A体系 + v2/v4` 的组合：`zero` 负责热力学/稳定性/结构上限，`{STAGE_A_SYSTEM_LABEL}` 负责电子结构主线，`v2/v4` 负责稀疏弹性补位。",
        f"- 从胜场看，`zero` 为全局第一：`VAL={val_wins['zero']}`、`TEST={test_wins['zero']}`；`{STAGE_A_SYSTEM_LABEL}` 合计 `VAL={stage_a_family_val}`、`TEST={stage_a_family_test}`，稳居第二梯队但主导电子结构簇。",
        "",
        "## 决策板",
        "",
        markdown_table(["动作", "方法", "理由"], route_rows),
        "",
        "## 关键数字",
        "",
        markdown_table(["对象", "VAL 胜场", "TEST 胜场", "解读"], win_rows),
        "",
        "## 任务簇归属",
        "",
        markdown_table(["任务簇", "VAL 最优", "TEST 最优"], family_rows),
        "",
        "## 当前最难任务",
        "",
        markdown_table(["Split", "Task", "当前最好方法", "最好分数"], hardest_rows),
        "",
        "## 结论落地",
        "",
        "1. 主线继续押 `zero`，它是当前唯一同时在热力学、稳定性、结构上形成系统优势的方法。",
        f"2. 电子结构不应放弃 `{STAGE_A_SYSTEM_LABEL}`，它仍是 `band_gap/cbm/vbm/efermi/is_metal` 的最优来源。",
        "3. 弹性任务不要再追求单一统一头，继续保留 `v2/v4`，Stage C 仅保留 `hybrid` 做下一轮验证。",
        "",
        f"完整报告见：`{MASTER_REPORT.relative_to(PROJECT_ROOT)}`",
    ]
    md_text = "\n".join(md_lines) + "\n"

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>老板看的一页执行摘要</title>
  <style>
    @page {{
      size: A4 landscape;
      margin: 8mm;
    }}
    body {{
      font-family: "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", sans-serif;
      color: #172121;
      background: #f8f6f1;
      margin: 0;
      font-size: 9.2pt;
      line-height: 1.25;
    }}
    .page {{
      height: 100%;
      box-sizing: border-box;
      padding: 8mm 9mm 7mm 9mm;
      background:
        radial-gradient(circle at top right, rgba(40,122,106,.08), transparent 22%),
        radial-gradient(circle at bottom left, rgba(186,82,34,.08), transparent 24%),
        #f8f6f1;
    }}
    .header {{
      display: grid;
      grid-template-columns: 1.3fr .9fr;
      gap: 10px;
      align-items: start;
      margin-bottom: 8px;
    }}
    .title {{
      font-size: 22pt;
      font-weight: 800;
      letter-spacing: .3px;
      margin: 0 0 3px 0;
      color: #16302b;
    }}
    .subtitle {{
      margin: 0;
      color: #4a5856;
      font-size: 9pt;
    }}
    .hero {{
      background: linear-gradient(135deg, #173f35, #245b4f);
      color: white;
      border-radius: 10px;
      padding: 10px 12px;
    }}
    .hero p {{
      margin: 0;
      font-size: 10pt;
      line-height: 1.35;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.12fr .88fr .9fr;
      gap: 9px;
      align-items: start;
    }}
    .card {{
      background: rgba(255,255,255,.72);
      border: 1px solid rgba(22, 48, 43, .10);
      border-radius: 10px;
      padding: 8px 9px;
    }}
    h2 {{
      margin: 0 0 6px 0;
      font-size: 11pt;
      color: #173f35;
      border-bottom: 1px solid rgba(23, 63, 53, .12);
      padding-bottom: 3px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 8.1pt;
    }}
    th, td {{
      border-bottom: 1px solid rgba(23, 63, 53, .10);
      text-align: left;
      vertical-align: top;
      padding: 3px 4px;
    }}
    th {{
      color: #173f35;
      font-weight: 700;
      background: rgba(23, 63, 53, .04);
    }}
    .small {{
      font-size: 7.7pt;
      color: #5a6664;
    }}
    .badge {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 8pt;
      margin-right: 5px;
      margin-bottom: 4px;
    }}
    .b-red {{ background: #f5d7d2; color: #8f2d18; }}
    .b-green {{ background: #dbece4; color: #1d5a48; }}
    .b-gold {{ background: #f2e6c9; color: #7c5a15; }}
    .b-gray {{ background: #e8ecec; color: #516160; }}
    ul, ol {{
      margin: 5px 0 0 18px;
      padding: 0;
    }}
    li {{
      margin: 2px 0;
    }}
    .footer {{
      margin-top: 8px;
      font-size: 7.8pt;
      color: #546260;
      display: flex;
      justify-content: space-between;
    }}
    code {{
      font-family: "DejaVu Sans Mono", monospace;
      font-size: 7.9pt;
      background: rgba(23, 63, 53, .06);
      padding: 1px 3px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div>
        <h1 class="title">老板看的一页执行摘要</h1>
        <p class="subtitle">覆盖 Stage A / Stage A PyG / Stage B v1-v4 / Stage C h1-h2-hybrid / zero，统一按任务级 R2/ACC 横向比较。</p>
      </div>
      <div class="hero">
        <p><strong>一句话结论：</strong>当前不应押单一路线，而应采用 <strong>zero + Stage A体系 + v2/v4</strong> 的组合。<strong>zero</strong> 提供热力学/稳定性/结构上限，<strong>{html.escape(STAGE_A_SYSTEM_LABEL)}</strong> 负责电子结构，<strong>v2/v4</strong> 负责稀疏弹性补位。</p>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>1. 决策结论</h2>
        <div>
          <span class="badge b-red">优先推进: zero</span>
          <span class="badge b-green">保留主线: Stage A</span>
          <span class="badge b-gray">保留参考: Stage A PyG</span>
          <span class="badge b-gold">弹性补位: v2 / v4</span>
        </div>
        {html_table(["动作", "方法", "理由"], route_rows)}
        <p class="small">解释口径：<code>best_val_loss</code> 只用于训练过程参考，不用于跨方法主结论。</p>
      </div>

      <div class="card">
        <h2>2. 关键数字</h2>
        {html_table(["对象", "VAL 胜场", "TEST 胜场", "解读"], win_rows)}
        <h2 style="margin-top:7px;">3. 最难任务</h2>
        {html_table(["Split", "Task", "最好方法", "最好分数"], hardest_rows)}
      </div>

      <div class="card">
        <h2>4. 任务簇归属</h2>
        {html_table(["任务簇", "VAL 最优", "TEST 最优"], family_rows)}
        <h2 style="margin-top:7px;">5. 立即执行</h2>
        <ol>
          <li>把 zero 作为下一轮主增长点，继续围绕热力学/稳定性/结构做容量与数据配比优化。</li>
          <li>保留 {html.escape(STAGE_A_SYSTEM_LABEL)} 电子结构主线，不要把 band_gap/cbm/vbm/efermi/is_metal 迁出共享主干。</li>
          <li>弹性路线继续分治，保留 v2/v4；Stage C 只留 hybrid 做解冻 backbone 的下一轮验证。</li>
        </ol>
      </div>
    </div>

    <div class="footer">
      <div>完整报告：<code>{html.escape(str(MASTER_REPORT.relative_to(PROJECT_ROOT)))}</code></div>
      <div>生成时间：2026-03-10</div>
    </div>
  </div>
</body>
</html>
"""
    return md_text, html_text


def main() -> None:
    md_text, html_text = build_outputs()
    OUT_MD.write_text(md_text)
    OUT_HTML.write_text(html_text)
    print(f"saved_md={OUT_MD}")
    print(f"saved_html={OUT_HTML}")


if __name__ == "__main__":
    main()
