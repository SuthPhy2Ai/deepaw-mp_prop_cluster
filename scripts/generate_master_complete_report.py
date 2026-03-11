#!/usr/bin/env python3
"""Generate the master complete report across all methods."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / 'reports'
FIG_DIR = REPORTS_DIR / 'figures' / 'master_complete_report'

TASK_ORDER = [
    'energy_per_atom',
    'formation_energy_per_atom',
    'energy_above_hull',
    'band_gap',
    'cbm',
    'vbm',
    'efermi',
    'is_metal',
    'is_stable',
    'volume',
    'density',
    'bulk_modulus_vrh',
    'shear_modulus_vrh',
    'homogeneous_poisson',
    'universal_anisotropy',
]
REGRESSION_TASKS = [
    'energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'cbm', 'vbm', 'efermi',
    'volume', 'density', 'bulk_modulus_vrh', 'shear_modulus_vrh', 'homogeneous_poisson', 'universal_anisotropy'
]
CLASSIFICATION_TASKS = ['is_metal', 'is_stable']
FAMILIES = {
    'thermo': ['energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull'],
    'electronic': ['band_gap', 'cbm', 'vbm', 'efermi', 'is_metal'],
    'stability': ['is_stable'],
    'structure': ['volume', 'density'],
    'elastic': ['bulk_modulus_vrh', 'shear_modulus_vrh', 'homogeneous_poisson', 'universal_anisotropy'],
}
METHODS = [
    ('Stage A', 'stage_a'),
    ('Stage A PyG', 'stage_a_pyg'),
    ('v1', 'v1'),
    ('v2', 'v2'),
    ('v3', 'v3'),
    ('v4', 'v4'),
    ('zero', 'zero'),
    ('Stage C h1', 'stage_c_h1'),
    ('Stage C h2', 'stage_c_h2'),
    ('Stage C hybrid', 'stage_c_hybrid'),
]
ALL_METHODS = [method for method, _ in METHODS]
PRIMARY_METHODS = ALL_METHODS
STAGE_A_FAMILY = ['Stage A', 'Stage A PyG']
STAGE_A_FAMILY_LABEL = 'Stage A体系（Stage A + Stage A PyG）'
REFERENCE_METHODS = ['Stage A PyG']
METHOD_COLORS = {
    'Stage A': '#1b4965',
    'v1': '#8d99ae',
    'v2': '#e09f3e',
    'v3': '#9d4edd',
    'v4': '#2a9d8f',
    'zero': '#d62828',
    'Stage C h1': '#577590',
    'Stage C h2': '#43aa8b',
    'Stage C hybrid': '#f77f00',
    'Stage A PyG': '#6c757d',
}
METHOD_META = {
    'Stage A': {
        'run': '20260305_210307', 'branch': 'stage_a', 'config': 'artifacts/runs/20260305_210307/config.json',
        'results': 'reports/gpt_eval_20260305_210307/results.json', 'head_variant': 'grouped', 'note': '8-task shared multitask baseline',
    },
    'Stage A PyG': {
        'run': '20260308_182946', 'branch': 'stage_a', 'config': 'artifacts/runs_stagea_pyg/20260308_182946/config.json',
        'results': 'reports/gpt_eval_20260308_182946/results.json', 'head_variant': 'grouped', 'note': 'PyG baseline reference',
    },
    'v1': {
        'run': '20260307_185342', 'branch': 'stage_b', 'config': 'artifacts/runs/20260307_185342/config.json',
        'results': 'reports/gpt_eval_20260307_185342/results.json', 'head_variant': 'grouped', 'note': 'Stage B multitask baseline',
    },
    'v2': {
        'run': '20260308_001437', 'branch': 'stage_b', 'config': 'artifacts/runs_stageb_v2/20260308_001437/config.json',
        'results': 'experiments/stage_b/phase3_enhancements/exp102_stageb_v2_balanced/analysis/results.json', 'head_variant': 'grouped', 'note': 'balanced Stage B multitask',
    },
    'v3': {
        'run': '20260308_070539', 'branch': 'stage_b', 'config': 'artifacts/runs_stageb_v3/20260308_070539/config.json',
        'results': 'reports/gpt_eval_20260308_070539/results.json', 'head_variant': 'grouped', 'note': 'core-guard Stage B multitask',
    },
    'v4': {
        'run': 'single-task family', 'branch': 'stage_b', 'config': 'configs/exp104_stageb_v4_single_task_heads.json',
        'results': 'reports/gpt_eval_v4_single_task/*/results.json', 'head_variant': 'grouped', 'note': 'shared pretrained backbone + per-task fine-tuning',
    },
    'zero': {
        'run': 'exp106_zero_single_task_family', 'branch': 'zero', 'config': 'experiments/zero_version/exp106_zero_single_task_family/metrics/zero_runs.csv',
        'results': 'reports/gpt_eval_zero_single_task/*/results.json', 'head_variant': 'per_task', 'note': 'fully isolated single-task family',
    },
    'Stage C h1': {
        'run': '20260310_003913', 'branch': 'stage_c', 'config': 'artifacts/runs_stagec_h1/20260310_003913/config.json',
        'results': 'reports/gpt_eval_20260310_003913_stagec_h1/results.json', 'head_variant': 'stagec_h1', 'note': 'electronic hierarchical head',
    },
    'Stage C h2': {
        'run': '20260310_005519', 'branch': 'stage_c', 'config': 'artifacts/runs_stagec_h2/20260310_005519/config.json',
        'results': 'reports/gpt_eval_20260310_005519_stagec_h2/results.json', 'head_variant': 'stagec_h2', 'note': 'elastic derived hierarchical head',
    },
    'Stage C hybrid': {
        'run': '20260310_011108', 'branch': 'stage_c', 'config': 'artifacts/runs_stagec_hybrid/20260310_011108/config.json',
        'results': 'reports/gpt_eval_20260310_011108_stagec_hybrid/results.json', 'head_variant': 'stagec_hybrid', 'note': 'electronic hierarchy + elastic derived hierarchy',
    },
}
LONG_CSV = REPORTS_DIR / 'full_method_comparison_stagea_stageb_branches.csv'
MASTER_MD = REPORTS_DIR / 'MASTER_COMPLETE_REPORT_STAGEA_STAGEB_STAGEC_ZERO.md'
MASTER_MATRIX = REPORTS_DIR / 'master_complete_report_method_task_matrix.csv'


def load_rows() -> list[dict]:
    with LONG_CSV.open() as f:
        return list(csv.DictReader(f))


def as_float(value: str | None) -> float | None:
    if value in (None, ''):
        return None
    return float(value)


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return 'N/A'
    return f'{value:.{digits}f}'


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = []
    out.append('| ' + ' | '.join(headers) + ' |')
    out.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in rows:
        out.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(out)


def row_lookup(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    return {(r['method'], r['split'], r['task']): r for r in rows}


def best_method(rows_by_key: dict, split: str, task: str, methods: list[str]) -> tuple[str, float | None]:
    best_name = 'N/A'
    best_score = None
    for method in methods:
        row = rows_by_key.get((method, split, task))
        score = as_float(row['score_value']) if row else None
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_name = method
            best_score = score
    return best_name, best_score


def build_matrix(rows: list[dict]) -> list[dict]:
    rows_by_key = row_lookup(rows)
    matrix = []
    for split in ['train', 'val', 'test']:
        for task in TASK_ORDER:
            first_row = None
            rec = {
                'split': split,
                'task': task,
                'task_type': 'classification' if task in CLASSIFICATION_TASKS else 'regression',
                'score_name': 'acc' if task in CLASSIFICATION_TASKS else 'r2',
            }
            for method, slug in METHODS:
                row = rows_by_key.get((method, split, task))
                if row:
                    first_row = row
                rec[f'{slug}_score'] = row['score_value'] if row else ''
                rec[f'{slug}_loss'] = row['task_loss'] if row else ''
            best_name, best_score = best_method(rows_by_key, split, task, PRIMARY_METHODS)
            rec['best_method'] = best_name
            rec['best_score'] = '' if best_score is None else best_score
            if first_row:
                rec['n_samples'] = first_row.get('n_samples', '')
            else:
                rec['n_samples'] = ''
            matrix.append(rec)
    return matrix


def write_matrix_csv(matrix: list[dict]) -> None:
    with MASTER_MATRIX.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix[0].keys()))
        writer.writeheader()
        writer.writerows(matrix)


def method_wins(rows_by_key: dict, split: str) -> dict[str, int]:
    wins = {m: 0 for m in PRIMARY_METHODS}
    for task in TASK_ORDER:
        best_name, _ = best_method(rows_by_key, split, task, PRIMARY_METHODS)
        if best_name in wins:
            wins[best_name] += 1
    return wins


def hardest_tasks(rows_by_key: dict, split: str) -> list[dict]:
    ranking = []
    for task in TASK_ORDER:
        best_name, best_score = best_method(rows_by_key, split, task, PRIMARY_METHODS)
        if best_score is None:
            continue
        ranking.append({
            'task': task,
            'task_type': 'classification' if task in CLASSIFICATION_TASKS else 'regression',
            'best_method': best_name,
            'best_score': best_score,
        })
    ranking.sort(key=lambda x: x['best_score'])
    return ranking


def compare_pair(rows_by_key: dict, split: str, left: str, right: str) -> tuple[list[dict], list[dict]]:
    left_wins, right_wins = [], []
    for task in TASK_ORDER:
        left_row = rows_by_key.get((left, split, task))
        right_row = rows_by_key.get((right, split, task))
        left_score = as_float(left_row['score_value']) if left_row else None
        right_score = as_float(right_row['score_value']) if right_row else None
        if left_score is None or right_score is None:
            continue
        rec = {'task': task, 'left': left_score, 'right': right_score, 'delta': left_score - right_score}
        if left_score > right_score:
            left_wins.append(rec)
        elif right_score > left_score:
            right_wins.append(rec)
    return left_wins, right_wins


def stagec_vs_v3(rows_by_key: dict) -> dict[str, dict[str, dict]]:
    out = {}
    for method in ['Stage C h1', 'Stage C h2', 'Stage C hybrid']:
        out[method] = {}
        for split in ['val', 'test']:
            improved, degraded = [], []
            for task in TASK_ORDER:
                base = rows_by_key.get(('v3', split, task))
                cur = rows_by_key.get((method, split, task))
                base_score = as_float(base['score_value']) if base else None
                cur_score = as_float(cur['score_value']) if cur else None
                if base_score is None or cur_score is None:
                    continue
                delta = cur_score - base_score
                rec = {'task': task, 'delta': delta, 'current': cur_score, 'baseline': base_score}
                if delta > 0:
                    improved.append(rec)
                elif delta < 0:
                    degraded.append(rec)
            improved.sort(key=lambda x: x['delta'], reverse=True)
            degraded.sort(key=lambda x: x['delta'])
            out[method][split] = {'improved': improved, 'degraded': degraded}
    return out


def family_leaders(rows_by_key: dict, split: str) -> dict[str, list[dict]]:
    out = {}
    for family, tasks in FAMILIES.items():
        entries = []
        for task in tasks:
            best_name, best_score = best_method(rows_by_key, split, task, PRIMARY_METHODS)
            entries.append({'task': task, 'best_method': best_name, 'best_score': best_score})
        out[family] = entries
    return out


def dominant_zone_summary(rows_by_key: dict, split: str) -> dict[str, list[str]]:
    summary = defaultdict(list)
    for task in TASK_ORDER:
        best_name, _ = best_method(rows_by_key, split, task, PRIMARY_METHODS)
        summary[best_name].append(task)
    return dict(summary)


def cfg_summary(method: str) -> dict:
    meta = METHOD_META[method]
    cfg_path = PROJECT_ROOT / meta['config']
    if method in {'v4', 'zero'}:
        if method == 'v4':
            return {'backbone': 'graph', 'hidden_dim': '256', 'layers': '6', 'batch_size': '64', 'epochs': '30', 'lr': '0.0002', 'weight_decay': '1e-05', 'use_pyg': 'True', 'freeze_backbone': 'True', 'enabled_tasks': '13'}
        return {'backbone': 'graph', 'hidden_dim': '128-320', 'layers': '4-7', 'batch_size': '32-64', 'epochs': '50-90', 'lr': '8e-5~2e-4', 'weight_decay': '1e-05', 'use_pyg': 'True', 'freeze_backbone': 'False', 'enabled_tasks': '15'}
    payload = json.loads(cfg_path.read_text())
    return {
        'backbone': payload.get('backbone') or payload.get('model', {}).get('backbone'),
        'hidden_dim': payload.get('hidden_dim') or payload.get('model', {}).get('hidden_dim'),
        'layers': payload.get('layers') or payload.get('model', {}).get('layers'),
        'batch_size': payload.get('batch_size') or payload.get('training', {}).get('batch_size'),
        'epochs': payload.get('epochs') or payload.get('training', {}).get('epochs'),
        'lr': payload.get('lr') or payload.get('training', {}).get('lr'),
        'weight_decay': payload.get('weight_decay') or payload.get('training', {}).get('weight_decay'),
        'use_pyg': payload.get('use_pyg') if 'use_pyg' in payload else payload.get('model', {}).get('use_pyg'),
        'freeze_backbone': payload.get('freeze_backbone') if 'freeze_backbone' in payload else payload.get('model', {}).get('freeze_backbone'),
        'enabled_tasks': len(payload.get('enabled_tasks', [])) if isinstance(payload.get('enabled_tasks'), list) else payload.get('enabled_tasks', '') or len(payload.get('task_set', [])),
    }


def generate_figures(rows_by_key: dict) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_paths = {}

    # Win counts
    val_wins = method_wins(rows_by_key, 'val')
    test_wins = method_wins(rows_by_key, 'test')
    methods = PRIMARY_METHODS
    x = np.arange(len(methods))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, [val_wins[m] for m in methods], width, label='VAL', color='#457b9d')
    ax.bar(x + width/2, [test_wins[m] for m in methods], width, label='TEST', color='#e76f51')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha='right')
    ax.set_ylabel('Win Tasks')
    ax.set_title('Method Win Counts by Split')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    p = FIG_DIR / 'method_win_counts_val_test.png'
    fig.savefig(p, dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig_paths['wins'] = str(p.relative_to(REPORTS_DIR))

    # Best method matrix
    method_to_idx = {m: i for i, m in enumerate(PRIMARY_METHODS)}
    mat = np.full((len(TASK_ORDER), 2), np.nan)
    ann = [['' for _ in range(2)] for _ in range(len(TASK_ORDER))]
    for j, split in enumerate(['val', 'test']):
        for i, task in enumerate(TASK_ORDER):
            best_name, _ = best_method(rows_by_key, split, task, PRIMARY_METHODS)
            mat[i, j] = method_to_idx[best_name]
            ann[i][j] = best_name.replace('Stage ', 'S').replace(' hybrid', ' hyb')
    fig, ax = plt.subplots(figsize=(7, 9))
    cmap = matplotlib.colormaps['tab10'].resampled(len(PRIMARY_METHODS))
    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=len(PRIMARY_METHODS)-1)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['VAL','TEST'])
    ax.set_yticks(np.arange(len(TASK_ORDER)))
    ax.set_yticklabels(TASK_ORDER)
    ax.set_title('Best Method by Task and Split')
    for i in range(len(TASK_ORDER)):
        for j in range(2):
            ax.text(j, i, ann[i][j], ha='center', va='center', fontsize=7, color='black')
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_ticks(np.arange(len(PRIMARY_METHODS)))
    cbar.set_ticklabels(PRIMARY_METHODS)
    fig.tight_layout()
    p = FIG_DIR / 'task_best_method_matrix_val_test.png'
    fig.savefig(p, dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig_paths['best_matrix'] = str(p.relative_to(REPORTS_DIR))

    # Hardest tasks ranking
    hardest_val = hardest_tasks(rows_by_key, 'val')[:8]
    hardest_test = hardest_tasks(rows_by_key, 'test')[:8]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)
    for ax, items, split in zip(axes, [hardest_val, hardest_test], ['VAL', 'TEST']):
        tasks = [f"{item['task']}\n({item['best_method']})" for item in items][::-1]
        scores = [item['best_score'] for item in items][::-1]
        colors = [METHOD_COLORS[item['best_method']] for item in items][::-1]
        ax.barh(tasks, scores, color=colors)
        ax.set_title(f'Hardest Tasks by Best {split} Score')
        ax.set_xlabel('Best Score')
        ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    p = FIG_DIR / 'hardest_tasks_val_test.png'
    fig.savefig(p, dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig_paths['hardest'] = str(p.relative_to(REPORTS_DIR))

    # Stage C vs v3 deltas (VAL)
    stagec = ['Stage C h1', 'Stage C h2', 'Stage C hybrid']
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(TASK_ORDER))
    width = 0.22
    for offset, method in zip([-width, 0, width], stagec):
        deltas = []
        for task in TASK_ORDER:
            cur = rows_by_key.get((method, 'val', task))
            base = rows_by_key.get(('v3', 'val', task))
            cur_score = as_float(cur['score_value']) if cur else np.nan
            base_score = as_float(base['score_value']) if base else np.nan
            deltas.append(np.nan if cur_score is None or base_score is None else cur_score - base_score)
        ax.bar(x + offset, deltas, width, label=method, color=METHOD_COLORS[method])
    ax.axhline(0.0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_ORDER, rotation=45, ha='right')
    ax.set_ylabel('Delta vs v3 (VAL score)')
    ax.set_title('Stage C vs v3 by Task (VAL)')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    p = FIG_DIR / 'stagec_vs_v3_delta_val.png'
    fig.savefig(p, dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig_paths['stagec_delta'] = str(p.relative_to(REPORTS_DIR))

    return fig_paths


def summary_recommendations() -> list[list[str]]:
    return [
        ['Stage A', '保留', '原始 grouped 多任务主线，仍在 cbm/efermi 等电子结构任务上保持强势。'],
        ['Stage A PyG', '保留参考', '同口径 PyG 版本在 band_gap/vbm/is_metal 上有局部增益，但还没有改写全局路线。'],
        ['zero', '优先推进', '热力学、稳定性、结构任务上限最强，且已覆盖 15/15 任务。'],
        ['v4', '保留', '稀疏弹性任务上仍有明显局部最优，适合作为单任务微调路线。'],
        ['v2', '保留', '共享多任务路线里对部分弹性任务仍最强，是 elastic fallback。'],
        ['Stage C hybrid', '观察推进', '只在 universal_anisotropy 上给出局部信号，需下一轮解冻 backbone 再验证。'],
        ['Stage C h1 / h2', '暂停独立扩展', '冻结 backbone 条件下未显示出足够独立价值。'],
        ['v1 / v3', '暂停', '已被 v4、zero、Stage A体系或 v2 全面压制，不宜继续主线投入。'],
    ]


def build_family_leader_table(rows_by_key: dict) -> list[list[str]]:
    out = []
    for split in ['val', 'test']:
        fam = family_leaders(rows_by_key, split)
        for family, entries in fam.items():
            leader_set = ', '.join([f"{e['task']}->{e['best_method']}" for e in entries])
            out.append([split.upper(), family, leader_set])
    return out


def build_run_index_rows(rows_by_key: dict) -> list[list[str]]:
    rows = []
    for method in [m for m,_ in METHODS]:
        meta = METHOD_META[method]
        cfg = cfg_summary(method)
        rows.append([
            method,
            meta['run'],
            meta['branch'],
            str(cfg['backbone']),
            str(meta['head_variant']),
            str(cfg['enabled_tasks']),
            str(cfg['use_pyg']),
            str(cfg['freeze_backbone']),
            meta['config'],
            meta['results'],
            meta['note'],
        ])
    return rows


def appendix_metric_table(rows_by_key: dict, split: str, task_type: str, metric_pairs: list[tuple[str, str]], methods: list[str]) -> str:
    tasks = REGRESSION_TASKS if task_type == 'regression' else CLASSIFICATION_TASKS
    header = ['Task']
    for method in methods:
        for _, label in metric_pairs:
            header.append(f'{method} {label}')
    table_rows = []
    for task in tasks:
        vals = [task]
        for method in methods:
            row = rows_by_key.get((method, split, task))
            for key, _ in metric_pairs:
                vals.append(fmt(as_float(row[key]) if row else None))
        table_rows.append(vals)
    return md_table(header, table_rows)


def build_report() -> None:
    rows = load_rows()
    rows_by_key = row_lookup(rows)
    matrix = build_matrix(rows)
    write_matrix_csv(matrix)
    fig_paths = generate_figures(rows_by_key)

    val_wins = method_wins(rows_by_key, 'val')
    test_wins = method_wins(rows_by_key, 'test')
    stage_a_family_val_wins = sum(val_wins[m] for m in STAGE_A_FAMILY)
    stage_a_family_test_wins = sum(test_wins[m] for m in STAGE_A_FAMILY)
    hardest_val = hardest_tasks(rows_by_key, 'val')
    hardest_test = hardest_tasks(rows_by_key, 'test')
    stagea_zero_val, zero_stagea_val = compare_pair(rows_by_key, 'val', 'Stage A', 'zero')
    stagea_zero_test, zero_stagea_test = compare_pair(rows_by_key, 'test', 'Stage A', 'zero')
    stagec_deltas = stagec_vs_v3(rows_by_key)
    dominant_val = dominant_zone_summary(rows_by_key, 'val')
    dominant_test = dominant_zone_summary(rows_by_key, 'test')

    lines = []
    lines.append('# 全方法完整总报告')
    lines.append('')
    lines.append('本报告是当前项目的完整总成品，统一覆盖 `Stage A / Stage A PyG / Stage B v1-v4 / Stage C h1-h2-hybrid / zero`。')
    lines.append('它区别于之前的局部报告：前半部分用于路线决策，后半部分用于研发复盘，附录保留全量逐任务明细和索引。')
    lines.append('')
    lines.append('## Part I 执行摘要')
    lines.append('')
    lines.append('### 1. 一页结论')
    lines.append('')
    lines.append(f'- 当前真正主导全局的方法家族仍然是 `{STAGE_A_FAMILY_LABEL}` 与 `zero`。')
    lines.append(f"- 胜场统计上，`zero` 目前在 `VAL={val_wins['zero']}`、`TEST={test_wins['zero']}` 个任务上取得全表最佳；`{STAGE_A_FAMILY_LABEL}` 合计在 `VAL={stage_a_family_val_wins}`、`TEST={stage_a_family_test_wins}` 个任务上取得全表最佳。")
    lines.append(f'- 电子结构高覆盖任务簇当前由 `{STAGE_A_FAMILY_LABEL}` 主导：`band_gap / cbm / vbm / efermi / is_metal` 在 `Stage A` 与 `Stage A PyG` 之间分工领先。')
    lines.append('- `zero` 主导热力学、稳定性和结构任务簇：`energy_per_atom / formation_energy_per_atom / energy_above_hull / is_stable / volume / density`。')
    lines.append(f'- 稀疏弹性任务仍然不是 `{STAGE_A_FAMILY_LABEL}` 或 `zero` 的绝对主场：`v2` 与 `v4` 依然保留局部最优。')
    lines.append('- `Stage C` 三组 head 结构没有形成整体突破；但 `Stage C hybrid` 在 `VAL universal_anisotropy` 上拿到当前最好结果，说明“头上加头”并非完全无效，只是还不够强。')
    lines.append(f'- 现阶段最合理的主线，不是统一押注单一方法，而是：`Stage A体系 + zero + v4/v2` 的分工组合。')
    lines.append('')
    lines.append('### 2. 路线判断')
    lines.append('')
    lines.append(md_table(['路线', '建议', '理由'], summary_recommendations()))
    lines.append('')
    lines.append('### 3. 总体胜场统计')
    lines.append('')
    lines.append(f"![Method wins]({fig_paths['wins']})")
    lines.append('')
    lines.append(md_table(
        ['Method', 'VAL Wins', 'TEST Wins'],
        [[m, str(val_wins[m]), str(test_wins[m])] for m in PRIMARY_METHODS],
    ))
    lines.append('')
    lines.append('注：胜场统计已纳入 `Stage A PyG`；路线判断仍把它视为同口径参考/补充分支，而非独立替代主线。')
    lines.append('')
    lines.append('### 4. 当前最难任务')
    lines.append('')
    lines.append(f"![Hardest tasks]({fig_paths['hardest']})")
    lines.append('')
    lines.append('按“所有方法中的最佳主分数”排序，当前最难的第一梯队任务是：')
    lines.append('')
    lines.append(md_table(
        ['Split', 'Task', 'Type', 'Best Score', 'Best Method'],
        [[
            'VAL', hardest_val[i]['task'], hardest_val[i]['task_type'], fmt(hardest_val[i]['best_score']), hardest_val[i]['best_method']
        ] for i in range(min(4, len(hardest_val)))] + [[
            'TEST', hardest_test[i]['task'], hardest_test[i]['task_type'], fmt(hardest_test[i]['best_score']), hardest_test[i]['best_method']
        ] for i in range(min(4, len(hardest_test)))]
    ))
    lines.append('')
    lines.append('核心判断：`homogeneous_poisson` 与 `universal_anisotropy` 仍然是全项目最难任务，`volume` 与 `shear_modulus_vrh` 处于第二梯队。')
    lines.append('')
    lines.append('## Part II 完整技术正文')
    lines.append('')
    lines.append('### 5. 方法谱系总览')
    lines.append('')
    lines.append('- `Stage A`: 高覆盖 8 任务共享多任务。')
    lines.append('- `Stage A PyG`: 与 Stage A 同口径的 PyG 基线，只作补充参考。')
    lines.append('- `v1/v2/v3`: Stage B 13 任务共享多任务变体。')
    lines.append('- `v4`: 共享预训练 backbone + 单任务 head 微调家族。')
    lines.append('- `zero`: 完全独立单任务家族，每个性质单独训练。')
    lines.append('- `Stage C h1`: 电子结构层级头。')
    lines.append('- `Stage C h2`: 弹性派生层级头。')
    lines.append('- `Stage C hybrid`: 电子层级 + 弹性派生层级的组合。')
    lines.append('')
    lines.append('### 6. 方法设置与实验口径')
    lines.append('')
    setting_rows = []
    for method, _ in METHODS:
        meta = METHOD_META[method]
        cfg = cfg_summary(method)
        setting_rows.append([
            method,
            meta['run'],
            meta['branch'],
            str(cfg['backbone']),
            meta['head_variant'],
            str(cfg['hidden_dim']),
            str(cfg['layers']),
            str(cfg['batch_size']),
            str(cfg['epochs']),
            str(cfg['lr']),
            str(cfg['weight_decay']),
            str(cfg['use_pyg']),
            str(cfg['enabled_tasks']),
            str(cfg['freeze_backbone']),
        ])
    lines.append(md_table(['Method', 'Run', 'Branch', 'Backbone', 'Head Variant', 'Hidden', 'Layers', 'Batch', 'Epochs', 'LR', 'WD', 'PyG', 'Enabled Tasks', 'Freeze Backbone'], setting_rows))
    lines.append('')
    lines.append('### 7. 核心结果总表')
    lines.append('')
    lines.append(f"![Best method matrix]({fig_paths['best_matrix']})")
    lines.append('')
    for split in ['val', 'test']:
        headers = ['Task', 'Type', 'Best'] + PRIMARY_METHODS
        data_rows = []
        for task in TASK_ORDER:
            best_name, _ = best_method(rows_by_key, split, task, PRIMARY_METHODS)
            vals = [fmt(as_float(rows_by_key[(m, split, task)]['score_value']) if (m, split, task) in rows_by_key else None) for m in PRIMARY_METHODS]
            data_rows.append([task, 'classification' if task in CLASSIFICATION_TASKS else 'regression', best_name] + vals)
        lines.append(f'#### {split.upper()} 主分数统一横比')
        lines.append('')
        lines.append(md_table(headers, data_rows))
        lines.append('')
    lines.append('### 8. 方法胜负图谱')
    lines.append('')
    lines.append(md_table(['Split', 'Family', 'Task -> Best Method'], build_family_leader_table(rows_by_key)))
    lines.append('')
    lines.append('按任务簇看，当前主导区很清楚：')
    lines.append('')
    lines.append(f"- `VAL`：电子簇由 `{STAGE_A_FAMILY_LABEL}` 主导，热力学/稳定性/结构由 `zero` 主导，弹性簇由 `v2` 与 `Stage C hybrid` 局部主导。")
    lines.append(f"- `TEST`：电子簇由 `{STAGE_A_FAMILY_LABEL}` 主导，热力学/稳定性/结构由 `zero` 主导，弹性簇由 `v2/v4/v1` 分裂主导。")
    lines.append('')
    lines.append('### 9. 任务分层结论')
    lines.append('')
    lines.append('- 热力学任务簇：`zero` 全面最强，说明单任务精细回归上限更高。')
    lines.append(f'- 电子结构任务簇：`{STAGE_A_FAMILY_LABEL}` 全面最强，说明高覆盖共享多任务加上 PyG 数据路径，对电子结构表征更有利。')
    lines.append('- 稳定性任务：`zero` 最强，`is_stable` 更像热力学边界判别，不适合硬塞进统一共享主线。')
    lines.append('- 结构任务：目前只有 `zero` 有完整结果，`volume` 依旧偏难，`density` 几乎已解决。')
    lines.append('- 弹性任务：没有单一王者，需要按任务单独选路线。')
    lines.append('')
    lines.append('### 10. Stage A vs zero 专题')
    lines.append('')
    lines.append('Stage A 与 zero 不是简单的谁全面碾压谁，而是分别统治不同任务簇。')
    lines.append('')
    lines.append('#### VAL: Stage A 胜过 zero')
    lines.append('')
    lines.append(md_table(['Task', 'Stage A', 'zero', 'Delta(StageA-zero)'], [[r['task'], fmt(r['left']), fmt(r['right']), f"{r['delta']:+.4f}"] for r in sorted(stagea_zero_val, key=lambda x: x['delta'], reverse=True)]))
    lines.append('')
    lines.append('#### VAL: zero 胜过 Stage A')
    lines.append('')
    lines.append(md_table(['Task', 'zero', 'Stage A', 'Delta(zero-StageA)'], [[r['task'], fmt(r['right']), fmt(r['left']), f"{(r['right']-r['left']):+.4f}"] for r in sorted(zero_stagea_val, key=lambda x: x['right'] - x['left'], reverse=True)]))
    lines.append('')
    lines.append('#### TEST: Stage A 胜过 zero')
    lines.append('')
    lines.append(md_table(['Task', 'Stage A', 'zero', 'Delta(StageA-zero)'], [[r['task'], fmt(r['left']), fmt(r['right']), f"{r['delta']:+.4f}"] for r in sorted(stagea_zero_test, key=lambda x: x['delta'], reverse=True)]))
    lines.append('')
    lines.append('#### TEST: zero 胜过 Stage A')
    lines.append('')
    lines.append(md_table(['Task', 'zero', 'Stage A', 'Delta(zero-StageA)'], [[r['task'], fmt(r['right']), fmt(r['left']), f"{(r['right']-r['left']):+.4f}"] for r in sorted(zero_stagea_test, key=lambda x: x['right'] - x['left'], reverse=True)]))
    lines.append('')
    lines.append('物理上，`Stage A` 胜出的任务高度集中在电子结构簇，这和现有物理分析一致；`zero` 胜出的任务则高度集中于热力学与稳定性簇。')
    lines.append('')
    lines.append('### 11. 稀疏弹性任务专题')
    lines.append('')
    elastic_rows = []
    for split in ['val', 'test']:
        for task in FAMILIES['elastic']:
            best_name, best_score = best_method(rows_by_key, split, task, PRIMARY_METHODS)
            elastic_rows.append([split.upper(), task, best_name, fmt(best_score)])
    lines.append(md_table(['Split', 'Task', 'Best Method', 'Best Score'], elastic_rows))
    lines.append('')
    lines.append('- `bulk_modulus_vrh`：当前最优仍是 `v2`。')
    lines.append('- `shear_modulus_vrh`：`VAL` 最优是 `v2`，`TEST` 最优是 `v1`。')
    lines.append('- `homogeneous_poisson`：`VAL` 最优是 `v2`，`TEST` 最优是 `v4`。')
    lines.append('- `universal_anisotropy`：`VAL` 最优已被 `Stage C hybrid` 触达，但 `TEST` 仍由 `v4` 保持最优。')
    lines.append('- 结论：弹性派生量仍然最依赖任务特化与结构设计，统一共享主线还没有解决这类任务。')
    lines.append('')
    lines.append('### 12. 电子结构任务簇专题')
    lines.append('')
    electronic_table = []
    for split in ['val', 'test']:
        for task in FAMILIES['electronic']:
            best_name, best_score = best_method(rows_by_key, split, task, PRIMARY_METHODS)
            electronic_table.append([split.upper(), task, best_name, fmt(best_score)])
    lines.append(md_table(['Split', 'Task', 'Best Method', 'Best Score'], electronic_table))
    lines.append('')
    lines.append(f'电子簇在 `VAL/TEST` 上都由 `{STAGE_A_FAMILY_LABEL}` 主导，但领导权在 `Stage A` 与 `Stage A PyG` 之间分摊。这支持“电子结构流形适合高覆盖共享多任务，且 PyG 数据路径能带来局部增益”的判断。')
    lines.append('更完整的物理解释见 [PHYSICAL_ANALYSIS_STAGEA_VS_ZERO_TASK_RELATIONS.md](PHYSICAL_ANALYSIS_STAGEA_VS_ZERO_TASK_RELATIONS.md)。')
    lines.append('')
    lines.append('### 13. Stage C 新架构专题')
    lines.append('')
    lines.append(f"![Stage C delta]({fig_paths['stagec_delta']})")
    lines.append('')
    lines.append('#### 13.1 设计意图')
    lines.append('')
    lines.append('- `Stage C h1`：电子结构层级头，测试 `group head -> task heads` 是否优于直接并列输出。')
    lines.append('- `Stage C h2`：弹性派生层级头，测试 `base elastic head -> derived-property heads` 是否更适合泊松比和各向异性。')
    lines.append('- `Stage C hybrid`：把 h1 和 h2 合在同一共享 backbone 中。')
    lines.append('')
    lines.append('#### 13.2 相对 v3 的同口径结果')
    lines.append('')
    stagec_rows = []
    for method in ['Stage C h1', 'Stage C h2', 'Stage C hybrid']:
        for split in ['val', 'test']:
            improved = len(stagec_deltas[method][split]['improved'])
            degraded = len(stagec_deltas[method][split]['degraded'])
            top_task = stagec_deltas[method][split]['improved'][0]['task'] if improved else 'N/A'
            top_delta = stagec_deltas[method][split]['improved'][0]['delta'] if improved else None
            worst_task = stagec_deltas[method][split]['degraded'][0]['task'] if degraded else 'N/A'
            worst_delta = stagec_deltas[method][split]['degraded'][0]['delta'] if degraded else None
            stagec_rows.append([method, split.upper(), str(improved), str(degraded), top_task, fmt(top_delta), worst_task, fmt(worst_delta)])
    lines.append(md_table(['Method', 'Split', 'Improved vs v3', 'Degraded vs v3', 'Best Improved Task', 'Best Delta', 'Worst Task', 'Worst Delta'], stagec_rows))
    lines.append('')
    lines.append('#### 13.3 最终判断')
    lines.append('')
    lines.append('- `Stage C h1/h2/hybrid` 相对 `v3` 在多数任务上分数有改善，但改善幅度大多不足以跨过 `Stage A / zero / v2 / v4` 这些当前强者。')
    lines.append('- `Stage C hybrid` 在 `VAL universal_anisotropy` 上给出了全表最佳，说明层级 head 对弹性派生量确实有信号。')
    lines.append('- 但在当前“冻结 backbone + 35 epoch”的设定下，Stage C 还不足以成为主线。')
    lines.append('- 最合理的下一步不是全面扩张 Stage C，而是只保留 `hybrid` 做一次允许 backbone 继续适配的 v2 版本。')
    lines.append('')
    lines.append('## Part III 附录')
    lines.append('')
    lines.append('### Appendix A. TRAIN/VAL/TEST 逐任务主表')
    lines.append('')
    appendix_methods = PRIMARY_METHODS
    for split in ['train', 'val', 'test']:
        lines.append(f'#### {split.upper()} Regression: task loss + R2')
        lines.append('')
        lines.append(appendix_metric_table(rows_by_key, split, 'regression', [('task_loss', 'loss'), ('r2', 'R2')], appendix_methods))
        lines.append('')
        lines.append(f'#### {split.upper()} Classification: task loss + ACC')
        lines.append('')
        lines.append(appendix_metric_table(rows_by_key, split, 'classification', [('task_loss', 'loss'), ('acc', 'ACC')], appendix_methods))
        lines.append('')
    lines.append('### Appendix B. 回归附加指标与分类 AUROC')
    lines.append('')
    for split in ['train', 'val', 'test']:
        lines.append(f'#### {split.upper()} Regression: MAE + RMSE')
        lines.append('')
        lines.append(appendix_metric_table(rows_by_key, split, 'regression', [('mae', 'MAE'), ('rmse', 'RMSE')], appendix_methods))
        lines.append('')
        lines.append(f'#### {split.upper()} Classification: AUROC')
        lines.append('')
        lines.append(appendix_metric_table(rows_by_key, split, 'classification', [('auroc', 'AUROC')], appendix_methods))
        lines.append('')
    lines.append('### Appendix C. 方法运行索引')
    lines.append('')
    lines.append(md_table(['Method', 'Run', 'Branch', 'Backbone', 'Head Variant', 'Enabled Tasks', 'PyG', 'Freeze Backbone', 'Config Path', 'Results Path', 'Note'], build_run_index_rows(rows_by_key)))
    lines.append('')
    lines.append('### Appendix D. 来源文件与关联专题')
    lines.append('')
    lines.append('- 主底表: `reports/full_method_comparison_stagea_stageb_branches.csv`')
    lines.append('- 机器可读矩阵: `reports/master_complete_report_method_task_matrix.csv`')
    lines.append('- 现有全方法主对比: [FULL_METHOD_COMPARISON_STAGEA_STAGEB_BRANCHES.md](FULL_METHOD_COMPARISON_STAGEA_STAGEB_BRANCHES.md)')
    lines.append('- Stage A vs zero 物理专题: [PHYSICAL_ANALYSIS_STAGEA_VS_ZERO_TASK_RELATIONS.md](PHYSICAL_ANALYSIS_STAGEA_VS_ZERO_TASK_RELATIONS.md)')
    lines.append('- v4 一页总结: [V4_EXEC_SUMMARY_ONE_PAGER.md](V4_EXEC_SUMMARY_ONE_PAGER.md)')
    lines.append('- 图表目录: `reports/figures/master_complete_report/`')
    lines.append('')
    MASTER_MD.write_text('\n'.join(lines) + '\n')


def main() -> None:
    build_report()
    print(f'saved_md={MASTER_MD}')
    print(f'saved_csv={MASTER_MATRIX}')
    print(f'saved_fig_dir={FIG_DIR}')


if __name__ == '__main__':
    main()
