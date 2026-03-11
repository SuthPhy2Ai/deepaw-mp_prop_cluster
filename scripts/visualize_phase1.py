#!/usr/bin/env python3
"""Generate visualization charts for Phase 1 comparison."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Paths
exp01_metrics = Path("artifacts/runs/20260303_211013/metrics/best_summary.json")
exp02_metrics = Path("artifacts/runs/20260304_005923/metrics/best_summary.json")
output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Load metrics
with open(exp01_metrics) as f:
    exp01 = json.load(f)
with open(exp02_metrics) as f:
    exp02 = json.load(f)

# Extract validation metrics
val01 = exp01["val_metrics"]
val02 = exp02["val_metrics"]

# Define tasks and their metrics
classification_tasks = [
    ("is_metal", "AUROC", "is_metal_auroc"),
    ("is_stable", "AUROC", "is_stable_auroc"),
]

regression_tasks = [
    ("energy_per_atom", "MAE (eV)", "energy_per_atom_mae"),
    ("formation_energy", "MAE (eV)", "formation_energy_per_atom_mae"),
    ("energy_above_hull", "MAE (eV)", "energy_above_hull_mae"),
    ("band_gap", "MAE (eV)", "band_gap_mae"),
    ("cbm", "MAE (eV)", "cbm_mae"),
    ("vbm", "MAE (eV)", "vbm_mae"),
    ("efermi", "MAE (eV)", "efermi_mae"),
    ("volume", "MAE (Å³)", "volume_mae"),
    ("density", "MAE (g/cm³)", "density_mae"),
]

# Figure 1: Classification tasks comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Phase 1: Classification Tasks Comparison", fontsize=16, fontweight='bold')

for idx, (task, metric, key) in enumerate(classification_tasks):
    ax = axes[idx]

    comp_val = val01[key]
    graph_val = val02[key]

    x = np.arange(2)
    values = [comp_val, graph_val]
    colors = ['#2ecc71' if comp_val > graph_val else '#e74c3c',
              '#e74c3c' if comp_val > graph_val else '#2ecc71']

    bars = ax.bar(x, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['Composition', 'Graph'], fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"{task.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
    ax.set_ylim([0.7, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add winner indicator
    winner = "Composition" if comp_val > graph_val else "Graph"
    diff = abs(comp_val - graph_val) / comp_val * 100
    ax.text(0.5, 0.95, f"Winner: {winner}\nΔ: {diff:.1f}%",
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "phase1_classification_comparison.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'phase1_classification_comparison.png'}")
plt.close()

# Figure 2: Regression tasks comparison (grouped)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle("Phase 1: Regression Tasks Comparison (Lower is Better)",
             fontsize=16, fontweight='bold')
axes = axes.flatten()

for idx, (task, metric, key) in enumerate(regression_tasks):
    ax = axes[idx]

    comp_val = val01[key]
    graph_val = val02[key]

    x = np.arange(2)
    values = [comp_val, graph_val]
    colors = ['#2ecc71' if comp_val < graph_val else '#e74c3c',
              '#e74c3c' if comp_val < graph_val else '#2ecc71']

    bars = ax.bar(x, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['Comp', 'Graph'], fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(f"{task.replace('_', ' ').title()}", fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}' if val < 10 else f'{val:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add winner indicator
    winner = "Comp" if comp_val < graph_val else "Graph"
    diff = abs(comp_val - graph_val) / comp_val * 100
    ax.text(0.5, 0.95, f"Winner: {winner}\nΔ: {diff:.1f}%",
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=8)

# Hide unused subplots
for idx in range(len(regression_tasks), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(output_dir / "phase1_regression_comparison.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'phase1_regression_comparison.png'}")
plt.close()

# Figure 3: Win/Loss summary
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Phase 1: Overall Win/Loss Summary", fontsize=16, fontweight='bold')

categories = ['Classification\n(2 tasks)', 'Energy\n(3 tasks)',
              'Electronic\n(4 tasks)', 'Structural\n(2 tasks)',
              'Overall\n(11 tasks)']
comp_wins = [2, 2, 4, 1, 9]
graph_wins = [0, 1, 0, 1, 2]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, comp_wins, width, label='Composition Wins',
               color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, graph_wins, width, label='Graph Wins',
               color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Number of Tasks Won', fontsize=12)
ax.set_title('Task-by-Task Comparison', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add win rate annotation
ax.text(0.98, 0.98, f"Composition Win Rate: 82% (9/11)\nGraph Win Rate: 18% (2/11)",
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
        fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "phase1_winloss_summary.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'phase1_winloss_summary.png'}")
plt.close()

# Figure 4: Relative performance (percentage change)
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle("Phase 1: Relative Performance (Graph vs Composition)",
             fontsize=16, fontweight='bold')

all_tasks = classification_tasks + regression_tasks
task_names = []
percent_changes = []

for task, metric, key in all_tasks:
    comp_val = val01[key]
    graph_val = val02[key]

    # For classification (higher is better), positive change is good
    # For regression (lower is better), negative change is good
    if "auroc" in key or "acc" in key:
        pct_change = (graph_val - comp_val) / comp_val * 100
    else:
        pct_change = (graph_val - comp_val) / comp_val * 100

    task_names.append(task.replace('_', ' ').title())
    percent_changes.append(pct_change)

y_pos = np.arange(len(task_names))
colors = ['#2ecc71' if pc < 0 else '#e74c3c' for pc in percent_changes]

bars = ax.barh(y_pos, percent_changes, color=colors, alpha=0.7,
               edgecolor='black', linewidth=1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(task_names, fontsize=10)
ax.set_xlabel('Percentage Change (%)', fontsize=12)
ax.set_title('Negative = Graph Better, Positive = Composition Better', fontsize=12)
ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, pc) in enumerate(zip(bars, percent_changes)):
    width = bar.get_width()
    label_x = width + (5 if width > 0 else -5)
    ax.text(label_x, bar.get_y() + bar.get_height()/2.,
            f'{pc:+.1f}%',
            ha='left' if width > 0 else 'right', va='center',
            fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "phase1_relative_performance.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'phase1_relative_performance.png'}")
plt.close()

print("\n✅ All Phase 1 visualization charts generated successfully!")
print(f"📁 Output directory: {output_dir.absolute()}")
