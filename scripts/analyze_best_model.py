#!/usr/bin/env python
"""Analyze best model performance with comprehensive visualizations and report."""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mp_data_pipeline.ml.dataset import AseGraphMultitaskDataset, collate_graph_samples
from mp_data_pipeline.ml.pyg_dataset import PyGMaterialsDataset, collate_pyg_batch
from mp_data_pipeline.ml.tasks import TASK_INDEX, CLASSIFICATION_TASKS
from mp_data_pipeline.models.multitask_model import MultitaskPropertyModel


def load_checkpoint(checkpoint_path):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return ckpt


def evaluate_model(model, loader, device, enabled_tasks):
    """Evaluate model and collect predictions."""
    model.eval()

    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            preds = model(batch)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch['targets'].cpu().numpy())
            all_masks.append(batch['masks'].cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    return preds, targets, masks


def compute_metrics(preds, targets, masks, task_name):
    """Compute metrics for a single task."""
    idx = TASK_INDEX[task_name]
    mask = masks[:, idx] > 0.5

    if not np.any(mask):
        return None

    y_pred = preds[mask, idx]
    y_true = targets[mask, idx]

    metrics = {'n_samples': int(mask.sum())}

    if task_name in CLASSIFICATION_TASKS:
        # Classification metrics
        probs = 1.0 / (1.0 + np.exp(-y_pred))
        y_pred_class = (probs >= 0.5).astype(int)
        y_true_class = (y_true >= 0.5).astype(int)

        metrics['accuracy'] = float((y_pred_class == y_true_class).mean())

        if len(np.unique(y_true_class)) >= 2:
            metrics['auroc'] = float(roc_auc_score(y_true_class, probs))
        else:
            metrics['auroc'] = float('nan')

        metrics['predictions'] = probs
        metrics['targets'] = y_true_class
    else:
        # Regression metrics
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        metrics['r2'] = float(r2_score(y_true, y_pred))
        metrics['predictions'] = y_pred
        metrics['targets'] = y_true

    return metrics


def create_visualization(results, enabled_tasks, output_path):
    """Create comprehensive visualization figure with train/val/test comparison."""
    # Separate regression and classification tasks
    reg_tasks = [t for t in enabled_tasks if t not in CLASSIFICATION_TASKS]
    cls_tasks = [t for t in enabled_tasks if t in CLASSIFICATION_TASKS]

    n_reg = len(reg_tasks)
    n_cls = len(cls_tasks)

    # Create figure: 3 columns (train/val/test) for each task
    n_cols = 3  # train, val, test
    n_rows = n_reg + n_cls

    fig = plt.figure(figsize=(18, 4 * n_rows))

    row_idx = 0

    # Plot regression tasks (scatter plots) - one row per task, 3 columns for splits
    for task in reg_tasks:
        for col_idx, split_name in enumerate(['train', 'val', 'test']):
            ax = plt.subplot(n_rows, n_cols, row_idx * n_cols + col_idx + 1)

            if split_name not in results:
                ax.text(0.5, 0.5, f'{split_name.upper()}\nNot Available',
                       ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            metrics = results[split_name]['metrics'][task]
            if metrics is None:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            y_true = metrics['targets']
            y_pred = metrics['predictions']

            # Scatter plot with smaller points for train set
            alpha = 0.1 if split_name == 'train' else 0.3
            s = 5 if split_name == 'train' else 10
            ax.scatter(y_true, y_pred, alpha=alpha, s=s, c='blue')

            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')

            ax.set_xlabel('True Value', fontsize=9)
            ax.set_ylabel('Predicted Value', fontsize=9)

            title = f'{task} ({split_name.upper()})\n'
            title += f'MAE={metrics["mae"]:.4f}, R²={metrics["r2"]:.3f}'
            ax.set_title(title, fontsize=10, fontweight='bold')

            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        row_idx += 1

    # Plot classification tasks (confusion matrices) - one row per task, 3 columns for splits
    for task in cls_tasks:
        for col_idx, split_name in enumerate(['train', 'val', 'test']):
            ax = plt.subplot(n_rows, n_cols, row_idx * n_cols + col_idx + 1)

            if split_name not in results:
                ax.text(0.5, 0.5, f'{split_name.upper()}\nNot Available',
                       ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            metrics = results[split_name]['metrics'][task]
            if metrics is None:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            y_true = metrics['targets']
            y_pred = (metrics['predictions'] >= 0.5).astype(int)

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       cbar=False)

            ax.set_xlabel('Predicted', fontsize=9)
            ax.set_ylabel('True', fontsize=9)

            title = f'{task} ({split_name.upper()})\n'
            title += f'Acc={metrics["accuracy"]:.4f}, AUROC={metrics["auroc"]:.4f}'
            ax.set_title(title, fontsize=10, fontweight='bold')

        row_idx += 1

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Visualization saved to {output_path}")


def generate_report(results, config, output_path):
    """Generate comprehensive analysis report."""
    report = []

    report.append("# Model Performance Analysis Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n" + "="*80 + "\n")

    # Model configuration
    report.append("## Model Configuration\n")
    report.append(f"- **Backbone**: {config['backbone']}")
    report.append(f"- **Hidden Dimension**: {config['hidden_dim']}")
    report.append(f"- **Layers**: {config['layers']}")
    report.append(f"- **Learning Rate**: {config['lr']}")
    report.append(f"- **Batch Size**: {config['batch_size']}")
    report.append(f"- **Training Epochs**: {config['epochs']}")
    report.append(f"- **Stage**: {config['stage']}")
    report.append(f"- **Enabled Tasks**: {len(config['enabled_tasks'])}")
    report.append("")

    # Overall performance
    report.append("## Overall Performance\n")
    report.append("| Split | Loss |")
    report.append("|-------|------|")
    for split in ['train', 'val', 'test']:
        if split in results:
            report.append(f"| {split.capitalize()} | {results[split]['loss']:.4f} |")
    report.append("")

    # Per-task performance
    report.append("## Per-Task Performance\n")

    # Regression tasks
    reg_tasks = [t for t in config['enabled_tasks'] if t not in CLASSIFICATION_TASKS]
    if reg_tasks:
        report.append("### Regression Tasks\n")
        report.append("| Task | Split | MAE | RMSE | R² | Samples |")
        report.append("|------|-------|-----|------|----|---------| ")

        for task in reg_tasks:
            for split in ['train', 'val', 'test']:
                if split not in results:
                    continue
                metrics = results[split]['metrics'].get(task)
                if metrics:
                    report.append(f"| {task} | {split} | {metrics['mae']:.4f} | "
                                f"{metrics['rmse']:.4f} | {metrics['r2']:.3f} | "
                                f"{metrics['n_samples']} |")
        report.append("")

    # Classification tasks
    cls_tasks = [t for t in config['enabled_tasks'] if t in CLASSIFICATION_TASKS]
    if cls_tasks:
        report.append("### Classification Tasks\n")
        report.append("| Task | Split | Accuracy | AUROC | Samples |")
        report.append("|------|-------|----------|-------|---------|")

        for task in cls_tasks:
            for split in ['train', 'val', 'test']:
                if split not in results:
                    continue
                metrics = results[split]['metrics'].get(task)
                if metrics:
                    auroc_str = f"{metrics['auroc']:.4f}" if not np.isnan(metrics['auroc']) else "N/A"
                    report.append(f"| {task} | {split} | {metrics['accuracy']:.4f} | "
                                f"{auroc_str} | {metrics['n_samples']} |")
        report.append("")

    # Key findings
    report.append("## Key Findings\n")

    # Best and worst tasks
    val_metrics = results['val']['metrics']

    reg_maes = [(t, m['mae']) for t, m in val_metrics.items()
                if m and t in reg_tasks]
    if reg_maes:
        reg_maes.sort(key=lambda x: x[1])
        report.append("### Best Regression Tasks (by MAE):")
        for task, mae in reg_maes[:3]:
            report.append(f"- **{task}**: MAE = {mae:.4f}")
        report.append("")

        report.append("### Worst Regression Tasks (by MAE):")
        for task, mae in reg_maes[-3:]:
            report.append(f"- **{task}**: MAE = {mae:.4f}")
        report.append("")

    cls_accs = [(t, m['accuracy']) for t, m in val_metrics.items()
                if m and t in cls_tasks]
    if cls_accs:
        cls_accs.sort(key=lambda x: x[1], reverse=True)
        report.append("### Classification Tasks Performance:")
        for task, acc in cls_accs:
            auroc = val_metrics[task]['auroc']
            auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "N/A"
            report.append(f"- **{task}**: Accuracy = {acc:.4f}, AUROC = {auroc_str}")
        report.append("")

    # Overfitting analysis
    report.append("## Overfitting Analysis\n")
    for task in config['enabled_tasks']:
        train_m = results['train']['metrics'].get(task)
        val_m = results['val']['metrics'].get(task)

        if not train_m or not val_m:
            continue

        if task in CLASSIFICATION_TASKS:
            train_metric = train_m['accuracy']
            val_metric = val_m['accuracy']
            metric_name = "Accuracy"
        else:
            train_metric = train_m['mae']
            val_metric = val_m['mae']
            metric_name = "MAE"

        gap = abs(train_metric - val_metric)
        if task in CLASSIFICATION_TASKS:
            if gap > 0.1:
                report.append(f"- **{task}**: Train {metric_name}={train_metric:.4f}, "
                            f"Val {metric_name}={val_metric:.4f} (Gap: {gap:.4f}) ⚠️")
        else:
            if val_metric > train_metric * 1.2:
                report.append(f"- **{task}**: Train {metric_name}={train_metric:.4f}, "
                            f"Val {metric_name}={val_metric:.4f} (Gap: {gap:.4f}) ⚠️")

    report.append("")

    # Write report
    output_path.write_text('\n'.join(report))
    print(f"✅ Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--db', type=str, default='data/db/mp_materials.db')
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--use-pyg', dest='use_pyg', action='store_true', help='Use PyG InMemoryDataset backend')
    parser.add_argument('--no-pyg', dest='use_pyg', action='store_false', help='Disable PyG backend')
    parser.set_defaults(use_pyg=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)

    # Load config from run directory
    run_dir = Path(args.checkpoint).parent.parent
    config_path = run_dir / 'config.json'
    with open(config_path) as f:
        config = json.load(f)

    enabled_tasks = config['enabled_tasks']

    # Load model
    model = MultitaskPropertyModel(
        backbone_name=config['backbone'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['layers'],
        cutoff=config.get('cutoff', 6.0),
        n_rbf=config.get('n_rbf', 64),
        use_angles=config.get('use_angles', False),
        use_edge_update=config.get('use_edge_update', False),
        head_variant=config.get('head_variant', 'grouped'),
    )
    model.load_state_dict(ckpt['model_state'])

    device = torch.device(args.device)
    model.to(device)

    print(f"Model loaded, best epoch: {ckpt['epoch']}, best val loss: {ckpt['best_metric']:.4f}")

    # Load data splits
    with open(args.split) as f:
        split_data = json.load(f)

    # Extract mp_ids dict
    splits = split_data['mp_ids']
    split_tag = Path(args.split).stem

    results = {}

    # Evaluate on each split
    for split_name in ['train', 'val', 'test']:
        if split_name not in splits:
            continue

        try:
            print(f"\nEvaluating on {split_name} set...")

            if args.use_pyg:
                pyg_root = Path("data/pyg_cache") / split_tag / split_name
                dataset = PyGMaterialsDataset(
                    root=str(pyg_root),
                    db_path=str(args.db),
                    mp_ids=splits[split_name],
                    cutoff=config.get('cutoff', 6.0),
                    max_neighbors=config.get('max_neighbors', 24),
                )
                collate_fn = collate_pyg_batch
            else:
                dataset = AseGraphMultitaskDataset(
                    db_path=args.db,
                    mp_ids=splits[split_name],
                    cutoff=config.get('cutoff', 6.0),
                    max_neighbors=config.get('max_neighbors', 24),
                )
                collate_fn = collate_graph_samples

            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
            )

            preds, targets, masks = evaluate_model(model, loader, device, enabled_tasks)

            # Compute metrics for each task
            task_metrics = {}
            for task in enabled_tasks:
                metrics = compute_metrics(preds, targets, masks, task)
                task_metrics[task] = metrics

            # Compute overall loss (simple average of task losses)
            losses = []
            for task, metrics in task_metrics.items():
                if metrics:
                    if task in CLASSIFICATION_TASKS:
                        losses.append(1.0 - metrics['accuracy'])
                    else:
                        losses.append(metrics['mae'])

            results[split_name] = {
                'loss': np.mean(losses) if losses else float('nan'),
                'metrics': task_metrics,
            }

            print(f"✅ {split_name} evaluation complete: loss={results[split_name]['loss']:.4f}")

        except Exception as e:
            print(f"❌ Error evaluating {split_name} set: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Check if we have results to visualize
    if not results:
        print("❌ No results to visualize. Evaluation failed for all splits.")
        return

    # Generate visualization
    viz_path = output_dir / 'performance_visualization.png'
    create_visualization(results, enabled_tasks, viz_path)

    # Generate report
    report_path = output_dir / 'analysis_report.md'
    generate_report(results, config, report_path)

    # Save raw results
    results_path = output_dir / 'results.json'
    # Convert numpy arrays to lists for JSON serialization
    results_json = {}
    for split, data in results.items():
        results_json[split] = {
            'loss': data['loss'],
            'metrics': {}
        }
        for task, metrics in data['metrics'].items():
            if metrics:
                metrics_json = {k: v.tolist() if isinstance(v, np.ndarray) else v
                               for k, v in metrics.items()}
                results_json[split]['metrics'][task] = metrics_json

    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n✅ Analysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
