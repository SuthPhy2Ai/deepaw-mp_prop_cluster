#!/usr/bin/env python3
"""持续监控训练进度"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_process_info(pid):
    """获取进程信息"""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid,etime,%cpu,%mem,cmd"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            return lines[1]
        return None
    except subprocess.CalledProcessError:
        return None


def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
        return {
            "utilization": int(gpu_util),
            "memory_used": int(mem_used),
            "memory_total": int(mem_total)
        }
    except subprocess.CalledProcessError:
        return None


def get_latest_run():
    """获取最新的运行目录"""
    runs_dir = PROJECT_ROOT / "artifacts" / "runs"
    if not runs_dir.exists():
        return None
    runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    return runs[-1] if runs else None


def get_training_progress(run_dir):
    """获取训练进度"""
    if not run_dir:
        return None

    # 检查config
    config_file = run_dir / "config.json"
    if not config_file.exists():
        return {"stage": "initializing", "message": "配置文件未创建"}

    # 检查metrics
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists():
        return {"stage": "loading", "message": "数据加载中"}

    # 检查epoch文件
    epoch_files = sorted(metrics_dir.glob("epoch_*.json"))
    if not epoch_files:
        return {"stage": "loading", "message": "数据加载中"}

    # 读取最新epoch
    latest_epoch_file = epoch_files[-1]
    with open(latest_epoch_file) as f:
        epoch_data = json.load(f)

    epoch_num = int(latest_epoch_file.stem.split('_')[1])

    # 读取config获取总epoch数
    with open(config_file) as f:
        config = json.load(f)
    total_epochs = config.get("epochs", 50)

    return {
        "stage": "training",
        "current_epoch": epoch_num,
        "total_epochs": total_epochs,
        "train_loss": epoch_data.get("train_loss"),
        "val_loss": epoch_data.get("val_loss")
    }


def main():
    """主函数"""
    print("持续监控启动...")
    print("=" * 60)

    # 读取主控进程PID
    automation_pid = None
    try:
        result = subprocess.run(
            ["pgrep", "-f", "phase1_automation.py"],
            capture_output=True,
            text=True,
            check=True
        )
        automation_pid = int(result.stdout.strip().split()[0])
        print(f"检测到主控进程: PID {automation_pid}")
    except (subprocess.CalledProcessError, ValueError, IndexError):
        print("未检测到主控进程")

    check_count = 0
    last_run_dir = None

    while True:
        check_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n[检查 #{check_count}] {timestamp}")
        print("-" * 60)

        # 检查主控进程
        if automation_pid:
            proc_info = get_process_info(automation_pid)
            if proc_info:
                print(f"主控进程: 运行中")
            else:
                print(f"主控进程: 已停止")
                automation_pid = None

        # 检查训练进程
        try:
            result = subprocess.run(
                ["pgrep", "-f", "train_multitask.py"],
                capture_output=True,
                text=True,
                check=True
            )
            train_pids = result.stdout.strip().split()
            print(f"训练进程: {len(train_pids)} 个运行中")
            for pid in train_pids:
                proc_info = get_process_info(pid)
                if proc_info:
                    print(f"  PID {pid}: {proc_info}")
        except subprocess.CalledProcessError:
            print("训练进程: 无")

        # 检查GPU
        gpu_info = get_gpu_info()
        if gpu_info:
            print(f"GPU: {gpu_info['utilization']}% 使用率, "
                  f"{gpu_info['memory_used']}/{gpu_info['memory_total']} MB 显存")

        # 检查最新run
        latest_run = get_latest_run()
        if latest_run:
            if latest_run != last_run_dir:
                print(f"新运行目录: {latest_run.name}")
                last_run_dir = latest_run

            progress = get_training_progress(latest_run)
            if progress:
                if progress["stage"] == "training":
                    print(f"训练进度: Epoch {progress['current_epoch']}/{progress['total_epochs']}")
                    print(f"  Train Loss: {progress['train_loss']:.4f}")
                    print(f"  Val Loss: {progress['val_loss']:.4f}")
                else:
                    print(f"状态: {progress['message']}")
        else:
            print("运行目录: 未创建")

        # 等待5分钟
        print(f"\n下次检查: {(datetime.now().timestamp() + 300):.0f} (5分钟后)")
        time.sleep(300)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n监控已停止")
