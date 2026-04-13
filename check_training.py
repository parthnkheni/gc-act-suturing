#!/usr/bin/env python3
"""Training progress monitor  -- shows current epoch, loss, ETA."""
import os, re, time, sys
from datetime import timedelta

LOG_FILES = [
    ("GC-ACT KT OOD", "/home/exouser/logs/train_kt_gcact_ood.log", 1000),
    ("GC-ACT NT OOD", "/home/exouser/logs/train_nt_gcact_ood.log", 1000),
]

def parse_log(log_path, total_epochs):
    if not os.path.exists(log_path):
        return None

    epochs = []
    val_losses = []
    best_epoch = None
    best_loss = None
    first_epoch_time = None
    last_epoch_time = None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Match "Epoch N" from print statement
        m = re.search(r'Epoch\s+(\d+)', line)
        if m:
            ep = int(m.group(1))
            if ep not in epochs:
                epochs.append(ep)

        # Match tqdm progress: "  3%|...| 30/1000 [08:15<..."
        m = re.search(r'(\d+)/\d+\s+\[', line)
        if m:
            ep = int(m.group(1))
            if not epochs or ep > max(epochs):
                epochs.append(ep)

        # Match val loss
        m = re.search(r'Val loss:\s+([\d.]+)', line)
        if m:
            val_losses.append(float(m.group(1)))

        # Match best checkpoint
        m = re.search(r'Best ckpt, val loss ([\d.]+) @ epoch (\d+)', line)
        if m:
            best_loss = float(m.group(1))
            best_epoch = int(m.group(2))

    # Estimate timing from file modification
    if os.path.exists(log_path):
        stat = os.stat(log_path)
        file_age = time.time() - stat.st_mtime  # seconds since last write

    # Try to get start time from /tmp marker or file creation
    start_time_file = "/tmp/ood_kt_train_start"
    if os.path.exists(start_time_file):
        with open(start_time_file) as f:
            start_ts = float(f.read().strip())
    else:
        start_ts = os.stat(log_path).st_ctime

    current_epoch = max(epochs) if epochs else 0
    elapsed = time.time() - start_ts

    if current_epoch > 0:
        secs_per_epoch = elapsed / (current_epoch + 1)
        remaining_epochs = total_epochs - current_epoch - 1
        eta_secs = remaining_epochs * secs_per_epoch
    else:
        secs_per_epoch = 0
        eta_secs = 0

    return {
        'current_epoch': current_epoch,
        'total_epochs': total_epochs,
        'pct': (current_epoch + 1) / total_epochs * 100,
        'val_loss': val_losses[-1] if val_losses else None,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'elapsed': elapsed,
        'eta': eta_secs,
        'secs_per_epoch': secs_per_epoch,
        'file_age': file_age,
    }

def format_time(secs):
    return str(timedelta(seconds=int(secs)))

def main():
    print("=" * 60)
    print("  OOD KT Training Monitor")
    print("=" * 60)

    active_found = False
    for name, log_path, total_epochs in LOG_FILES:
        info = parse_log(log_path, total_epochs)
        if info is None:
            print(f"\n  {name}: not started yet")
            continue

        active_found = True
        print(f"\n  {name}")
        print(f"  {'-' * 50}")

        # Progress bar
        bar_width = 30
        filled = int(bar_width * info['pct'] / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"  [{bar}] {info['pct']:.1f}%")

        print(f"  Epoch: {info['current_epoch']}/{info['total_epochs']}")
        if info['val_loss'] is not None:
            print(f"  Val loss: {info['val_loss']:.5f}")
        if info['best_loss'] is not None:
            print(f"  Best: {info['best_loss']:.6f} @ epoch {info['best_epoch']}")
        print(f"  Elapsed: {format_time(info['elapsed'])}")
        if info['eta'] > 0:
            print(f"  ETA: {format_time(info['eta'])}")
            print(f"  Speed: {info['secs_per_epoch']:.1f}s/epoch")
        if info['file_age'] > 120:
            print(f"  [WARN] Log stale ({format_time(info['file_age'])} since last write)")

    if not active_found:
        print("\n  No training logs found yet. Training may still be starting up.")

    # Check GPU
    gpu_line = os.popen("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null").read().strip()
    if gpu_line:
        parts = gpu_line.split(', ')
        print(f"\n  GPU: {parts[0]}% util | {parts[1]}/{parts[2]} MiB")

    print()

if __name__ == '__main__':
    main()
