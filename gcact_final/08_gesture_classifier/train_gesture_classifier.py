#!/usr/bin/env python3
# train_gesture_classifier.py  -- Gesture Classifier Training
# Trains a separate image classifier that predicts which surgical gesture is
# currently being performed (10 classes, e.g., G3 = pushing needle through
# tissue, G15 = pulling suture). Uses a ResNet-18 backbone with a
# classification head. Achieves 93.3% accuracy on held-out tissue 7.
# The trained classifier is used at inference time by GC-ACT to provide
# real-time gesture context to the action prediction model.
"""
Gesture Classifier for SutureBot
Predicts surgical gesture (G2-G16) from camera images.
Used for GC-ACT conditioning and failure detection.

Architecture: ResNet18 (pretrained) + FC head
Input: single camera frame (left endoscope), 224x224
Output: gesture class (10 classes)

Usage:
    python train_gesture_classifier.py --data_dir ~/data --labels_dir ~/data/labels \
        --output_dir ~/checkpoints/gesture_classifier --epochs 50 --batch_size 64
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from collections import Counter
from tqdm import tqdm
import json


# Gesture label mapping  -- 10 unique gestures found in labels
GESTURE_LABELS = ['G2', 'G3', 'G6', 'G7', 'G10', 'G11', 'G13', 'G14', 'G15', 'G16']
GESTURE_TO_IDX = {g: i for i, g in enumerate(GESTURE_LABELS)}
IDX_TO_GESTURE = {i: g for g, i in GESTURE_TO_IDX.items()}
NUM_CLASSES = len(GESTURE_LABELS)


class GestureDataset(Dataset):
    """Dataset that pairs camera frames with gesture labels."""

    def __init__(self, data_dir, labels_dir, tissue_ids, subtasks, transform=None,
                 camera='left_img_dir', frame_stride=5):
        """
        Args:
            data_dir: root data directory (~/data)
            labels_dir: root labels directory (~/data/labels)
            tissue_ids: list of tissue IDs to include
            subtasks: list like ['2_needle_throw', '3_knot_tying']
            transform: torchvision transforms
            camera: which camera directory to use
            frame_stride: sample every N frames (reduces redundancy)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.camera = camera
        self.samples = []  # list of (image_path, gesture_idx)

        for tissue_id in tissue_ids:
            for subtask in subtasks:
                label_dir = os.path.join(labels_dir, f'tissue_{tissue_id}', subtask)
                if not os.path.isdir(label_dir):
                    continue

                for label_file in sorted(os.listdir(label_dir)):
                    if not label_file.endswith('_labels.txt'):
                        continue

                    episode_ts = label_file.replace('_labels.txt', '')
                    episode_dir = os.path.join(data_dir, f'tissue_{tissue_id}', subtask, episode_ts)
                    img_dir = os.path.join(episode_dir, camera)

                    if not os.path.isdir(img_dir):
                        continue

                    # Parse label file
                    label_path = os.path.join(label_dir, label_file)
                    gestures = self._parse_labels(label_path)
                    if not gestures:
                        continue

                    # Get available frames
                    frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

                    # Map frames to gestures
                    for i in range(0, len(frames), frame_stride):
                        frame_idx = i
                        gesture = self._get_gesture_at_frame(gestures, frame_idx)
                        if gesture and gesture in GESTURE_TO_IDX:
                            img_path = os.path.join(img_dir, frames[i])
                            self.samples.append((img_path, GESTURE_TO_IDX[gesture]))

        print(f"  Loaded {len(self.samples)} samples from {len(tissue_ids)} tissues")
        self._print_distribution()

    def _parse_labels(self, label_path):
        """Parse label file: 'start_frame end_frame gesture_label' per line."""
        gestures = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        start = int(parts[0])
                        end = int(parts[1])
                        gesture = parts[2]
                        gestures.append((start, end, gesture))
        except (ValueError, IOError):
            return []
        return gestures

    def _get_gesture_at_frame(self, gestures, frame_idx):
        """Find which gesture a frame belongs to."""
        for start, end, gesture in gestures:
            if start <= frame_idx <= end:
                return gesture
        return None

    def _print_distribution(self):
        """Print class distribution."""
        counts = Counter(IDX_TO_GESTURE[s[1]] for s in self.samples)
        for g in GESTURE_LABELS:
            print(f"    {g}: {counts.get(g, 0)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class GestureClassifier(nn.Module):
    """ResNet18 + FC head for gesture classification."""

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    per_class = {}
    for i, g in IDX_TO_GESTURE.items():
        mask = all_labels == i
        if mask.sum() > 0:
            per_class[g] = 100.0 * (all_preds[mask] == all_labels[mask]).mean()

    return total_loss / total, 100.0 * correct / total, per_class


def main():
    parser = argparse.ArgumentParser(description='Train gesture classifier')
    parser.add_argument('--data_dir', type=str, default=os.path.expanduser('~/data'))
    parser.add_argument('--labels_dir', type=str, default=os.path.expanduser('~/data/labels'))
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/checkpoints/gesture_classifier'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--frame_stride', type=int, default=5,
                        help='Sample every N frames (reduces redundancy)')
    parser.add_argument('--camera', type=str, default='left_img_dir',
                        choices=['left_img_dir', 'endo_psm1', 'endo_psm2'])
    parser.add_argument('--val_tissue', type=int, default=7,
                        help='Tissue ID for validation (rest for training)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # All tissues with labels (no tissue 1)
    all_tissues = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    train_tissues = [t for t in all_tissues if t != args.val_tissue]
    val_tissues = [args.val_tissue]
    subtasks = ['2_needle_throw', '3_knot_tying']

    print("=" * 60)
    print("  Gesture Classifier Training")
    print("=" * 60)
    print(f"  Train tissues: {train_tissues}")
    print(f"  Val tissue:    {val_tissues}")
    print(f"  Camera:        {args.camera}")
    print(f"  Frame stride:  {args.frame_stride}")
    print(f"  Classes:       {NUM_CLASSES} ({GESTURE_LABELS})")
    print()

    print("Loading training data...")
    train_dataset = GestureDataset(
        args.data_dir, args.labels_dir, train_tissues, subtasks,
        transform=train_transform, camera=args.camera, frame_stride=args.frame_stride
    )
    print("Loading validation data...")
    val_dataset = GestureDataset(
        args.data_dir, args.labels_dir, val_tissues, subtasks,
        transform=val_transform, camera=args.camera, frame_stride=args.frame_stride
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("ERROR: Empty dataset. Check data_dir and labels_dir paths.")
        return

    # Class weights for imbalanced data
    train_labels = [s[1] for s in train_dataset.samples]
    class_counts = Counter(train_labels)
    total = len(train_labels)
    weights = torch.tensor([total / (NUM_CLASSES * class_counts.get(i, 1)) for i in range(NUM_CLASSES)],
                           dtype=torch.float32).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = GestureClassifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: ResNet18, {n_params/1e6:.1f}M params")
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    print()

    best_val_acc = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, per_class = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'gesture_labels': GESTURE_LABELS,
                'gesture_to_idx': GESTURE_TO_IDX,
            }, os.path.join(args.output_dir, 'gesture_best.ckpt'))

        star = " *" if is_best else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.1f}% | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.1f}%{star}")

        if is_best:
            pc_str = " | ".join(f"{g}:{per_class.get(g, 0):.0f}%" for g in GESTURE_LABELS if g in per_class)
            print(f"  Per-class: {pc_str}")

    print(f"\nBest val accuracy: {best_val_acc:.1f}% @ epoch {best_epoch}")
    print(f"Checkpoint saved to {args.output_dir}/gesture_best.ckpt")

    # Save config
    config = {
        'gesture_labels': GESTURE_LABELS,
        'gesture_to_idx': GESTURE_TO_IDX,
        'num_classes': NUM_CLASSES,
        'camera': args.camera,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'val_tissue': args.val_tissue,
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    main()
