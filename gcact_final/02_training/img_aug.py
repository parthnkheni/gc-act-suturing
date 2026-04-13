# img_aug.py  -- Image Augmentation
# Applies random visual transformations (brightness, contrast, color shifts,
# crops) to camera images during training. This makes the model more robust
# to lighting and camera variations it might encounter on a different dVRK
# system than the one used to collect training data.

import numpy as np
import torch
import os
import random

import sys
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations, batch_rotations, transformations, trajectories
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

import seaborn as sns
from tqdm import tqdm
import json
import time
PATH_TO_SUTUREBOT = os.getenv('PATH_TO_SUTUREBOT')
if PATH_TO_SUTUREBOT:
    sys.path.append(os.path.join(PATH_TO_SUTUREBOT, 'src'))

import IPython
e = IPython.embed

class DataAug(object):
    def __init__(self, img_hw, mask_prob=0.07, mask_sketch_prob=0.5):
        self.img_hw = img_hw  # (H, W)
        self.ratio = 0.95
        self.mask_prob = mask_prob
        self.mask_sketch_prob = mask_sketch_prob

        # Spatial transforms (crop, resize, rotation), synced across 'left' and 'mask'
        self.spatial_transforms = A.Compose([
            A.RandomCrop(height=int(img_hw[0] * self.ratio), width=int(img_hw[1] * self.ratio)),
            A.Resize(height=img_hw[0], width=img_hw[1]),
            A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT_101),
        ], additional_targets={'mask': 'image'})

        # Color jitter (only for RGB)
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.5, hue=0.08)

        # Albumentations for pixel dropout
        min_height = max(1, img_hw[0] // 40)
        min_width = max(1, img_hw[1] // 40)
        max_height = min(img_hw[0] // 30, img_hw[0])
        max_width = min(img_hw[1] // 30, img_hw[1])

        self.pixel_dropout = A.Compose([
            A.CoarseDropout(max_holes=128, max_height=max_height, max_width=max_width,
                            min_holes=1, min_height=min_height, min_width=min_width,
                            fill_value=0, p=0.5),
        ], additional_targets={'mask': 'image'})

    def random_shift(self, img, shift_x=0, shift_y=0):
        max_shift_x = int(self.img_hw[1] * 0.2)
        max_shift_y = int(self.img_hw[0] * 0.2)

        if shift_x == 0 and shift_y == 0:
            shift_x = np.random.randint(-max_shift_x, max_shift_x)
            shift_y = np.random.randint(-max_shift_y, max_shift_y)

        img = T.functional.affine(img, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)
        return img, shift_x, shift_y

    def __call__(self, sample):
        # Convert tensors to numpy for Albumentations
        sample_np = {k: v.permute(1, 2, 0).cpu().numpy() for k, v in sample.items()}

        # Apply spatial transforms (crop, resize, rotate) consistently
        if 'left' in sample_np and 'mask' in sample_np:
            aug = self.spatial_transforms(image=sample_np['left'], mask=sample_np['mask'])
            sample_np['left'] = aug['image']
            sample_np['mask'] = aug['mask']
        else:
            sample_np['left'] = self.spatial_transforms(image=sample_np['left'])['image']

        # Apply pixel dropout consistently
        if 'left' in sample_np and 'mask' in sample_np:
            aug = self.pixel_dropout(image=sample_np['left'], mask=sample_np['mask'])
            sample_np['left'] = aug['image']
            sample_np['mask'] = aug['mask']
        else:
            sample_np['left'] = self.pixel_dropout(image=sample_np['left'])['image']

        # Convert back to torch tensors
        processed = {}
        shift_x = shift_y = 0
        for key, img_np in sample_np.items():
            img_t = torch.from_numpy(img_np).permute(2, 0, 1)

            # Apply color jitter only to RGB images
            if key in ['left', 'right', 'img_l_hist', 'img_lw', 'img_rw'] and img_t.shape[0] == 3:
                img_t = self.color_jitter(img_t)

            # Apply shift (shared between 'left' and 'mask')
            if key == 'left' or key == 'right':
                img_t, shift_x, shift_y = self.random_shift(img_t, shift_x=shift_x, shift_y=shift_y)
            elif key == 'mask':
                img_t, _, _ = self.random_shift(img_t, shift_x=shift_x, shift_y=shift_y)

            processed[key] = img_t

        # Optional sketch/history masking
        if 'img_l_hist' in processed and np.random.rand() < self.mask_sketch_prob:
            processed['img_l_hist'] = torch.zeros_like(processed['img_l_hist'])

        # Random full masking
        if np.random.rand() < self.mask_prob:
            mask_choice = np.random.choice(list(processed.keys()))
            processed[mask_choice] = torch.zeros_like(processed[mask_choice])

        return processed