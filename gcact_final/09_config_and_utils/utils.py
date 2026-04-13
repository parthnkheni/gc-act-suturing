# utils.py  -- Shared Helper Functions
# Common utility functions used across training, inference, and evaluation:
# data loading, random seed setting, dictionary operations, image
# preprocessing, and normalization/denormalization of robot actions.

import numpy as np
import torch
import os
import random
# import h5py
import torch.utils.data
from torch.utils.data import DataLoader, ConcatDataset, Sampler, WeightedRandomSampler
from collections import Counter
import cv2
import json
from torchvision import transforms
import sys

path_to_suturebot = os.getenv("PATH_TO_SUTUREBOT")

if path_to_suturebot:
    sys.path.append(os.path.join(path_to_suturebot, 'src'))
from aloha_pro.aloha_scripts.utils import crop_resize
# from auto_label.auto_label_func import get_auto_label
from generic_dataset import *
CROP_TOP = True  # hardcode
FILTER_MISTAKES = True  # Filter out mistakes from the dataset even if not use_language

def get_norm_stats(dataset_dirs, num_episodes_list):
    all_qpos_data = []
    all_action_data = []

    # Iterate over each directory and the corresponding number of episodes
    for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list):
        for episode_idx in range(num_episodes):
            dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
            with h5py.File(dataset_path, "r") as root:
                qpos = root["/observations/qpos"][()]
                action = root["/action"][()]
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))

    # Concatenate data from all directories
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # Normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf)

    # Normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()
    eps = 0.0001

    stats = {
        "action_mean": action_mean.numpy(),
        "action_std": action_std.numpy(),
        "action_min": action_min.numpy() - eps,
        "action_max": action_max.numpy() + eps,
        "qpos_mean": qpos_mean.numpy(),
        "qpos_std": qpos_std.numpy(),
        "example_qpos": all_qpos_data[-1].numpy(),
    }  # example from the last loaded qpos

    return stats

def load_data_dvrk(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    task_config,
    chunk_size,
    use_language=False,
    use_gesture=False,
    labels_dir=None):
    
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    num_episodes_val = task_config['num_episodes_val']
    train_indices = np.random.permutation(num_episodes)
    val_indices = np.random.permutation(num_episodes_val)

    # obtain normalization stats for qpos and action
    norm_stats = None
    action_mode = task_config['action_mode'][0]
    dataset_path = task_config['dataset_dir']
    tissue_samples_ids = task_config['tissue_samples_ids']
    tissue_samples_ids_val = task_config['tissue_samples_ids_val']
    camera_file_suffixes = task_config['camera_file_suffixes']

    print("\n-------------loading training data-------------\n")

    train_datasets = EpisodicDatasetDvrkGeneric(
            train_indices,
            tissue_samples_ids,
            dataset_dir,
            camera_names,
            camera_file_suffixes,
            task_config,
            chunk_size=chunk_size,
            use_language=use_language,
            use_gesture=use_gesture,
            labels_dir=labels_dir,
        )
    print("\n-------------loading validation data-------------\n")

    val_datasets = EpisodicDatasetDvrkGeneric(
            val_indices,
            tissue_samples_ids_val,
            dataset_dir,
            camera_names,
            camera_file_suffixes,
            task_config,
            use_language=use_language,
            use_gesture=use_gesture,
            labels_dir=labels_dir,
        )

    # Get task labels for all samples
    task_labels = train_datasets.sample_task_labels
    task_counts = Counter(task_labels)  # e.g., {'1': 500, '2': 200, '3': 500}
    print(f"task count:{task_counts}")
    total_samples = len(task_labels)

    # Compute weights
    weights = [1.0 / task_counts[task] for task in task_labels]

    # Create sampler
    sampler = WeightedRandomSampler(weights, num_samples=total_samples, replacement=True)

    train_dataloader = DataLoader(train_datasets, batch_size=batch_size_train, sampler=sampler,
                              pin_memory=True, num_workers=16, prefetch_factor=4, persistent_workers=True)

    # train_dataloader = DataLoader(train_datasets, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=16, prefetch_factor=4, persistent_workers=True)
    val_dataloader = DataLoader(val_datasets, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=16, prefetch_factor=4, persistent_workers=True)

    return train_dataloader, val_dataloader, norm_stats, False



### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach().cpu()
    return new_d


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def number_to_one_hot(number, size=501):
    one_hot_array = np.zeros(size)
    one_hot_array[number] = 1
    return one_hot_array
