# constants_dvrk.py  -- Task Configurations & Normalization Constants
# Central configuration file for all training and inference. Defines each
# subtask (needle pickup, needle throw, knot tying) with its settings:
# which tissues to train on, which to validate on, normalization statistics,
# camera names, episode counts, and action format. When you run training or
# inference, it looks up the task name here to get all its parameters.

import pathlib
import numpy as np
import os

DATA_DIR = os.getenv("PATH_TO_DATASET")

TASK_CONFIGS = {
    'suturing_dot': {
        'dataset_dir': DATA_DIR,
        'goal_condition_style': None,
        'num_episodes': 615,
        'num_episodes_val': 30,
        'tissue_samples_ids': [5, 7],
        'tissue_samples_ids_val': [7],
        'camera_file_suffixes': ["_left.jpg", "_psm2.jpg", "_psm1.jpg"],
        'episode_len': 600,
        'action_mode': ['hybrid', {
            'mean': np.array([ 0.02625816, -0.0110879 ,  0.03568867,  0.31292207,  0.17532917,
                              -0.85406159,  0.14030828,  0.83284442,  0.24811407, -0.21028069,
                              -0.00848454,  0.00621005,  0.04818589, -0.15450319,  0.40315658,
                              -0.85163067, -0.1758566 , -0.86021874, -0.37486684,  0.08481652]),
            'std': np.array([0.01      , 0.01      , 0.01      , 0.19912399, 0.29502391,
                            0.12340494, 0.34657023, 0.1639507 , 0.27952175, 0.32551449,
                            0.01      , 0.01      , 0.01      , 0.1645081 , 0.22718365,
                            0.09820093, 0.17983071, 0.10066894, 0.21470842, 0.29919435]),
        }],
        'norm_scheme': 'std',
        'save_frequency': 200,
        'camera_names': ['left', 'left_wrist', 'right_wrist'],
    },

    'needle_pickup_all': {
        'dataset_dir': DATA_DIR,
        'goal_condition_style': None,
        'num_episodes': 629,
        'num_episodes_val': 86,
        'tissue_samples_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'tissue_samples_ids_val': [7],
        'phase_prefixes': ['1_'],
        'camera_file_suffixes': ["_left.jpg", "_psm2.jpg", "_psm1.jpg"],
        'episode_len': 600,
        'action_mode': ['hybrid', {
            'mean': np.array([ 0.02117121, -0.01061033,  0.04433738,  0.22251957,  0.1971502 ,
                              -0.90597265, -0.17570475,  0.91686429,  0.13694032,  0.08506552,
                              -0.01144972,  0.00619323,  0.04760892, -0.16096897,  0.45806132,
                              -0.77968459, -0.19713571, -0.78929362, -0.41484009,  0.03692078]),
            'std': np.array([0.01028275, 0.01613198, 0.03551102, 0.14531227, 0.24641732,
                            0.09483275, 0.19915273, 0.10352518, 0.24363046, 0.41263211,
                            0.01      , 0.01569558, 0.02975965, 0.18217486, 0.32791292,
                            0.12508477, 0.21523964, 0.13445343, 0.31883376, 0.33758943]),
        }],
        'norm_scheme': 'std',
        'save_frequency': 200,
        'camera_names': ['left', 'left_wrist', 'right_wrist'],
    },

    'needle_throw_all': {
        'dataset_dir': DATA_DIR,
        'goal_condition_style': None,
        'num_episodes': 310,
        'num_episodes_val': 84,
        'tissue_samples_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'tissue_samples_ids_val': [7],
        'phase_prefixes': ['2_'],
        'camera_file_suffixes': ["_left.jpg", "_psm2.jpg", "_psm1.jpg"],
        'episode_len': 600,
        'action_mode': ['hybrid', {
            'mean': np.array([ 0.03055553, -0.0079745 ,  0.03942896,  0.20950523,  0.2459589 ,
                              -0.83687713,  0.02511587,  0.84121708,  0.25258415, -0.19935053,
                              -0.00144373,  0.0049653 ,  0.04490225, -0.16504303,  0.40365707,
                              -0.88094752, -0.16531958, -0.89206828, -0.38058983,  0.10116849]),
            'std': np.array([0.01      , 0.01      , 0.01      , 0.18217966, 0.37271865,
                            0.1521178 , 0.22397943, 0.15792516, 0.39091832, 0.38089521,
                            0.01      , 0.01      , 0.01      , 0.10402391, 0.13901681,
                            0.06005486, 0.0981895 , 0.06387606, 0.13532799, 0.11621434]),
        }],
        'norm_scheme': 'std',
        'save_frequency': 200,
        'camera_names': ['left', 'left_wrist', 'right_wrist'],
    },

    'knot_tying_all': {
        'dataset_dir': DATA_DIR,
        'goal_condition_style': None,
        'num_episodes': 952,
        'num_episodes_val': 133,
        'tissue_samples_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'tissue_samples_ids_val': [7],
        'phase_prefixes': ['3_'],
        'camera_file_suffixes': ["_left.jpg", "_psm2.jpg", "_psm1.jpg"],
        'episode_len': 600,
        'action_mode': ['hybrid', {
            'mean': np.array([ 0.0249953 , -0.01066993,  0.03099244,  0.35583106,  0.09714196,
                              -0.88176762,  0.25547242,  0.85854408,  0.2132891 , -0.3488589 ,
                              -0.00550976,  0.00859079,  0.04786375, -0.11541289,  0.52012943,
                              -0.81531284, -0.27246644, -0.80284463, -0.4707253 ,  0.00315498]),
            'std': np.array([0.01      , 0.01      , 0.01      , 0.20154828, 0.1873807 ,
                            0.10344235, 0.31842194, 0.1565692 , 0.1619791 , 0.01224801,
                            0.01      , 0.01      , 0.01      , 0.15112859, 0.13838071,
                            0.09706259, 0.1765505 , 0.10569934, 0.13144507, 0.33264002]),
        }],
        'norm_scheme': 'std',
        'save_frequency': 200,
        'camera_names': ['left', 'left_wrist', 'right_wrist'],
    },
}
