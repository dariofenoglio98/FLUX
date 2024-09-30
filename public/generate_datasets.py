# ANDA interface


import config as cfg

import numpy as np
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from ANDA import anda

# valid dataset names
assert cfg.dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST', 'FMNIST', 'EMNIST'], \
        "Dataset not found! Please check the ANDA page for more details."

anda_dataset = []

# special static mode using unique fn
if cfg.drifting_type == 'static':
    assert cfg.non_iid_type in ['feature_skew',
                                'label_skew',
                                'feature_condition_skew',
                                'label_condition_skew',
                                'split_unbalanced',
                                'feature_label_skew',
                                'feature_condition_skew_with_label_skew',
                                'label_condition_skew_with_label_skew',
                                'label_condition_skew_unbalanced',
                                'feature_condition_skew_unbalanced',
                                'label_skew_unbalanced',
                                'feature_skew_unbalanced'
    ], "Non-IID type not supported in static mode! Please check the ANDA page for more details."
    anda_dataset = anda.load_split_datasets(
        dataset_name = cfg.dataset_name,
        client_number = cfg.n_clients,
        non_iid_type = cfg.non_iid_type,
        mode = "manual",
        show_features = cfg.show_features,
        show_labels = cfg.show_labels,
        random_seed = cfg.random_seed,
        **cfg.args
    )
elif cfg.drifting_type in ['trND_teDR','trDA_teDR','trDA_teND','trDR_teDR','trDR_teND']:
    # dynamic mode using same fn
    anda_dataset = anda.load_split_datasets_dynamic(
        dataset_name = cfg.dataset_name,
        client_number = cfg.n_clients,
        non_iid_type = cfg.non_iid_type,
        drfting_type = cfg.drifting_type,
        show_features = cfg.show_features,
        show_labels = cfg.show_labels,
        random_seed = cfg.random_seed,
        **cfg.args
    )
else:
    raise ValueError("Drifting type not found! Please check the ANDA page for more details.")

# Save anda_dataset
# simple format as training not drifting
if cfg.drifting_type in ['static', 'trND_teDR']:
    for i in range(cfg.n_clients):
        np.save(f'./data/cur_datasets/client_{i+1}', anda_dataset[i])
        print(f"Data for client {i+1} saved")

# complex format as training drifting
else:
    count = 1
    for dataset in anda_dataset:
        client_number = dataset['client_number']
        cluster = dataset['cluster']
        order = dataset['epoch_locker_order']

        # change name to better conform        
        filename = f'./data/cur_datasets/client_{client_number}_cluster_{cluster}_order_{order}.npy'
        
        np.save(filename, dataset)

        print(f"Dataset piece {count} saved!")
        count += 1

print("Datasets saved successfully!")
