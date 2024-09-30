import config as cfg
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
    )

    print("finished!")

elif cfg.drifting_type == 'trND_teDR':
    pass
elif cfg.drifting_type == 'trDA_teDR':
    pass
elif cfg.drifting_type == 'trDA_teND':
    pass
elif cfg.drifting_type == 'trDR_teDR':
    pass
elif cfg.drifting_type == 'trDR_teND':
    pass
else:
    print("Drifting type not found! Please check the ANDA page for more details.")

