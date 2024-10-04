# Overall settings
strategy = 'fedprox' # ['fedavg', 'fedprox', 'cfl_oneshot', 'cfl_drift']
random_seed = 42
gpu = 1 # set the GPU to use, if -1 use CPU

# Strategy cfl_oneshot
cfl_oneshot_CLIENT_SCALING_METHOD = 1
cfl_oneshot_CLIENT_CLUSTER_METHOD = 1
# Strategy fedprox
fedprox_proximal_mu = 0.1

# Dataset settings
dataset_name = "MNIST" # ["CIFAR10", "CIFAR100", "MNIST", "FMNIST", "EMNIST"]
drifting_type = 'static' # ['static', 'trND_teDR', 'trDA_teDR', 'trDA_teND', 'trDR_teDR', 'trDR_teND'] refer to ANDA page for more details
non_iid_type = 'feature_skew' # refer to ANDA page for more details
n_clients = 5
show_features = False # show generated feature details if any
show_labels = False # show distribution of data if any
# careful with the args applying to your settings above
args = {
    # 'scaling_label_low': 0.7,
    # 'scaling_label_high': 1
    # 'set_color':True,
    # 'colors':3,
    # 'set_rotation':True,
    # 'rotations':4,
    # 'scaling_color_low':0.8,
    # 'scaling_color_high':1.0
    # 'DA_epoch_locker_num': 3,
    # 'DA_random_locker':True,
    # 'DA_continual_divergence':False,
    # 'DA_epoch_locker_num':10,
    # 'DA_max_dist':10,
    # 'rotation_bank':4,
    # 'color_bank':3,
}

# Training model settings
model_name = "LeNet5"   # ["LeNet5", "ResNet9"]
batch_size = 64
test_batch_size = 1024
client_eval_ratio = 0.2
n_rounds = 6
local_epochs = 2
lr = 0.005
momentum = 0.9
transform = None


# self-defined settings
n_classes_dict = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "MNIST": 10,
    "FMNIST": 10
}
n_classes = n_classes_dict[dataset_name]

input_size_dict = {
    "CIFAR10": (32, 32),
    "CIFAR100": (32, 32),
    "MNIST": (28, 28),
    "FMNIST": (28, 28)
}
input_size = input_size_dict[dataset_name]

acceptable_accuracy = {
    "CIFAR10": 0.5,
    "CIFAR100": 0.1,
    "MNIST": 0.8,
    "FMNIST": 0.8
}
th_accuracy = acceptable_accuracy[dataset_name]
training_drifting = False if drifting_type in ['static', 'trND_teDR'] else True # to be identified
