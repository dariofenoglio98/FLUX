# Overall settings
strategy = 'fedavg' # ['fedavg', 'fedprox', 'cfl_oneshot', 'cfl_drift']
random_seed = 42
gpu = 1 # set the GPU to use, if -1 use CPU
n_clients = 10

# Strategy cfl_oneshot
cfl_oneshot_CLIENT_SCALING_METHOD = 1
cfl_oneshot_CLIENT_CLUSTER_METHOD = 3
extended_descriptors = True
# Strategy fedprox
fedprox_proximal_mu = 0.001

# Dataset settings
dataset_name = "MNIST" # ["CIFAR10", "CIFAR100", "MNIST", "FMNIST", "EMNIST"]
drifting_type = 'static' # ['static', 'trND_teDR', 'trDA_teDR', 'trDA_teND', 'trDR_teDR', 'trDR_teND'] refer to ANDA page for more details
non_iid_type = 'label_skew_strict' # refer to ANDA page for more details
show_features = True # show generated feature details if any
show_labels = True # show distribution of data if any
# careful with the args applying to your settings above
args = {
    # 'set_rotation':True,
    # 'rotations':2,
    # 'set_color':True,
    # 'colors':2,
    # 'show_distribution':True,
    # 'client_n_class':2,
    # 'py_bank':3,
    'random_mode':True,
    'mixing_label_number':3,
    'scaling_label_low':1.0,
    'scaling_label_high':1.0,
    'verbose':True
    # 'scaling_label_low':0.5,
    # 'scaling_label_high':0.9
}

# Training model settings
model_name = "LeNet5"   # ["LeNet5", "ResNet9"]
batch_size = 64
test_batch_size = 64
client_eval_ratio = 0.2
n_rounds = 5
local_epochs = 2
lr = 0.005
momentum = 0.9

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
