# Overall settings
model_name = "LeNet5"   # ["LeNet5", "ResNet9"]
dataset_name = "MNIST" # ["CIFAR10", "CIFAR100" "MNIST", "FMNIST"]
strategy = 'fedavg' # ['fedavg', 'cfl_oneshot', 'cfl_drift']
non_iid_type = 'TODO' # []
n_clients = 2
client_number = n_clients
random_seed = 42

# Dataset settings




# Training model settings
batch_size = 64
test_batch_size = 1024
client_eval_ratio = 0.2
n_rounds = 2
local_epochs = 2
lr = 0.01
momentum = 0.9
seed = random_seed
transform = None

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

