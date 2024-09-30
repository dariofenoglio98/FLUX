# Config

# Overall settings
random_seed = 42
strategy = 'fedavg' # ['fedavg', 'OneShot', 'Our']  


# Training settings
model_name = "LeNet5"   # ["LeNet5", "ResNet9"]
batch_size = 64
test_batch_size = 1024
client_eval_ratio = 0.2
n_rounds = 2
local_epochs = 2
lr = 0.01
momentum = 0.9
seed = random_seed
transform = None

# Dataset settings
dataset_name = "MNIST" # Options: "CIFAR10", "CIFAR100" "MNIST", "FMNIST"
client_number = 2
set_rotation = False
rotations = 4
scaling_rotation_low = 1
scaling_rotation_high = 1
set_color = True
colors = 3
scaling_color_low = 1
scaling_color_high = 1
random_order = True
show_distribution = True

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

