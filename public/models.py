"""
LeNet-5 (around 60k parameters)
ResNet-9 (around 6M parameters)

Check gpu function
Training functions to test the models

"""

import numpy as np  
from math import prod
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#############################################################################################################
# Models 
#############################################################################################################
# LeNet-5 model
class LeNet5(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, input_size=(28, 28)):
        super(LeNet5, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2)  # Convolutional layer with 6 feature maps of size 5x5
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # Subsampling layer with 6 feature maps of size 2x2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # Convolutional layer with 16 feature maps of size 5x5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # Subsampling layer with 16 feature maps of size 2x2
        
        # Dinamically calculate the size of the features after convolutional layers
        dummy_input = torch.zeros(1, in_channels, *input_size)
        dummy_output = self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))
        self.feature_size = prod(dummy_output.size()[1:])

        self.fc1 = nn.Linear(self.feature_size, 120)  # Fully connected layer, output size 120
        self.fc2 = nn.Linear(120, 84)  # Fully connected layer, output size 84
        self.fc3 = nn.Linear(84, num_classes)  # Fully connected layer, output size num_classes

    def forward(self, x, latent=False):
        x = F.relu(self.conv1(x))  # Apply ReLU after conv1
        x = self.pool1(x)  # Apply subsampling pool1
        x = F.relu(self.conv2(x))  # Apply ReLU after conv2
        x = self.pool2(x)  # Apply subsampling pool2
        x_l = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x_l))  # Apply ReLU after fc1
        x = F.relu(self.fc2(x))  # Apply ReLU after fc2
        x = self.fc3(x)  # Output layer
        if latent:
            return x, x_l
        else:
            return x

# Resnet-9 layer
def residual_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# ResNet-9 model
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes, input_size=(28, 28)):
        super().__init__()
        self.num_classes = num_classes
        self.prep = residual_block(in_channels, 64)
        self.layer1_head = residual_block(64, 128, pool=True)
        self.layer1_residual = nn.Sequential(residual_block(128, 128), residual_block(128, 128))
        self.layer2 = residual_block(128, 256, pool=True)
        self.layer3_head = residual_block(256, 512, pool=True)
        self.layer3_residual = nn.Sequential(residual_block(512, 512), residual_block(512, 512))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Changed to adaptive average pooling:         self.MaxPool2d = nn.Sequential(nn.MaxPool2d(4))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the features after the convolutional layers
        dummy_input = torch.zeros(1, in_channels, *input_size)
        dummy_output = self.pool(self.layer3_head(self.layer2(self.layer1_head(self.prep(dummy_input)))))
        self.feature_size = dummy_output.size(1) * dummy_output.size(2) * dummy_output.size(3)

        # Output layer
        self.linear = nn.Linear(self.feature_size, num_classes)

    def forward(self, x, latent=False):
        x = self.prep(x)
        x = self.layer1_head(x)
        x = self.layer1_residual(x) + x
        x = self.layer2(x)
        x = self.layer3_head(x)
        x = self.layer3_residual(x) + x
        x = self.pool(x)  # Changed to adaptive average pooling
        x_l = x.view(x.size(0), -1)
        x = self.linear(x_l)
        if latent:
            return x, x_l
        else:
            return x
    
models = {
    'LeNet5': LeNet5,
    'ResNet9': ResNet9,
}    

#############################################################################################################
# Helper functions 
#############################################################################################################

# simple train function
def simple_train(model, device, train_loader, optimizer, epoch, client_id=None):
    model.train()
    loss_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % 10 == 0:
        #     print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
        #           f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        loss_list.append(loss.item())
    # print(f'Client: {client_id} - Train Epoch: {epoch} \tLoss: {sum(loss_list)/len(loss_list):.6f}')

# simple test function
def simple_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    # print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
    #       f'({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return test_loss, accuracy

# ModelEvaluator class
class ModelEvaluator:
    def __init__(self, test_loader, device):
        """
        Initializes the ModelEvaluator with the model, device, and number of classes.
        
        Args:
            test_loader: DataLoader with test data
            device: Device to run the evaluation on
        """
        self.test_loader = test_loader
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.criterion_trad = torch.nn.CrossEntropyLoss() 

    def extract_descriptors(self,
                            model,
                            client_id: int = 0,
                            max_latent_space: int = 2
                            ):
        """
        Evaluates the model on the provided test data and returns the descriptors.
        Descriptors:
            latent space representation, traditional metrics, and metrics per class

        Args:
            model: Model to evaluate
            client_id: Client ID
            max_latent_space: Maximum value of the latent space (used for scaling/PCA)
        """
        
        # Set model to evaluation mode
        model.eval()
        num_classes = model.num_classes

        # Initialize storage for metrics
        precision_per_class = [0] * num_classes
        recall_per_class = [0] * num_classes
        f1_per_class = [0] * num_classes
        accuracy_per_class = [0] * num_classes
        loss_per_class = [0] * num_classes
        class_counts = [0] * num_classes

        y_true_all = []
        y_pred_all = []
        loss_all = []
        latent_all = []
        latent_mean = []
        loss_trad = 0
        total_samples = 0

        # Accumulate predictions and targets over batches
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # model output
                output, latent_space = model(data, latent=True)
                latent_all.extend(latent_space.cpu().numpy())
                    
                y_pred_batch = output.argmax(dim=1, keepdim=False)  # Predicted class labels
                
                # Store the true and predicted labels for the batch
                y_true_all.extend(target.cpu().numpy())
                y_pred_all.extend(y_pred_batch.cpu().numpy())
                
                # Compute per-sample loss for the batch
                batch_loss = self.criterion(output, target).cpu().numpy()
                loss_all.extend(batch_loss)
                
                # Compute traditional loss for the batch
                loss_trad += self.criterion_trad(output, target).item()
                
                # Accumulate the total number of samples
                total_samples += len(target)

        # Convert collected predictions and true labels into tensors for processing
        y_true_all = torch.tensor(y_true_all)
        y_pred_all = torch.tensor(y_pred_all)
        loss_all = torch.tensor(loss_all)
        
        # Average traditional loss over the total number of samples
        loss_trad /= total_samples
        
        # Calculate traditional accuracy on the entire test set
        accuracy_trad = accuracy_score(y_true_all, y_pred_all)
        
        # Average latent
        latent_all = np.array(latent_all)
        new_max_latent_space = np.max(latent_all)
        # SCALE OR NOT TRY BOTH
        # scaler = MinMaxScaler(feature_range=(0, max_latent_space)) # maybe try also StandardScaler
        # latent_all = scaler.fit_transform(latent_all) # Sample, Dim_latent_space
        # print(f"Min-Max values of latent_all: {np.min(latent_all)}, {np.max(latent_all)}")
        # create random_points to fit PCA (min=0, max=max_latent_space)
        np.random.seed(seed=1)
        random_points = np.random.uniform(0, max_latent_space, size=(200, latent_all.shape[1]))
        pca = PCA(n_components=num_classes)
        # fit PCA on random_points
        pca.fit(random_points)
        # transform latent_all
        latent_all = pca.transform(latent_all)
        # Mean on first dimension
        latent_mean = list(np.mean(latent_all, axis=0))
            
        # Iterate through each class (for MNIST, classes are 0 to 9 by default)
        for class_idx in range(num_classes):
            # Get all predictions and ground truths for the current class
            class_mask = (y_true_all == class_idx)  # Mask for this class
            
            y_true_class = (y_true_all == class_idx).numpy().astype(int)  # Binary labels for the current class
            y_pred_class = (y_pred_all == class_idx).numpy().astype(int)  # Binary predictions for the current class
            
            # Only calculate if there are samples for this class
            if class_mask.sum() > 0:
                # Compute precision, recall, and F1-score for this class
                precision = precision_score(y_true_class, y_pred_class, zero_division=0)
                recall = recall_score(y_true_class, y_pred_class, zero_division=0)
                f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
                accuracy = accuracy_score(y_true_class, y_pred_class)

                # Compute the loss for this class (average the loss of samples in this class)
                class_loss = loss_all[class_mask].mean().item()

                # Update class counts and metrics
                precision_per_class[class_idx] = precision
                recall_per_class[class_idx] = recall
                f1_per_class[class_idx] = f1
                accuracy_per_class[class_idx] = accuracy
                loss_per_class[class_idx] = class_loss
                class_counts[class_idx] = class_mask.sum().item()

        res = {
            "num_examples_val": len(self.test_loader.dataset),
            "loss_val": float(loss_trad),
            "accuracy": float(accuracy_trad),
            "precision_pc": json.dumps(precision_per_class), # use json.dumps to serialize the list - read with json.loads
            "recall_pc": json.dumps(recall_per_class),
            "f1_pc": json.dumps(f1_per_class),
            "accuracy_pc": json.dumps(accuracy_per_class),
            "loss_pc": json.dumps(loss_per_class),
            "latent_space": json.dumps(latent_mean),
            "max_latent_space": float(new_max_latent_space),
            "cid": int(client_id)
        }

        return res

    def evaluate(self, model):
        """
        Evaluates the model on the provided test data and returns various metrics.

        Args:
            model: Model to evaluate
        """
        
        # client-enhanced evaluation function
        # def evaluate_model_per_class(model, device, test_loader, latent=False):
        # Set model to evaluation mode
        model.eval()
        num_classes = model.num_classes

        y_true_all = []
        y_pred_all = []
        loss_all = []
        latent_all = []
        loss_trad = 0
        total_samples = 0

        # Accumulate predictions and targets over batches
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output, latent_space = model(data, latent=True)
                latent_all.extend(latent_space.cpu().numpy())
                    
                y_pred_batch = output.argmax(dim=1, keepdim=False)  # Predicted class labels
                
                # Store the true and predicted labels for the batch
                y_true_all.extend(target.cpu().numpy())
                y_pred_all.extend(y_pred_batch.cpu().numpy())
                
                # Compute per-sample loss for the batch
                batch_loss = self.criterion(output, target).cpu().numpy()
                loss_all.extend(batch_loss)
                
                # Compute traditional loss for the batch
                loss_trad += self.criterion_trad(output, target).item()
                
                # Accumulate the total number of samples
                total_samples += len(target)

        # Convert collected predictions and true labels into tensors for processing
        y_true_all = torch.tensor(y_true_all)
        y_pred_all = torch.tensor(y_pred_all)
        loss_all = torch.tensor(loss_all)
        
        # Average traditional loss over the total number of samples
        loss_trad /= total_samples
        
        # Calculate traditional accuracy on the entire test set
        accuracy_trad = accuracy_score(y_true_all, y_pred_all)
        f1_score_trad = f1_score(y_true_all, y_pred_all, average='weighted') # Calculate metrics for each label, and find their average weighted by support. NOT traditional F1-score
        
        # Take the next round max latent space value
        latent_all = np.array(latent_all)
        new_max_latent_space = np.max(latent_all)

        return loss_trad, accuracy_trad, f1_score_trad, new_max_latent_space

# Dataset class
class CombinedDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


#############################################################################################################
# test the models
#############################################################################################################
def main():
    # TODO ANDA deployment
    # For simplicity, may find the old version of ANDA to test
    return

    # Training settings
    model_name = "ResNet9"   # Options: "LeNet5", "ResNet9"
    batch_size = 64
    test_batch_size = 1000
    epochs = 10
    lr = 0.01
    momentum = 0.9
    seed = 1
    transform = None
    # dataset settings
    dataset_name = "CIFAR10"
    client_number = 10
    set_rotation = True
    rotations = 4
    scaling_rotation_low = 0.1
    scaling_rotation_high = 0.2
    set_color = True
    colors = 3
    scaling_color_low = 0.1
    scaling_color_high = 0.2
    random_order = True

    print(f"\n\033[94mTraining {model_name} on {dataset_name} with {client_number} clients\033[0m\n")

    device = utils.check_gpu(manual_seed=True, print_info=True)
    torch.manual_seed(seed)

    # load data
    # deprecated soon as using ANDA
    train_images, train_labels, test_images, test_labels = noniidgen.load_full_datasets(dataset_name)

    # create data: split_feature_skew
    clients_data = noniidgen.split_feature_skew(
        train_features = train_images,
        train_labels = train_labels,
        test_features = test_images,
        test_labels = test_labels,
        client_number = client_number,
        set_rotation = set_rotation,
        rotations = rotations,
        scaling_rotation_low = scaling_rotation_low,
        scaling_rotation_high = scaling_rotation_high,
        set_color = set_color,
        colors = colors,
        scaling_color_low = scaling_color_low,
        scaling_color_high = scaling_color_high,
        random_order = random_order
    )

    # merge the data (for Centralized Learning Simulation)
    train_features, train_labels, test_features, test_labels = merge_data(clients_data)

    # Create the datasets
    train_dataset = CombinedDataset(train_features, train_labels, transform=transform)
    test_dataset = CombinedDataset(test_features, test_labels, transform=transform)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # model = LeNet5(in_channels=3, num_classes=10, input_size=(32,32)).to(device)
    model = models[model_name](in_channels=3, num_classes=10, input_size=(32,32)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


    for epoch in range(1, epochs + 1):
        simple_train(model, device, train_loader, optimizer, epoch)
        _, _ = simple_test(model, device, test_loader)

if __name__ == '__main__':
    main()
