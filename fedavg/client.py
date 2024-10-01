"""
This code creates a Flower client that can be used to train a model locally and share the updated 
model with the server. When it is started, it connects to the Flower server and waits for instructions.
If the server sends a model, the client trains the model locally and sends back the updated model.
If abilitated, at the end of the training the client evaluates the last model, and plots the 
metrics during the training.

This is code is set to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
have this code running.
"""

import argparse
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import flwr as fl

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import public.config as cfg
import public.utils as utils
import public.models as models


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
        model,
        train_loader, 
        val_loader, # input of ModelEvaluator
        optimizer, 
        num_examples, 
        client_id, 
        train_fn, 
        device
        ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.num_examples = num_examples
        self.client_id = client_id
        self.train_fn = train_fn
        self.evaluate_fn = models.ModelEvaluator(test_loader=val_loader, device=device)
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Extract descriptors
        descriptors = self.evaluate_fn.extract_descriptors(model=self.model, client_id=self.client_id, \
                                                        max_latent_space=config["max_latent_space"])

        # Train the model   
        for epoch in range(config["local_epochs"]):
            self.train_fn(self.model, self.device, self.train_loader, self.optimizer, epoch, self.client_id)


        return self.get_parameters(config), self.num_examples["train"], descriptors, #{}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        try:
            # loss, accuracy, precision_pc, recall_pc, f1_pc, accuracy_pc, loss_pc = self.evaluate_fn(self.model, self.device, self.val_loader)
            loss_trad, accuracy_trad, f1_score_trad, new_max_latent_space = self.evaluate_fn.evaluate(self.model)

            return float(loss_trad), self.num_examples["val"], {
                "accuracy": float(accuracy_trad),
                "f1_score": float(f1_score_trad),
                "max_latent_space": float(new_max_latent_space),
                "cid": int(self.client_id)
            }
            
        except Exception as e:
            print(f"An error occurred during inference of client {self.client_id}: {e}, returning same zero metrics") 
            return float(10000), self.num_examples["val"], {
                "accuracy": float(0),
                "f1_score": float(0),
                "max_latent_space": float(config["max_latent_space"]),
                "cid": int(self.client_id)
            }

# main
def main() -> None:
    # Get client id
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--id",
        type=int,
        choices=range(1, cfg.n_clients+1),
        required=True,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()

    # Load device, model and data
    device = utils.check_gpu()
    model = models.models[cfg.model_name](in_channels=3, num_classes=cfg.n_classes, \
                                          input_size=cfg.input_size).to(device)
    data = np.load(f'./data/client_{args.id}.npy', allow_pickle=True).item()

    # Split the data into training and testing subsets
    train_features, val_features, train_labels, val_labels = train_test_split(
        data['train_features'], data['train_labels'], \
        test_size=cfg.client_eval_ratio, random_state=cfg.random_seed
    )

    num_examples = {
        "train": train_features.shape[0],
        "val": val_features.shape[0]
    }

    # Create the datasets
    train_dataset = models.CombinedDataset(train_features, train_labels, transform=None)
    val_dataset = models.CombinedDataset(val_features, val_labels, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.test_batch_size, shuffle=False)

    # Optimizer and Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    # Start Flower client
    client = FlowerClient(model, train_loader, val_loader, optimizer, num_examples, args.id, 
                           models.simple_train, device).to_client()
    
    fl.client.start_client(server_address="[::]:8098", client=client) # local host

if __name__ == "__main__":
    main()