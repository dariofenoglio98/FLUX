"""
This code implements the FedAvg, when it starts, the server waits for the clients to connect. When the established number 
of clients is reached, the learning process starts. The server sends the model to the clients, and the clients train the 
model locally. After training, the clients send the updated model back to the server. Then client models are aggregated 
with FedAvg. The aggregated model is then sent to the clients for the next round of training. The server saves the model 
and metrics after each round.

This is code is set to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
run the appopriate client code (client.py).

"""

# Libraries
import flwr as fl
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import Parameters, Scalar, Metrics
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes
import argparse
import torch
from torch.utils.data import DataLoader
import os
from logging import WARNING
from flwr.common.logger import log
from collections import OrderedDict
import json
import time
import pandas as pd
import config as cfg
import utils
import models
from sklearn.model_selection import train_test_split
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitRes,
    FitIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common import NDArray, NDArrays
from functools import reduce
from flwr.server.client_manager import ClientManager
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


# Define the max latent space as global variable
max_latent_space = 2

# Config_client
def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
        "local_epochs": cfg.local_epochs,
        "tot_rounds": cfg.n_rounds,
        "min_latent_space": 0,
        "max_latent_space": max_latent_space,
    }
    return config

# Custom weighted average function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    # validities = [num_examples * m["validity"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples

def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

def aggregate_fit(
    self,
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    """Aggregate fit results using weighted average."""
    if not results:
        return None, {}
    # Do not aggregate if there are failures and failures are not accepted
    if not self.accept_failures and failures:
        return None, {}

    # Convert results
    weights_results = [
        (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        for _, fit_res in results
    ]
    parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

    # Aggregate custom metrics if aggregation fn was provided
    metrics_aggregated = {}
    if self.fit_metrics_aggregation_fn:
        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
    elif server_round == 1:  # Only log this warning once
        log(WARNING, "No fit_metrics_aggregation_fn provided")

    return parameters_aggregated, metrics_aggregated

# Custom strategy to save model after each round
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.dataset = dataset
        self.client_cid_list = []
        self.aggregated_cluster_parameters = []
        self.cluster_labels = {}

    # Override aggregate_fit method to add saving functionality
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        
        
        ################################################################################
        # Federated averaging aggregation
        ################################################################################
        # Federated averaging - from traditional code
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_parameters_global = ndarrays_to_parameters(aggregate(weights_results))   # Global aggregation - traditional - no clustering
        
        # Aggregate custom metrics if aggregation fn was provided   NO FIT METRICS AGGREGATION FN PROVIDED - SKIPPED FOR NOW
        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
            
            
        ################################################################################
        # Save model
        ################################################################################
        if aggregated_parameters_global is not None:

            print(f"Saving round {server_round} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters_global)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            # Save the model
            torch.save(self.model.state_dict(), f"checkpoints/{cfg.model_name}/{cfg.dataset_name}/model_{server_round}.pth")
        
        return aggregated_parameters_global, aggregated_metrics
    
    
    ############################################################################################################
    # Aggregate evaluation results
    ############################################################################################################
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
    


    



# Main
def main() -> None:

    # Start time
    start_time = time.time()

    # Create directories 
    utils.create_folders()

    # Create all data for clients
    utils.generate_dataset()

    # Pick the independent test set from each client
    test_x, test_y = [], []
    for client_id in range(cfg.client_number):
        data = np.load(f'./data/client_{client_id+1}.npy', allow_pickle=True).item()
        test_x.append(data['test_features'])
        test_y.append(data['test_labels'])
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)
    # Split the data into test subsets (final evaluation & server validation)
    test_x_server, test_x, test_y_server, test_y = train_test_split(test_x, test_y, test_size=0.5, random_state=42)

    # Create the datasets
    test_dataset = models.CombinedDataset(test_x, test_y, transform=None)

    # Create the data loaders
    test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)

    # model and history folder
    device = utils.check_gpu(manual_seed=True, print_info=True)
    model = models.models[cfg.model_name](in_channels=3, num_classes=cfg.n_classes, input_size=cfg.input_size).to(device)

    # Define strategy
    strategy = SaveModelStrategy(
        model=model, # model to be trained
        min_fit_clients=cfg.client_number, #+cfg.n_attackers, # Never sample less than 10 clients for training
        min_evaluate_clients=cfg.client_number, #+cfg.n_attackers,  # Never sample less than 5 clients for evaluation
        min_available_clients=cfg.client_number, #+cfg.n_attackers, # Wait until all 10 clients are available
        fraction_fit=1.0, # Sample 100 % of available clients for training
        fraction_evaluate=1.0, # Sample 100 % of available clients for evaluation
        evaluate_metrics_aggregation_fn=weighted_average,
        on_evaluate_config_fn=fit_config,
        on_fit_config_fn=fit_config,
        dataset=cfg.dataset_name,
    )

    print(f"\n\033[94mTraining {cfg.model_name} on {cfg.dataset_name} with {cfg.client_number} clients\033[0m\n")

    # Start Flower server for three rounds of federated learning
    history = fl.server.start_server(
        server_address="0.0.0.0:8098",   # 0.0.0.0 listens to all available interfaces
        config=fl.server.ServerConfig(num_rounds=cfg.n_rounds),
        strategy=strategy,
    )
    # convert history to list
    loss = [k[1] for k in history.losses_distributed]
    accuracy = [k[1] for k in history.metrics_distributed['accuracy']]

    # Save loss and accuracy to a file
    print(f"Saving metrics to as .json in histories folder...")
    with open(f'histories/{cfg.model_name}/{cfg.dataset_name}/distributed_metrics.json', 'w') as f:
        json.dump({'loss': loss, 'accuracy': accuracy}, f)

    # Single Plot
    best_loss_round, best_acc_round = utils.plot_loss_and_accuracy(loss, accuracy, show=False)

    # Load the best model
    model.load_state_dict(torch.load(f"checkpoints/{cfg.model_name}/{cfg.dataset_name}/model_{best_loss_round}.pth", weights_only=False))

    # Evaluate the model on the test set
    loss_test, accuracy_test = models.simple_test(model, device, test_loader)
    print(f"\n\033[93mTest Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f}\033[0m\n")

    # Print training time in minutes (grey color)
    print(f"\033[90mTraining time: {round((time.time() - start_time)/60, 2)} minutes\033[0m")
    time.sleep(1)
    
if __name__ == "__main__":
    main()
