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

    # Override aggregate_fit method to add saving functionality
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        
        
        ################################################################################
        # Client descriptors analysis
        ################################################################################
        # Check if client cid list available 
        if len(self.client_cid_list) == 0:
            for client in results:
                self.client_cid_list.append(client[0].cid)
        # print(f"Client cid list: {self.client_cid_list}")
        
        # Extract client descriptors
        client_descr = []
        for _, res in results:
            # res.num_examples, res.metrics
            loss_pc = json.loads(res.metrics["loss_pc"])
            latent_space = json.loads(res.metrics["latent_space"])
            # concatenate the descriptors
            client_descr.append(loss_pc + latent_space)
        
        # Normalizing descriptor matrix
        client_descr = np.array(client_descr)
        # 1. Normalize each column between 0 and 1 #TODO: test both 1 and 2 methods
        scaler = MinMaxScaler() # StandardScaler()
        scaled_data = scaler.fit_transform(client_descr)
        # print(f"scaled_data: {scaled_data}")
        # 2. Normalize by group of descriptors #TODO: test both 1 and 2 methods
        loss_pc = client_descr[:, :cfg.n_classes]
        latent_space = client_descr[:, cfg.n_classes:]
        scaled_loss_pc = scaler.fit_transform(loss_pc.reshape(-1, 1)).reshape(loss_pc.shape)  
        latent_space_pc = scaler.fit_transform(latent_space.reshape(-1, 1)).reshape(latent_space.shape)
        scaled_data_2 = np.hstack((scaled_loss_pc, latent_space_pc)) # TODO: we can also weight them (loss and latent) differently here, by multiplying them by a factor
        # print(f"scaled_data_2: {scaled_data_2}")
        
        # Visualization: reduce dimensionality to 2D and plot the data
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(scaled_data_2)
                    
        # Clustering
        # Transpose the data so that the clients (first dimension) become rows (3x20)
        transposed_data = scaled_data_2

        # Range of cluster numbers to try
        range_n_clusters = range(2, scaled_data_2.shape[0])  # Adjust based on your data size

        # Store inertia (sum of squared distances to centroids) and silhouette scores
        inertia = []
        silhouette_scores = []
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(transposed_data)
            inertia.append(kmeans.inertia_)
            
            # Calculate silhouette score and append
            silhouette_avg = silhouette_score(transposed_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        if not os.path.exists(f"images/{cfg.model_name}/{cfg.dataset_name}/plots_descriptors"):
            os.makedirs(f"images/{cfg.model_name}/{cfg.dataset_name}/plots_descriptors")
        # Plot inertia (Elbow Method)
        plt.figure(figsize=(10, 5))
        plt.plot(range_n_clusters, inertia, marker='o', label='Inertia')
        plt.title(f'Elbow Method (Optimal Cluster:{range_n_clusters[np.argmax(inertia)]}) - R.{server_round}', fontsize=18)  # Title font size 18
        plt.xlabel('Number of clusters', fontsize=16)  # X label font size 16
        plt.ylabel('Inertia', fontsize=16)  # Y label font size 16
        plt.savefig(f"images/{cfg.model_name}/{cfg.dataset_name}/plots_descriptors/elbow_method_{server_round}.png")

        # Plot silhouette scores
        plt.figure(figsize=(10, 5))
        plt.plot(range_n_clusters, silhouette_scores, marker='o', label='Silhouette Score')
        plt.title(f'Silhouette Scores (Optimal Cluster:{range_n_clusters[np.argmax(silhouette_scores)]}) - R.{server_round}', fontsize=18)  # Title font size 18
        plt.xlabel('Number of clusters', fontsize=16)  # X label font size 16
        plt.ylabel('Silhouette Score', fontsize=16)  # Y label font size 16
        plt.savefig(f"images/{cfg.model_name}/{cfg.dataset_name}/plots_descriptors/silhouette_scores_{server_round}.png")

        # Identify the best number of clusters based on the highest silhouette score
        best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]

        # Apply PCA to reduce the data to 2D for visualization
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(transposed_data)

        # Apply KMeans with the best number of clusters
        kmeans_best = KMeans(n_clusters=best_n_clusters, random_state=42)
        cluster_labels_best = kmeans_best.fit_predict(transposed_data)
        print(f"KMeans cluster_labels: {cluster_labels_best}")

        # Plot the identified clusters with the best score/number of clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=cluster_labels_best, palette="deep")
        plt.title(f'KMEANS ({best_n_clusters} Clusters) - R.{server_round}', fontsize=18)
        plt.xlabel('PC1', fontsize=16)
        plt.ylabel('PC2', fontsize=16)
        plt.savefig(f"images/{cfg.model_name}/{cfg.dataset_name}/plots_descriptors/kmeans_cluster_visualization_{server_round}.png")

        # Apply DBSCAN (doesn't require specifying the number of clusters)
        clustering = DBSCAN(eps=0.5, min_samples=2)  # You can tune the parameters `eps` and `min_samples`
        cluster_labels_dbscan = clustering.fit_predict(transposed_data)
        print(f"DBSCAN cluster_labels: {cluster_labels_dbscan}")

        # Plot the identified DBSCAN clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=cluster_labels_dbscan, palette="deep", legend="full")
        plt.title(f'DBSCAN ({len(set(cluster_labels_dbscan))} Clusters) - R.{server_round}', fontsize=18)
        plt.xlabel('PC1', fontsize=16)
        plt.ylabel('PC2', fontsize=16)
        plt.savefig(f"images/{cfg.model_name}/{cfg.dataset_name}/plots_descriptors/dbscan_cluster_visualization_{server_round}.png")
        
        # HDBSCAN
        clustering = HDBSCAN(min_cluster_size=2)  # You can tune the parameters `min_cluster_size` and `min_samples`
        cluster_labels_hdbscan = clustering.fit_predict(transposed_data)
        print(f"HDBSCAN cluster_labels: {cluster_labels_hdbscan}")
        
        # Plot the identified HDBSCAN clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=cluster_labels_hdbscan, palette="deep", legend="full")
        plt.title(f'HDBSCAN ({len(set(cluster_labels_hdbscan))} Clusters) - R.{server_round}', fontsize=18)
        plt.xlabel('PC1', fontsize=16)
        plt.ylabel('PC2', fontsize=16)
        plt.savefig(f"images/{cfg.model_name}/{cfg.dataset_name}/plots_descriptors/hdbscan_cluster_visualization_{server_round}.png")

        
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
        aggregated_parameters = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
            
            
        ################################################################################
        # Save model
        ################################################################################
        if aggregated_parameters is not None:

            print(f"Saving round {server_round} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            # Save the model
            torch.save(self.model.state_dict(), f"checkpoints/{cfg.model_name}/{cfg.dataset_name}/model_{server_round}.pth")
        
        return aggregated_parameters, aggregated_metrics
    
    
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
        
        # Update the max_latent_space for the next round
        max_client_latent_space = max([res.metrics["max_latent_space"] for _, res in results])
        global max_latent_space 
        max_latent_space = 1.02 * max_client_latent_space 
        print(f"Max latent space (evaluation): {max_latent_space}")

        return loss_aggregated, metrics_aggregated
    
    # # Override configure_fit method to add custom configuration
    # def configure_fit(
    #     self, server_round: int, parameters: Parameters, client_manager: ClientManager
    # ) -> List[Tuple[ClientProxy, FitIns]]:
    #     """Configure the next round of training."""
    #     config = {}
    #     if self.on_fit_config_fn is not None:
    #         # Custom fit config function provided
    #         config = self.on_fit_config_fn(server_round)      # Config sent to clients during training
    #         # print(f"Server Config: {config}")
    #     fit_ins = FitIns(parameters, config)

    #     # Sample clients
    #     sample_size, min_num_clients = self.num_fit_clients(
    #         client_manager.num_available()
    #     )
    #     clients = client_manager.sample(
    #         num_clients=sample_size, min_num_clients=min_num_clients
    #     )

    #     # Return client/config pairs
    #     return [(client, fit_ins) for client in clients]

    # # Override configure_evaluate method to add custom configuration
    # def configure_evaluate(
    #     self, server_round: int, parameters: Parameters, client_manager: ClientManager
    # ) -> List[Tuple[ClientProxy, EvaluateIns]]:
    #     """Configure the next round of evaluation."""
    #     # Do not configure federated evaluation if fraction eval is 0.
    #     if self.fraction_evaluate == 0.0:
    #         return []

    #     # Parameters and config
    #     config = {}
    #     if self.on_evaluate_config_fn is not None:
    #         # Custom evaluation config function provided
    #         config = self.on_evaluate_config_fn(server_round)      # Config sent to clients during evaluation
    #     evaluate_ins = EvaluateIns(parameters, config)

    #     # Sample clients
    #     sample_size, min_num_clients = self.num_evaluation_clients(
    #         client_manager.num_available()
    #     )
    #     clients = client_manager.sample(
    #         num_clients=sample_size, min_num_clients=min_num_clients
    #     )

    #     # Return client/config pairs
    #     return [(client, evaluate_ins) for client in clients]




# Main
def main() -> None:
    # parser = argparse.ArgumentParser(description="Flower")
    # parser.add_argument(
    #     "--rounds",
    #     type=int,
    #     default=20,
    #     help="Specifies the number of FL rounds",
    # )
    # args = parser.parse_args()

    # Start time
    start_time = time.time()

    # Create directories 
    utils.create_folders()

    # Create all data for clients
    utils.generate_dataset()

    # Pick the indipendent test set from each client
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
    model.load_state_dict(torch.load(f"checkpoints/{cfg.model_name}/{cfg.dataset_name}/model_{best_loss_round}.pth"))

    # Evaluate the model on the test set
    loss_test, accuracy_test = models.simple_test(model, device, test_loader)
    print(f"\n\033[93mTest Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f}\033[0m\n")

    # Print training time in minutes (grey color)
    print(f"\033[90mTraining time: {round((time.time() - start_time)/60, 2)} minutes\033[0m")
    time.sleep(1)
    
if __name__ == "__main__":
    main()
