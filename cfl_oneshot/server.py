"""
This code implements the FedAvg, when it starts, the server waits for the clients to connect. When the established number 
of clients is reached, the learning process starts. The server sends the model to the clients, and the clients train the 
model locally. After training, the clients send the updated model back to the server. Then client models are aggregated 
with FedAvg. The aggregated model is then sent to the clients for the next round of training. The server saves the model 
and metrics after each round.

This is code is set to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
run the appopriate client code (client.py).

METHOD: in the first rounds, FedAvg is used until the global model reaches a pre-defined accuracy. After that the 
current global model is utilized to extract client descriptors and perform the one-shot clustering. After the clustering,
each client receives only the assigned cluster model, which its local model will be aggregated with other client models
in the same clusters. The training continues until the end. 
"""

# Libraries
import json
import time
import torch
import numpy as np
from functools import reduce
from logging import WARNING
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import List, Tuple, Union, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import public.config as cfg
import public.utils as utils
import public.models as models

import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common.logger import log
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitRes,
    FitIns,
    Parameters,
    Scalar,
    Metrics,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
)

VERBOSE = True
# Define the max latent space as global variable
MAX_LATENT_SPACE = 2
CLIENT_SCALING_METHOD = 1
CLIENT_CLUSTER_METHOD = 1

# TODO DARIO
# 1. Save the cluster centroids (i already have the function in the dynamic
# 2. Add std accuracy per class - descriptors

# client_descr_scaled
def client_descr_scaling(
        client_descr: np.ndarray = None,
        scaling_method: int = 1,
        scaler = None
        ) -> np.ndarray:
    
    # 1. Normalize each column between 0 and 1 
    if scaling_method == 1:
        return scaler.fit_transform(client_descr)
    
    # 2. Normalize by group of descriptors
    elif scaling_method == 2:
        loss_pc = client_descr[:, :cfg.n_classes]
        latent_space = client_descr[:, cfg.n_classes:]
        scaled_loss_pc = scaler.fit_transform(loss_pc.reshape(-1, 1)).reshape(loss_pc.shape)  
        latent_space_pc = scaler.fit_transform(latent_space.reshape(-1, 1)).reshape(latent_space.shape)
        return np.hstack((scaled_loss_pc, latent_space_pc)) # TODO weighted
    else:
        print("Invalid scaling method!")
        return None

# Config_client
def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
        "local_epochs": cfg.local_epochs,
        "tot_rounds": cfg.n_rounds,
        "min_latent_space": 0,
        "max_latent_space": MAX_LATENT_SPACE,
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

# Custom strategy to save model after each round
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model # used for saving checkpoints
        self.path = path # saving model path

        self.aggregated_cluster_parameters = {} # [cluster_label] = model parameters
        self.cluster_labels = {}    # [cid] = cluster_label
        self.aggregated_parameters_global = None
        self.cluster_status = 0 # 0: not started, 1: to cluster, 2: done
        self.n_clusters = 1 # current number of clusters

    # Override configure_fit method to add custom configuration
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)      # Config sent to clients during training 
            
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # Clustered training
        if self.cluster_status == 2:
            return [(client,
                     FitIns(self.aggregated_cluster_parameters[self.cluster_labels[client.cid]], \
                            config)) for client in clients]
        
        # Global training if not yet clustering
        else:
            fit_ins = FitIns(parameters, config)
            return [(client, fit_ins) for client in clients]

    
    # Override aggregate_fit method to add saving functionality
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        
        ###############################################################################            
        # Clustering
        ###############################################################################
        if self.cluster_status == 1:
            print(f"\033[91mRound {server_round} - Clustering clients...\033[0m")
            self.cluster_status = 2

            # Extract & scale client descriptors and self-assigned client ids, FLWR cids
            client_descr = []
            client_id_plot = []
            client_cid_list = []
            for proxy, res in results:
                client_descr.append(json.loads(res.metrics["loss_pc"]) + \
                        json.loads(res.metrics["latent_space"]))
                client_id_plot.append(res.metrics["cid"])
                client_cid_list.append(proxy.cid)
            # scaling
            client_descr = client_descr_scaling(np.array(client_descr), 
                                                CLIENT_SCALING_METHOD,  
                                                MinMaxScaler())
            
            # Apply PCA to reduce the data to 2D for visualization
            X_reduced = PCA(n_components=2).fit_transform(client_descr)

            # KMeans
            if CLIENT_CLUSTER_METHOD == 1:
                range_n_clusters = range(2, cfg.n_classes)
                # Store inertia (sum of squared distances to centroids) and silhouette scores
                inertia, silhouette_scores = [], []
                for n_clusters in range_n_clusters:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=cfg.random_seed)
                    cluster_labels = kmeans.fit_predict(client_descr)
                    inertia.append(kmeans.inertia_)
                    # Calculate silhouette score and append
                    silhouette_avg = silhouette_score(client_descr, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
            
                # Plot inertia (Elbow Method) and silhouette scores
                utils.plot_elbow_and_silhouette(range_n_clusters, inertia, silhouette_scores, server_round)

                # Identify the best number of clusters based on the highest silhouette score
                best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
                kmeans_best = KMeans(n_clusters=best_n_clusters, random_state=cfg.random_seed)
                cluster_labels = kmeans_best.fit_predict(client_descr)
                utils.cluster_plot(X_reduced, cluster_labels, client_id_plot, server_round, name="KMeans")

            # DBSCAN
            elif CLIENT_CLUSTER_METHOD == 2:
                clustering = DBSCAN(eps=0.5, min_samples=2)  # You can tune the parameters `eps` and `min_samples`
                cluster_labels = clustering.fit_predict(client_descr)
                utils.cluster_plot(X_reduced, cluster_labels, client_id_plot, server_round, name="DBSCAN")
            
            # HDBSCAN
            elif CLIENT_CLUSTER_METHOD == 3:
                clustering = HDBSCAN(min_cluster_size=2)  # You can tune the parameters `min_cluster_size` and `min_samples`
                cluster_labels = clustering.fit_predict(client_descr) # Note negative values are outliers, here I make them positive for visualization
                if min(cluster_labels) < 0:
                    cluster_labels = cluster_labels + abs(min(cluster_labels))
                utils.cluster_plot(X_reduced, cluster_labels, client_id_plot, server_round, name="HDBSCAN")
            
            else:
                print("Invalid clustering method!")

            # Update results and assign clusters
            self.n_clusters = max(cluster_labels) + 1
            self.cluster_labels = {cid: cluster_labels[i] for i, cid in enumerate(client_cid_list)}

            print(f"\033[91mRound {server_round} - Identified {self.n_clusters} clusters ({self.cluster_labels.values()})\033[0m")

            # TODO: save centroids (see dynamic code because i already have the centroid funciton)
        
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
        
        # Clustered, update to cluster models
        if self.cluster_status == 2:
            # Split aggregation into clusters
            client_clusters = {i: [] for i in range(self.n_clusters)}
            for i, cluster in enumerate(self.cluster_labels.values()):
                client_clusters[cluster].append(weights_results[i])
            # Aggregate each cluster
            for cl, param in client_clusters.items():
                self.aggregated_cluster_parameters[cl] = ndarrays_to_parameters(aggregate(param))
        # Not yet clustered, update to global model
        else:
            self.aggregated_parameters_global = ndarrays_to_parameters(aggregate(weights_results))
        
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
        # Clustered, save cluster models
        if self.cluster_status == 2:
            # Save the aggregated cluster models
            for cl, params in enumerate(self.aggregated_cluster_parameters):
                print(f"Saving round {server_round} aggregated_cluster_parameters_{cl}...")
                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(params)
                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)
                # Overwrite the model
                torch.save(self.model.state_dict(), f"checkpoints/{self.path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_cluster_{cl}.pth")
        # Not yet clustered, save global model
        else:
            print(f"Saving round {server_round} aggregated_parameters...")
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(self.aggregated_parameters_global)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            # Overwrite the model 
            torch.save(self.model.state_dict(), f"checkpoints/{self.path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_server.pth")
        
        return self.aggregated_parameters_global, aggregated_metrics
   
    # Override configure_evaluate method to add custom configuration
    def configure_evaluate(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)      # Config sent to clients during evaluation
            
        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # Clustered eval
        if self.cluster_status == 2:
            return [(client,
                     EvaluateIns(self.aggregated_cluster_parameters[self.cluster_labels[client.cid]], \
                            config)) for client in clients]
                
        # Global eval if not yet clustering
        else:
            evaluate_ins = EvaluateIns(parameters, config)
            return [(client, evaluate_ins) for client in clients]

 
    # Override
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

        # Clustering requirements detection
        print(f"\033[93mRound {server_round} - Aggregated loss: {loss_aggregated} - Aggregated metrics: {metrics_aggregated}\033[0m")
        if metrics_aggregated["accuracy"] > cfg.th_accuracy and self.cluster_status == 0:
            self.cluster_status = 1
            print(f"\033[93mRound {server_round} - Will be clustering next round\033[0m")
        
        # Update the max_latent_space for the next round
        max_client_latent_space = max([res.metrics["max_latent_space"] for _, res in results])
        global MAX_LATENT_SPACE 
        MAX_LATENT_SPACE = 1.02 * max_client_latent_space 

        return loss_aggregated, metrics_aggregated

# Main
def main() -> None:

    utils.set_seed(cfg.random_seed)
    start_time = time.time()
    exp_path = utils.create_folders()
    device = utils.check_gpu()
    in_channels = utils.get_in_channels()
    model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                          input_size=cfg.input_size).to(device)

    # Define strategy
    strategy = SaveModelStrategy(
        # self defined
        model=model,
        path=exp_path,
        # super
        min_fit_clients=cfg.n_clients, # always all training
        min_evaluate_clients=cfg.n_clients, # always all evaluating
        min_available_clients=cfg.n_clients, # always all available
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
    )

    # Start Flower server for three rounds of federated learning
    history = fl.server.start_server(
        server_address="0.0.0.0:8098",   # 0.0.0.0 listens to all available interfaces
        config=fl.server.ServerConfig(num_rounds=cfg.n_rounds),
        strategy=strategy,
    )

    # Convert history to list
    loss = [k[1] for k in history.losses_distributed]
    accuracy = [k[1] for k in history.metrics_distributed['accuracy']]

    # Save loss and accuracy to a file
    print(f"Saving metrics to as .json in histories folder...")
    with open(f'histories/{exp_path}/distributed_metrics.json', 'w') as f:
        json.dump({'loss': loss, 'accuracy': accuracy}, f)

    # Plots and Evaluation the model on the client datasets
    best_loss_round, best_acc_round = utils.plot_loss_and_accuracy(loss, accuracy, show=False)
    model.load_state_dict(torch.load(f"checkpoints/{exp_path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_server.pth", weights_only=False))

    # The following evaluation is incorrect
    # TODO: load the saved evaluation global model - give to each new client the right cluster model
    # 1. Load the evaluation model
    # 2. Extract the descriptors () from each test client datasset
    # 3. Check for each client dataset which centroids is the closest
    # 4. Evaluate the respective (closest) cluster model on that client dataset 
    # 5. Aggregate the metrics and keep client metrics for further analysis

    # Generate server-side test dataset from clients test datasets
    test_x, test_y = [], []
    for client_id in range(cfg.n_clients):
        if not cfg.training_drifting:
            cur_data = np.load(f'../data/cur_datasets/client_{client_id+1}.npy', allow_pickle=True).item()
            test_x.append(cur_data['test_features']) if in_channels == 3 else test_x.append(cur_data['test_features'].unsqueeze(1))
            test_y.append(cur_data['test_labels'])
        else:
            cur_data = np.load(f'../data/cur_datasets/client_{client_id}_round_-1.npy', allow_pickle=True).item()
            test_x.append(cur_data['features']) if in_channels == 3 else test_x.append(cur_data['features'].unsqueeze(1))
            test_y.append(cur_data['labels'])
    test_dataset = models.CombinedDataset(np.concatenate(test_x, axis=0), \
                                          np.concatenate(test_y, axis=0), \
                                          transform=None)
    test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)

    loss_test, accuracy_test = models.simple_test(model, device, test_loader)
    print(f"\n\033[93mTest Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f}\033[0m")

    print(f"\033[90mTraining time: {round((time.time() - start_time)/60, 2)} minutes\033[0m")
    time.sleep(1)
    
if __name__ == "__main__":
    main()