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
import copy
import time
import torch
import argparse
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
    

# Config_client
def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
        "local_epochs": cfg.local_epochs,
        "tot_rounds": cfg.n_rounds,
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
        
        # In the first round sent the same model to all clients
        print(f"\033[93mRound {server_round} - Sending model to {len(clients)} clients\033[0m")
        if server_round == 1:
            return [(client, FitIns(parameters, config)) for client in clients]
        else:
            # Clustered training
            return [(client,
                        FitIns(self.aggregated_cluster_parameters[self.cluster_labels[client.cid]], \
                        config)) for client in clients]
        


    
    # Override aggregate_fit method to add saving functionality
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        
        
        ################################################################################
        # Clustered federated averaging aggregation
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

        cur_round_cids = [proxy.cid for proxy, _ in results]
        cur_cluster_labels = [res.metrics['cluster'] for _, res in results]
        # store cluster labels
        self.cluster_labels = {cid: cur_cluster_labels[i] for i, cid in enumerate(cur_round_cids)}
        
        # Clustered, update to cluster models
        n_clusters = max(cur_cluster_labels) + 1
        # client_clusters = {i: [] for i in range(n_clusters)}
        print(f"Unique clusters: {np.unique(cur_cluster_labels)}")
        client_clusters = {i: [] for i in np.unique(cur_cluster_labels)}
        for i in range(cfg.n_clients):
            client_clusters[cur_cluster_labels[i]].append(weights_results[i])

        # Aggregate each cluster
        for cl, param in client_clusters.items():
            self.aggregated_cluster_parameters[cl] = ndarrays_to_parameters(aggregate(param))

        
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
        for cl, params in self.aggregated_cluster_parameters.items():
            print(f"Saving round {server_round} aggregated_cluster_parameters_{cl}...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(params)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            # Overwrite the model
            torch.save(self.model.state_dict(), f"checkpoints/{self.path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_cluster_{cl}.pth")
        
        return None, aggregated_metrics
   
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
        return [(client,
                    EvaluateIns(self.aggregated_cluster_parameters[self.cluster_labels[client.cid]], \
                    config)) for client in clients]

 
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

        return loss_aggregated, metrics_aggregated

# Main
def main() -> None:
    # Get arguments
    parser = argparse.ArgumentParser(description='Clustered Federated Learning - Server')
    parser.add_argument('--fold', type=int, default=0, help='Fold number of the cross-validation')
    args = parser.parse_args()

    utils.set_seed(cfg.random_seed + args.fold)
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
        server_address=f"{cfg.ip}:{cfg.port}",   # 0.0.0.0 listens to all available interfaces
        config=fl.server.ServerConfig(num_rounds=cfg.n_rounds),
        strategy=strategy,
    )

    # Convert history to list
    loss = [k[1] for k in history.losses_distributed]
    accuracy = [k[1] for k in history.metrics_distributed['accuracy']]

    # Save loss and accuracy to a file
    print(f"Saving metrics to as .json in histories folder...")
    with open(f'histories/{exp_path}/distributed_metrics_{args.fold}.json', 'w') as f:
        json.dump({'loss': loss, 'accuracy': accuracy}, f)

    # Plot client training loss and accuracy
    utils.plot_all_clients_metrics()

    # Plots and Evaluation the model on the client datasets
    best_loss_round, best_acc_round = utils.plot_loss_and_accuracy(loss, accuracy, show=False)
        
    # Evaluate the model on the client datasets    
    losses, accuracies = [], []
    for client_id in range(cfg.n_clients):
        test_x, test_y = [], []
        if not cfg.training_drifting:
            cur_data = np.load(f'../data/cur_datasets/client_{client_id}.npy', allow_pickle=True).item()
            test_x = cur_data['test_features'] if in_channels == 3 else cur_data['test_features'].unsqueeze(1)
            test_y = cur_data['test_labels']
        else:
            cur_data = np.load(f'../data/cur_datasets/client_{client_id}_round_-1.npy', allow_pickle=True).item()
            test_x = cur_data['features'] if in_channels == 3 else cur_data['features'].unsqueeze(1)
            test_y = cur_data['labels']
        
        # Create test dataset and loader
        test_dataset = models.CombinedDataset(test_x, test_y, transform=None)
        test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)

        # Load respective cluster model
        known_cluster = cur_data['cluster'] 
        cluster_model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                          input_size=cfg.input_size).to(device)
        cluster_model.load_state_dict(torch.load(f"checkpoints/{exp_path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_cluster_{known_cluster}.pth", weights_only=False))
        
        # Evaluate
        loss_test, accuracy_test = models.simple_test(cluster_model, device, test_loader)
        print(f"\033[93mClient {client_id} - Test Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f} - Closest centroid {known_cluster}\033[0m")
        accuracies.append(accuracy_test)
        losses.append(loss_test)

    # print average loss and accuracy
    print(f"\n\033[93mAverage Loss: {np.mean(losses):.3f}, Average Accuracy: {np.mean(accuracies)*100:.2f}\033[0m")
    print(f"\033[90mTraining time: {round((time.time() - start_time)/60, 2)} minutes\033[0m")
    
    # Save metrics as numpy array
    metrics = {
        "loss": losses,
        "accuracy": accuracies,
        "average_loss": np.mean(losses),
        "average_accuracy": np.mean(accuracies),
        "time": round((time.time() - start_time)/60, 2)
    }
    np.save(f'results/{exp_path}/test_metrics_fold_{args.fold}.npy', metrics)    
    
    time.sleep(1)
    
if __name__ == "__main__":
    main()
