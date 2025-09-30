
"""
central_fedavg.py
=================
End-to-end simulation of **centralised** Federated Averaging (FedAvg)
without Hydra, Flower networking, or custom engine abstractions.  Each
client runs locally in a single Python process; after every round the
server aggregates their weights with a weighted mean.

Assumptions
-----------
* The **public** package you already use is available and unchanged:
  - `public.config`   → `cfg`   (hyper-parameters)
  - `public.utils`    → `utils` (helper functions)
  - `public.models`   → `models` (model zoo, train/test helpers)
* Per-client datasets live in `../data/cur_datasets/` as NumPy files,
  exactly as in your current workflow.

Run
---
```bash
python central_fedavg.py --fold 0
```

Feel free to tweak any `cfg.*` value (e.g. `n_rounds`, `patience`,
`local_epochs`) and re-run.
"""
from __future__ import annotations
from sklearn.preprocessing import MinMaxScaler
import random
import json 
import argparse
import copy
import os
import time
from collections import OrderedDict
from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator # type: ignore

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# Local imports (unchanged from your code-base)
# ---------------------------------------------------------------------
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import public.config as cfg
import public.utils as utils
import public.models as models
# ---------------------------------------------------------------------


# -------------------------- FedAvg helpers ---------------------------
class client_descr_scaling:
    def __init__(self, 
                 scaling_method: int = 1, 
                 scaler = None, # MinMaxScaler() or StandardScaler()
                 *args,
                 **kwargs):
        self.scaling_method = scaling_method
        self.scaler = scaler
        self.scalers = None
        self.fitted = False 
        if cfg.selected_descriptors == 'Px':
            self.descriptors_dim = [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors
            self.num_scalers = cfg.n_latent_space_descriptors
        elif cfg.selected_descriptors == 'Py':
            self.descriptors_dim = [cfg.len_metric_descriptor] * cfg.n_metrics_descriptors 
            self.num_scalers = cfg.n_metrics_descriptors
        elif cfg.selected_descriptors == 'Px_cond':
            self.descriptors_dim = [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors * 2
            self.num_scalers = cfg.n_latent_space_descriptors * 2
        elif cfg.selected_descriptors == 'Pxy_cond':
            self.descriptors_dim = [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors * 2 + [cfg.len_metric_descriptor] * cfg.n_metrics_descriptors
            self.num_scalers = cfg.n_latent_space_descriptors * 2 + cfg.n_metrics_descriptors
        elif cfg.selected_descriptors == 'Px_label_long':
            self.descriptors_dim = [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors * (cfg.n_classes + 1)
            self.num_scalers = cfg.n_latent_space_descriptors * (cfg.n_classes + 1)
        elif cfg.selected_descriptors == 'Px_label_short':
            self.descriptors_dim = [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors * 2
            self.num_scalers = cfg.n_latent_space_descriptors * 2
        else:
            self.descriptors_dim = [cfg.len_metric_descriptor] * cfg.n_metrics_descriptors + [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors
            self.num_scalers = cfg.n_metrics_descriptors + cfg.n_latent_space_descriptors

        print(f"n scalers: {self.num_scalers} - desc dim {self.descriptors_dim}")

    def scale(self, client_descr: np.ndarray = None) -> np.ndarray:
        # Normalize by group of descriptors
        if self.scaling_method == 1:
            if self.scalers is None:
                self.scalers = [copy.deepcopy(self.scaler) for _ in range(self.num_scalers)]
                self.dim = client_descr.shape[1]
             
            if self.fitted:
                if client_descr.shape[1] != self.dim:
                    raise ValueError("Client descriptors dimension mismatch!")
                scaled_client_descr = np.zeros(client_descr.shape)
                start_idx = 0
                for i, (scaler, descr_dim) in enumerate(zip(self.scalers, self.descriptors_dim)):
                    end_idx = start_idx + descr_dim
                    single_client_descr = client_descr[:, start_idx:end_idx]
                    scaled_client_descr[:, start_idx:end_idx] = scaler.transform(
                        single_client_descr.reshape(-1, 1)).reshape(single_client_descr.shape)
                    start_idx = end_idx
            else:
                self.fitted = True
                scaled_client_descr = np.zeros(client_descr.shape)
                start_idx = 0
                for i, (scaler, descr_dim) in enumerate(zip(self.scalers, self.descriptors_dim)):
                    end_idx = start_idx + descr_dim
                    single_client_descr = client_descr[:, start_idx:end_idx]
                    scaled_client_descr[:, start_idx:end_idx] = scaler.fit_transform(
                        single_client_descr.reshape(-1, 1)).reshape(single_client_descr.shape)
                    start_idx = end_idx
                
            return scaled_client_descr
        
        elif self.scaling_method == 2:
            # TODO weighted scaling
            return None
        
        elif self.scaling_method == 3:
            # No scaling
            return client_descr
        
        else:
            print("Invalid scaling method!")
            return None

def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """Load NumPy weights into *model* (in-place)."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """Return model weights as a list of **NumPy** arrays (CPU)."""
    return [v.cpu().numpy() for _, v in model.state_dict().items()]


def aggregate(client_results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """Weighted FedAvg aggregation."""
    total_samples = sum(n for _, n in client_results)
    n_layers = len(client_results[0][0])
    aggregated: List[np.ndarray] = []
    for layer in range(n_layers):
        layer_sum = sum(params[layer] * n for params, n in client_results)
        aggregated.append(layer_sum / total_samples)
    return aggregated

def assign_cluster_if_needed(
        client: FLClient,
        params_for_desc: List[np.ndarray],
        rnd: int,
        cluster_labels_inference: dict[int, int] = {},
        cluster_centroids_dict: dict[int, np.ndarray] = {},
        max_latent_space: float = 2.0,
        descriptors_scaler: client_descr_scaling = None,
) -> int:
    """
    Return the cluster id for `client`.  
    If that client has never been clustered, compute its descriptors,
    pick the closest centroid and *store* the mapping so we never
    recompute it.
    """
    if client.client_id not in cluster_labels_inference:
        # 1. descriptor extraction                                        <-- NEW
        desc = client.descriptor_extraction(params_for_desc, rnd,
                                            max_latent_space=max_latent_space)
        
        desc_scaled = descriptors_scaler.scale(desc.reshape(1, -1))

        # 2. choose nearest centroid                                      <-- NEW
        label = min(cluster_centroids_dict,
                    key=lambda k: np.linalg.norm(
                        desc_scaled - cluster_centroids_dict[k]))
        cluster_labels_inference[client.client_id] = label               # cache
    return cluster_labels_inference[client.client_id], cluster_labels_inference

# ---------------------------- Client --------------------------------

class FLClient:
    """A *very* light-weight federated client."""

    def __init__(self, model: torch.nn.Module, client_id: int, device: torch.device):
        self.model = model.to(device)
        self.client_id = client_id
        self.device = device

        # For quick post-hoc plotting
        self.metrics = {"round": [], "loss": [], "accuracy": []}

        # Concept drift support (optional)
        self.drifting = cfg.training_drifting
        self.drifting_log = []
        if self.drifting:
            log = np.load("../data/cur_datasets/drifting_log.npy", allow_pickle=True).item()
            self.drifting_log = log[self.client_id]

    # ----------------------- data utilities -------------------------

    def _load_raw_numpy(self, cur_round: int):
        """Return raw dict with keys that match your pipeline."""
        if not self.drifting:
            path = f"../data/cur_datasets/client_{self.client_id}.npy"
        else:
            idx = max([i for i in self.drifting_log if i <= cur_round], default=0)
            path = f"../data/cur_datasets/client_{self.client_id}_round_{idx}.npy"
        return np.load(path, allow_pickle=True).item()

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        t = torch.tensor(arr, dtype=torch.float32 if arr.dtype != np.int64 else torch.int64)
        if utils.get_in_channels() == 1:
            t = t.unsqueeze(1)  # add channel dim if needed
        return t

    def _get_loaders(self, cur_round: int) -> tuple[DataLoader, DataLoader]:
        raw = self._load_raw_numpy(cur_round)
        if self.drifting:
            feats, labs = raw["features"], raw["labels"]
        else:
            feats, labs = raw["train_features"], raw["train_labels"]

        feats = self._to_tensor(feats)
        labs = self._to_tensor(labs).squeeze()

        tr_f, val_f, tr_l, val_l = train_test_split(
            feats,
            labs,
            test_size=cfg.client_eval_ratio,
            random_state=cfg.random_seed,
            stratify=labs,
        )

        # Optional data reduction for small-sample experiments
        if cfg.n_samples_clients > 0:
            tr_f, tr_l = tr_f[: cfg.n_samples_clients], tr_l[: cfg.n_samples_clients]

        train_ds = models.CombinedDataset(tr_f, tr_l, transform=None)
        val_ds = models.CombinedDataset(val_f, val_l, transform=None)
        return (
            DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=cfg.test_batch_size, shuffle=False),
        )

    # -------------------- FL primitives -----------------------------

    def fit(self, parameters: List[np.ndarray], cur_round: int, local_epochs: int, extract_d: bool = False, max_latent_space: float = 2.) -> Tuple[List[np.ndarray], int]:
        set_parameters(self.model, parameters)
        train_loader, _ = self._get_loaders(cur_round)
        
        # Extract descriptors
        descriptors = {}
        if extract_d:
            descriptors = models.ModelEvaluator(test_loader=train_loader, device=self.device).extract_descriptors(model=self.model, \
                                                        client_id=self.client_id, max_latent_space=max_latent_space)

        optim = torch.optim.SGD(self.model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
        for epoch in range(local_epochs):
            models.simple_train(
                model=self.model,
                device=self.device,
                train_loader=train_loader,
                optimizer=optim,
                epoch=epoch,
                client_id=self.client_id,
            )
        return get_parameters(self.model), len(train_loader.dataset), descriptors

    def descriptor_extraction(self, parameters: List[np.ndarray], cur_round: int, max_latent_space: float = 2.) -> dict:
        set_parameters(self.model, parameters)
        train_loader, _ = self._get_loaders(cur_round)
        
        # Extract descriptors
        descriptors = {}
        descriptors = models.ModelEvaluator(test_loader=train_loader, device=self.device).extract_descriptors(model=self.model, \
                                                        client_id=self.client_id, max_latent_space=max_latent_space)
        
        descriptors = json.loads(descriptors["latent_space_mean"]) + \
                                            json.loads(descriptors["latent_space_std"]) + \
                                            json.loads(descriptors["latent_space_mean_by_label"]) + \
                                            json.loads(descriptors["latent_space_std_by_label"])
        return np.array(descriptors)

    def evaluate(self, parameters: List[np.ndarray], cur_round: int) -> Tuple[float, int]:
        set_parameters(self.model, parameters)
        _, val_loader = self._get_loaders(cur_round)
        evaluator = models.ModelEvaluator(val_loader, device=self.device)
        loss, acc, f1, new_max_latent_space = evaluator.evaluate(self.model)

        # save small log (optional)
        self.metrics["round"].append(cur_round)
        self.metrics["loss"].append(loss)
        self.metrics["accuracy"].append(acc)
        os.makedirs("results/" + cfg.default_path, exist_ok=True)
        np.save(f"results/{cfg.default_path}/client_{self.client_id}_metrics.npy", self.metrics)

        print(f"Client {self.client_id:02d} | round {cur_round:03d} | val_loss={loss:.4f} | val_acc={acc:.4f}")
        return float(loss), len(val_loader.dataset), new_max_latent_space


# --------------------------- main loop ------------------------------


def main() -> None:
    parser = argparse.ArgumentParser("Centralised FedAvg baseline")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold")
    # parser.add_argument("--scaling", type=int, default=0, help="Scaling factor for partial aggregation")
    args = parser.parse_args()

    # Reproducibility & misc
    utils.set_seed(cfg.random_seed + args.fold)
    device = utils.check_gpu()
    exp_path = utils.create_folders()
    in_channels = utils.get_in_channels()

    # Global reference model (used only by the server)
    global_model = models.models[cfg.model_name](
        in_channels=in_channels,
        num_classes=cfg.n_classes,
        input_size=cfg.input_size,
    ).to(device)
    
    # descriptor scaler
    descriptors_scaler = client_descr_scaling(scaling_method=cfg.flux_CLIENT_SCALING_METHOD,
                                              scaler=MinMaxScaler(),
                                              )

    # Spawn *independent* clients
    clients: list[FLClient] = [
        FLClient(model=copy.deepcopy(global_model), client_id=i, device=device)
        for i in range(cfg.n_clients)
    ]

    global_params = get_parameters(global_model)

    best_loss, best_round, no_improvement = float("inf"), 0, 0
    history = {"round": [], "loss_val_avg": []}
    t0 = time.time()
    max_latent_space = 2.0
    client_descriptors = []

    for rnd in range(1, cfg.n_rounds + 1):
        print(f"\n{'='*70}\n         --> ROUND {rnd}/{cfg.n_rounds}\n{'='*70}")
        
        # ---------------- client selection ---------------------------
        n_total = len(clients)
        # n_participants = max(1, int(cfg.partial_aggregation_ratio[args.scaling] * n_total))
        n_participants = max(1, int(cfg.partial_aggregation_ratio * n_total))
        participants = random.sample(clients, n_participants)
        print(f"Using {n_participants}/{n_total} clients this round")

        # ---------------- local training ---------------------------
        client_results: List[Tuple[List[np.ndarray], int]] = []
        for client in participants:
            if rnd < 5:
                new_params, n_samples, _ = client.fit(global_params, rnd, cfg.local_epochs)
            # elif rnd == 5: # old version without participation
            #     new_params, n_samples, descriptors = client.fit(global_params, rnd, cfg.local_epochs, extract_d=True, max_latent_space=max_latent_space)
            #     client_descriptors.append(json.loads(descriptors["latent_space_mean"]) + \
            #                                 json.loads(descriptors["latent_space_std"]) + \
            #                                 json.loads(descriptors["latent_space_mean_by_label"]) + \
            #                                 json.loads(descriptors["latent_space_std_by_label"]))
            # else:
            #     set_parameters(global_model, cluster_specific_global_params[cluster_labels_inference[client.client_id]])
            #     new_params, n_samples, _ = client.fit(global_params, rnd, cfg.local_epochs, extract_d=False, max_latent_space=max_latent_space)
            elif rnd == 5:
                new_params, n_samples, descriptors = client.fit(global_params, rnd, cfg.local_epochs, extract_d=True, max_latent_space=max_latent_space)
                client_descriptors.append(json.loads(descriptors["latent_space_mean"]) + \
                                            json.loads(descriptors["latent_space_std"]) + \
                                            json.loads(descriptors["latent_space_mean_by_label"]) + \
                                            json.loads(descriptors["latent_space_std_by_label"]))
            else:
                # descriptors = client.descriptor_extraction(global_params, rnd, max_latent_space=max_latent_space)
                # Find closest cluster centroid if client id is not in cluster_labels_inference
                client_cluster, cluster_labels_inference = assign_cluster_if_needed(client, global_params, rnd, cluster_labels_inference, cluster_centroids_dict, max_latent_space, descriptors_scaler)
                # Local training 
                client_specific_params = cluster_specific_global_params[client_cluster]
                new_params, n_samples, _ = client.fit(client_specific_params, rnd, cfg.local_epochs, extract_d=False, max_latent_space=max_latent_space)
                
            client_results.append((new_params, n_samples))

        # ---------------- aggregation ------------------------------
        if rnd < 5:  # FedAvg
            global_params = aggregate(client_results)
            set_parameters(global_model, global_params)
            # save model
            os.makedirs(f"checkpoints/{exp_path}", exist_ok=True)
            torch.save(global_model.state_dict(), f"checkpoints/{exp_path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_server.pth")

        elif rnd == 5: # clustering
            client_descriptors = descriptors_scaler.scale(np.array(client_descriptors))
            
            nbrs = NearestNeighbors(n_neighbors=2).fit(client_descriptors)
            distances, _ = nbrs.kneighbors(client_descriptors)
            sorted_distances = np.sort(distances[:, 1])
            kneedle = KneeLocator(range(len(sorted_distances)), sorted_distances, curve='convex', direction='increasing')
            eps = sorted_distances[kneedle.elbow] * cfg.eps_scaling  
            clustering = DBSCAN(eps=eps, min_samples=2)
            dbscan_cluster_labels = clustering.fit_predict(client_descriptors)
            dbscan_valid_clusters = len(set(dbscan_cluster_labels)) - (1 if -1 in dbscan_cluster_labels else 0)
            cluster_labels = dbscan_cluster_labels.copy()
            next_cluster_label = dbscan_valid_clusters  # Start assigning new cluster numbers
            # Loop over the labels and assign new labels to noise points
            for i, label in enumerate(dbscan_cluster_labels):
                if label == -1:  # Noise point detected
                    cluster_labels[i] = next_cluster_label
                    next_cluster_label += 1  # Move to the next cluster number

            # Output the final eps, the number of clusters, and the new labels
            print(f"Number of clusters (including reassigned noise points): {len(set(cluster_labels))}")
            print(f"Cluster labels after reassigning noise points: {cluster_labels}")
            real_n_clusters = np.load(f'../data/cur_datasets/n_clusters.npy').item()

            cluster_centroids_dict = utils.calculate_centroids(client_descriptors, clustering, cluster_labels)

            # Update results and assign clusters
            n_clusters = max(cluster_labels) + 1
            # Save cluster labels
            # cluster_labels_inference = {i: cluster_labels[i] for i in range(cfg.n_clients)}
            cluster_labels_inference = {client.client_id: cluster_labels[i] for i, client in enumerate(participants)}
            np.save(f'results/{exp_path}/cluster_labels_inference_{cfg.non_iid_type}_n_clients_{cfg.n_clients}.npy', cluster_labels_inference)
    
            # Split aggregation into clusters
            client_clusters_results = {i: [] for i in range(n_clusters)}
            # for i in range(cfg.n_clients):
            for idx, client in enumerate(participants):
                cur_cluster = cluster_labels_inference[client.client_id]
                client_clusters_results[cur_cluster].append(client_results[idx])

            # Aggregate each cluster
            cluster_specific_global_params = {}
            for cl, param in client_clusters_results.items():
                cluster_specific_global_params[cl] = aggregate(param)
                
            # calculate all cluster_labels_inference
            for client in clients:
                client_cluster, cluster_labels_inference = assign_cluster_if_needed(client, global_params, rnd, cluster_labels_inference, cluster_centroids_dict, max_latent_space, descriptors_scaler)

            
        else:  # FedAvg with clusters - 
            # Split aggregation into clusters
            # client_clusters_results = {i: [] for i in range(n_clusters)}
            # for i in range(cfg.n_clients):
            #     cur_cluster = cluster_labels_inference[i]
            #     client_clusters_results[cur_cluster].append(client_results[i])

            for idx, client in enumerate(participants):
                cl, cluster_labels_inference = assign_cluster_if_needed(client, global_params, rnd, cluster_labels_inference, cluster_centroids_dict, max_latent_space, descriptors_scaler)
                client_clusters_results[cl].append(client_results[idx])

            # Aggregate each cluster
            cluster_specific_global_params = {}
            for cl, param in client_clusters_results.items():
                cluster_specific_global_params[cl] = aggregate(param)
            
            # save cluster-specific global model
            for cl, params in cluster_specific_global_params.items():
                set_parameters(global_model, params)
                os.makedirs(f"checkpoints/{exp_path}", exist_ok=True)
                torch.save(global_model.state_dict(), f"checkpoints/{exp_path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_cluster_{cl}.pth")

                
        # ---------------- validation -------------------------------
        val_losses, sizes = [], []
        max_latents = []
        for client in participants:
            if rnd < 5:
                loss, n, new_max_latent_space = client.evaluate(global_params, rnd)
            else:
                loss, n, new_max_latent_space = client.evaluate(cluster_specific_global_params[cluster_labels_inference[client.client_id]], rnd)
            val_losses.append(loss)
            sizes.append(n)
            max_latents.append(new_max_latent_space)
        
        if rnd == 5:
            max_latent_space = 1.02*max(max_latents)

        w_loss = sum(l * n for l, n in zip(val_losses, sizes)) / sum(sizes)
        history["round"].append(rnd)
        history["loss_val_avg"].append(w_loss)
        print(f"Aggregated val_loss = {w_loss:.4f}")

        # ---------------- early stopping ---------------------------
        # if w_loss < best_loss:
        #     best_loss, best_round = w_loss, rnd
        #     no_improvement = 0
        #     ckpt_dir = os.path.join(exp_path, "checkpoints")
        #     os.makedirs(ckpt_dir, exist_ok=True)
        #     torch.save(global_model.state_dict(), os.path.join(ckpt_dir, f"model_round_{rnd}.pth"))
        # else:
        #     no_improvement += 1

        # if no_improvement >= cfg.patience:
        #     print(f"Early stopping triggered at round {rnd}.")
        #     break

    # ---------------- post-training -------------------------------
    print("\nFinal evaluation with the *best* global cluster-specific model")

   # Read cluster centroids from json - for test-time inference
    cluster_centroids = np.load(f'results/{exp_path}/centroids_{cfg.non_iid_type}_n_clients_{cfg.n_clients}.npy', allow_pickle=True).item()
    if cfg.selected_descriptors in ["Px_cond", "Pxy_cond", "Px_label_long", "Px_label_short"]:
        cluster_centroids = {label: centroid[:cfg.n_latent_space_descriptors*cfg.len_latent_space_descriptor] for label, centroid in cluster_centroids.items()}
        print(f"\033[93mCluster centroids: {cluster_centroids}\033[0m\n") # only latent space

    # Read cluster assignement during training for inference (known)
    print(f"\033[93mRead cluster assignement during training for inference\033[0m\n")
    print(f"\033[93mCluster labels: {cluster_labels_inference}\033[0m\n")
    
    # Load global model for evaluation
    evaluation_model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                          input_size=cfg.input_size).to(device)
    evaluation_model.load_state_dict(torch.load(f"checkpoints/{exp_path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_server.pth", weights_only=False))

    # Evaluate the model on the client datasets    
    losses, accuracies = [], []
    losses_known, accuracies_known = [], []
    for client_id in range(cfg.n_clients):
        test_x, test_y = [], []
        if not cfg.training_drifting:
            cur_data = np.load(f'../data/cur_datasets/client_{client_id}.npy', allow_pickle=True).item()
            test_x = torch.tensor(cur_data['test_features'], dtype=torch.float32) if in_channels == 3 else torch.tensor(cur_data['test_features'], dtype=torch.float32).unsqueeze(1)
            test_y = torch.tensor(cur_data['test_labels'], dtype=torch.int64)
        
        # Create test dataset and loader
        test_dataset = models.CombinedDataset(test_x, test_y, transform=None)
        test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)
    
        # # --- Test-time inference: check closest cluster ---
        # # Extract descriptors, scaling
        # descriptors = models.ModelEvaluator(test_loader=test_loader, device=device).extract_descriptors_inference(
        #                                             model=evaluation_model, max_latent_space=max_latent_space)
        
        # if cfg.selected_descriptors in ["Px_cond", "Pxy_cond", "Px_label_long", "Px_label_short"]:
        #     descriptors = descriptors_scaler.scale(descriptors.reshape(1,-1))
        #     descriptors = descriptors[:, :cfg.n_latent_space_descriptors*cfg.len_latent_space_descriptor] # only latent space
    
        # # Find the closest cluster centroid
        # client_cluster = min(cluster_centroids, key=lambda k: np.linalg.norm(descriptors - cluster_centroids[k]))
        
        # # Load respective cluster model
        # cluster_model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
        #                                 input_size=cfg.input_size).to(device)
        # cluster_model.load_state_dict(torch.load(f"checkpoints/{exp_path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_cluster_{client_cluster}.pth", weights_only=False))
        
        # # Evaluate
        # loss_test, accuracy_test = models.simple_test(cluster_model, device, test_loader)
        # print(f"\033[93mClient {client_id} - Test Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f} - Closest centroid {client_cluster}\033[0m")
        # accuracies.append(accuracy_test)
        # losses.append(loss_test)
        
        
        # --- Participating clients: assign known cluster ---
        client_cluster = cluster_labels_inference[client_id]

        # Load respective cluster model
        cluster_model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                        input_size=cfg.input_size).to(device)
        cluster_model.load_state_dict(torch.load(f"checkpoints/{exp_path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_cluster_{client_cluster}.pth", weights_only=False))
        
        # Evaluate
        loss_test, accuracy_test = models.simple_test(cluster_model, device, test_loader)
        print(f"\033[93mClient (known) {client_id} - Test Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f} - Closest centroid {client_cluster}\033[0m")
        accuracies_known.append(accuracy_test)
        losses_known.append(loss_test)

    # print average loss and accuracy
    # print(f"\n\033[93mAverage Loss: {np.mean(losses):.3f}, Average Accuracy: {np.mean(accuracies)*100:.2f}\033[0m")
    print(f"\033[93mAverage Loss (known): {np.mean(losses_known):.3f}, Average Accuracy (known): {np.mean(accuracies_known)*100:.2f}\033[0m")
    
    # Save metrics as numpy array
    metrics = {
        # "loss": losses,
        # "accuracy": accuracies,
        # "average_loss": np.mean(losses),
        # "average_accuracy": np.mean(accuracies),
        "loss_known": losses_known,
        "accuracy_known": accuracies_known,
        "average_loss_known": np.mean(losses_known),
        "average_accuracy_known": np.mean(accuracies_known),
        "time": round((time.time() - t0)/60, 2),
        "identified_clusters": n_clusters,
        "real_clusters": real_n_clusters,
    }
    np.save(f'results/{exp_path}/test_metrics_fold_{args.fold}.npy', metrics)
     
    print(f"Total training time: {(time.time() - t0)/60:.2f} min")


if __name__ == "__main__":
    main()
