
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

import argparse
import copy
import os
import time
from collections import OrderedDict
from typing import List, Tuple
import random

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

    def fit(self, parameters: List[np.ndarray], cur_round: int, local_epochs: int) -> Tuple[List[np.ndarray], int]:
        set_parameters(self.model, parameters)
        train_loader, _ = self._get_loaders(cur_round)

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
        return get_parameters(self.model), len(train_loader.dataset)

    def evaluate(self, parameters: List[np.ndarray], cur_round: int) -> Tuple[float, int]:
        set_parameters(self.model, parameters)
        _, val_loader = self._get_loaders(cur_round)
        evaluator = models.ModelEvaluator(val_loader, device=self.device)
        loss, acc, f1, _ = evaluator.evaluate(self.model)

        # save small log (optional)
        self.metrics["round"].append(cur_round)
        self.metrics["loss"].append(loss)
        self.metrics["accuracy"].append(acc)
        os.makedirs("results/" + cfg.default_path, exist_ok=True)
        np.save(f"results/{cfg.default_path}/client_{self.client_id}_metrics.npy", self.metrics)

        print(f"Client {self.client_id:02d} | round {cur_round:03d} | val_loss={loss:.4f} | val_acc={acc:.4f}")
        return float(loss), len(val_loader.dataset)


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

    # Spawn *independent* clients
    clients: list[FLClient] = [
        FLClient(model=copy.deepcopy(global_model), client_id=i, device=device)
        for i in range(cfg.n_clients)
    ]

    global_params = get_parameters(global_model)

    best_loss, best_round, no_improvement = float("inf"), 0, 0
    history = {"round": [], "loss_val_avg": []}
    t0 = time.time()

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
            new_params, n_samples = client.fit(global_params, rnd, cfg.local_epochs)
            client_results.append((new_params, n_samples))

        # ---------------- aggregation ------------------------------
        global_params = aggregate(client_results)
        set_parameters(global_model, global_params)
        os.makedirs(os.path.join(exp_path, "checkpoints"), exist_ok=True)
        torch.save(global_model.state_dict(), os.path.join(os.path.join(exp_path, "checkpoints"), f"model_round_{rnd}.pth"))

        # ---------------- validation -------------------------------
        val_losses, sizes = [], []
        for client in participants:
            loss, n = client.evaluate(global_params, rnd)
            val_losses.append(loss)
            sizes.append(n)

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
    print("\nFinal evaluation with the *best* global model")
    best_round = cfg.n_rounds  # Use the last round as the best one
    print(f"Best round: {best_round} | best val_loss: {best_loss:.4f}")

    ckpt_file = os.path.join(exp_path, "checkpoints", f"model_round_{best_round}.pth")
    global_model.load_state_dict(torch.load(ckpt_file, map_location=device))

    # Evaluate the model on the client datasets    
    losses, accuracies = [], []
    for client_id in range(cfg.n_clients):
        test_x, test_y = [], []
        if not cfg.training_drifting:
            cur_data = np.load(f'../data/cur_datasets/client_{client_id}.npy', allow_pickle=True).item()
            test_x = torch.tensor(cur_data['test_features'], dtype=torch.float32) if in_channels == 3 else torch.tensor(cur_data['test_features'], dtype=torch.float32).unsqueeze(1)
            test_y = torch.tensor(cur_data['test_labels'], dtype=torch.int64)
        
        # Create test dataset and loader
        test_dataset = models.CombinedDataset(test_x, test_y, transform=None)
        test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)

        # Evaluate on client
        loss_test, accuracy_test = models.simple_test(global_model, device, test_loader)
        print(f"\033[93mClient {client_id} - Test Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f}\033[0m")
        accuracies.append(accuracy_test)
        losses.append(loss_test)
    
    # Averaged accuracy across clients   
    print(f"\n\033[93mAverage Test Loss: {np.mean(losses):.3f}, Average Test Accuracy: {np.mean(accuracies)*100:.2f}\033[0m\n")
    
    # Save metrics as numpy array
    metrics = {
        "loss": losses,
        "accuracy": accuracies,
        "average_loss": np.mean(losses),
        "average_accuracy": np.mean(accuracies),
        "time": round((time.time() - t0)/60, 2)
    }
    np.save(f'results/{exp_path}/test_metrics_fold_{args.fold}.npy', metrics)
    
    print(f"Total training time: {(time.time() - t0)/60:.2f} min")


if __name__ == "__main__":
    main()
