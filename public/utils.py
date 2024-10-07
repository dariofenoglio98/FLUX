import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from typing import List
import public.config as cfg

# Create folders
def create_folders():
    os.makedirs(f"results/{cfg.random_seed}/{cfg.model_name}/{cfg.dataset_name}/{cfg.drifting_type}", exist_ok=True)
    os.makedirs(f"histories/{cfg.random_seed}/{cfg.model_name}/{cfg.dataset_name}/{cfg.drifting_type}", exist_ok=True)
    os.makedirs(f"checkpoints/{cfg.random_seed}/{cfg.model_name}/{cfg.dataset_name}/{cfg.drifting_type}", exist_ok=True)
    os.makedirs(f"images/{cfg.random_seed}/{cfg.model_name}/{cfg.dataset_name}/{cfg.drifting_type}", exist_ok=True)

    return f"{cfg.random_seed}/{cfg.model_name}/{cfg.dataset_name}/{cfg.drifting_type}"

# define device
def check_gpu():
    torch.manual_seed(cfg.random_seed)
    if cfg.gpu == -1:
        device = 'cpu'
    elif torch.cuda.is_available():
        device = 'cuda:' + str(cfg.gpu)
        torch.cuda.manual_seed_all(cfg.random_seed) 
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.mps.manual_seed(cfg.random_seed)
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    return device

# plot and save plot on server side
def plot_loss_and_accuracy(
        loss: List[float],
        accuracy: List[float],
        show: bool = True):
    
    # # Plot loss separately
    # plt.figure(figsize=(12, 6))
    # plt.plot(loss, label='Loss', color='blue')
    # min_loss_index = loss.index(min(loss))
    # plt.scatter(min_loss_index, loss[min_loss_index], color='red', marker='*', s=100, label='Min Loss')
    
    # # Labels and title for loss
    # plt.xlabel('Rounds')
    # plt.ylabel('Loss')
    # plt.title('Distributed Loss (Weighted Average on Test-Set)')
    # plt.legend()
    
    # # Save the loss plot
    # loss_plot_path = f"images/{cfg.random_seed}/{cfg.model_name}/{cfg.dataset_name}/{cfg.drifting_type}/{cfg.non_iid_type}_loss_n_clients_{cfg.n_clients}_n_rounds_{cfg.n_rounds}.png"
    # plt.savefig(loss_plot_path)
    # if show:
    #     plt.show()




    plt.plot(loss, label='Loss')
    plt.plot(accuracy, label='Accuracy')
    min_loss_index = loss.index(min(loss))
    max_accuracy_index = accuracy.index(max(accuracy))
    print(f"\n\033[1;34mServer Side\033[0m \nMinimum Loss occurred at round {min_loss_index + 1} with a loss value of {loss[min_loss_index]:.3f} \n \
          Maximum Accuracy occurred at round {max_accuracy_index + 1} with an accuracy value of {accuracy[max_accuracy_index]*100:.2f}\n")
    plt.scatter(min_loss_index, loss[min_loss_index], color='blue', marker='*', s=100, label='Min Loss')
    plt.scatter(max_accuracy_index, accuracy[max_accuracy_index], color='orange', marker='*', s=100, label='Max Accuracy')
    
    # Labels and title
    plt.xlabel('Rounds')
    plt.ylabel('Metrics')
    plt.title('Distributed Metrics (Weighted Average on Test-Set)')
    plt.legend()
    plt.savefig(f"images/{cfg.random_seed}/{cfg.model_name}/{cfg.dataset_name}/{cfg.drifting_type}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_n_rounds_{cfg.n_rounds}.png")

    plt.show() if show else None
    return min_loss_index+1, max_accuracy_index+1

# Cluster plot
def cluster_plot(X_reduced, cluster_labels, client_cid, server_round, name="KMeans"):
    # Create a folder to save the plots
    if not os.path.exists(f"images/{cfg.random_seed}/{cfg.model_name}/{cfg.dataset_name}/{cfg.drifting_type}/plots_descriptors"):
        os.makedirs(f"images/{cfg.random_seed}/{cfg.model_name}/{cfg.dataset_name}/{cfg.drifting_type}/plots_descriptors")
    
    # number of clusters - only number of cluster_labels - no string element
    # n_clusters = np.unique([n for n in cluster_labels if n.isnumeric()]).shape[0]
    n_clusters = np.unique(cluster_labels).shape[0]
    
    # Plot the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=cluster_labels, palette="deep", legend="full", s=100)
    plt.title(f'{name} ({n_clusters} Clusters) - R.{server_round}', fontsize=18)
    plt.xlabel('PC1', fontsize=16)
    plt.ylabel('PC2', fontsize=16)
    # Annotate client id
    for i, cid in enumerate(client_cid):
        plt.text(X_reduced[i, 0], X_reduced[i, 1], str(cid), fontsize=10, ha='right')
    # Save the plot
    plt.savefig(f"images/{cfg.random_seed}/{cfg.model_name}/{cfg.dataset_name}/{cfg.drifting_type}/plots_descriptors/{name.lower()}_cluster_visualization_{server_round}.png")
    plt.close()
    
# Plot the elbow and silhouette scores
def plot_elbow_and_silhouette(range_n_clusters, inertia, silhouette_scores, server_round):
    # Create a folder to save the plots
    if not os.path.exists(f"images/{cfg.random_seed}/{cfg.model_name}/{cfg.dataset_name}/{cfg.drifting_type}/plots_descriptors"):
        os.makedirs(f"images/{cfg.random_seed}/{cfg.model_name}/{cfg.dataset_name}/{cfg.drifting_type}/plots_descriptors")
        
    # Create figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))  # Two plots side by side, width is larger (20) to accommodate both plots

    # Plot inertia (Elbow Method) on the first subplot
    axs[0].plot(range_n_clusters, inertia, marker='o', label='Inertia')
    axs[0].set_title(f'Elbow Method (Optimal Cluster: {range_n_clusters[np.argmin(inertia)]}) - R.{server_round}', fontsize=18)
    axs[0].set_xlabel('Number of clusters', fontsize=16)
    axs[0].set_ylabel('Inertia', fontsize=16)

    # Plot silhouette scores on the second subplot
    axs[1].plot(range_n_clusters, silhouette_scores, marker='o', label='Silhouette Score')
    axs[1].set_title(f'Silhouette Scores (Optimal Cluster: {range_n_clusters[np.argmax(silhouette_scores)]}) - R.{server_round}', fontsize=18)
    axs[1].set_xlabel('Number of clusters', fontsize=16)
    axs[1].set_ylabel('Silhouette Score', fontsize=16)

    # Save the combined figure to the appropriate directory
    plt.savefig(f"images/{cfg.random_seed}/{cfg.model_name}/{cfg.dataset_name}/{cfg.drifting_type}/plots_descriptors/elbow_and_silhouette_{server_round}.png")
    plt.close()

# Get cur dataset in_channels
def get_in_channels():
    for file_name in ['../data/cur_datasets/client_1.npy', '../data/cur_datasets/client_1_round_-1.npy']:
        if os.path.exists(file_name):
            cur_data = np.load(file_name, allow_pickle=True).item()
            break
    cur_features = cur_data['train_features'] if not cfg.training_drifting else cur_data['features']

    return 3 if len(cur_features.size()) == 4 else 1

def set_seed(seed):
    # Set seed for torch
    torch.manual_seed(seed)
    
    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(cfg.random_seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set deterministic behavior for CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set PYTHONHASHSEED
    os.environ['PYTHONHASHSEED'] = str(seed)

# Calculate centroids
def calculate_centroids(data: np.ndarray,
                        clustering_method,
                        cluster_labels,
                        save=True):
    """
    Calculate the centroids of the clusters.
    
    Args:
    data: np.ndarray
        The data points.
    clustering_method: sklearn.cluster object
        The clustering method used.
    cluster_labels: np.ndarray
        The cluster labels.
    
    Returns:
        A dictionary containing the cluster label as the key and the centroid as the value.
    """
    
    # Kmeans
    if cfg.cfl_oneshot_CLIENT_CLUSTER_METHOD == 1:
        centroids = clustering_method.cluster_centers_
        centroids_dict = {label: np.array(centroid) for label, centroid in zip(np.unique(cluster_labels), centroids)}
    
    # DBSCAN and HDBSCAN
    elif cfg.cfl_oneshot_CLIENT_CLUSTER_METHOD in [2, 3]:
        centroids_dict = {}
        for label in np.unique(cluster_labels):
            cluster_points = data[cluster_labels == label]
            centroids_dict[label] = np.array(cluster_points.mean(axis=0))
    
    # Save
    if save:
        path = f"results/{cfg.random_seed}/{cfg.model_name}/{cfg.dataset_name}/{cfg.drifting_type}"
        np.save(f"{path}/centroids_{cfg.non_iid_type}_n_clients_{cfg.n_clients}.npy", centroids_dict, allow_pickle=True)
    
    return centroids_dict
