import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os

import public.config as cfg

# Create folders
def create_folders():
    # Clean cache
    os.system(f"rm -r results/{cfg.model_name}/{cfg.dataset_name}")

    # Create directories for results
    os.makedirs(f"results/{cfg.model_name}/{cfg.dataset_name}")
    os.makedirs(f"histories/{cfg.model_name}/{cfg.dataset_name}", exist_ok=True)
    os.makedirs(f"checkpoints/{cfg.model_name}/{cfg.dataset_name}", exist_ok=True)
    os.makedirs(f"images/{cfg.model_name}/{cfg.dataset_name}", exist_ok=True)

# define device
def check_gpu(manual_seed=True, print_info=True):
    if manual_seed:
        torch.manual_seed(0)
    if torch.cuda.is_available():
        if print_info:
            print("CUDA is available")
        device = 'cuda'
        torch.cuda.manual_seed_all(0) 
    elif torch.backends.mps.is_available():
        if print_info:
            print("MPS is available")
        device = torch.device("mps")
        torch.mps.manual_seed(0)
    else:
        if print_info:
            print("CUDA is not available")
        device = 'cpu'
    return device

# plot and save plot on server side
def plot_loss_and_accuracy(loss, accuracy,  show=True):
    # read args
    rounds = cfg.n_rounds
    
    # Plot loss and accuracy
    plt.figure(figsize=(12, 6))

    plt.plot(loss, label='Loss')
    plt.plot(accuracy, label='Accuracy')
    min_loss_index = loss.index(min(loss))
    max_accuracy_index = accuracy.index(max(accuracy))
    print(f"\n\033[1;34mServer Side\033[0m \nMinimum Loss occurred at round {min_loss_index + 1} with a loss value of {loss[min_loss_index]:.3f} \nMaximum Accuracy occurred at round {max_accuracy_index + 1} with an accuracy value of {accuracy[max_accuracy_index]*100:.2f}\n")
    plt.scatter(min_loss_index, loss[min_loss_index], color='blue', marker='*', s=100, label='Min Loss')
    plt.scatter(max_accuracy_index, accuracy[max_accuracy_index], color='orange', marker='*', s=100, label='Max Accuracy')
    
    # Labels and title
    plt.xlabel('Rounds')
    plt.ylabel('Metrics')
    plt.title('Distributed Metrics (Weighted Average on Test-Set)')
    plt.legend()
    plt.savefig(f"images/{cfg.model_name}/{cfg.dataset_name}/training_{rounds}_rounds.png")
    if show:
        plt.show()
    return min_loss_index+1, max_accuracy_index+1

# Cluster plot
def cluster_plot(X_reduced, cluster_labels, client_cid, server_round, name="KMeans"):
    # Create a folder to save the plots
    if not os.path.exists(f"images/{cfg.model_name}/{cfg.dataset_name}/plots_descriptors"):
        os.makedirs(f"images/{cfg.model_name}/{cfg.dataset_name}/plots_descriptors")
    
    # number of clusters - only number of cluster_labels - no string element
    n_clusters = np.unique([n for n in cluster_labels if n.isnumeric()]).shape[0]
    
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
    plt.savefig(f"images/{cfg.model_name}/{cfg.dataset_name}/plots_descriptors/{name.lower()}_cluster_visualization_{server_round}.png")
    plt.close()
    
# Plot the elbow and silhouette scores
def plot_elbow_and_silhouette(range_n_clusters, inertia, silhouette_scores, server_round):
    # Create a folder to save the plots
    if not os.path.exists(f"images/{cfg.model_name}/{cfg.dataset_name}/plots_descriptors"):
        os.makedirs(f"images/{cfg.model_name}/{cfg.dataset_name}/plots_descriptors")
        
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
    plt.savefig(f"images/{cfg.model_name}/{cfg.dataset_name}/plots_descriptors/elbow_and_silhouette_{server_round}.png")
    plt.close()
