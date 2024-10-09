import numpy as np
import config as cfg


def calculate_mean_std_metrics(metrics):
    # Initialize a dictionary to hold the means of all keys
    mean_std_metrics = {}

    # Extract the keys from the first entry in the metrics list
    keys = metrics[0].keys()

    for key in keys:
        # Check if the value corresponding to the key is a list
        if isinstance(metrics[0][key], list):
            # Compute the mean for each element across all entries for this key
            mean_value = np.mean([metric[key] for metric in metrics], axis=0).tolist()
            std_value = np.std([metric[key] for metric in metrics], axis=0).tolist()
        else:
            # Compute the mean for the scalar values across all entries for this key
            mean_value = np.mean([metric[key] for metric in metrics])
            std_value = np.std([metric[key] for metric in metrics])
        
        # Store the computed mean in the mean_metrics dictionary with the '_mean' suffix
        mean_std_metrics[f'{key}_mean'] = mean_value
        mean_std_metrics[f'{key}_std'] = std_value
        
    return mean_std_metrics


# Load metrics from all folds
metrics = []
for i in range(cfg.k_folds):
    # Load metrics
    metrics.append(
        np.load(f'{cfg.strategy}/results/{cfg.default_path}/test_metrics_fold_{i}.npy',
                allow_pickle=True
                ).item()
        )

# Calculate the mean metrics
result = calculate_mean_std_metrics(metrics)

# Save the mean metrics to a file
np.save(f'{cfg.strategy}/results/{cfg.default_path}/mean_std_test_metrics.npy', result)


