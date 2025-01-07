# read data json
import json
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
import argparse
import config as cfg


###################################################################################################
# ARGUMENTS
###################################################################################################
parser = argparse.ArgumentParser(description='Plot accuracy and derivative across folds')
parser.add_argument('--show', default=False, help='Show the plots', choices=[True, False])
parser.add_argument('--dataset', default='CIFAR10', help='Dataset name', choices=['CIFAR10', 'CIFAR100', 'MNIST', 'FMNIST'])
args = parser.parse_args()


###################################################################################################
# READ DATA
###################################################################################################
data = []
for i in range(cfg.k_folds):
    with open(f'{cfg.strategy}/histories/{cfg.default_path}/distributed_metrics_{i}.json') as f:
        data.append(json.load(f))



###################################################################################################
# CONFIDENCE INTERVALS AND DERIVATIVES - TRAINING ACCURACY
###################################################################################################
# Extract accuracy lists from each run
accuracy_runs = np.array([x['accuracy'] for x in data])

# Number of runs and steps
num_runs, num_steps = accuracy_runs.shape

# Compute mean accuracy per step
mean_accuracy = np.mean(accuracy_runs, axis=0)

# Compute standard error per step
sem_accuracy = stats.sem(accuracy_runs, axis=0)

# Compute 95% confidence intervals
confidence_level = 0.95
degrees_freedom = num_runs - 1
confidence_intervals = sem_accuracy * stats.t.ppf((1 + confidence_level) / 2., degrees_freedom)

# Compute derivatives for each run
derivatives = np.diff(accuracy_runs, axis=1) 

# Compute mean derivative per step
mean_derivative = np.mean(derivatives, axis=0)

# Compute standard error for derivatives
sem_derivative = stats.sem(derivatives, axis=0)

# Compute 95% confidence intervals for derivatives
confidence_intervals_derivative = sem_derivative * stats.t.ppf((1 + confidence_level) / 2., degrees_freedom)

# Define step indices
steps_accuracy = np.arange(1, num_steps + 1)
steps_derivative = np.arange(2, num_steps + 1)  

# Create a figure with two subplots
plt.figure(figsize=(7, 9))

import seaborn as sns
# Optional: Enhance plot aesthetics
sns.set(style="whitegrid")
plt.rcParams.update({
    'font.size': 14,             # Base font size
    'figure.figsize': (18, 8),   # Increased width for side-by-side plots
    'axes.labelsize': 16,        # Axis labels font size
    'axes.titlesize': 19,        # Titles font size
    'legend.fontsize': 14,       # Legend font size
    'xtick.labelsize': 14,       # X-ticks font size
    'ytick.labelsize': 14        # Y-ticks font size
})

# ---- Top Plot: Accuracy ----
plt.subplot(2, 1, 1)
plt.plot(steps_accuracy, mean_accuracy, marker='o', linestyle='-', color='blue', label='Mean Accuracy')
plt.fill_between(steps_accuracy,
                 mean_accuracy - confidence_intervals,
                 mean_accuracy + confidence_intervals,
                 color='blue', alpha=0.2, label='95% Confidence Interval')
plt.title(f'Mean Accuracy Over Rounds - {args.dataset}')
plt.xlabel('Training Rounds')
plt.ylabel('Accuracy')
plt.xticks(steps_accuracy)
plt.legend()
plt.grid(True)

# ---- Bottom Plot: Derivative of Accuracy ----
plt.subplot(2, 1, 2)
plt.plot(steps_derivative, mean_derivative, marker='x', linestyle='-', color='red', label='Mean Derivative')
plt.fill_between(steps_derivative,
                 mean_derivative - confidence_intervals_derivative,
                 mean_derivative + confidence_intervals_derivative,
                 color='red', alpha=0.2, label='95% Confidence Interval')
plt.title(f'Mean Derivative of Accuracy - {args.dataset}')
plt.xlabel('Training Rounds')
plt.ylabel('Change in Accuracy')
plt.xticks(steps_derivative)
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.show()
plt.savefig(f'{cfg.strategy}/images/{cfg.default_path}/accuracy_derivative_{cfg.non_iid_type}.png')
if args.show:
    plt.show()
else:
    plt.close()
