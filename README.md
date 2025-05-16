# FLUX: Efficient Descriptor-Driven Clustered Federated Learning

FLUX is a novel, efficient Clustered Federated Learning (CFL) framework designed to tackle the four most common types of distribution shifts encountered in real-world federated learning scenarios:

- Feature distribution shifts \((P(X))\)
- Label distribution shifts \((P(Y))\)
- Conditional concept shifts \((P(Y|X)\) and \(P(X|Y))\)

By leveraging compact, privacy-preserving client-side descriptors and an adaptive, density-based clustering algorithm, FLUX automatically discovers client clusters without any prior knowledge of the number of distributions, and supports test-time adaptation for truly unseen and unlabeled clients. FLUX maintains computational and communication costs on par with FedAvg while delivering significant accuracy gains across benchmarks.


## 📦 Features
- *Descriptor-Driven Clustering:* Extracts low-dimensional descriptors of each client’s empirical data distribution (marginal and class-conditional moments) to capture heterogeneity.
- *Automatic Cluster Discovery:* Utilizes an adaptive density-based clustering method to infer both the number of clusters and client assignments in an unsupervised manner.
- *Test-Time Adaptation:* Assigns previously unseen clients to the nearest cluster-specific model using label-agnostic descriptors, enabling robust performance on unlabeled data.
- *Minimal Overhead:* Adds negligible computational cost and only a small descriptor communicated per client (L/p ≪ 1) compared to standard FedAvg.
- *Scalable & Flexible:* Demonstrated scalability to hundreds of clients and robustness across MNIST, Fashion‑MNIST, CIFAR-10/100, and a real-world CheXpert medical dataset.


## 🚀 Installation

1. Clone the repository:
   ```
   git clone https://github.com/...
   cd ...
   ```
2. Create the Conda environment (recommended)
    ```
    conda env create -f environment.yml
    ```
3. (Optional) Install via pip
    ```
    pip install -r requirements.txt
    ```

## ⚙️ Configuration
All experimental settings are managed via the `public/config.py` and `run.sh` files:
- `dataset`: Choose from MNIST, FMNIST, CIFAR10, CIFAR100, or CheXpert.
- `model`: Select model architectures (e.g., simple LeNet, ResNet).
- `shift_type`: Specify one of \(P(X), P(Y), P(Y|X), P(X|Y)\), or combinations.
- `strategy`: Choose the method (e.g., FedAvg, FLUX)
- Heterogeneity levels: Control the severity of non-IID partitions (1–8).
- Training hyperparameters: Learning rate, batch size, local epochs, communication rounds, etc.



## 🏃‍♂️ Running Experiments
Once configured, launch experiments with:
```bash
bash run.sh
```
Results—including metrics, training history, visualisations, and model checkpoints—are saved under the `strategy/results`, `strategy/history`, `strategy/images`, and `strategy/checkpoints` directories, respectively.

## License
This project is licensed under the MIT License – see the LICENSE.md file for details.

## Citation
