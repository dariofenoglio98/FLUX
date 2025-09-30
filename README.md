# FLUX: Efficient Descriptor-Driven Clustered Federated Learning under Arbitrary Distribution Shifts

This is the official implementation of the paper accepted at **NeurIPS 2025**:

> **FLUX: Efficient Descriptor-Driven Clustered Federated Learning under Arbitrary Distribution Shifts**  
> Dario Fenoglio, Mohan Li, Pietro Barbiero, Nicholas D. Lane, Marc Langheinrich, Martin Gjoreski

---

## Overview

**FLUX** is a clustering-based Federated Learning (FL) framework that leverages compact, privacy-preserving descriptors to dynamically cluster clients and enable robust training under heterogeneous, non-IID data. Unlike previous CFL approaches, FLUX:
- requires **no prior knowledge** of the number of clusters or distribution types,  
- handles **all four major distribution shifts** during both training and test time, and  
- supports **test-time adaptation**, automatically assigning unseen and unlabeled clients to the most suitable cluster-specific models.  

Extensive experiments across six datasets (MNIST, FMNIST, CIFAR-10, CIFAR-100, CheXpert, Office-Home) show that FLUX improves robustness and scalability while keeping computation and communication overhead comparable to FedAvg.

---

## Key Features
- **Descriptor Extraction:** Client-side extraction of compact descriptors that approximate the 2-Wasserstein distance between distributions.  
  Captures:
  - Feature distribution shifts: \(P(X)\) varies across clients.  
  - Label distribution shifts: \(P(Y)\) varies across clients.  
  - Concept shifts (same features, different labels): \(P(Y|X)\) varies across clients.  
  - Concept shifts (same labels, different features): \(P(X|Y)\) varies across clients.  

- **Unsupervised Clustering:** Adaptive clustering of clients with no need to specify the number of clusters.  

- **Test-Time Adaptation:** Automatic assignment of unseen, unlabeled clients to cluster-specific models.  

- **Scalability & Efficiency:** Lightweight communication and computation overhead, comparable to FedAvg, enabling deployment with large client populations.  

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dariofenoglio98/CFL.git
   cd CFL
   ```
2. Set Conda Environment and necessary Libraries
    ```bash
    conda env create -f environment.yml
    ```
3. or install only the required Python packages
    ```bash
    pip install -r requirements.txt
    ```

---

## Runing Experiments
1. Configure your training setup in 'public/config.py'.
2. Lunch experiments with: 
   ```bash
   bash run.sh 
   ```
   Results and checkpoints will be saved under:
      - 'strategy/images/'
      - 'strategy/histories/'
      - 'strategy/results/'
      - 'strategy/checkpoints/'


---


## License
This project is licensed under the MIT License – see the LICENSE.md file for details.


---


## Citation
If you use this code, please cite our paper:
```bibtex
@inproceedings{fenoglio2025flux,
  title     = {FLUX: Efficient Descriptor-Driven Clustered Federated Learning under Arbitrary Distribution Shifts},
  author    = {Fenoglio, Dario and Li, Mohan and Barbiero, Pietro and Lane, Nicholas D. and Langheinrich, Marc and Gjoreski, Martin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

