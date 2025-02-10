# CFL: Clustering-based Federated Learning Framework

**CFL** is a Federated Learning (FL) framework designed to dynamically cluster clients based on their data distribution, optimizing collaboration and model performance. CFL addresses distribution shifts in FL by ensuring efficient clustering and test-time adaptation, without requiring prior knowledge of client distributions.

## Key Features
- **Unsupervised Client Clustering:** Clients are grouped dynamically using descriptor extraction and clustering, handling all four common types of distribution shifts:
  - Feature distribution shift: The marginal distributions \(P(X)\) vary across clients.
  - Label distribution shift: The marginal distributions \(P(Y)\) vary across clients.
  - Concept shift (Same features, different label): The conditional distributions \(P(Y|X)\) vary across clients.
  - Concept shift (Same label, different features): The conditional distributions \(P(X|Y)\) vary across clients.

- **Test-Time Adaptation:** Enables unseen and unlabeled clients to be assigned to the most suitable cluster-specific model during inference.
- **Scalability & Efficiency:** Maintains low communication and computational overhead, making it practical for large-scale FL deployments.


## Installation

1. Clone the repository:
   ```
   git clone https://github.com/dariofenoglio98/CFL.git
   cd CFL
   ```
2. Set Conda Environment and necessary Libraries
    ```
    conda env create -f environment.yml
    ```
3. or install only the required Python packages
    ```
    pip install -r requirements.txt
    ```

## Run Experiments
Set desired training configuration in 'public/config.py'. 
Run: 
```
bash run.sh 
```

## License
This project is licensed under the MIT License – see the LICENSE.md file for details.

## Citation
...
