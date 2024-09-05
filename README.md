# CFL: Clustering-based Federated Learning Framework

**CFL** is a Federated Learning (FL) framework designed for dynamic clustering of clients during training based on their data distribution. This framework aims to enhance the efficiency and accuracy of FL by grouping clients with similar data characteristics. Additionally, CFL addresses data shift and drifting, considering variations in:

- \(P(x)\): Distribution of inputs
- \(P(y)\): Distribution of labels
- \(P(x|y)\): Conditional distribution of inputs given labels
- \(P(y|x)\): Conditional distribution of labels given inputs

## Features

- **Dynamic Client Clustering**: Clients are clustered during training based on their data distribution, optimizing collaboration and model convergence.
- **Data Shift & Drift Handling**: Adapts to data shifts across clients by considering the distributional changes in both input and output spaces (See ANDA for more information).
- **Scalable FL Framework**: Designed to scale across large numbers of clients while maintaining computation and communication efficiency.

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

## Usage - Tutorial

## Run Experiments

## License
This project is licensed under the MIT License – see the LICENSE.md file for details.

## Citation
