# DW-ILP: Dynamic-Window Integer Linear Programming for Edge-AI Resource Allocation in G-ITS

This repository contains the simulation code for the experiments presented in the paper *"Dynamic Windowing for Edge-AI Resource Allocation in Green Intelligent Transportation Systems (G-ITS)"*. It implements the **Dynamic-Window Integer Linear Programming (DW-ILP) framework** and baseline schedulers for collaborative task offloading in consumer-edge vehicular environments.

## Key Features
- **Core Mechanisms**: Implements traffic-adaptive dynamic windowing and energy-aware constraint strengthening.
- **Scheduler Comparison**: Includes Greedy, Fixed-Window, Fixed-Window with pruning, Dynamic-Window, Dynamic-Window with pruning, and Race-mode schedulers.
- **Learning Baseline**: Provides a lightweight Graph Neural Network (GNN) model as a reference baseline.
- **Configurable Experiments**: Easily adjust task load, node resources, energy models, and scheduling policies via YAML configuration files.

## Requirements
- **Python**: >= 3.8
- **Main Dependencies**: See `requirements.txt` (install via `pip install -r requirements.txt`).
- An ILP solver (e.g., `ortools` or `mip`) is required for the core optimization; the code provides the interface.

## Quick Start

1.  **Clone the repo and install dependencies:**
bash
git clone <repository-url>
cd dw-ilp-repro
pip install -r requirements.txt
2.  **Run the main comparison experiment:**
The following command runs experiments under the default configuration, comparing all schedulers.
bash
python -m src.main --config config/default.yaml
3.  **Run other experiments:**
Different experiments from the paper can be executed by specifying corresponding configuration files:
bash
# Run scalability tests
python -m src.main --config config/exp_dynamic_pruning_large_v4.yaml
# Run experiments with the energy model enabled
python -m src.main --config config/exp_dynamic_race_large_v1.yaml
## Project Structure
dw-ilp-repro/
├── config/ # Experiment configuration files
│ ├── default.yaml # Main comparison experiment
│ ├── exp_dynamic_*.yaml # Dynamic pruning & large-scale experiments
│ └── ...
├── src/ # Source code
│ ├── main.py # Main experiment entry point
│ ├── simulator/ # Task and node generators
│ ├── scheduler/ # All scheduler implementations
│ ├── baselines/ # Baselines (Greedy, GNN, etc.)
│ ├── learning_baselines/ # GNN model definition & training
│ └── utils/ # Utility functions
└── results/ # Experiment results (generated after runs)
## Usage
All experiment parameters (e.g., number of tasks, arrival rate, number of nodes, window size, pruning thresholds) are defined in the YAML files under the `config/` directory. Modify the config file and re-run `src.main` to execute.

- Results (including average completion time, SLA violation rate, allocation time, energy consumption, etc.) are saved in JSON and CSV formats in the configured `output_dir`.
- The code is designed for reproducibility, with random seeds fixed for each run.

## Note
This implementation focuses on simulating the algorithmic workflow and comparative evaluation. System-level parameters (e.g., energy coefficients, transmission rates) are normalized. The reported results reflect relative performance trends.
