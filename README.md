# Boston Housing Price Prediction — MLOps Assignment 1

Predicts house prices on the Boston Housing dataset using two classical
scikit-learn models: DecisionTreeRegressor and KernelRidge.

## Installation

```bash
# 1. Create and activate a conda environment
conda create -n mlops python=3.10 -y
conda activate mlops

# 2. Install dependencies
pip install -r requirements.txt
```

## How to run

```bash
# Decision Tree model
python train.py

# Kernel Ridge model
python train2.py
```

Each script loads the Boston Housing data, trains the model, and prints the
test-set MSE and the 5-fold cross-validated average MSE.

## Branches
- `main` — README and merged code.
- `dtree` — Decision Tree (train.py) + misc.py + requirements.txt.
- `kernelridge` — Kernel Ridge (train2.py) + GitHub Actions CI.