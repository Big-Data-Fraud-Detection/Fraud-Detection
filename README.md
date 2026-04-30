# Enterprise Fraud Detection
This project builds an end-to-end fraud detection pipeline for financial transactions using the PaySim synthetic mobile money dataset. The pipeline ingests 6.36 million transactions, engineers 34 behavioral, temporal, and balance-based features, and trains a LightGBM classifier to identify fraudulent transactions. The model handles extreme class imbalance (0.13% fraud rate) using cost-sensitive learning (scale_pos_weight = 773) and achieves a PR-AUC of 0.9986 and an F1 score of 0.9988 on a held-out test set. The pipeline is reproducible, tracked with MLflow, and includes SHAP explainability for individual predictions.

## Group Members
* Israel De La Mothe  - 816037345
* Kieron Seepersad    - 816041436 
* Dylan Sinkia        - 816042623 
* Tyrese Des Vignes   - 816042764 


## Setup
### Prerequisites

- Python 3.12 or higher
- pip (Python package installer)

### Install Dependancies
pip install -r requirements.txt


## Pipeline


## Reproducing Results


## Data
The PaySim synthetic mobile money dataset is available on Kaggle:
* URL: https://www.kaggle.com/datasets/ealaxi/paysim1
* Citation: Lopez-Rojas, E. and Elmir, A., "PaySim: Synthetic Financial Dataset for Fraud Detection", 2017

## AI Used
