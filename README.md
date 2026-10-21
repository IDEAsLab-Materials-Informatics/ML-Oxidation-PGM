# Data-driven Models and Insights on the Oxidation Behavior of Pt Group Metals Substituted \(\beta\)-NiAl

Official repository containing the source code, dataset, featurization scripts, and machine learning models for the manuscript [**"Data-driven Models and Insights on the Oxidation Behavior of Pt Group Metals Substituted \(\beta\)-NiAl"** [1.1].](https://doi.org/10.1016/j.commatsci.2026.115175)

This repository provides full end-to-end support for reproducing the results outlined in the paper, including the materials database, feature engineering routines, machine learning model development (ANN, Random Forest, and XGBoost), evaluation metrics, and visualization workflows [1.1].

---

##  Project Structure

The repository is structured as follows to map out the different feature spaces and model architectures used in our study:

```text
├── Data_collected_PGM.xlsx                  # Primary materials database for PGM-substituted β-NiAl
├── Featurization/                           # Feature engineering scripts
│   ├── Featurisation_CBFV_01.py             # Composition-based feature vector generation [1.2]
│   └── Featurisation_Composition_01.py      # Compositional descriptor computation [1.2]
├── ANN_Elemental_Features/                  # Neural Network models built on elemental features
├── ANN_Physical_Features/                   # Neural Network models built on physical features
├── RF_Elemental_Features/                   # Random Forest models built on elemental features
├── RF_Physical_Features/                    # Random Forest models built on physical features
├── XGBOOST_Elemental_Features/              # XGBoost models built on elemental features
├── XGBOOST_Physical_Features/               # XGBoost models built on physical features
├── Trained Models Physical Descriptors XGB/ # Saved production checkpoints for XGBoost physical models
├── CoSMoR/                                  # Component support/validation module
└── system_versions                          # Environment and dependencies version log
```

---

## Getting Started

### 1. Installation
Clone this repository to your local directory:
```bash
git clone https://github.com
cd ML-Oxidation-PGM
```

### 2. Prerequisites
Before executing code, please inspect the `system_versions` file to verify package compatibility. Core requirements include:
- Python 3.x
- NumPy & Pandas (Data processing)
- Scikit-Learn & XGBoost (Traditional ML)
- TensorFlow / Keras (Neural Networks)

---

## Execution Pipeline

### Step 1: Feature Engineering
Extract physical, chemical, and structural material features from `Data_collected_PGM.xlsx` by running the scripts within the `Featurization/` folder:

```bash
# Generate compositional descriptors
python Featurization/Featurisation_Composition_01.py

# Generate composition-based feature vectors
python Featurization/Featurisation_CBFV_01.py
```

### Step 2: Training & Reproducing Models
Navigate to any specific algorithm folder to initialize model training or validate baseline configurations on either *Elemental* or *Physical* feature distributions:

```bash
# Example: Navigating to XGBoost Physical Descriptors workflow
cd XGBOOST_Physical_Features
```

---


