# Neuro-Fuzzy Based Diabetes Prediction with Activation Function Analysis 

## Folder Structure
```
FUZZY/
├── data/
│   ├── download_data.py      # Auto-downloads the diabetes dataset
│   └── diabetes.csv           # Dataset (auto-downloaded or manual)
├── anfis_module/
│   ├── __init__.py            # Package init
│   ├── membership.py          # Fuzzy membership functions
│   └── anfis.py               # ANFIS implementation (5-layer architecture)
├── neural_network.py          # Keras NN with activation function comparison
├── main.py                    # Main pipeline — run this file
├── requirements.txt           # Python dependencies
├── results_comparison.png     # Generated comparison plot
└── README.md                  # This file
```

## Setup Instructions

### 1. Install Python
Make sure you have Python 3.8 or higher installed.

### 2. Install Dependencies
Open a terminal in the FUZZY folder and run:
```bash
pip install -r requirements.txt
```

### 3. Get the Dataset
**Option A (automatic):** The script downloads it automatically when you run `main.py`.

**Option B (manual):**
1. Go to https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Download `diabetes.csv`
3. Place it in the `data/` folder

### 4. Run the Project
```bash
python main.py
```

## What This Project Does

1. **Loads** the Pima Indians Diabetes Dataset (768 patient records, 8 medical features)
2. **Preprocesses** the data (handles missing values, normalizes features, splits 80/20)
3. **Trains Neural Networks** with 4 different activation functions (Sigmoid, Tanh, ReLU, Leaky ReLU) and compares their accuracy
4. **Trains an ANFIS model** (Adaptive Neuro-Fuzzy Inference System) that combines neural network learning with fuzzy logic reasoning
5. **Runs a Hybrid Pipeline** where ANFIS fuzzifies the inputs (converts crisp values to fuzzy membership degrees) and feeds them into a Neural Network
6. **Plots a comparison chart** showing all model accuracies side by side

## How ANFIS Works in This System

ANFIS has 5 layers:
- **Layer 1 (Fuzzification):** Converts crisp input values into fuzzy membership degrees using Gaussian functions
- **Layer 2 (Rules):** Computes the firing strength of each fuzzy rule
- **Layer 3 (Normalization):** Normalizes the rule strengths so they sum to 1
- **Layer 4 (Consequent):** Computes weighted linear outputs for each rule
- **Layer 5 (Aggregation):** Sums all rule outputs to produce the final prediction

## References
- ANFIS Repository: https://github.com/twmeggs/anfis
- Dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- SIH 2025: https://www.sih.gov.in/sih2025PS
