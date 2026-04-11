"""
=============================================================
  SIH-Inspired Neuro-Fuzzy Disease Prediction System
  with Activation Function Analysis
=============================================================

Pipeline:
  Step 1 — Load & preprocess the Pima Indians Diabetes Dataset
  Step 2 — Train Neural Networks with 4 activation functions
  Step 3 — Train standalone ANFIS (Neuro-Fuzzy)
  Step 4 -- Hybrid pipeline: ANFIS fuzzification -> NN
  Step 5 — Print results summary with detailed metrics
  Step 6 — Plot comparison chart & confusion matrices

Dataset : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
ANFIS   : https://github.com/twmeggs/anfis
SIH ref : https://www.sih.gov.in/sih2025PS
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — avoids plt.show() blocking
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from anfis_module.anfis import ANFIS
from neural_network import train_nn_with_activation, build_model

# ── reproducibility ──────────────────────────────────────────
np.random.seed(42)

# Column names matching the Pima dataset
COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]


# ==============================================================
# STEP 1 : Load & Preprocess
# ==============================================================
def load_dataset():
    """Load CSV from data/ folder; download if missing."""
    data_path = os.path.join('data', 'diabetes.csv')

    if not os.path.exists(data_path):
        print("  Dataset not found — attempting download...")
        from data.download_data import download_dataset
        result = download_dataset()
        if result is None:
            raise FileNotFoundError(
                "Could not get dataset. See README.md for manual instructions."
            )

    df = pd.read_csv(data_path)
    # If file has no header, assign column names
    if 'Outcome' not in df.columns:
        df = pd.read_csv(data_path, header=None, names=COLUMNS)
    return df


def preprocess(df):
    """Handle missing values, normalize, split."""
    # Columns where 0 is biologically impossible → treat as missing
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan)
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def select_top_features(X_train, y_train, X_test, k=4):
    """Pick top-k features by absolute correlation with target."""
    corrs = [abs(np.corrcoef(X_train[:, i], y_train)[0, 1])
             for i in range(X_train.shape[1])]
    idx = np.argsort(corrs)[-k:]
    return X_train[:, idx], X_test[:, idx], idx


# ==============================================================
# Detailed Metrics Helper
# ==============================================================
def compute_metrics(y_true, y_pred, model_name):
    """Compute accuracy, precision, recall, F1-score and confusion matrix."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    return {
        'name': model_name,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'confusion_matrix': cm
    }


def print_metrics_table(all_metrics):
    """Print a formatted comparison table of all model metrics."""
    header = f"  {'Model':<16s} {'Accuracy':>9s} {'Precision':>10s} {'Recall':>8s} {'F1-Score':>9s}"
    print(header)
    print("  " + "-" * 54)
    for m in all_metrics:
        print(f"  {m['name']:<16s} {m['accuracy']:>9.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1_score']:>9.4f}")
    print()


def print_confusion_matrices(all_metrics):
    """Print confusion matrices for each model."""
    for m in all_metrics:
        cm = m['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        print(f"  {m['name']}:")
        print(f"    +-----------------+----------+----------+")
        print(f"    |                 | Pred: 0  | Pred: 1  |")
        print(f"    +-----------------+----------+----------+")
        print(f"    | Actual: 0 (Neg) |  TN={tn:<4d} |  FP={fp:<4d} |")
        print(f"    | Actual: 1 (Pos) |  FN={fn:<4d} |  TP={tp:<4d} |")
        print(f"    +-----------------+----------+----------+")
        print()


# ==============================================================
# STEP 6 : Plotting
# ==============================================================
def plot_results(all_metrics):
    """Bar chart comparing all model accuracies + metrics grouped chart."""
    names   = [m['name'] for m in all_metrics]
    acc_vals = [m['accuracy'] for m in all_metrics]
    colors  = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

    # --- Plot 1: Accuracy bar chart ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, acc_vals,
                   color=colors[:len(names)],
                   edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, acc_vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.xlabel('Model / Activation Function', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Disease Prediction - Model Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results_comparison.png', dpi=150, bbox_inches='tight')
    print("\n  [PLOT 1] Saved as: results_comparison.png")
    plt.close()

    # --- Plot 2: Grouped metric comparison ---
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (key, label) in enumerate(zip(metrics_keys, metric_labels)):
        vals = [m[key] for m in all_metrics]
        bars = ax.bar(x + i * width, vals, width, label=label,
                      edgecolor='black', linewidth=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Detailed Metrics Comparison - All Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
    print("  [PLOT 2] Saved as: metrics_comparison.png")
    plt.close(fig)

    # --- Plot 3: Confusion matrix heatmaps ---
    n_models = len(all_metrics)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for idx, m in enumerate(all_metrics):
        sns.heatmap(m['confusion_matrix'], annot=True, fmt='d',
                    cmap='Blues', ax=axes[idx],
                    xticklabels=['Non-Diabetic', 'Diabetic'],
                    yticklabels=['Non-Diabetic', 'Diabetic'])
        axes[idx].set_title(m['name'], fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    # Hide unused subplot slots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
    print("  [PLOT 3] Saved as: confusion_matrices.png")
    plt.close(fig)


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("=" * 65)
    print("   SIH-Inspired Neuro-Fuzzy Disease Prediction System")
    print("   with Activation Function Analysis")
    print("=" * 65)

    # -- Step 1 ------------------------------------------------
    print("\n[Step 1] Loading and preprocessing dataset...")
    df = load_dataset()
    X_train, X_test, y_train, y_test = preprocess(df)
    print(f"  Training samples : {X_train.shape[0]}")
    print(f"  Testing samples  : {X_test.shape[0]}")
    print(f"  Features         : {X_train.shape[1]}")

    # -- Step 2 ------------------------------------------------
    print("\n" + "=" * 65)
    print("[Step 2] Neural Network - Activation Function Comparison")
    print("=" * 65)

    import tensorflow as tf
    activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu']
    nn_predictions = {}   # store predictions for metrics
    nn_results = {}       # store accuracy

    for act in activations:
        print(f"\n  > Training with {act.upper()} ...", end=" ", flush=True)
        tf.random.set_seed(42)
        np.random.seed(42)
        model = build_model(X_train.shape[1], act)
        from tensorflow.keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10,
                                   restore_best_weights=True, verbose=0)
        model.fit(X_train, y_train, epochs=100, batch_size=32,
                  validation_split=0.2, callbacks=[early_stop], verbose=0)
        y_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        acc = round(accuracy_score(y_test, y_pred), 4)
        nn_results[act] = acc
        nn_predictions[act] = y_pred
        print(f"Accuracy: {acc:.4f}")

    # -- Step 3 ------------------------------------------------
    print("\n" + "=" * 65)
    print("[Step 3] Standalone ANFIS (Neuro-Fuzzy System)")
    print("=" * 65)

    # Use top 4 features to keep number of rules manageable (2^4 = 16)
    X_tr_anfis, X_te_anfis, feat_idx = select_top_features(
        X_train, y_train, X_test, k=4
    )
    feature_names = [COLUMNS[i] for i in feat_idx]
    print(f"  Selected features: {feature_names}")
    print(f"  Number of rules  : {2 ** 4}")

    anfis = ANFIS(n_features=4, n_mfs=2)
    anfis.train(X_tr_anfis, y_train, epochs=200, lr=0.05, verbose=True)
    anfis_pred = anfis.predict(X_te_anfis)
    anfis_acc = round(accuracy_score(y_test, anfis_pred), 4)
    print(f"\n  [OK] ANFIS Test Accuracy: {anfis_acc:.4f}")

    # -- Step 4 ------------------------------------------------
    print("\n" + "=" * 65)
    print("[Step 4] Hybrid Pipeline - ANFIS Fuzzification -> Neural Network")
    print("=" * 65)

    # Use ANFIS Layer-1 to convert 8 crisp features → 16 fuzzy features
    anfis_full = ANFIS(n_features=8, n_mfs=2)
    anfis_full.initialize_params(X_train)

    X_tr_fuzzy = anfis_full.get_membership_features(X_train)
    X_te_fuzzy = anfis_full.get_membership_features(X_test)
    print(f"  Original features : {X_train.shape[1]}")
    print(f"  Fuzzy features    : {X_tr_fuzzy.shape[1]}  (8 inputs x 2 MFs)")

    best_act = max(nn_results, key=nn_results.get)
    print(f"  Using best activation: {best_act}")

    tf.random.set_seed(42)
    np.random.seed(42)
    hybrid_model = build_model(X_tr_fuzzy.shape[1], best_act)
    early_stop = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True, verbose=0)
    hybrid_model.fit(X_tr_fuzzy, y_train, epochs=100, batch_size=32,
                     validation_split=0.2, callbacks=[early_stop], verbose=0)
    hybrid_prob = hybrid_model.predict(X_te_fuzzy, verbose=0).flatten()
    hybrid_pred = (hybrid_prob >= 0.5).astype(int)
    hybrid_acc = round(accuracy_score(y_test, hybrid_pred), 4)
    print(f"\n  [OK] Hybrid Test Accuracy: {hybrid_acc:.4f}")

    # -- Step 5 ------------------------------------------------
    print("\n" + "=" * 65)
    print("[Step 5] DETAILED RESULTS & METRICS")
    print("=" * 65)

    # Build metrics for every model
    all_metrics = []
    for act in activations:
        all_metrics.append(compute_metrics(y_test, nn_predictions[act], f"NN-{act}"))
    all_metrics.append(compute_metrics(y_test, anfis_pred, "ANFIS"))
    all_metrics.append(compute_metrics(y_test, hybrid_pred, f"Hybrid({best_act})"))

    print("\n  -- Metrics Comparison Table --\n")
    print_metrics_table(all_metrics)

    print("  -- Confusion Matrices --\n")
    print_confusion_matrices(all_metrics)

    # Quick accuracy bar display
    print("  -- Accuracy Summary --\n")
    for m in all_metrics:
        bar = "#" * int(m['accuracy'] * 40)
        print(f"  {m['name']:<16s} : {m['accuracy']:.4f}  {bar}")

    # ── Step 6 ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("[Step 6] Saving comparison plots...")
    print("=" * 65)
    plot_results(all_metrics)

    print("\n[DONE] All results generated successfully.\n")
    return all_metrics


if __name__ == '__main__':
    main()
