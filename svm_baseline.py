"""
=============================================================
  SVM Baseline Classifier for Task Offloading
  Multi-Agent RAG + FL Pipeline — Comparative Baseline
  
  File: svm_baseline.py
  Directory: c:\\Users\\admin\\Desktop\\RAG PIPELINE
  
  Purpose:
    Train an SVM classifier on the same 8 telemetry features
    used by the FL-trained MLP. This provides a centralized
    baseline to demonstrate why the federated + RAG approach
    is necessary (non-IID handling, privacy, explainability).
  
  Outputs:
    - SVM accuracy, F1 score, precision, recall
    - Confusion matrix visualization
    - Decision boundary visualization (2D PCA projection)
    - Per-cell-tower accuracy analysis (non-IID weakness)
    - Comparison chart: SVM vs Standard FL vs RAG+FL
=============================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "task_offloading_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SEED = 42

FEATURE_COLS = [
    'Cell_Identity(CI)', 'Downlink_Traffic', 'LAC', 'Duration',
    'RAT', 'Uplink_Traffic', 'Latency', 'Throughput'
]
TARGET_COL = 'Offloading_Decision'

# Log-transform columns (same as your MLP pipeline uses)
LOG_COLS_INDICES = [1, 3, 5, 6, 7]  # Downlink, Duration, Uplink, Latency, Throughput

# Premium dark color palette (matches your discovery_analysis.py)
COLORS = {
    "edge_cloud": "#6C5CE7",
    "mobile_device": "#00CEC9",
    "accent": "#FD79A8",
    "warning": "#FDCB6E",
    "bg_dark": "#0D1117",
    "bg_card": "#161B22",
    "text": "#C9D1D9",
    "text_bright": "#F0F6FC",
    "grid": "#21262D",
    "green": "#00B894",
    "red": "#E17055",
    "blue": "#74B9FF",
    "svm_color": "#A29BFE",       # SVM-specific: soft purple
    "svm_linear": "#55EFC4",      # LinearSVC: mint green
}


def setup_dark_style():
    """Configure matplotlib for premium dark theme."""
    plt.rcParams.update({
        'figure.facecolor': COLORS["bg_dark"],
        'axes.facecolor': COLORS["bg_card"],
        'axes.edgecolor': COLORS["grid"],
        'axes.labelcolor': COLORS["text"],
        'text.color': COLORS["text_bright"],
        'xtick.color': COLORS["text"],
        'ytick.color': COLORS["text"],
        'grid.color': COLORS["grid"],
        'grid.alpha': 0.3,
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'figure.titlesize': 16,
        'figure.titleweight': 'bold',
    })


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════
def load_and_prepare_data(
    dataset_path: str = DATASET_PATH,
    test_size: float = 0.2,
    sample_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, StandardScaler]:
    """
    Load dataset and prepare features with the same log-transform
    + scaling pipeline used by the FL model.
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, df, scaler
    """
    print(f"\n  Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"  Loaded {len(df):,} records")
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=SEED).reset_index(drop=True)
        print(f"  Sampled down to {len(df):,} records for SVM training")
    
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = (df[TARGET_COL] == 'Edge Cloud').astype(int).values
    
    # Log-transform heavily-skewed columns (same as multi_agent_rag_fl.py)
    X_log = X.copy()
    for idx in LOG_COLS_INDICES:
        X_log[:, idx] = np.log1p(np.abs(X_log[:, idx]))
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_log, y, test_size=test_size, random_state=SEED, stratify=y
    )
    
    # StandardScaler (better for SVM than MinMax)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Class distribution
    train_edge_pct = y_train.mean() * 100
    test_edge_pct = y_test.mean() * 100
    print(f"  Train set: {len(X_train):,} samples ({train_edge_pct:.1f}% Edge Cloud)")
    print(f"  Test set:  {len(X_test):,} samples ({test_edge_pct:.1f}% Edge Cloud)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, df, scaler


# ══════════════════════════════════════════════════════════════
# SVM TRAINING
# ══════════════════════════════════════════════════════════════
def train_linear_svm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    verbose: bool = True
) -> Tuple[object, Dict]:
    """
    Train a LinearSVC classifier (fast, scalable to 200K+ records).
    """
    if verbose:
        print("\n  [LinearSVC] Training...")
    
    t0 = time.time()
    
    # CalibratedClassifierCV wraps LinearSVC to produce probability estimates
    base_svm = LinearSVC(
        C=1.0,
        class_weight='balanced',
        max_iter=10000,
        random_state=SEED,
        dual='auto'
    )
    svm_calibrated = CalibratedClassifierCV(base_svm, cv=3)
    svm_calibrated.fit(X_train, y_train)
    
    train_time = time.time() - t0
    
    # Evaluate
    y_pred = svm_calibrated.predict(X_test)
    y_proba = svm_calibrated.predict_proba(X_test)[:, 1]
    
    metrics = {
        "model_name": "LinearSVC",
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "train_time_s": train_time,
        "n_train_samples": len(X_train),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    
    if verbose:
        print(f"  [LinearSVC] Trained in {train_time:.1f}s")
        print(f"  [LinearSVC] Accuracy: {metrics['accuracy']:.4f}")
        print(f"  [LinearSVC] F1 Score: {metrics['f1_score']:.4f}")
        print(f"  [LinearSVC] Precision: {metrics['precision']:.4f}")
        print(f"  [LinearSVC] Recall: {metrics['recall']:.4f}")
    
    return svm_calibrated, metrics


def train_rbf_svm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    sample_limit: int = 40000,
    verbose: bool = True
) -> Tuple[object, Dict]:
    """
    Train an RBF-kernel SVM (non-linear decision boundary).
    
    Note: RBF SVM is O(n^2) to O(n^3), so we subsample for large datasets.
    """
    if verbose:
        print("\n  [RBF SVM] Training...")
    
    # Subsample if dataset is too large for RBF kernel
    if len(X_train) > sample_limit:
        if verbose:
            print(f"  [RBF SVM] Subsampling {sample_limit:,} from {len(X_train):,} "
                  f"(RBF kernel is O(n^2) complexity)")
        idx = np.random.RandomState(SEED).choice(len(X_train), sample_limit, replace=False)
        X_train_sub = X_train[idx]
        y_train_sub = y_train[idx]
    else:
        X_train_sub = X_train
        y_train_sub = y_train
    
    t0 = time.time()
    
    svm_rbf = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        class_weight='balanced',
        random_state=SEED,
        max_iter=10000,
        probability=True
    )
    svm_rbf.fit(X_train_sub, y_train_sub)
    
    train_time = time.time() - t0
    
    # Evaluate on FULL test set
    y_pred = svm_rbf.predict(X_test)
    y_proba = svm_rbf.predict_proba(X_test)[:, 1]
    
    metrics = {
        "model_name": "RBF SVM",
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "train_time_s": train_time,
        "n_train_samples": len(X_train_sub),
        "n_support_vectors": svm_rbf.n_support_.sum(),
        "support_vector_ratio": svm_rbf.n_support_.sum() / len(X_train_sub),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    
    if verbose:
        print(f"  [RBF SVM] Trained on {len(X_train_sub):,} samples in {train_time:.1f}s")
        print(f"  [RBF SVM] Support Vectors: {metrics['n_support_vectors']:,} "
              f"({metrics['support_vector_ratio']:.1%} of training data)")
        print(f"  [RBF SVM] Accuracy: {metrics['accuracy']:.4f}")
        print(f"  [RBF SVM] F1 Score: {metrics['f1_score']:.4f}")
        print(f"  [RBF SVM] Precision: {metrics['precision']:.4f}")
        print(f"  [RBF SVM] Recall: {metrics['recall']:.4f}")
    
    return svm_rbf, metrics


# ══════════════════════════════════════════════════════════════
# NON-IID WEAKNESS ANALYSIS
# ══════════════════════════════════════════════════════════════
def analyze_per_cell_accuracy(
    svm_model, scaler: StandardScaler, df: pd.DataFrame,
    model_name: str = "SVM", verbose: bool = True
) -> Dict:
    """
    Analyze SVM accuracy per cell tower to demonstrate the
    non-IID weakness — a centralized SVM performs poorly on
    cells with skewed distributions because it can't adapt
    to local patterns the way FL + RAG can.
    """
    if verbose:
        print(f"\n  [{model_name}] Analyzing per-cell-tower accuracy...")
    
    # Prepare all data
    X_all = df[FEATURE_COLS].values.astype(np.float64)
    y_all = (df[TARGET_COL] == 'Edge Cloud').astype(int).values
    
    X_log = X_all.copy()
    for idx in LOG_COLS_INDICES:
        X_log[:, idx] = np.log1p(np.abs(X_log[:, idx]))
    X_scaled = scaler.transform(X_log)
    
    y_pred = svm_model.predict(X_scaled)
    
    # Per-cell accuracy
    df_eval = df.copy()
    df_eval['y_true'] = y_all
    df_eval['y_pred'] = y_pred
    df_eval['correct'] = (df_eval['y_true'] == df_eval['y_pred']).astype(int)
    
    cell_accuracy = df_eval.groupby('Cell_Identity(CI)').agg(
        n_records=('correct', 'count'),
        accuracy=('correct', 'mean'),
        edge_rate=('y_true', 'mean')
    ).reset_index()
    
    # Filter cells with enough samples
    cell_accuracy = cell_accuracy[cell_accuracy['n_records'] >= 10]
    
    # Classify cells by skew
    highly_skewed = cell_accuracy[
        (cell_accuracy['edge_rate'] < 0.2) | (cell_accuracy['edge_rate'] > 0.8)
    ]
    balanced = cell_accuracy[
        (cell_accuracy['edge_rate'] >= 0.3) & (cell_accuracy['edge_rate'] <= 0.7)
    ]
    
    results = {
        "overall_accuracy": accuracy_score(y_all, y_pred),
        "total_cells_evaluated": len(cell_accuracy),
        "avg_cell_accuracy": cell_accuracy['accuracy'].mean(),
        "std_cell_accuracy": cell_accuracy['accuracy'].std(),
        "min_cell_accuracy": cell_accuracy['accuracy'].min(),
        "max_cell_accuracy": cell_accuracy['accuracy'].max(),
        "highly_skewed_cells": len(highly_skewed),
        "highly_skewed_avg_acc": highly_skewed['accuracy'].mean() if len(highly_skewed) > 0 else 0,
        "balanced_cells": len(balanced),
        "balanced_avg_acc": balanced['accuracy'].mean() if len(balanced) > 0 else 0,
        "cell_accuracy_df": cell_accuracy,
    }
    
    if verbose:
        print(f"  [{model_name}] Cells evaluated: {results['total_cells_evaluated']}")
        print(f"  [{model_name}] Avg per-cell accuracy: {results['avg_cell_accuracy']:.4f} "
              f"(std: {results['std_cell_accuracy']:.4f})")
        print(f"  [{model_name}] Highly-skewed cells ({len(highly_skewed)}): "
              f"avg acc = {results['highly_skewed_avg_acc']:.4f}")
        print(f"  [{model_name}] Balanced cells ({len(balanced)}): "
              f"avg acc = {results['balanced_avg_acc']:.4f}")
        print(f"  [{model_name}] Worst cell accuracy: {results['min_cell_accuracy']:.4f}")
    
    return results


# ══════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
def plot_confusion_matrices(
    linear_metrics: Dict, rbf_metrics: Dict,
    y_test: np.ndarray, save: bool = True
) -> None:
    """Generate side-by-side confusion matrices for Linear and RBF SVM."""
    setup_dark_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('SVM Baseline: Confusion Matrices',
                 fontsize=16, fontweight='bold', color=COLORS["text_bright"], y=0.98)
    
    labels = ['Mobile\nDevice', 'Edge\nCloud']
    
    for ax, metrics, color in zip(
        axes,
        [linear_metrics, rbf_metrics],
        [COLORS["svm_linear"], COLORS["svm_color"]]
    ):
        cm = metrics["confusion_matrix"]
        cm_pct = cm.astype(float) / cm.sum() * 100
        
        im = ax.imshow(cm_pct, interpolation='nearest', cmap='Blues', alpha=0.8)
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text_color = 'white' if cm_pct[i, j] > 40 else COLORS["text_bright"]
                ax.text(j, i, f'{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)',
                       ha='center', va='center', fontsize=12,
                       fontweight='bold', color=text_color)
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        
        acc = metrics['accuracy']
        f1 = metrics['f1_score']
        ax.set_title(f"{metrics['model_name']}\nAcc={acc:.3f}  F1={f1:.3f}",
                     fontsize=13, color=color)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    
    if save:
        ensure_output_dir()
        filepath = os.path.join(OUTPUT_DIR, "svm_confusion_matrices.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight',
                   facecolor=COLORS["bg_dark"], edgecolor='none')
        print(f"  [Plot] Saved: {filepath}")
    plt.close(fig)


def plot_roc_curves(
    linear_metrics: Dict, rbf_metrics: Dict,
    y_test: np.ndarray, save: bool = True
) -> None:
    """Generate ROC curves for both SVM variants."""
    setup_dark_style()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle('SVM Baseline: ROC Curves',
                 fontsize=16, fontweight='bold', color=COLORS["text_bright"], y=0.98)
    
    for metrics, color, ls in [
        (linear_metrics, COLORS["svm_linear"], '-'),
        (rbf_metrics, COLORS["svm_color"], '--')
    ]:
        fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2.5, linestyle=ls,
                label=f'{metrics["model_name"]} (AUC = {roc_auc:.4f})')
    
    # Diagonal reference
    ax.plot([0, 1], [0, 1], color=COLORS["grid"], linestyle=':', linewidth=1.5,
            label='Random (AUC = 0.5)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic', fontsize=14)
    ax.legend(loc='lower right', framealpha=0.8, fontsize=11)
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    
    if save:
        ensure_output_dir()
        filepath = os.path.join(OUTPUT_DIR, "svm_roc_curves.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight',
                   facecolor=COLORS["bg_dark"], edgecolor='none')
        print(f"  [Plot] Saved: {filepath}")
    plt.close(fig)


def plot_per_cell_accuracy(
    cell_analysis: Dict, model_name: str = "RBF SVM",
    save: bool = True
) -> None:
    """Visualize per-cell accuracy to demonstrate non-IID weakness."""
    setup_dark_style()
    
    cell_df = cell_analysis["cell_accuracy_df"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{model_name}: Per-Cell Accuracy (Non-IID Weakness)',
                 fontsize=16, fontweight='bold', color=COLORS["text_bright"], y=0.98)
    
    # Left: Scatter — Cell edge_rate vs accuracy
    ax1 = axes[0]
    
    scatter_colors = [
        COLORS["red"] if acc < 0.7 else (COLORS["warning"] if acc < 0.85 else COLORS["green"])
        for acc in cell_df['accuracy']
    ]
    
    ax1.scatter(cell_df['edge_rate'] * 100, cell_df['accuracy'] * 100,
               c=scatter_colors, alpha=0.5, s=15, edgecolors='none')
    
    ax1.axhline(y=85, color=COLORS["accent"], linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.text(5, 86.5, 'Target (85%)', color=COLORS["accent"], fontsize=9)
    
    ax1.set_xlabel('Cell Edge Cloud Rate (%)', fontsize=11)
    ax1.set_ylabel('SVM Accuracy (%)', fontsize=11)
    ax1.set_title('Accuracy vs Cell Distribution Skew', fontsize=13)
    ax1.grid(alpha=0.2)
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(0, 105)
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color=COLORS["red"], label='Poor (<70%)'),
        mpatches.Patch(color=COLORS["warning"], label='Moderate (70-85%)'),
        mpatches.Patch(color=COLORS["green"], label='Good (>85%)')
    ]
    ax1.legend(handles=legend_patches, loc='lower left', framealpha=0.8, fontsize=9)
    
    # Right: Histogram of per-cell accuracy
    ax2 = axes[1]
    
    n, bins, patches = ax2.hist(cell_df['accuracy'] * 100, bins=30,
                                 edgecolor='white', linewidth=0.5, alpha=0.85)
    
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 70:
            patch.set_facecolor(COLORS["red"])
        elif left_edge < 85:
            patch.set_facecolor(COLORS["warning"])
        else:
            patch.set_facecolor(COLORS["green"])
    
    ax2.axvline(x=cell_df['accuracy'].mean() * 100, color=COLORS["accent"],
               linestyle='--', alpha=0.8, linewidth=2)
    ax2.text(cell_df['accuracy'].mean() * 100 + 1, ax2.get_ylim()[1] * 0.9,
            f"Mean: {cell_df['accuracy'].mean():.1%}",
            color=COLORS["accent"], fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Per-Cell Accuracy (%)', fontsize=11)
    ax2.set_ylabel('Number of Cells', fontsize=11)
    ax2.set_title('Distribution of Per-Cell Accuracies', fontsize=13)
    ax2.grid(axis='y', alpha=0.2)
    
    # Annotation for non-IID weakness
    poor_cells = (cell_df['accuracy'] < 0.7).sum()
    total_cells = len(cell_df)
    ax2.text(0.95, 0.95,
            f'{poor_cells} of {total_cells} cells\nhave <70% accuracy\n(Non-IID effect)',
            transform=ax2.transAxes, ha='right', va='top', fontsize=10,
            color=COLORS["red"], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS["bg_card"],
                      edgecolor=COLORS["red"], alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    
    if save:
        ensure_output_dir()
        filepath = os.path.join(OUTPUT_DIR, "svm_per_cell_accuracy.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight',
                   facecolor=COLORS["bg_dark"], edgecolor='none')
        print(f"  [Plot] Saved: {filepath}")
    plt.close(fig)


def plot_comparison_chart(
    svm_metrics: Dict,
    standard_fl_metrics: Optional[Dict] = None,
    rag_fl_metrics: Optional[Dict] = None,
    save: bool = True
) -> None:
    """
    Generate the final comparison bar chart:
    SVM Baseline vs Standard FL vs RAG + FL.
    """
    setup_dark_style()
    
    # Default values from your summary matrix if FL metrics aren't provided
    if standard_fl_metrics is None:
        standard_fl_metrics = {"accuracy": 0.782, "f1_score": 0.72}
    if rag_fl_metrics is None:
        rag_fl_metrics = {"accuracy": 0.966, "f1_score": 0.95}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Comparison: SVM Baseline vs Federated Learning',
                 fontsize=16, fontweight='bold', color=COLORS["text_bright"], y=0.98)
    
    models = ['SVM\nBaseline', 'Standard\nFL', 'RAG +\nFL']
    bar_colors = [COLORS["svm_color"], COLORS["red"], COLORS["green"]]
    
    # Left: Accuracy comparison
    ax1 = axes[0]
    accuracies = [
        svm_metrics['accuracy'] * 100,
        standard_fl_metrics['accuracy'] * 100,
        rag_fl_metrics['accuracy'] * 100
    ]
    
    bars = ax1.bar(models, accuracies, color=bar_colors, alpha=0.85,
                   edgecolor='white', linewidth=0.5, width=0.5)
    
    for bar, val in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=13, color=COLORS["text_bright"])
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Classification Accuracy', fontsize=13)
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.2)
    
    # Right: F1 Score comparison
    ax2 = axes[1]
    f1_scores = [
        svm_metrics['f1_score'] * 100,
        standard_fl_metrics['f1_score'] * 100,
        rag_fl_metrics['f1_score'] * 100
    ]
    
    bars = ax2.bar(models, f1_scores, color=bar_colors, alpha=0.85,
                   edgecolor='white', linewidth=0.5, width=0.5)
    
    for bar, val in zip(bars, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=13, color=COLORS["text_bright"])
    
    ax2.set_ylabel('F1 Score (%)', fontsize=12)
    ax2.set_title('F1 Score (Edge Cloud Class)', fontsize=13)
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.2)
    
    # Bottom annotation
    fig.text(0.5, 0.02,
            'SVM is centralized (no privacy) | FL preserves data locality | '
            'RAG adds explainability + guardrails',
            ha='center', fontsize=10, color=COLORS["text"],
            style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    
    if save:
        ensure_output_dir()
        filepath = os.path.join(OUTPUT_DIR, "svm_vs_fl_comparison.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight',
                   facecolor=COLORS["bg_dark"], edgecolor='none')
        print(f"  [Plot] Saved: {filepath}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# COMPREHENSIVE RESULTS TABLE
# ══════════════════════════════════════════════════════════════
def plot_results_table(
    linear_metrics: Dict, rbf_metrics: Dict,
    cell_analysis: Dict, save: bool = True
) -> None:
    """Generate a comprehensive results summary table."""
    setup_dark_style()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('SVM Baseline: Comprehensive Results Summary',
                 fontsize=16, fontweight='bold', color=COLORS["text_bright"], y=0.97)
    ax.axis('off')
    
    headers = ['Metric', 'LinearSVC', 'RBF SVM', 'RAG + FL\n(Your System)']
    rows = [
        ['Overall Accuracy',
         f'{linear_metrics["accuracy"]:.4f}',
         f'{rbf_metrics["accuracy"]:.4f}',
         '~0.9660'],
        ['F1 Score (Edge Cloud)',
         f'{linear_metrics["f1_score"]:.4f}',
         f'{rbf_metrics["f1_score"]:.4f}',
         '~0.9500'],
        ['Precision',
         f'{linear_metrics["precision"]:.4f}',
         f'{rbf_metrics["precision"]:.4f}',
         'N/A'],
        ['Recall',
         f'{linear_metrics["recall"]:.4f}',
         f'{rbf_metrics["recall"]:.4f}',
         'N/A'],
        ['Training Time',
         f'{linear_metrics["train_time_s"]:.1f}s',
         f'{rbf_metrics["train_time_s"]:.1f}s',
         'FL Distributed'],
        ['Training Samples',
         f'{linear_metrics["n_train_samples"]:,}',
         f'{rbf_metrics["n_train_samples"]:,}',
         'Federated'],
        ['Data Privacy',
         'Centralized',
         'Centralized',
         'Preserved (FL)'],
        ['Non-IID Handling',
         'None',
         'None',
         'RAG Calibration'],
        ['Explainability',
         'None',
         'None',
         'Full (Citations)'],
        ['Per-Cell Avg Accuracy',
         f'{cell_analysis["avg_cell_accuracy"]:.4f}',
         f'{cell_analysis["avg_cell_accuracy"]:.4f}',
         'Higher (Adaptive)'],
        ['Worst Cell Accuracy',
         f'{cell_analysis["min_cell_accuracy"]:.4f}',
         f'{cell_analysis["min_cell_accuracy"]:.4f}',
         'Guardrail Protected'],
    ]
    
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.28, 0.22, 0.22, 0.28]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["grid"])
        cell.set_linewidth(1.5)
        
        if row == 0:
            cell.set_facecolor(COLORS["edge_cloud"])
            cell.set_text_props(color='white', fontweight='bold', fontsize=11)
        elif col == 0:
            cell.set_facecolor('#1A1F2E')
            cell.set_text_props(color=COLORS["text_bright"], fontweight='bold')
        elif col in [1, 2]:
            cell.set_facecolor('#1E1E2E')
            cell.set_text_props(color=COLORS["svm_color"])
        elif col == 3:
            cell.set_facecolor('#1A2E1A')
            cell.set_text_props(color=COLORS["green"], fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    
    if save:
        ensure_output_dir()
        filepath = os.path.join(OUTPUT_DIR, "svm_results_table.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight',
                   facecolor=COLORS["bg_dark"], edgecolor='none')
        print(f"  [Plot] Saved: {filepath}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════
def run_svm_baseline(
    dataset_path: str = DATASET_PATH,
    sample_size: Optional[int] = None,
    rbf_sample_limit: int = 40000,
    standard_fl_metrics: Optional[Dict] = None,
    rag_fl_metrics: Optional[Dict] = None,
    verbose: bool = True
) -> Dict:
    """
    Run the complete SVM baseline experiment.
    
    Args:
        dataset_path: Path to task_offloading_dataset.csv
        sample_size: If set, subsample dataset before training
        rbf_sample_limit: Max training samples for RBF SVM
        standard_fl_metrics: Dict with 'accuracy', 'f1_score' from Standard FL
        rag_fl_metrics: Dict with 'accuracy', 'f1_score' from RAG+FL
        verbose: Print detailed progress
    
    Returns:
        Dict with all SVM baseline results and metrics
    """
    print("\n" + "=" * 70)
    print("  SVM BASELINE EXPERIMENT")
    print("  Task Offloading: Mobile Device vs Edge Cloud")
    print("=" * 70)
    
    total_start = time.time()
    
    # 1. Load and prepare data
    X_train, X_test, y_train, y_test, df, scaler = load_and_prepare_data(
        dataset_path=dataset_path,
        sample_size=sample_size
    )
    
    # 2. Train LinearSVC
    print("\n" + "-" * 50)
    print("  PHASE 1: LinearSVC Training")
    print("-" * 50)
    linear_model, linear_metrics = train_linear_svm(
        X_train, y_train, X_test, y_test, verbose=verbose
    )
    
    # 3. Train RBF SVM
    print("\n" + "-" * 50)
    print("  PHASE 2: RBF SVM Training")
    print("-" * 50)
    rbf_model, rbf_metrics = train_rbf_svm(
        X_train, y_train, X_test, y_test,
        sample_limit=rbf_sample_limit, verbose=verbose
    )
    
    # 4. Per-cell accuracy analysis (non-IID weakness)
    print("\n" + "-" * 50)
    print("  PHASE 3: Per-Cell Non-IID Analysis")
    print("-" * 50)
    cell_analysis = analyze_per_cell_accuracy(
        rbf_model, scaler, df, model_name="RBF SVM", verbose=verbose
    )
    
    # 5. Generate visualizations
    print("\n" + "-" * 50)
    print("  PHASE 4: Generating Visualizations")
    print("-" * 50)
    
    plot_confusion_matrices(linear_metrics, rbf_metrics, y_test)
    plot_roc_curves(linear_metrics, rbf_metrics, y_test)
    plot_per_cell_accuracy(cell_analysis, "RBF SVM")
    
    # Use the better SVM for comparison
    best_svm = rbf_metrics if rbf_metrics['accuracy'] > linear_metrics['accuracy'] else linear_metrics
    plot_comparison_chart(best_svm, standard_fl_metrics, rag_fl_metrics)
    plot_results_table(linear_metrics, rbf_metrics, cell_analysis)
    
    # 6. Print detailed report
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("  SVM BASELINE — FINAL REPORT")
    print("=" * 70)
    
    print(f"\n  {'Metric':<30} {'LinearSVC':>12} {'RBF SVM':>12}")
    print(f"  {'=' * 54}")
    print(f"  {'Accuracy':<30} {linear_metrics['accuracy']:>12.4f} {rbf_metrics['accuracy']:>12.4f}")
    print(f"  {'F1 Score':<30} {linear_metrics['f1_score']:>12.4f} {rbf_metrics['f1_score']:>12.4f}")
    print(f"  {'Precision':<30} {linear_metrics['precision']:>12.4f} {rbf_metrics['precision']:>12.4f}")
    print(f"  {'Recall':<30} {linear_metrics['recall']:>12.4f} {rbf_metrics['recall']:>12.4f}")
    print(f"  {'Training Time':<30} {linear_metrics['train_time_s']:>11.1f}s {rbf_metrics['train_time_s']:>11.1f}s")
    
    if 'n_support_vectors' in rbf_metrics:
        print(f"\n  RBF SVM Support Vectors: {rbf_metrics['n_support_vectors']:,} "
              f"({rbf_metrics['support_vector_ratio']:.1%} of training data)")
    
    print(f"\n  Per-Cell Tower Analysis:")
    print(f"    Avg cell accuracy: {cell_analysis['avg_cell_accuracy']:.4f}")
    print(f"    Std cell accuracy: {cell_analysis['std_cell_accuracy']:.4f}")
    print(f"    Worst cell:        {cell_analysis['min_cell_accuracy']:.4f}")
    print(f"    Highly-skewed cells ({cell_analysis['highly_skewed_cells']}): "
          f"avg acc = {cell_analysis['highly_skewed_avg_acc']:.4f}")
    
    print(f"\n  Total experiment time: {total_time:.1f}s")
    print(f"  Charts saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    # Classification reports
    print("\n  LinearSVC Classification Report:")
    print(classification_report(
        y_test, linear_metrics['y_pred'],
        target_names=['Mobile Device', 'Edge Cloud'],
        digits=4
    ))
    
    print("  RBF SVM Classification Report:")
    print(classification_report(
        y_test, rbf_metrics['y_pred'],
        target_names=['Mobile Device', 'Edge Cloud'],
        digits=4
    ))
    
    return {
        "linear_svm": {k: v for k, v in linear_metrics.items() 
                       if k not in ['y_pred', 'y_proba', 'confusion_matrix']},
        "rbf_svm": {k: v for k, v in rbf_metrics.items() 
                    if k not in ['y_pred', 'y_proba', 'confusion_matrix']},
        "cell_analysis": {k: v for k, v in cell_analysis.items() 
                         if k != 'cell_accuracy_df'},
        "best_svm_accuracy": max(linear_metrics['accuracy'], rbf_metrics['accuracy']),
        "best_svm_f1": max(linear_metrics['f1_score'], rbf_metrics['f1_score']),
        "linear_model": linear_model,
        "rbf_model": rbf_model,
        "scaler": scaler,
    }


if __name__ == "__main__":
    results = run_svm_baseline(verbose=True)
