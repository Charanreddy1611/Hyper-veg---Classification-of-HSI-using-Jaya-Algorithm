"""
Evaluation Module

Scientific evaluation of classification results with comprehensive metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    precision_recall_fscore_support, classification_report
)
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Dict
) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels (n_pixels,)
        y_pred: Predicted labels (n_pixels,)
        class_names: Dictionary mapping class ID to name
        
    Returns:
        Dictionary with metrics:
        - overall_accuracy: Overall accuracy (OA)
        - kappa: Cohen's Kappa coefficient
        - per_class_precision: Precision per class
        - per_class_recall: Recall per class
        - per_class_f1: F1 score per class
        - macro_f1: Macro-averaged F1
        - confusion_matrix: Confusion matrix (raw)
        - confusion_matrix_norm: Normalized confusion matrix
    """
    # Overall accuracy
    oa = accuracy_score(y_true, y_pred)
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro-averaged F1
    macro_f1 = np.mean(f1)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # Get class IDs
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Build per-class dictionaries
    per_class_precision = {int(cid): float(prec) for cid, prec in zip(unique_classes, precision)}
    per_class_recall = {int(cid): float(rec) for cid, rec in zip(unique_classes, recall)}
    per_class_f1 = {int(cid): float(f) for cid, f in zip(unique_classes, f1)}
    
    metrics = {
        'overall_accuracy': float(oa),
        'kappa': float(kappa),
        'macro_f1': float(macro_f1),
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm,
        'confusion_matrix_norm': cm_norm,
        'class_ids': [int(cid) for cid in unique_classes]
    }
    
    logger.info(f"Overall Accuracy: {oa:.4f}")
    logger.info(f"Kappa: {kappa:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: str = None
) -> None:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix (n_classes, n_classes)
        class_names: List of class name strings
        normalize: If True, show percentages; if False, show counts
        title: Plot title
        save_path: Optional path to save figure
    """
    if normalize:
        cm_plot = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        fmt = '.2f'
        label = 'Percentage'
    else:
        cm_plot = cm
        fmt = 'd'
        label = 'Count'
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': label}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    # Always show in notebook
    plt.show()


def plot_classification_map(
    predictions: np.ndarray,
    labels_true: np.ndarray,
    class_names: Dict,
    title: str = "Classification Map",
    save_path: str = None
) -> None:
    """
    Plot side-by-side ground truth and predictions.
    
    Args:
        predictions: Predicted labels (y, x) or (n_pixels,)
        labels_true: True labels (y, x) or (n_pixels,)
        class_names: Dictionary mapping class ID to name
        title: Plot title
        save_path: Optional path to save figure
    """
    # Reshape if needed
    if predictions.ndim == 1:
        n_pixels = len(predictions)
        n_side = int(np.sqrt(n_pixels))
        
        # Check if it's a perfect square
        if n_side * n_side == n_pixels:
            # Perfect square - reshape directly
            predictions = predictions.reshape(n_side, n_side)
            labels_true = labels_true.reshape(n_side, n_side)
        else:
            # Not a perfect square - find best rectangular layout
            # Find the factor pair closest to square
            best_h = int(np.sqrt(n_pixels))
            while best_h > 0 and n_pixels % best_h != 0:
                best_h -= 1
            
            if best_h > 0:
                # Found a factor - use it
                best_w = n_pixels // best_h
                predictions = predictions.reshape(best_h, best_w)
                labels_true = labels_true.reshape(best_h, best_w)
            else:
                # No exact factor found - use approximate square with padding
                # Pad to next square
                next_square = (n_side + 1) ** 2
                pad_size = next_square - n_pixels
                predictions = np.pad(predictions, (0, pad_size), mode='constant', constant_values=0)
                labels_true = np.pad(labels_true, (0, pad_size), mode='constant', constant_values=0)
                predictions = predictions.reshape(n_side + 1, n_side + 1)
                labels_true = labels_true.reshape(n_side + 1, n_side + 1)
    
    # Get unique classes
    all_classes = np.unique(np.concatenate([predictions.flatten(), labels_true.flatten()]))
    all_classes = all_classes[all_classes > 0]  # Exclude background
    
    # Create colormap
    n_classes = len(all_classes)
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    cmap = mcolors.ListedColormap(colors)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Ground truth
    im1 = ax1.imshow(labels_true, cmap=cmap, interpolation='nearest', vmin=0, vmax=n_classes)
    ax1.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Predictions
    im2 = ax2.imshow(predictions, cmap=cmap, interpolation='nearest', vmin=0, vmax=n_classes)
    ax2.set_title('Predictions', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Create legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor='black')
        for i, class_id in enumerate(all_classes)
    ]
    legend_labels = [class_names.get(int(cid), f"Class {cid}") for cid in all_classes]
    
    fig.legend(legend_elements, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.02),
              ncol=min(8, len(all_classes)), fontsize=9)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved classification map to {save_path}")
    
    # Always show in notebook
    plt.show()


def compare_models(
    results: Dict[str, Dict]
) -> None:
    """
    Compare multiple models side-by-side.
    
    Args:
        results: Dictionary of {model_name: metrics_dict}
    """
    model_names = list(results.keys())
    
    # Extract metrics
    oa_values = [results[name]['overall_accuracy'] for name in model_names]
    kappa_values = [results[name]['kappa'] for name in model_names]
    
    # Print table
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<20} {'OA':<12} {'Kappa':<12} {'Macro F1':<12}")
    print("-"*60)
    
    for name in model_names:
        metrics = results[name]
        print(f"{name:<20} {metrics['overall_accuracy']:<12.4f} "
              f"{metrics['kappa']:<12.4f} {metrics.get('macro_f1', 0.0):<12.4f}")
    
    print("="*60 + "\n")
    
    # Bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1.bar(x, oa_values, width, label='Overall Accuracy', alpha=0.8)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Overall Accuracy', fontsize=12)
    ax1.set_title('Overall Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(x, kappa_values, width, label='Kappa', alpha=0.8, color='orange')
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel("Cohen's Kappa", fontsize=12)
    ax2.set_title("Kappa Comparison", fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
