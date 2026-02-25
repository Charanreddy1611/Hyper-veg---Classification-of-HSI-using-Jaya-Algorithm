"""
SVM Classifier Module

SVM classification with proper spatial cross-validation.
Uses spatial blocks instead of random splits to avoid spatial autocorrelation.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def create_spatial_blocks(
    labels: np.ndarray,
    n_blocks: int = 5
) -> np.ndarray:
    """
    Divide spatial grid into n_blocks × n_blocks blocks.
    
    Assigns each pixel a block ID (0 to n_blocks²-1) based on its
    spatial position. This ensures spatial separation for cross-validation.
    
    Args:
        labels: Ground truth labels (y, x)
        n_blocks: Number of blocks per dimension (total = n_blocks²)
        
    Returns:
        Block ID map (y, x) with values 0 to n_blocks²-1
    """
    n_y, n_x = labels.shape
    
    # Compute block size
    block_size_y = n_y / n_blocks
    block_size_x = n_x / n_blocks
    
    # Create block IDs
    block_ids = np.zeros((n_y, n_x), dtype=int)
    
    for y in range(n_y):
        for x in range(n_x):
            block_y = int(y / block_size_y)
            block_x = int(x / block_size_x)
            # Clamp to valid range
            block_y = min(block_y, n_blocks - 1)
            block_x = min(block_x, n_blocks - 1)
            block_id = block_y * n_blocks + block_x
            block_ids[y, x] = block_id
    
    logger.info(f"Created {n_blocks}×{n_blocks} = {n_blocks**2} spatial blocks")
    
    return block_ids


def spatial_cross_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    spatial_positions: np.ndarray,
    block_ids: np.ndarray,
    test_block: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data by holding out one spatial block as test set.
    
    Args:
        X: Pixel matrix (n_pixels, n_bands)
        y: Label vector (n_pixels,)
        spatial_positions: Pixel positions (n_pixels, 2) with [y, x] coordinates
        block_ids: Block ID map (y, x)
        test_block: Block ID to use as test set
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Get block ID for each pixel
    pixel_block_ids = block_ids[spatial_positions[:, 0], spatial_positions[:, 1]]
    
    # Split
    test_mask = pixel_block_ids == test_block
    train_mask = ~test_mask
    
    X_train = X[train_mask, :]
    X_test = X[test_mask, :]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    logger.info(f"Split: train={len(y_train)}, test={len(y_test)} (block {test_block})")
    
    return X_train, X_test, y_train, y_test


def train_svm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    C: float = 10.0,
    gamma: str = "scale",
    optimize: bool = True
) -> Pipeline:
    """
    Train SVM classifier with optional hyperparameter optimization.
    
    Builds sklearn Pipeline: StandardScaler → SVC
    
    Args:
        X_train: Training pixel matrix (n_pixels, n_bands)
        y_train: Training labels (n_pixels,)
        kernel: SVM kernel type
        C: Regularization parameter
        gamma: Kernel coefficient
        optimize: If True, run GridSearchCV for hyperparameter tuning
        
    Returns:
        Fitted sklearn Pipeline
    """
    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42))
    ])
    
    if optimize:
        logger.info("Running GridSearchCV for hyperparameter optimization...")
        
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01]
        }
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,  # 3-fold CV on training set
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        pipeline = grid_search.best_estimator_
    else:
        logger.info("Training SVM with fixed hyperparameters...")
        pipeline.fit(X_train, y_train)
    
    return pipeline


def run_spatial_cv(
    X: np.ndarray,
    y: np.ndarray,
    block_ids: np.ndarray,
    n_blocks: int = 5
) -> Dict:
    """
    Run full spatial cross-validation.
    
    For each spatial block: train on rest, evaluate on held-out block.
    Aggregates results across all folds.
    
    Args:
        X: Pixel matrix (n_pixels, n_bands)
        y: Label vector (n_pixels,)
        block_ids: Block ID map (y, x)
        n_blocks: Number of blocks per dimension
        
    Returns:
        Dictionary with aggregated metrics:
        - overall_accuracy: Mean/std OA across folds
        - kappa: Mean/std Kappa across folds
        - per_class_f1: Mean/std F1 per class
        - all_fold_results: List of per-fold results
    """
    from .evaluation import compute_metrics
    
    # Get spatial positions for each pixel
    # Assume pixels are in row-major order
    n_y, n_x = block_ids.shape
    spatial_positions = np.zeros((len(y), 2), dtype=int)
    for i in range(len(y)):
        y_pos = i // n_x
        x_pos = i % n_x
        spatial_positions[i] = [y_pos, x_pos]
    
    all_oa = []
    all_kappa = []
    all_per_class_f1 = []
    all_fold_results = []
    
    n_total_blocks = n_blocks ** 2
    
    for test_block in range(n_total_blocks):
        logger.info(f"\n=== Fold {test_block + 1}/{n_total_blocks} (test block {test_block}) ===")
        
        # Split
        X_train, X_test, y_train, y_test = spatial_cross_validation_split(
            X, y, spatial_positions, block_ids, test_block
        )
        
        if len(y_test) == 0:
            logger.warning(f"No test pixels in block {test_block}, skipping")
            continue
        
        # Train
        pipeline = train_svm_classifier(X_train, y_train, optimize=False)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        metrics = compute_metrics(y_test, y_pred, {})
        
        all_oa.append(metrics['overall_accuracy'])
        all_kappa.append(metrics['kappa'])
        all_per_class_f1.append(metrics['per_class_f1'])
        all_fold_results.append(metrics)
        
        logger.info(f"Fold {test_block + 1}: OA = {metrics['overall_accuracy']:.4f}, Kappa = {metrics['kappa']:.4f}")
    
    # Aggregate results
    results = {
        'overall_accuracy': {
            'mean': np.mean(all_oa),
            'std': np.std(all_oa)
        },
        'kappa': {
            'mean': np.mean(all_kappa),
            'std': np.std(all_kappa)
        },
        'per_class_f1': {},
        'all_fold_results': all_fold_results
    }
    
    # Aggregate per-class F1
    if len(all_per_class_f1) > 0:
        class_ids = list(all_per_class_f1[0].keys())
        for class_id in class_ids:
            f1_values = [fold[class_id] for fold in all_per_class_f1 if class_id in fold]
            if len(f1_values) > 0:
                results['per_class_f1'][class_id] = {
                    'mean': np.mean(f1_values),
                    'std': np.std(f1_values)
                }
    
    logger.info(f"\n=== Spatial CV Summary ===")
    logger.info(f"Overall Accuracy: {results['overall_accuracy']['mean']:.4f} ± {results['overall_accuracy']['std']:.4f}")
    logger.info(f"Kappa: {results['kappa']['mean']:.4f} ± {results['kappa']['std']:.4f}")
    
    return results
