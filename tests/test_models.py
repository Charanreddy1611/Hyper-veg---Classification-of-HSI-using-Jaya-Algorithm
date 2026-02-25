"""
Unit tests for machine learning models.
"""

import numpy as np
import pytest
import torch
from src.models.svm_classifier import create_spatial_blocks, spatial_cross_validation_split
from src.models.cnn_classifier import SpectralCNN, HyperspectralDataset, get_device


@pytest.fixture
def synthetic_data():
    """Create synthetic pixel matrix and labels for testing."""
    n_pixels = 100
    n_bands = 50
    n_classes = 5
    
    X = np.random.randn(n_pixels, n_bands)
    y = np.random.randint(1, n_classes + 1, size=n_pixels)
    
    # Create spatial positions (assume 10x10 grid)
    n_y, n_x = 10, 10
    labels_2d = y.reshape(n_y, n_x)
    
    return X, y, labels_2d, n_y, n_x


def test_spatial_blocks_coverage(synthetic_data):
    """Test that all pixels are assigned a block."""
    _, _, labels_2d, n_y, n_x = synthetic_data
    n_blocks = 3
    
    block_ids = create_spatial_blocks(labels_2d, n_blocks=n_blocks)
    
    # All pixels should have a block ID
    assert block_ids.shape == (n_y, n_x)
    assert np.all(block_ids >= 0)
    assert np.all(block_ids < n_blocks ** 2)


def test_svm_pipeline_runs(synthetic_data):
    """Test that SVM pipeline trains and predicts without error."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    
    X, y, _, _, _ = synthetic_data
    
    # Simple train/test split
    split_idx = len(X) // 2
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    assert len(y_pred) == len(y_test)
    assert np.all(y_pred >= 1)  # Labels are 1-indexed


def test_cnn_forward_pass():
    """Test that CNN forward pass produces correct output shape."""
    n_bands = 50
    n_classes = 5
    batch_size = 8
    
    model = SpectralCNN(n_bands, n_classes)
    
    # Create dummy input
    x = torch.randn(batch_size, 1, n_bands)
    
    # Forward pass
    output = model(x)
    
    assert output.shape == (batch_size, n_classes)


def test_cnn_output_classes():
    """Test that CNN output has n_classes logits."""
    n_bands = 50
    n_classes = 10
    batch_size = 4
    
    model = SpectralCNN(n_bands, n_classes)
    x = torch.randn(batch_size, 1, n_bands)
    output = model(x)
    
    assert output.shape[1] == n_classes


def test_hyperspectral_dataset(synthetic_data):
    """Test HyperspectralDataset."""
    X, y, _, _, _ = synthetic_data
    
    dataset = HyperspectralDataset(X, y)
    
    assert len(dataset) == len(y)
    
    # Test __getitem__
    spectrum, label = dataset[0]
    
    assert spectrum.shape == (1, X.shape[1])  # (1, n_bands)
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long
