"""
1D CNN Classifier Module

PyTorch implementation of 1D CNN for hyperspectral classification.
Treats each pixel spectrum as a 1D signal, using convolution to detect
local spectral features (absorption dips, reflectance peaks).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SpectralCNN(nn.Module):
    """
    1D CNN for hyperspectral spectral classification.
    
    Architecture:
    - Input: (batch, 1, n_bands) - 1 channel, n_bands "timesteps"
    - Conv1d layers with BatchNorm and ReLU
    - GlobalAveragePooling
    - Fully connected layers with dropout
    """
    
    def __init__(self, n_bands: int, n_classes: int):
        """
        Initialize SpectralCNN.
        
        Args:
            n_bands: Number of spectral bands
            n_classes: Number of output classes
        """
        super(SpectralCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, n_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 1, n_bands)
            
        Returns:
            Logits tensor (batch, n_classes)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)
        
        # Fully connected
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
    
    def get_spectral_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 256-dimensional spectral feature embedding.
        
        Args:
            x: Input tensor (batch, 1, n_bands)
            
        Returns:
            Feature tensor (batch, 256)
        """
        # Forward through conv layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        return x


class HyperspectralDataset(Dataset):
    """
    PyTorch Dataset for hyperspectral pixel data.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Pixel matrix (n_pixels, n_bands)
            y: Label vector (n_pixels,)
        """
        self.X = torch.FloatTensor(X)
        # Convert labels to 0-indexed (subtract 1 since labels are 1-indexed)
        self.y = torch.LongTensor(y - 1)
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reshape to (1, n_bands) for 1D conv
        spectrum = self.X[idx].unsqueeze(0)  # (1, n_bands)
        label = self.y[idx]
        return spectrum, label


def get_device(device: str = "auto") -> torch.device:
    """
    Get appropriate device (CUDA, MPS, or CPU).
    
    Args:
        device: Device string ("auto", "cuda", "mps", or "cpu")
        
    Returns:
        torch.device object
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "auto"
) -> Tuple[SpectralCNN, Dict]:
    """
    Train 1D CNN classifier.
    
    Args:
        X_train: Training pixel matrix (n_pixels, n_bands)
        y_train: Training labels (n_pixels,)
        X_val: Validation pixel matrix (n_pixels, n_bands)
        y_val: Validation labels (n_pixels,)
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device string ("auto", "cuda", "mps", or "cpu")
        
    Returns:
        Tuple of:
        - best_model: Trained model (best validation performance)
        - history: Dictionary with training history
    """
    device_obj = get_device(device)
    logger.info(f"Using device: {device_obj}")
    
    n_bands = X_train.shape[1]
    n_classes = len(np.unique(np.concatenate([y_train, y_val])))
    
    # Create datasets
    train_dataset = HyperspectralDataset(X_train, y_train)
    val_dataset = HyperspectralDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = SpectralCNN(n_bands, n_classes).to(device_obj)
    
    # Loss and optimizer
    # Compute class weights for imbalanced data
    unique, counts = np.unique(y_train, return_counts=True)
    class_weights = len(y_train) / (n_classes * counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device_obj)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    logger.info(f"Training CNN for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for spectra, labels in train_loader:
            spectra = spectra.to(device_obj)
            labels = labels.to(device_obj)
            
            optimizer.zero_grad()
            outputs = model(spectra)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for spectra, labels in val_loader:
                spectra = spectra.to(device_obj)
                labels = labels.to(device_obj)
                
                outputs = model(spectra)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{n_epochs}: "
                       f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                       f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    history['best_epoch'] = best_epoch
    history['best_val_acc'] = best_val_acc
    
    logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
    
    return model, history


def evaluate_cnn(
    model: SpectralCNN,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Dict,
    device: str = "auto"
) -> Dict:
    """
    Evaluate CNN on test set.
    
    Args:
        model: Trained SpectralCNN model
        X_test: Test pixel matrix (n_pixels, n_bands)
        y_test: Test labels (n_pixels,)
        class_names: Dictionary mapping class ID to name
        device: Device string
        
    Returns:
        Dictionary with predictions, probabilities, and metrics
    """
    from .evaluation import compute_metrics
    
    device_obj = get_device(device)
    model.eval()
    
    # Create dataset and loader
    test_dataset = HyperspectralDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for spectra, labels in test_loader:
            spectra = spectra.to(device_obj)
            outputs = model(spectra)
            
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert predictions back to 1-indexed
    y_pred = np.array(all_predictions) + 1
    y_true = np.array(all_labels) + 1
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics['probabilities'] = np.array(all_probabilities)
    metrics['predictions'] = y_pred  # Add predictions for visualization
    
    return metrics
