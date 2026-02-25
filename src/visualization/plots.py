"""
Visualization Module

All visualization functions for the HyperVeg project.
Produces publication-quality figures.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def plot_false_color_composite(
    cube: xr.DataArray,
    r_nm: float = 680.0,
    g_nm: float = 550.0,
    b_nm: float = 450.0,
    title: str = "False Color Composite",
    save_path: Optional[str] = None
) -> None:
    """
    Create false color RGB composite from three spectral bands.
    
    Args:
        cube: Hyperspectral cube (band, y, x)
        r_nm: Wavelength for red channel (nm)
        g_nm: Wavelength for green channel (nm)
        b_nm: Wavelength for blue channel (nm)
        title: Plot title
        save_path: Optional path to save figure
    """
    # Get available wavelengths
    available_wavelengths = cube.coords['band'].values
    
    # Find nearest band indices (works even if non-monotonic)
    r_idx = np.argmin(np.abs(available_wavelengths - r_nm))
    g_idx = np.argmin(np.abs(available_wavelengths - g_nm))
    b_idx = np.argmin(np.abs(available_wavelengths - b_nm))
    
    # Select bands using isel (works with non-monotonic indices)
    r_band = cube.isel(band=r_idx).values
    g_band = cube.isel(band=g_idx).values
    b_band = cube.isel(band=b_idx).values
    
    # Get actual wavelengths used (for display)
    r_actual = available_wavelengths[r_idx]
    g_actual = available_wavelengths[g_idx]
    b_actual = available_wavelengths[b_idx]
    
    # Stack into RGB
    rgb = np.stack([r_band, g_band, b_band], axis=2)
    
    # Stretch to 2nd-98th percentile for visualization
    for i in range(3):
        band_data = rgb[:, :, i]
        p2, p98 = np.nanpercentile(band_data, [2, 98])
        rgb[:, :, i] = np.clip((band_data - p2) / (p98 - p2 + 1e-10), 0, 1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, interpolation='nearest')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.text(0.02, 0.98, f"R: {r_actual:.0f}nm, G: {g_actual:.0f}nm, B: {b_actual:.0f}nm",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved false color composite to {save_path}")
    
    # Always show in notebook
    plt.show()


def plot_spectral_signatures(
    cube: xr.DataArray,
    labels: np.ndarray,
    class_names: Dict[int, str],
    class_ids: Optional[List[int]] = None,
    wavelengths: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot mean spectral signatures ± 1 std for each class.
    
    Marks water vapor absorption regions and key wavelengths.
    
    Args:
        cube: Reflectance cube (band, y, x)
        labels: Ground truth labels (y, x)
        class_names: Dictionary mapping class ID to name
        class_ids: List of class IDs to plot (if None, plots all labeled classes)
        wavelengths: Wavelength array (if None, uses cube.coords['band'])
        save_path: Optional path to save figure
    """
    if wavelengths is None:
        wavelengths = cube.coords['band'].values
    
    if class_ids is None:
        # Plot all labeled classes (exclude 0)
        class_ids = [cid for cid in np.unique(labels) if cid > 0]
    
    # Water vapor absorption regions
    water_vapor_regions = [(1350, 1450), (1800, 1960), (2450, 2500)]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot water vapor regions as grey shaded areas
    for wv_min, wv_max in water_vapor_regions:
        ax.axvspan(wv_min, wv_max, alpha=0.2, color='grey', label='Water vapor absorption' if wv_min == 1350 else '')
    
    # Key wavelengths
    ax.axvline(670, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Chlorophyll absorption')
    ax.axvline(865, color='green', linestyle='--', alpha=0.5, linewidth=1, label='NIR plateau')
    
    # Plot spectra for each class
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_ids)))
    
    for i, class_id in enumerate(class_ids):
        class_mask = (labels == class_id)
        if np.sum(class_mask) == 0:
            continue
        
        # Extract spectra for this class
        class_spectra = cube.values[:, class_mask]  # (n_bands, n_pixels)
        
        # Compute mean and std
        mean_spectrum = np.nanmean(class_spectra, axis=1)
        std_spectrum = np.nanstd(class_spectra, axis=1)
        
        # Plot mean ± std
        ax.plot(wavelengths, mean_spectrum, color=colors[i], linewidth=2,
                label=class_names.get(class_id, f"Class {class_id}"))
        ax.fill_between(wavelengths, mean_spectrum - std_spectrum,
                        mean_spectrum + std_spectrum, color=colors[i], alpha=0.2)
    
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Reflectance", fontsize=12)
    ax.set_title("Spectral Signatures by Class", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(wavelengths.min(), wavelengths.max())
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved spectral signatures to {save_path}")
    
    # Always show in notebook
    plt.show()


def plot_spectral_indices_maps(
    indices_dataset: xr.Dataset,
    save_path: Optional[str] = None
) -> None:
    """
    Plot grid of 2D maps for each spectral index.
    
    Args:
        indices_dataset: xr.Dataset with index variables (ndvi, evi, nbr, ndwi)
        save_path: Optional path to save figure
    """
    indices_to_plot = ['ndvi', 'evi', 'nbr', 'ndwi']
    colormaps = {
        'ndvi': 'RdYlGn',
        'evi': 'RdYlGn',
        'nbr': 'RdBu_r',
        'ndwi': 'Blues'
    }
    
    n_indices = len([idx for idx in indices_to_plot if idx in indices_dataset.data_vars])
    
    if n_indices == 0:
        logger.warning("No indices found in dataset")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    for idx_name in indices_to_plot:
        if idx_name not in indices_dataset.data_vars:
            continue
        
        ax = axes[plot_idx]
        index_data = indices_dataset[idx_name].values
        
        # Choose colormap
        cmap = colormaps.get(idx_name, 'viridis')
        
        im = ax.imshow(index_data, cmap=cmap, interpolation='nearest')
        ax.set_title(f"{idx_name.upper()}", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Index Value', fontsize=10)
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Spectral Indices Maps", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved indices maps to {save_path}")
    
    # Always show in notebook
    plt.show()


def plot_pca_components(
    pca_cube: xr.DataArray,
    n_show: int = 6,
    save_path: Optional[str] = None
) -> None:
    """
    Show first n_show PCA component images as spatial maps.
    
    Args:
        pca_cube: PCA-transformed cube (component, y, x)
        n_show: Number of components to display
        save_path: Optional path to save figure
    """
    n_components = min(n_show, pca_cube.shape[0])
    
    # Get explained variance from attrs if available
    if 'explained_variance_ratio' in pca_cube.attrs:
        explained_var = pca_cube.attrs['explained_variance_ratio']
    else:
        explained_var = None
    
    n_cols = 3
    n_rows = (n_components + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i in range(n_components):
        ax = axes[i]
        component_data = pca_cube.isel(component=i).values
        
        # Stretch to 2nd-98th percentile
        p2, p98 = np.nanpercentile(component_data, [2, 98])
        component_stretched = np.clip((component_data - p2) / (p98 - p2 + 1e-10), 0, 1)
        
        im = ax.imshow(component_stretched, cmap='gray', interpolation='nearest')
        ax.axis('off')
        
        var_str = ""
        if explained_var is not None:
            var_pct = explained_var[i] * 100
            var_str = f" ({var_pct:.1f}%)"
        
        ax.set_title(f"PC {i+1}{var_str}", fontsize=11, fontweight='bold')
    
    # Hide unused subplots
    for i in range(n_components, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Principal Component Images", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PCA components to {save_path}")
    
    # Always show in notebook
    plt.show()


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history: loss and accuracy curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss curves
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    if 'train_acc' in history:
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Accuracy', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Mark early stopping if present
    if 'best_epoch' in history:
        best_epoch = history['best_epoch']
        ax1.axvline(best_epoch, color='green', linestyle='--', alpha=0.7, label='Best Model')
        ax2.axvline(best_epoch, color='green', linestyle='--', alpha=0.7, label='Best Model')
        ax1.legend()
        ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history to {save_path}")
    
    # Always show in notebook
    plt.show()


def plot_abundance_maps(
    abundances: xr.Dataset,
    save_path: Optional[str] = None
) -> None:
    """
    Plot abundance maps for each endmember.
    
    Args:
        abundances: xr.Dataset with endmember abundance variables (y, x)
        save_path: Optional path to save figure
    """
    endmember_names = list(abundances.data_vars)
    n_endmembers = len(endmember_names)
    
    n_cols = 2
    n_rows = (n_endmembers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, endmember_name in enumerate(endmember_names):
        ax = axes[i]
        abundance_map = abundances[endmember_name].values
        
        im = ax.imshow(abundance_map, cmap='YlOrRd', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f"{endmember_name} Abundance", fontsize=11, fontweight='bold')
        ax.axis('off')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Fraction', fontsize=9)
    
    # Hide unused subplots
    for i in range(n_endmembers, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Spectral Unmixing Abundance Maps", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved abundance maps to {save_path}")
    
    # Always show in notebook
    plt.show()
