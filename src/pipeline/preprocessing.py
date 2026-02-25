"""
Preprocessing Module

Cleans the reflectance cube before analysis: removes bad bands, water vapor
absorption regions, normalizes data, and prepares for machine learning.
"""

import numpy as np
import xarray as xr
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def remove_water_vapor_bands(
    cube: xr.DataArray,
    wavelengths: np.ndarray,
    absorption_regions: List[Tuple[float, float]] = [(1350, 1450), (1800, 1960), (2450, 2500)]
) -> Tuple[xr.DataArray, np.ndarray]:
    """
    Remove spectral bands in water vapor absorption regions.
    
    Water vapor strongly absorbs radiation at specific wavelengths, making
    these bands unreliable for analysis. Standard regions to remove:
    - 1350-1450 nm: Strong water vapor absorption
    - 1800-1960 nm: Strong water vapor absorption  
    - 2450-2500 nm: End of SWIR, low SNR
    
    Args:
        cube: Hyperspectral cube (band, y, x)
        wavelengths: Wavelength array in nm (n_bands,)
        absorption_regions: List of (min, max) wavelength tuples in nm
        
    Returns:
        Tuple of:
        - cleaned_cube: Cube with absorption bands removed (n_clean_bands, y, x)
        - remaining_wavelengths: Wavelengths of remaining bands (n_clean_bands,)
    """
    # Create mask for bands to keep
    keep_mask = np.ones(len(wavelengths), dtype=bool)
    
    bands_removed_per_region = {}
    
    for region_min, region_max in absorption_regions:
        region_mask = (wavelengths >= region_min) & (wavelengths <= region_max)
        n_removed = np.sum(region_mask)
        keep_mask[region_mask] = False
        bands_removed_per_region[f"{region_min}-{region_max}nm"] = n_removed
        if n_removed > 0:
            logger.info(f"Removed {n_removed} bands in water vapor region {region_min}-{region_max} nm")
    
    # Apply mask
    cleaned_cube = cube.isel(band=keep_mask)
    remaining_wavelengths = wavelengths[keep_mask]
    
    total_removed = len(wavelengths) - len(remaining_wavelengths)
    logger.info(f"Removed {total_removed} water vapor absorption bands ({len(remaining_wavelengths)} bands remaining)")
    
    return cleaned_cube, remaining_wavelengths


def remove_noisy_bands(
    cube: xr.DataArray,
    snr_threshold: float = 50.0
) -> xr.DataArray:
    """
    Remove spectral bands with low signal-to-noise ratio.
    
    Estimates per-band SNR from spatial variance across pixels.
    SNR ≈ mean / std across all pixels per band.
    
    Args:
        cube: Hyperspectral cube (band, y, x)
        snr_threshold: Minimum SNR to keep a band
        
    Returns:
        Cleaned cube with low-SNR bands removed
    """
    # Compute per-band statistics
    # Reshape to (n_bands, n_pixels)
    cube_2d = cube.values.reshape(cube.shape[0], -1)
    
    # Remove NaN pixels for SNR calculation
    valid_mask = ~np.isnan(cube_2d)
    
    band_means = np.zeros(cube.shape[0])
    band_stds = np.zeros(cube.shape[0])
    
    for band_idx in range(cube.shape[0]):
        band_data = cube_2d[band_idx, valid_mask[band_idx, :]]
        if len(band_data) > 0:
            band_means[band_idx] = np.mean(band_data)
            band_stds[band_idx] = np.std(band_data)
        else:
            band_means[band_idx] = 0.0
            band_stds[band_idx] = np.inf
    
    # Estimate SNR: mean / std
    snr_estimates = np.where(band_stds > 0, band_means / band_stds, 0.0)
    
    # Keep bands above threshold
    keep_mask = snr_estimates >= snr_threshold
    
    # Ensure at least some bands remain (keep top 50% if all would be removed)
    n_kept = np.sum(keep_mask)
    if n_kept == 0:
        logger.warning(f"All bands would be removed with SNR threshold {snr_threshold}. Keeping top 50% of bands.")
        # Keep top 50% by SNR
        n_keep = max(1, cube.shape[0] // 2)
        top_indices = np.argsort(snr_estimates)[-n_keep:]
        keep_mask = np.zeros(cube.shape[0], dtype=bool)
        keep_mask[top_indices] = True
        n_kept = np.sum(keep_mask)
    
    n_removed = np.sum(~keep_mask)
    
    if n_removed > 0:
        logger.info(f"Removed {n_removed} noisy bands (SNR < {snr_threshold})")
        if n_removed < len(snr_estimates):
            logger.info(f"SNR range of removed bands: [{np.min(snr_estimates[~keep_mask]):.2f}, {np.max(snr_estimates[~keep_mask]):.2f}]")
    
    cleaned_cube = cube.isel(band=keep_mask)
    
    if cleaned_cube.shape[0] == 0:
        raise ValueError("No bands remaining after noise removal. Check SNR threshold.")
    
    logger.info(f"Kept {n_kept} bands after noise removal")
    
    return cleaned_cube


def normalize_reflectance(
    cube: xr.DataArray,
    method: str = "standardize",
    train_mask: Optional[np.ndarray] = None
) -> Tuple[xr.DataArray, Dict]:
    """
    Normalize reflectance cube for machine learning.
    
    IMPORTANT: Only compute statistics from training pixels to avoid
    data leakage. Use train_mask to specify which pixels to use for
    computing normalization parameters.
    
    Args:
        cube: Hyperspectral cube (band, y, x)
        method: Normalization method - "standardize", "minmax", or "l2"
        train_mask: Optional boolean mask (y, x) indicating training pixels.
                   If None, uses all valid (non-NaN) pixels.
        
    Returns:
        Tuple of:
        - normalized_cube: Normalized cube (same shape as input)
        - norm_params: Dictionary with normalization parameters for inverse transform
        
    Raises:
        ValueError: If method is not recognized
    """
    if method not in ["standardize", "minmax", "l2"]:
        raise ValueError(f"Unknown normalization method: {method}. Must be 'standardize', 'minmax', or 'l2'")
    
    # Check if cube is empty
    if cube.shape[0] == 0:
        raise ValueError("Cannot normalize empty cube (0 bands). Check preprocessing steps.")
    
    # Create training mask
    if train_mask is None:
        # Use all valid pixels (non-NaN)
        # Check first band if available, otherwise check any band
        if cube.shape[0] > 0:
            valid_pixels = ~np.isnan(cube.values[0, :, :])  # Check first band
        else:
            # Fallback: check all bands and combine
            valid_pixels = ~np.isnan(cube.values).any(axis=0)
        train_mask = np.broadcast_to(valid_pixels[np.newaxis, :, :], cube.shape)
    else:
        # Broadcast train_mask to (band, y, x)
        train_mask = np.broadcast_to(train_mask[np.newaxis, :, :], cube.shape)
    
    # Reshape to (n_bands, n_pixels)
    cube_2d = cube.values.reshape(cube.shape[0], -1)
    train_mask_2d = train_mask.reshape(cube.shape[0], -1)
    
    norm_params = {"method": method}
    normalized_values = np.zeros_like(cube.values)
    
    if method == "standardize":
        # Per-band: subtract mean, divide by std
        # Compute stats only from training pixels
        band_means = np.zeros(cube.shape[0])
        band_stds = np.zeros(cube.shape[0])
        
        for band_idx in range(cube.shape[0]):
            train_data = cube_2d[band_idx, train_mask_2d[band_idx, :]]
            train_data = train_data[~np.isnan(train_data)]
            
            if len(train_data) > 0:
                band_means[band_idx] = np.mean(train_data)
                band_stds[band_idx] = np.std(train_data)
                if band_stds[band_idx] < 1e-10:
                    band_stds[band_idx] = 1.0  # Avoid division by zero
            else:
                band_means[band_idx] = 0.0
                band_stds[band_idx] = 1.0
        
        # Apply normalization: (x - mean) / std
        for band_idx in range(cube.shape[0]):
            normalized_values[band_idx, :, :] = (
                cube.values[band_idx, :, :] - band_means[band_idx]
            ) / band_stds[band_idx]
        
        norm_params["mean"] = band_means
        norm_params["std"] = band_stds
        
        logger.info(f"Standardized: mean range [{np.min(band_means):.4f}, {np.max(band_means):.4f}], std range [{np.min(band_stds):.4f}, {np.max(band_stds):.4f}]")
    
    elif method == "minmax":
        # Per-band: scale to [0, 1]
        band_mins = np.zeros(cube.shape[0])
        band_maxs = np.zeros(cube.shape[0])
        
        for band_idx in range(cube.shape[0]):
            train_data = cube_2d[band_idx, train_mask_2d[band_idx, :]]
            train_data = train_data[~np.isnan(train_data)]
            
            if len(train_data) > 0:
                band_mins[band_idx] = np.min(train_data)
                band_maxs[band_idx] = np.max(train_data)
                band_range = band_maxs[band_idx] - band_mins[band_idx]
                if band_range < 1e-10:
                    band_range = 1.0  # Avoid division by zero
            else:
                band_mins[band_idx] = 0.0
                band_maxs[band_idx] = 1.0
                band_range = 1.0
        
        # Apply normalization: (x - min) / (max - min)
        for band_idx in range(cube.shape[0]):
            normalized_values[band_idx, :, :] = (
                cube.values[band_idx, :, :] - band_mins[band_idx]
            ) / (band_maxs[band_idx] - band_mins[band_idx])
        
        norm_params["min"] = band_mins
        norm_params["max"] = band_maxs
        
        logger.info(f"Min-max normalized: range [{np.min(band_mins):.4f}, {np.max(band_maxs):.4f}]")
    
    elif method == "l2":
        # Per-pixel: normalize each spectrum to unit L2 norm
        # Reshape to (n_pixels, n_bands)
        cube_pixels = cube.values.transpose(1, 2, 0).reshape(-1, cube.shape[0])
        
        # Compute L2 norm per pixel
        l2_norms = np.linalg.norm(cube_pixels, axis=1, keepdims=True)
        l2_norms = np.where(l2_norms < 1e-10, 1.0, l2_norms)  # Avoid division by zero
        
        # Normalize
        normalized_pixels = cube_pixels / l2_norms
        
        # Reshape back to (band, y, x)
        normalized_values = normalized_pixels.reshape(cube.shape[1], cube.shape[2], cube.shape[0]).transpose(2, 0, 1)
        
        norm_params["l2_norms"] = l2_norms.reshape(cube.shape[1], cube.shape[2])
        
        logger.info(f"L2 normalized: mean norm = {np.mean(l2_norms):.4f}")
    
    # Preserve NaN values
    normalized_values = np.where(np.isnan(cube.values), np.nan, normalized_values)
    
    normalized_cube = xr.DataArray(
        normalized_values,
        dims=cube.dims,
        coords=cube.coords,
        attrs={
            **cube.attrs,
            "normalization_method": method,
            "normalized": True
        }
    )
    
    return normalized_cube, norm_params


def extract_pixel_matrix(
    cube: xr.DataArray,
    labels: np.ndarray,
    exclude_unlabeled: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract pixel matrix from hyperspectral cube for machine learning.
    
    Reshapes cube from (band, y, x) to (n_pixels, n_bands) and extracts
    corresponding labels. Optionally filters out unlabeled pixels.
    
    Args:
        cube: Hyperspectral cube (band, y, x)
        labels: Ground truth labels (y, x)
        exclude_unlabeled: If True, exclude pixels with label == 0
        
    Returns:
        Tuple of:
        - X: Pixel matrix (n_pixels, n_bands)
        - y: Label vector (n_pixels,)
    """
    # Reshape cube to (n_pixels, n_bands)
    # Transpose from (band, y, x) to (y, x, band), then reshape
    cube_3d = cube.values.transpose(1, 2, 0)  # (y, x, band)
    n_pixels = cube_3d.shape[0] * cube_3d.shape[1]
    X = cube_3d.reshape(n_pixels, cube_3d.shape[2])
    
    # Reshape labels to (n_pixels,)
    y = labels.reshape(n_pixels)
    
    # Filter out unlabeled pixels if requested
    if exclude_unlabeled:
        labeled_mask = y > 0
        X = X[labeled_mask, :]
        y = y[labeled_mask]
        logger.info(f"Extracted {len(y)} labeled pixels (excluded {n_pixels - len(y)} unlabeled)")
    else:
        logger.info(f"Extracted {len(y)} pixels (including unlabeled)")
    
    return X, y
