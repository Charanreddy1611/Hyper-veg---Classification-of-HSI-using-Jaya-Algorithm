"""
Spectral Unmixing Module

Decomposes mixed pixels into endmember fractions using Fully Constrained
Least Squares (FCLS) linear unmixing.

Physics: Linear Mixing Model
    x = E × a + ε
    subject to: a_k >= 0 (non-negativity)
                sum(a_k) = 1 (sum-to-one)
"""

import numpy as np
import xarray as xr
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def extract_endmembers_manual(
    cube: xr.DataArray,
    labels: np.ndarray,
    class_ids: List[int],
    class_names: Dict[int, str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract endmember spectra from labeled pixels.
    
    Computes mean spectrum for each requested class from ground truth,
    providing physically meaningful endmembers.
    
    Args:
        cube: Reflectance cube (band, y, x)
        labels: Ground truth labels (y, x)
        class_ids: List of class IDs to extract as endmembers
        class_names: Dictionary mapping class ID to name
        
    Returns:
        Tuple of:
        - endmember_matrix: Endmember spectra (n_bands, n_endmembers)
        - endmember_names: List of endmember names
    """
    n_bands = cube.shape[0]
    n_endmembers = len(class_ids)
    
    endmember_matrix = np.zeros((n_bands, n_endmembers))
    endmember_names = []
    
    for i, class_id in enumerate(class_ids):
        # Find pixels of this class
        class_mask = (labels == class_id)
        
        if np.sum(class_mask) == 0:
            logger.warning(f"No pixels found for class {class_id}, skipping")
            continue
        
        # Extract spectra for this class
        class_spectra = cube.values[:, class_mask]  # (n_bands, n_pixels)
        
        # Compute mean spectrum
        mean_spectrum = np.mean(class_spectra, axis=1)
        endmember_matrix[:, i] = mean_spectrum
        
        class_name = class_names.get(class_id, f"Class_{class_id}")
        endmember_names.append(class_name)
        
        logger.info(f"Extracted endmember {i+1}/{n_endmembers}: {class_name} ({np.sum(class_mask)} pixels)")
    
    return endmember_matrix, endmember_names


def fully_constrained_unmix_pixel(
    pixel_spectrum: np.ndarray,
    endmember_matrix: np.ndarray
) -> np.ndarray:
    """
    Solve Fully Constrained Least Squares (FCLS) for a single pixel.
    
    Minimizes ||x - E×a||² subject to:
    - a_k >= 0 (non-negativity)
    - sum(a_k) = 1 (sum-to-one)
    
    Args:
        pixel_spectrum: Pixel spectrum vector (n_bands,)
        endmember_matrix: Endmember matrix (n_bands, n_endmembers)
        
    Returns:
        Abundance vector (n_endmembers,) with fractions summing to 1
    """
    n_endmembers = endmember_matrix.shape[1]
    
    # Objective function: ||x - E×a||²
    def objective(abundances):
        reconstructed = endmember_matrix @ abundances
        residual = pixel_spectrum - reconstructed
        return np.sum(residual ** 2)
    
    # Constraints
    # Sum-to-one: sum(a) = 1
    constraints = {
        'type': 'eq',
        'fun': lambda a: np.sum(a) - 1.0
    }
    
    # Bounds: non-negativity a_k >= 0
    bounds = [(0.0, 1.0)] * n_endmembers
    
    # Initial guess: uniform abundances
    x0 = np.ones(n_endmembers) / n_endmembers
    
    # Solve optimization
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    if not result.success:
        logger.debug(f"Optimization warning: {result.message}")
    
    abundances = np.clip(result.x, 0.0, 1.0)
    # Renormalize to ensure sum-to-one
    abundances = abundances / (np.sum(abundances) + 1e-10)
    
    return abundances


def unmix_cube(
    cube: xr.DataArray,
    endmember_matrix: np.ndarray,
    endmember_names: List[str],
    valid_mask: Optional[np.ndarray] = None
) -> xr.Dataset:
    """
    Apply FCLS unmixing to every valid pixel in the cube.
    
    Args:
        cube: Reflectance cube (band, y, x)
        endmember_matrix: Endmember spectra (n_bands, n_endmembers)
        endmember_names: List of endmember names
        valid_mask: Optional boolean mask (y, x) indicating valid pixels
        
    Returns:
        xr.Dataset with one DataArray per endmember (y, x shape)
        Each variable is the fractional abundance map for that endmember
    """
    n_endmembers = endmember_matrix.shape[1]
    n_y, n_x = cube.shape[1], cube.shape[2]
    
    # Create valid mask
    if valid_mask is None:
        valid_mask = ~np.isnan(cube.values[0, :, :])
    
    # Initialize abundance maps
    abundance_maps = {}
    
    for i, endmember_name in enumerate(endmember_names):
        abundance_maps[endmember_name] = np.zeros((n_y, n_x))
    
    # Reshape cube to (n_pixels, n_bands)
    cube_2d = cube.values.transpose(1, 2, 0).reshape(-1, cube.shape[0])
    valid_mask_flat = valid_mask.reshape(-1)
    
    # Unmix each valid pixel
    valid_indices = np.where(valid_mask_flat)[0]
    
    logger.info(f"Unmixing {len(valid_indices)} valid pixels...")
    
    for pixel_idx in tqdm(valid_indices, desc="Unmixing pixels"):
        pixel_spectrum = cube_2d[pixel_idx, :]
        
        # Skip if spectrum has NaN
        if np.isnan(pixel_spectrum).any():
            continue
        
        # Solve FCLS
        abundances = fully_constrained_unmix_pixel(pixel_spectrum, endmember_matrix)
        
        # Store abundances
        y_idx, x_idx = np.unravel_index(pixel_idx, (n_y, n_x))
        for i, endmember_name in enumerate(endmember_names):
            abundance_maps[endmember_name][y_idx, x_idx] = abundances[i]
    
    # Create xarray Dataset
    abundance_arrays = {}
    for endmember_name, abundance_map in abundance_maps.items():
        abundance_arrays[endmember_name] = xr.DataArray(
            abundance_map,
            dims=["y", "x"],
            coords={"y": cube.coords["y"], "x": cube.coords["x"]},
            attrs={
                "endmember": endmember_name,
                "units": "fraction",
                "description": f"Fractional abundance of {endmember_name}"
            }
        )
    
    abundance_dataset = xr.Dataset(abundance_arrays)
    abundance_dataset.attrs = {
        "description": "FCLS spectral unmixing results",
        "n_endmembers": n_endmembers,
        "endmembers": endmember_names
    }
    
    logger.info(f"Completed unmixing: {n_endmembers} endmembers")
    
    return abundance_dataset


def compute_reconstruction_error(
    cube: xr.DataArray,
    abundances: xr.Dataset,
    endmember_matrix: np.ndarray
) -> xr.DataArray:
    """
    Compute per-pixel reconstruction error from unmixing.
    
    Reconstructs spectra from abundances and compares to original.
    
    Args:
        cube: Original reflectance cube (band, y, x)
        abundances: Abundance Dataset with endmember variables (y, x)
        endmember_matrix: Endmember spectra (n_bands, n_endmembers)
        
    Returns:
        Reconstruction error map (y, x) as RMSE per pixel
    """
    n_y, n_x = cube.shape[1], cube.shape[2]
    n_bands = cube.shape[0]
    n_endmembers = endmember_matrix.shape[1]
    
    # Extract abundance arrays
    abundance_arrays = [abundances[name].values for name in abundances.data_vars]
    abundance_stack = np.stack(abundance_arrays, axis=0)  # (n_endmembers, y, x)
    
    # Reshape for matrix multiplication
    abundance_flat = abundance_stack.transpose(1, 2, 0).reshape(-1, n_endmembers)  # (n_pixels, n_endmembers)
    cube_flat = cube.values.transpose(1, 2, 0).reshape(-1, n_bands)  # (n_pixels, n_bands)
    
    # Reconstruct: x_hat = E × a
    reconstructed_flat = abundance_flat @ endmember_matrix.T  # (n_pixels, n_bands)
    
    # Compute RMSE per pixel
    error_flat = np.sqrt(np.mean((cube_flat - reconstructed_flat) ** 2, axis=1))
    
    # Reshape back to (y, x)
    error_map = error_flat.reshape(n_y, n_x)
    
    # Create DataArray
    error_da = xr.DataArray(
        error_map,
        dims=["y", "x"],
        coords={"y": cube.coords["y"], "x": cube.coords["x"]},
        attrs={
            "description": "Reconstruction RMSE from spectral unmixing",
            "units": "reflectance"
        }
    )
    
    logger.info(f"Reconstruction error: mean = {np.nanmean(error_map):.4f}, std = {np.nanstd(error_map):.4f}")
    
    return error_da
