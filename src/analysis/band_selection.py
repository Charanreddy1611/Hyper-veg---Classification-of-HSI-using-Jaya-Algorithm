"""
Jaya-Based Hyperspectral Band Selection Module

Implements the band selection method from:
Patro, R.N., Subudhi, S., Biswal, P.K. (2019). Spectral clustering
and spatial Frobenius norm-based Jaya optimisation for BS of
hyperspectral images. IET Image Processing, 13(2), 307-315.

This module replaces PCA/MNF with a feature SELECTION approach that
preserves the physical meaning of original spectral bands.
"""

import numpy as np
import xarray as xr
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def kmeans_spectral_clustering(
    cube: xr.DataArray,
    n_clusters: int = 17,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform K-means clustering on pixel spectra.
    
    Clusters all pixel spectra into k clusters to identify distinct
    spectral classes. We use k=17 (slightly above the 16 ground truth
    classes) to capture spectral signatures from unlabelled pixels that
    may form distinct clusters not covered by ground truth labels.
    
    Physics:
        K-means identifies natural groupings in the high-dimensional
        spectral space. Each cluster center represents a prototypical
        spectral signature. These centers are used to measure spectral
        distinctiveness between bands.
    
    Args:
        cube: Hyperspectral cube (band, y, x)
        n_clusters: Number of clusters (default 17)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of:
        - cluster_centres: Cluster center spectra (n_clusters, n_bands)
        - cluster_labels: Cluster assignment per pixel (n_pixels,)
    """
    # Reshape to (n_pixels, n_bands)
    cube_2d = cube.values.transpose(1, 2, 0).reshape(-1, cube.shape[0])
    
    # Remove NaN pixels
    valid_mask = ~np.isnan(cube_2d).any(axis=1)
    cube_valid = cube_2d[valid_mask, :]
    
    logger.info(f"Running K-means on {len(cube_valid)} valid pixels...")
    
    # Run K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels_full = np.zeros(len(cube_2d), dtype=int)
    cluster_labels_full[valid_mask] = kmeans.fit_predict(cube_valid)
    cluster_labels_full[~valid_mask] = -1  # Mark invalid pixels
    
    cluster_centres = kmeans.cluster_centers_  # (n_clusters, n_bands)
    
    logger.info(f"K-means complete: {n_clusters} clusters from {len(cube_valid)} pixels")
    
    return cluster_centres, cluster_labels_full


def gaussian_spatial_filter(
    cube: xr.DataArray,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply 2D Gaussian spatial filter to each spectral band.
    
    Gaussian filtering makes spatial texture patterns explicit, enabling
    the Frobenius norm to measure spatial dissimilarity between bands.
    This captures complementary spatial information beyond spectral
    differences.
    
    Physics:
        Gaussian blur reduces high-frequency noise while preserving
        spatial structure. The filtered bands reveal spatial texture
        patterns that differ between bands, even when spectral signatures
        are similar.
    
    Args:
        cube: Hyperspectral cube (band, y, x)
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Filtered cube as numpy array (n_bands, y, x)
    """
    filtered = np.zeros_like(cube.values)
    
    # Apply filter to each band independently
    for band_idx in range(cube.shape[0]):
        filtered[band_idx, :, :] = gaussian_filter(
            cube.values[band_idx, :, :],
            sigma=sigma,
            mode='reflect'
        )
    
    logger.info(f"Applied Gaussian spatial filter (sigma={sigma}) to {cube.shape[0]} bands")
    
    return filtered


def compute_spectral_distance(
    cluster_centres: np.ndarray,
    band_indices: np.ndarray
) -> float:
    """
    Compute spectral distinctiveness score for selected bands.
    
    Measures how different the spectral class signatures are between
    selected bands using L1 distance between cluster centers.
    
    Physics:
        For each pair of selected bands (i, j), compute the L1 distance
        between cluster center vectors at those bands. High distance
        indicates bands carry complementary spectral class information,
        not redundant.
    
    Args:
        cluster_centres: Cluster center spectra (n_clusters, n_bands)
        band_indices: Selected band indices (n_select,)
        
    Returns:
        Normalized sum of pairwise spectral distances (higher = better)
    """
    n_select = len(band_indices)
    
    if n_select < 2:
        return 0.0
    
    # Compute pairwise L1 distances
    D1_matrix = np.zeros((n_select, n_select))
    
    for i, band_i in enumerate(band_indices):
        for j, band_j in enumerate(band_indices):
            if i != j:
                # L1 distance between cluster centers at these bands
                D1_ij = np.sum(np.abs(cluster_centres[:, band_i] - cluster_centres[:, band_j]))
                D1_matrix[i, j] = D1_ij
    
    # Normalize by maximum value
    max_D1 = np.max(D1_matrix)
    if max_D1 > 0:
        D1_matrix_normalized = D1_matrix / max_D1
    else:
        D1_matrix_normalized = D1_matrix
    
    # Return sum of all pairwise distances (upper triangle only to avoid double counting)
    score = np.sum(np.triu(D1_matrix_normalized, k=1))
    
    return score


def compute_frobenius_distance(
    filtered_bands: np.ndarray,
    band_indices: np.ndarray
) -> float:
    """
    Compute spatial de-correlation score using Frobenius norm.
    
    Measures how different the spatial texture patterns are between
    selected bands after Gaussian filtering.
    
    Physics:
        Frobenius norm: ||A||_F = sqrt(tr(A^T A)) = sqrt(sum of squared elements)
        
        For each pair of bands (i, j), compute:
        D2_ij = ||B_f_i - B_f_j||_F
        where B_f_i is the spatially filtered 2D image of band i.
        
        High D2 indicates bands carry complementary spatial information.
    
    Args:
        filtered_bands: Spatially filtered cube (n_bands, y, x)
        band_indices: Selected band indices (n_select,)
        
    Returns:
        Normalized sum of pairwise Frobenius distances (higher = better)
    """
    n_select = len(band_indices)
    
    if n_select < 2:
        return 0.0
    
    # Compute pairwise Frobenius distances
    D2_matrix = np.zeros((n_select, n_select))
    
    for i, band_i in enumerate(band_indices):
        for j, band_j in enumerate(band_indices):
            if i != j:
                # Flatten the 2D band images to vectors
                vec_i = filtered_bands[band_i, :, :].ravel()
                vec_j = filtered_bands[band_j, :, :].ravel()
                
                # Frobenius norm: sqrt(sum of squared differences)
                diff = vec_i - vec_j
                D2_ij = np.sqrt(np.dot(diff, diff))
                D2_matrix[i, j] = D2_ij
    
    # Normalize by maximum value
    max_D2 = np.max(D2_matrix)
    if max_D2 > 0:
        D2_matrix_normalized = D2_matrix / max_D2
    else:
        D2_matrix_normalized = D2_matrix
    
    # Return sum of all pairwise distances (upper triangle only)
    score = np.sum(np.triu(D2_matrix_normalized, k=1))
    
    return score


def jaya_fitness(
    band_indices: np.ndarray,
    cluster_centres: np.ndarray,
    filtered_bands: np.ndarray
) -> float:
    """
    Compute combined fitness function for Jaya optimization.
    
    Combined fitness: f(B) = D1_score + D2_score
    
    This is a MAXIMISATION problem - we want selected bands to be
    maximally distinct both spectrally AND spatially.
    
    Args:
        band_indices: Selected band indices (n_select,)
        cluster_centres: Cluster center spectra (n_clusters, n_bands)
        filtered_bands: Spatially filtered cube (n_bands, y, x)
        
    Returns:
        Combined fitness score (higher = better), or -inf if duplicates
    """
    # Check for duplicates
    if len(band_indices) != len(np.unique(band_indices)):
        return -np.inf
    
    # Compute both objectives
    D1_score = compute_spectral_distance(cluster_centres, band_indices)
    D2_score = compute_frobenius_distance(filtered_bands, band_indices)
    
    # Combined fitness
    fitness = D1_score + D2_score
    
    return fitness


def run_jaya_single(
    cluster_centres: np.ndarray,
    filtered_bands: np.ndarray,
    n_bands_total: int = 200,
    n_select: int = 5,
    population_size: int = 10,
    max_iterations: int = 50,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, float, list[float]]:
    """
    Run a single Jaya optimization to select optimal bands.
    
    Implements the Jaya algorithm from Patro et al. (2019), Fig. 3.
    Jaya has no algorithm-specific hyperparameters - only random numbers
    r1 and r2 sampled each iteration.
    
    Physics:
        Jaya update equation:
        X'_j,k,i = X_j,k,i + r1*(X_j,best,i - |X_j,k,i|)
                              - r2*(X_j,worst,i - |X_j,k,i|)
        
        This moves solutions toward the best and away from the worst,
        with no tuning parameters needed.
    
    Args:
        cluster_centres: Cluster center spectra (n_clusters, n_bands)
        filtered_bands: Spatially filtered cube (n_bands, y, x)
        n_bands_total: Total number of bands
        n_select: Number of bands to select per run
        population_size: Number of candidate solutions
        max_iterations: Maximum iterations per run
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of:
        - G_sol: Best solution (selected band indices) (n_select,)
        - G_best: Best fitness value
        - cost_history: Fitness history per iteration (max_iterations,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize population: random integer matrix [population_size × n_select]
    # Each row is a candidate solution with n_select unique band indices
    population = np.zeros((population_size, n_select), dtype=int)
    
    for k in range(population_size):
        # Sample n_select unique bands
        population[k, :] = np.random.choice(
            n_bands_total, size=n_select, replace=False
        )
    
    # Evaluate initial fitness
    fitness_values = np.zeros(population_size)
    for k in range(population_size):
        fitness_values[k] = jaya_fitness(
            population[k, :], cluster_centres, filtered_bands
        )
    
    # Find best and worst
    best_idx = np.argmax(fitness_values)
    worst_idx = np.argmin(fitness_values)
    
    P_best = fitness_values[best_idx]
    P_sol_best = population[best_idx, :].copy()
    P_sol_worst = population[worst_idx, :].copy()
    
    G_best = P_best
    G_sol = P_sol_best.copy()
    
    cost_history = [G_best]
    
    # Main Jaya loop
    for iteration in range(max_iterations):
        # Update each candidate solution
        for k in range(population_size):
            new_solution = population[k, :].copy()
            
            # Update each variable (band position)
            for j in range(n_select):
                r1, r2 = np.random.uniform(0, 1, size=2)
                
                # Jaya update equation
                X_new = (population[k, j] +
                        r1 * (P_sol_best[j] - np.abs(population[k, j])) -
                        r2 * (P_sol_worst[j] - np.abs(population[k, j])))
                
                # Round and clip
                X_new = int(round(X_new))
                X_new = np.clip(X_new, 0, n_bands_total - 1)
                
                new_solution[j] = X_new
            
            # Resolve duplicates
            unique_vals, counts = np.unique(new_solution, return_counts=True)
            if len(unique_vals) < n_select:
                # Has duplicates - replace with random unused bands
                duplicates = unique_vals[counts > 1]
                used_bands = set(unique_vals)
                available_bands = [b for b in range(n_bands_total) if b not in used_bands]
                
                if len(available_bands) >= len(duplicates):
                    replacement_bands = np.random.choice(
                        available_bands, size=len(duplicates), replace=False
                    )
                    
                    # Replace first occurrence of each duplicate
                    for dup, repl in zip(duplicates, replacement_bands):
                        dup_indices = np.where(new_solution == dup)[0]
                        if len(dup_indices) > 1:
                            new_solution[dup_indices[0]] = repl
            
            population[k, :] = new_solution
        
        # Re-evaluate fitness
        for k in range(population_size):
            fitness_values[k] = jaya_fitness(
                population[k, :], cluster_centres, filtered_bands
            )
        
        # Update best and worst
        best_idx = np.argmax(fitness_values)
        worst_idx = np.argmin(fitness_values)
        
        P_best = fitness_values[best_idx]
        P_sol_best = population[best_idx, :].copy()
        P_sol_worst = population[worst_idx, :].copy()
        
        # Update global best
        if P_best > G_best:
            G_best = P_best
            G_sol = P_sol_best.copy()
        
        cost_history.append(G_best)
    
    return G_sol, G_best, cost_history


def run_jaya_with_ranking(
    cube: xr.DataArray,
    n_clusters: int = 17,
    population_size: int = 10,
    max_iterations: int = 50,
    bands_per_run: int = 5,
    n_evaluations: int = 50,
    final_n_bands: int = 18,
    gaussian_sigma: float = 1.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full Jaya band selection pipeline with ranking.
    
    Implements the complete pipeline from Patro et al. (2019):
    1. K-means spectral clustering
    2. Gaussian spatial filtering
    3. Multiple independent Jaya runs
    4. Ranking and thresholding to select final bands
    
    Args:
        cube: Hyperspectral cube (band, y, x)
        n_clusters: Number of K-means clusters
        population_size: Jaya population size
        max_iterations: Jaya max iterations per run
        bands_per_run: Bands selected per Jaya run
        n_evaluations: Number of independent Jaya runs
        final_n_bands: Final number of bands to select
        gaussian_sigma: Gaussian filter sigma
        seed: Random seed
        
    Returns:
        Tuple of:
        - selected_band_indices: Final selected bands (final_n_bands,)
        - band_frequencies: Selection frequency per band (n_bands,)
        - cost_histories: Cost history per evaluation (n_evaluations, max_iterations+1)
    """
    logger.info("="*70)
    logger.info("JAYA BAND SELECTION PIPELINE")
    logger.info("="*70)
    
    # Step 1: K-means spectral clustering
    logger.info("Step 1: K-means spectral clustering...")
    cluster_centres, cluster_labels = kmeans_spectral_clustering(
        cube, n_clusters=n_clusters, random_state=seed
    )
    
    # Step 2: Gaussian spatial filtering
    logger.info("Step 2: Gaussian spatial filtering...")
    filtered_bands = gaussian_spatial_filter(cube, sigma=gaussian_sigma)
    
    # Step 3: Run Jaya n_evaluations times
    logger.info(f"Step 3: Running {n_evaluations} independent Jaya evaluations...")
    all_selected_bands = []
    cost_histories = []
    
    for eval_idx in tqdm(range(n_evaluations), desc="Jaya evaluations"):
        eval_seed = seed + eval_idx if seed is not None else None
        selected, fitness, history = run_jaya_single(
            cluster_centres=cluster_centres,
            filtered_bands=filtered_bands,
            n_bands_total=cube.shape[0],
            n_select=bands_per_run,
            population_size=population_size,
            max_iterations=max_iterations,
            seed=eval_seed
        )
        all_selected_bands.append(selected)
        cost_histories.append(history)
    
    all_selected_bands = np.array(all_selected_bands)  # (n_evaluations, bands_per_run)
    cost_histories = np.array(cost_histories)  # (n_evaluations, max_iterations+1)
    
    # Step 4: Count frequency each band was selected
    band_frequencies = np.zeros(cube.shape[0], dtype=int)
    for selected in all_selected_bands:
        for band_idx in selected:
            band_frequencies[band_idx] += 1
    
    # Step 5: Rank bands by frequency and select top final_n_bands
    ranked_indices = np.argsort(band_frequencies)[::-1]  # Descending order
    selected_band_indices = ranked_indices[:final_n_bands]
    
    # Log summary
    logger.info(f"\nSelected {final_n_bands} bands from {cube.shape[0]} total")
    logger.info(f"Selected band indices: {selected_band_indices}")
    if 'band' in cube.coords:
        selected_wavelengths = cube.coords['band'].values[selected_band_indices]
        logger.info(f"Selected wavelengths: {selected_wavelengths.astype(int)} nm")
    logger.info(f"Selection frequencies: {band_frequencies[selected_band_indices]}")
    
    return selected_band_indices, band_frequencies, cost_histories


def apply_band_selection(
    cube: xr.DataArray,
    selected_indices: np.ndarray
) -> xr.DataArray:
    """
    Apply band selection to hyperspectral cube.
    
    Subsets the cube to only the selected bands, preserving physical
    meaning of original wavelengths.
    
    Args:
        cube: Hyperspectral cube (band, y, x)
        selected_indices: Selected band indices
        
    Returns:
        Reduced cube (n_selected_bands, y, x) with updated attrs
    """
    reduced_cube = cube.isel(band=selected_indices)
    
    # Update attributes
    reduced_cube.attrs = {
        **cube.attrs,
        "band_selection_method": "Jaya-Patro2019",
        "n_bands_selected": len(selected_indices),
        "selected_band_indices": selected_indices.tolist()
    }
    
    logger.info(f"Applied band selection: {cube.shape[0]} → {len(selected_indices)} bands")
    
    return reduced_cube


def plot_band_selection_results(
    band_frequencies: np.ndarray,
    selected_indices: np.ndarray,
    wavelengths: np.ndarray,
    cost_histories: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot comprehensive band selection results.
    
    Creates a 2×2 figure showing:
    - Band selection frequency bar chart
    - Selected bands on spectrum
    - Convergence curves
    - False color preview
    
    Args:
        band_frequencies: Selection frequency per band (n_bands,)
        selected_indices: Final selected band indices
        wavelengths: Wavelength array (n_bands,)
        cost_histories: Cost history per evaluation (n_evaluations, n_iterations)
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Band selection frequency
    ax1 = axes[0, 0]
    all_indices = np.arange(len(band_frequencies))
    colors = ['red' if i in selected_indices else 'grey' for i in all_indices]
    ax1.bar(all_indices, band_frequencies, color=colors, alpha=0.7)
    ax1.set_xlabel('Band Index', fontsize=12)
    ax1.set_ylabel('Selection Frequency', fontsize=12)
    ax1.set_title('Jaya Band Selection Frequency (50 evaluations)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Selected bands on spectrum
    ax2 = axes[0, 1]
    ax2.plot(wavelengths, band_frequencies, 'b-', linewidth=1, alpha=0.5, label='Frequency')
    
    # Mark water vapor absorption regions
    water_vapor_regions = [(1350, 1450), (1800, 1960), (2450, 2500)]
    for wv_min, wv_max in water_vapor_regions:
        ax2.axvspan(wv_min, wv_max, alpha=0.2, color='grey')
    
    # Mark selected bands
    selected_wavelengths = wavelengths[selected_indices]
    selected_freqs = band_frequencies[selected_indices]
    ax2.scatter(selected_wavelengths, selected_freqs, color='red', s=50, zorder=5, label='Selected')
    for wl, freq in zip(selected_wavelengths, selected_freqs):
        ax2.axvline(wl, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Selection Frequency', fontsize=12)
    ax2.set_title('Selected Bands by Wavelength', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Convergence curves
    ax3 = axes[1, 0]
    n_evaluations, n_iterations = cost_histories.shape
    
    # Plot all individual curves (light grey)
    for i in range(n_evaluations):
        ax3.plot(cost_histories[i, :], color='lightgrey', alpha=0.3, linewidth=0.5)
    
    # Plot mean curve (bold blue)
    mean_history = np.mean(cost_histories, axis=0)
    ax3.plot(mean_history, color='blue', linewidth=2, label='Mean')
    
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Fitness Value', fontsize=12)
    ax3.set_title('Jaya Convergence (50 evaluations)', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: False color preview (placeholder - would need cube data)
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.5, 'False Color Preview\n(First 3 Selected Bands)', 
             ha='center', va='center', fontsize=12, transform=ax4.transAxes)
    ax4.set_title('False Colour: First 3 Selected Bands', fontsize=13, fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved band selection results to {save_path}")
    
    # Always show in notebook
    plt.show()
