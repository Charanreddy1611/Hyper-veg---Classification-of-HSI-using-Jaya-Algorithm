"""
Unit tests for Jaya-based band selection module.
"""

import numpy as np
import pytest
import xarray as xr
from src.analysis.band_selection import (
    kmeans_spectral_clustering,
    gaussian_spatial_filter,
    compute_spectral_distance,
    compute_frobenius_distance,
    jaya_fitness,
    run_jaya_single,
    apply_band_selection
)


@pytest.fixture
def small_cube():
    """Create synthetic hyperspectral cube for testing."""
    n_bands = 20
    n_y, n_x = 10, 10
    wavelengths = np.linspace(400, 2500, n_bands)
    
    # Create synthetic reflectance (0-1 range)
    reflectance = np.random.uniform(0.1, 0.8, size=(n_bands, n_y, n_x))
    
    cube = xr.DataArray(
        reflectance,
        dims=["band", "y", "x"],
        coords={"band": wavelengths, "y": np.arange(n_y), "x": np.arange(n_x)}
    )
    
    return cube


def test_kmeans_output_shape(small_cube):
    """Test that K-means output has correct shape."""
    cube = small_cube
    n_clusters = 17
    
    cluster_centres, cluster_labels = kmeans_spectral_clustering(
        cube, n_clusters=n_clusters, random_state=42
    )
    
    assert cluster_centres.shape == (n_clusters, cube.shape[0])
    assert cluster_labels.shape == (cube.shape[1] * cube.shape[2],)


def test_gaussian_filter_shape_preserved(small_cube):
    """Test that Gaussian filter preserves input shape."""
    cube = small_cube
    filtered = gaussian_spatial_filter(cube, sigma=1.0)
    
    assert filtered.shape == cube.shape


def test_gaussian_filter_smooths(small_cube):
    """Test that Gaussian filter reduces spatial variance."""
    cube = small_cube
    filtered = gaussian_spatial_filter(cube, sigma=1.0)
    
    # Check that filtered band has lower std than original
    original_std = np.std(cube.values[0, :, :])
    filtered_std = np.std(filtered[0, :, :])
    
    assert filtered_std < original_std


def test_spectral_distance_positive(small_cube):
    """Test that spectral distance returns positive value."""
    cube = small_cube
    cluster_centres, _ = kmeans_spectral_clustering(cube, n_clusters=5, random_state=42)
    
    band_indices = np.array([0, 5, 10, 15, 19])
    distance = compute_spectral_distance(cluster_centres, band_indices)
    
    assert distance >= 0.0
    assert isinstance(distance, (float, np.floating))


def test_frobenius_distance_positive(small_cube):
    """Test that Frobenius distance returns positive value."""
    cube = small_cube
    filtered_bands = gaussian_spatial_filter(cube, sigma=1.0)
    
    band_indices = np.array([0, 5, 10, 15, 19])
    distance = compute_frobenius_distance(filtered_bands, band_indices)
    
    assert distance >= 0.0
    assert isinstance(distance, (float, np.floating))


def test_jaya_fitness_duplicate_bands(small_cube):
    """Test that fitness returns -inf for duplicate bands."""
    cube = small_cube
    cluster_centres, _ = kmeans_spectral_clustering(cube, n_clusters=5, random_state=42)
    filtered_bands = gaussian_spatial_filter(cube, sigma=1.0)
    
    # Test with duplicates
    band_indices_dup = np.array([0, 5, 10, 5, 19])  # 5 appears twice
    fitness_dup = jaya_fitness(band_indices_dup, cluster_centres, filtered_bands)
    
    assert fitness_dup == -np.inf
    
    # Test without duplicates
    band_indices_unique = np.array([0, 5, 10, 15, 19])
    fitness_unique = jaya_fitness(band_indices_unique, cluster_centres, filtered_bands)
    
    assert fitness_unique > -np.inf
    assert isinstance(fitness_unique, (float, np.floating))


def test_jaya_single_returns_correct_shape(small_cube):
    """Test that single Jaya run returns correct output shape."""
    cube = small_cube
    cluster_centres, _ = kmeans_spectral_clustering(cube, n_clusters=5, random_state=42)
    filtered_bands = gaussian_spatial_filter(cube, sigma=1.0)
    
    n_select = 5
    G_sol, G_best, cost_history = run_jaya_single(
        cluster_centres=cluster_centres,
        filtered_bands=filtered_bands,
        n_bands_total=cube.shape[0],
        n_select=n_select,
        population_size=10,
        max_iterations=20,  # Shorter for testing
        seed=42
    )
    
    assert G_sol.shape == (n_select,)
    assert np.all(G_sol >= 0)
    assert np.all(G_sol < cube.shape[0])
    assert len(np.unique(G_sol)) == n_select  # No duplicates
    assert isinstance(G_best, (float, np.floating))
    assert len(cost_history) == 21  # max_iterations + 1


def test_jaya_single_cost_increases(small_cube):
    """Test that Jaya fitness improves or holds."""
    cube = small_cube
    cluster_centres, _ = kmeans_spectral_clustering(cube, n_clusters=5, random_state=42)
    filtered_bands = gaussian_spatial_filter(cube, sigma=1.0)
    
    _, _, cost_history = run_jaya_single(
        cluster_centres=cluster_centres,
        filtered_bands=filtered_bands,
        n_bands_total=cube.shape[0],
        n_select=5,
        population_size=10,
        max_iterations=20,
        seed=42
    )
    
    # Best fitness should be non-decreasing (or at least final >= initial)
    assert cost_history[-1] >= cost_history[0]


def test_jaya_single_reproducible(small_cube):
    """Test that Jaya is reproducible with same seed."""
    cube = small_cube
    cluster_centres, _ = kmeans_spectral_clustering(cube, n_clusters=5, random_state=42)
    filtered_bands = gaussian_spatial_filter(cube, sigma=1.0)
    
    G_sol1, _, _ = run_jaya_single(
        cluster_centres=cluster_centres,
        filtered_bands=filtered_bands,
        n_bands_total=cube.shape[0],
        n_select=5,
        population_size=10,
        max_iterations=20,
        seed=42
    )
    
    G_sol2, _, _ = run_jaya_single(
        cluster_centres=cluster_centres,
        filtered_bands=filtered_bands,
        n_bands_total=cube.shape[0],
        n_select=5,
        population_size=10,
        max_iterations=20,
        seed=42
    )
    
    np.testing.assert_array_equal(G_sol1, G_sol2)


def test_apply_band_selection_shape(small_cube):
    """Test that band selection produces correct output shape."""
    cube = small_cube
    selected_indices = np.array([0, 5, 10, 15, 19, 2, 7, 12, 17, 3, 8, 13, 18, 1, 6, 11, 16, 4])
    
    reduced_cube = apply_band_selection(cube, selected_indices)
    
    assert reduced_cube.shape == (len(selected_indices), cube.shape[1], cube.shape[2])


def test_apply_band_selection_attrs_preserved(small_cube):
    """Test that band selection preserves and updates attributes."""
    cube = small_cube
    selected_indices = np.array([0, 5, 10, 15, 19])
    
    reduced_cube = apply_band_selection(cube, selected_indices)
    
    assert "band_selection_method" in reduced_cube.attrs
    assert reduced_cube.attrs["band_selection_method"] == "Jaya-Patro2019"
    assert reduced_cube.attrs["n_bands_selected"] == len(selected_indices)
    assert "selected_band_indices" in reduced_cube.attrs
