"""
Unit tests for vegetation indices module.
"""

import numpy as np
import pytest
import xarray as xr
from src.analysis.indices import (
    compute_ndvi,
    compute_evi,
    compute_ndwi,
    compute_nbr,
    compute_all_indices
)


@pytest.fixture
def synthetic_reflectance_cube():
    """Create synthetic reflectance cube with realistic vegetation spectrum."""
    n_bands = 200
    wavelengths = np.linspace(400, 2500, n_bands)
    n_y, n_x = 10, 10
    
    # Create synthetic vegetation spectrum
    # High in NIR (~865nm), low in red (~670nm)
    reflectance = np.zeros((n_bands, n_y, n_x))
    
    for i, wl in enumerate(wavelengths):
        if 400 <= wl <= 700:  # Visible: low reflectance
            reflectance[i, :, :] = 0.1
        elif 700 < wl <= 1300:  # NIR: high reflectance
            reflectance[i, :, :] = 0.6
        else:  # SWIR: medium reflectance
            reflectance[i, :, :] = 0.3
    
    cube = xr.DataArray(
        reflectance,
        dims=["band", "y", "x"],
        coords={"band": wavelengths, "y": np.arange(n_y), "x": np.arange(n_x)}
    )
    
    return cube


def test_ndvi_range(synthetic_reflectance_cube):
    """Test that NDVI values are in [-1, 1]."""
    cube = synthetic_reflectance_cube
    ndvi = compute_ndvi(cube)
    
    assert np.nanmin(ndvi.values) >= -1.0
    assert np.nanmax(ndvi.values) <= 1.0


def test_ndvi_vegetation_positive(synthetic_reflectance_cube):
    """Test that vegetated pixels have positive NDVI."""
    cube = synthetic_reflectance_cube
    ndvi = compute_ndvi(cube)
    
    # Our synthetic vegetation should have positive NDVI
    # (NIR=0.6 > Red=0.1)
    assert np.nanmean(ndvi.values) > 0.0


def test_ndvi_zero_division_handled(synthetic_reflectance_cube):
    """Test that zero denominator results in NaN, not inf."""
    cube = synthetic_reflectance_cube
    
    # Create cube where NIR + Red = 0 (shouldn't happen in practice, but test edge case)
    cube_zero = cube.copy()
    cube_zero.values[:, :, :] = 0.0
    
    ndvi = compute_ndvi(cube_zero)
    
    # Should be NaN, not inf
    assert np.all(np.isnan(ndvi.values))


def test_evi_range(synthetic_reflectance_cube):
    """Test that EVI values are reasonable."""
    cube = synthetic_reflectance_cube
    evi = compute_evi(cube)
    
    # EVI typically ranges from -1 to 1, but can exceed in extreme cases
    assert np.nanmin(evi.values) >= -2.0
    assert np.nanmax(evi.values) <= 2.0


def test_all_indices_returns_dataset(synthetic_reflectance_cube):
    """Test that compute_all_indices returns xr.Dataset."""
    cube = synthetic_reflectance_cube
    indices_dataset = compute_all_indices(cube)
    
    assert isinstance(indices_dataset, xr.Dataset)
    assert 'ndvi' in indices_dataset.data_vars
    assert 'evi' in indices_dataset.data_vars
    assert 'ndwi' in indices_dataset.data_vars
    assert 'nbr' in indices_dataset.data_vars
