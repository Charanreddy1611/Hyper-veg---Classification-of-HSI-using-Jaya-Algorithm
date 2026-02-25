"""
Unit tests for radiometric calibration module.
"""

import numpy as np
import pytest
import xarray as xr
from src.pipeline.calibration import (
    generate_synthetic_dn,
    apply_radiometric_calibration,
    validate_calibration
)


@pytest.fixture
def synthetic_reflectance_cube():
    """Create synthetic reflectance cube for testing."""
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
    
    return cube, wavelengths


def test_dn_shape_preserved(synthetic_reflectance_cube):
    """Test that DN generation preserves input shape."""
    cube, wavelengths = synthetic_reflectance_cube
    dn_cube, gain, offset = generate_synthetic_dn(cube, wavelengths, seed=42)
    
    assert dn_cube.shape == cube.shape
    assert len(gain) == cube.shape[0]
    assert len(offset) == cube.shape[0]


def test_calibration_linear_relationship(synthetic_reflectance_cube):
    """Test that calibration follows L = gain * DN + offset."""
    cube, wavelengths = synthetic_reflectance_cube
    dn_cube, gain, offset = generate_synthetic_dn(cube, wavelengths, seed=42)
    
    # Apply calibration
    radiance_cube = apply_radiometric_calibration(dn_cube, gain, offset)
    
    # Verify linear relationship for a few pixels
    for band_idx in [0, 5, 10]:
        dn_band = dn_cube.values[band_idx, 0, 0]
        radiance_band = radiance_cube.values[band_idx, 0, 0]
        expected = gain[band_idx] * float(dn_band) + offset[band_idx]
        
        np.testing.assert_allclose(radiance_band, expected, rtol=1e-5)


def test_bad_pixel_masking(synthetic_reflectance_cube):
    """Test that bad pixels are masked as NaN."""
    cube, wavelengths = synthetic_reflectance_cube
    dn_cube, gain, offset = generate_synthetic_dn(cube, wavelengths, seed=42)
    
    # Create bad pixel mask
    bad_mask = np.zeros((cube.shape[1], cube.shape[2]), dtype=bool)
    bad_mask[0, 0] = True  # Mark one pixel as bad
    
    radiance_cube = apply_radiometric_calibration(
        dn_cube, gain, offset, bad_pixel_mask=bad_mask
    )
    
    # Check that bad pixel is NaN
    assert np.isnan(radiance_cube.values[:, 0, 0]).all()


def test_valid_range_clipping(synthetic_reflectance_cube):
    """Test that values are clipped to valid range."""
    cube, wavelengths = synthetic_reflectance_cube
    dn_cube, gain, offset = generate_synthetic_dn(cube, wavelengths, seed=42)
    
    # Use very restrictive valid range
    radiance_cube = apply_radiometric_calibration(
        dn_cube, gain, offset, valid_range=(0.0, 10.0)
    )
    
    assert radiance_cube.values.max() <= 10.0
    assert radiance_cube.values.min() >= 0.0


def test_negative_gain_raises(synthetic_reflectance_cube):
    """Test that negative gain raises ValueError."""
    cube, wavelengths = synthetic_reflectance_cube
    dn_cube, _, _ = generate_synthetic_dn(cube, wavelengths, seed=42)
    
    # Create negative gain
    negative_gain = -np.ones(len(wavelengths))
    offset = np.zeros(len(wavelengths))
    
    with pytest.raises(ValueError, match="non-negative"):
        apply_radiometric_calibration(dn_cube, negative_gain, offset)
