"""
Radiometric Calibration Module

Simulates the conversion from Digital Numbers (DN) to at-sensor Radiance.
Since Indian Pines is already in reflectance, we reverse-engineer synthetic DN
from the reflectance data, then calibrate back to demonstrate the full pipeline.

Physics: L(λ) = gain(λ) × DN(λ) + offset(λ)
"""

import numpy as np
import xarray as xr
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def generate_synthetic_dn(
    reflectance_cube: xr.DataArray,
    wavelengths: np.ndarray,
    seed: int = 42
) -> Tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """
    Generate synthetic Digital Numbers from reflectance data.
    
    Reverse-engineers realistic DN values by applying inverse calibration
    with realistic gain/offset coefficients, then adds sensor noise and
    quantizes to uint16 range.
    
    Physics:
        We reverse the calibration equation: DN = (L - offset) / gain
        where L is derived from reflectance (simplified approximation).
        We add realistic sensor noise scaled by wavelength-dependent SNR.
    
    Args:
        reflectance_cube: Reflectance cube (band, y, x) with values in [0, 1]
        wavelengths: Wavelength array in nm (200,)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of:
        - dn_cube: Synthetic DN values as xr.DataArray (band, y, x), dtype uint16
        - gain_array: Per-band gain coefficients (200,)
        - offset_array: Per-band offset coefficients (200,)
    """
    np.random.seed(seed)
    
    # Generate realistic per-band gain and offset coefficients
    # Gain: AVIRIS typical range [0.005, 0.015] W/m²/sr/μm per DN
    gain_min, gain_max = 0.005, 0.015
    gain_array = np.random.uniform(gain_min, gain_max, size=len(wavelengths))
    
    # Offset: dark current simulation [-1.0, 1.0] W/m²/sr/μm
    offset_min, offset_max = -1.0, 1.0
    offset_array = np.random.uniform(offset_min, offset_max, size=len(wavelengths))
    
    # Convert reflectance to approximate radiance for DN generation
    # Simplified: assume solar irradiance ~1000 W/m²/μm and convert
    # This is a simplification - in reality we'd need full solar spectrum
    solar_irradiance_approx = 1000.0  # W/m²/μm (simplified constant)
    radiance_approx = reflectance_cube.values * solar_irradiance_approx / np.pi
    
    # Reverse calibration: DN = (L - offset) / gain
    # Use broadcasting: gain_array and offset_array shape (200,) broadcast with cube (200, 145, 145)
    dn_float = (radiance_approx - offset_array[:, np.newaxis, np.newaxis]) / gain_array[:, np.newaxis, np.newaxis]
    
    # Add realistic sensor noise
    # SNR varies with wavelength: high SNR (300:1) in VIS, lower (150:1) in SWIR
    wavelengths_um = wavelengths / 1000.0  # Convert to micrometers
    
    # SNR model: higher in visible (400-700nm), lower in SWIR (2000-2500nm)
    snr_vis = 300.0
    snr_swir = 150.0
    # Interpolate SNR based on wavelength
    snr_array = np.interp(
        wavelengths_um,
        [0.4, 0.7, 2.0, 2.5],
        [snr_vis, snr_vis, snr_swir, snr_swir]
    )
    
    # Noise standard deviation = signal / SNR
    noise_std = np.abs(dn_float) / snr_array[:, np.newaxis, np.newaxis]
    noise = np.random.normal(0, noise_std, size=dn_float.shape)
    dn_noisy = dn_float + noise
    
    # Quantize to uint16 (0-65535 range)
    dn_noisy = np.clip(dn_noisy, 0, 65535)
    dn_uint16 = dn_noisy.astype(np.uint16)
    
    # Create xarray DataArray
    dn_cube = xr.DataArray(
        dn_uint16,
        dims=reflectance_cube.dims,
        coords=reflectance_cube.coords,
        attrs={
            **reflectance_cube.attrs,
            "units": "DN",
            "dtype": "uint16",
            "description": "Synthetic Digital Numbers (reverse-engineered from reflectance)"
        }
    )
    
    logger.info(f"Generated synthetic DN cube: shape {dn_cube.shape}, range [{dn_uint16.min()}, {dn_uint16.max()}]")
    
    return dn_cube, gain_array, offset_array


def apply_radiometric_calibration(
    dn_cube: xr.DataArray,
    gain: np.ndarray,
    offset: np.ndarray,
    valid_range: Tuple[float, float] = (0.0, 1000.0),
    bad_pixel_mask: Optional[np.ndarray] = None
) -> xr.DataArray:
    """
    Apply radiometric calibration: convert DN to at-sensor Radiance.
    
    Physics:
        L(λ) = gain(λ) × DN(λ) + offset(λ)
        
        This is the standard linear radiometric calibration equation used
        by most hyperspectral sensors including AVIRIS.
    
    Args:
        dn_cube: Digital Numbers cube (band, y, x), dtype uint16
        gain: Per-band gain coefficients (n_bands,)
        offset: Per-band offset coefficients (n_bands,)
        valid_range: (min, max) tuple for valid radiance values in W/m²/sr/μm
        bad_pixel_mask: Optional boolean mask (y, x) where True = bad pixel
        
    Returns:
        Radiance cube as xr.DataArray (band, y, x) with units W/m²/sr/μm
        
    Raises:
        ValueError: If gain contains negative values
    """
    if np.any(gain < 0):
        raise ValueError("Gain coefficients must be non-negative")
    
    if len(gain) != dn_cube.shape[0] or len(offset) != dn_cube.shape[0]:
        raise ValueError(f"Gain/offset arrays must match number of bands: {dn_cube.shape[0]}")
    
    # Convert DN to float for calculation
    dn_float = dn_cube.values.astype(np.float32)
    
    # Apply calibration: L = gain * DN + offset
    # Broadcasting: gain/offset (n_bands,) with cube (n_bands, y, x)
    radiance = gain[:, np.newaxis, np.newaxis] * dn_float + offset[:, np.newaxis, np.newaxis]
    
    # Mask bad pixels as NaN
    if bad_pixel_mask is not None:
        if bad_pixel_mask.shape != (dn_cube.shape[1], dn_cube.shape[2]):
            raise ValueError(f"Bad pixel mask shape {bad_pixel_mask.shape} doesn't match spatial dimensions")
        radiance[:, bad_pixel_mask] = np.nan
        n_bad = np.sum(bad_pixel_mask)
        logger.warning(f"Masked {n_bad} bad pixels as NaN")
    
    # Clip to valid range
    radiance_clipped = np.clip(radiance, valid_range[0], valid_range[1])
    n_clipped_low = np.sum(radiance < valid_range[0])
    n_clipped_high = np.sum(radiance > valid_range[1])
    
    if n_clipped_low > 0:
        logger.warning(f"Clipped {n_clipped_low} pixels below valid range ({valid_range[0]})")
    if n_clipped_high > 0:
        logger.warning(f"Clipped {n_clipped_high} pixels above valid range ({valid_range[1]})")
    
    # Create xarray DataArray
    radiance_cube = xr.DataArray(
        radiance_clipped,
        dims=dn_cube.dims,
        coords=dn_cube.coords,
        attrs={
            **dn_cube.attrs,
            "units": "W/m^2/sr/um",
            "calibration_applied": True,
            "valid_range": valid_range,
            "description": "At-sensor radiance (radiometrically calibrated)"
        }
    )
    
    logger.info(f"Applied radiometric calibration: radiance range [{radiance_clipped.min():.2f}, {radiance_clipped.max():.2f}] W/m²/sr/μm")
    
    return radiance_cube


def validate_calibration(
    original_reflectance: xr.DataArray,
    calibrated_radiance: xr.DataArray,
    gain: np.ndarray,
    offset: np.ndarray
) -> Dict:
    """
    Validate calibration by comparing original reflectance with calibrated result.
    
    Computes per-band correlation and RMSE to verify that the calibration
    process preserves the spectral information correctly.
    
    Args:
        original_reflectance: Original reflectance cube (band, y, x)
        calibrated_radiance: Calibrated radiance cube (band, y, x)
        gain: Gain coefficients used (n_bands,)
        offset: Offset coefficients used (n_bands,)
        
    Returns:
        Dictionary with validation statistics:
        - per_band_correlation: Correlation coefficient per band (n_bands,)
        - per_band_rmse: RMSE per band (n_bands,)
        - mean_correlation: Mean correlation across all bands
        - mean_rmse: Mean RMSE across all bands
        - validation_passed: Boolean flag (True if mean_correlation > 0.9)
    """
    # Convert radiance back to approximate reflectance for comparison
    # Simplified conversion (in reality would need full atmospheric correction)
    solar_irradiance_approx = 1000.0
    reflectance_from_radiance = calibrated_radiance.values * np.pi / solar_irradiance_approx
    
    orig_flat = original_reflectance.values.reshape(original_reflectance.shape[0], -1)
    calib_flat = reflectance_from_radiance.reshape(calibrated_radiance.shape[0], -1)
    
    # Remove NaN pixels
    valid_mask = ~(np.isnan(orig_flat) | np.isnan(calib_flat))
    
    per_band_correlation = np.zeros(original_reflectance.shape[0])
    per_band_rmse = np.zeros(original_reflectance.shape[0])
    
    for band_idx in range(original_reflectance.shape[0]):
        band_valid = valid_mask[band_idx, :]
        if np.sum(band_valid) > 0:
            orig_band = orig_flat[band_idx, band_valid]
            calib_band = calib_flat[band_idx, band_valid]
            
            # Correlation
            if np.std(orig_band) > 0 and np.std(calib_band) > 0:
                corr = np.corrcoef(orig_band, calib_band)[0, 1]
                per_band_correlation[band_idx] = corr if not np.isnan(corr) else 0.0
            else:
                per_band_correlation[band_idx] = 0.0
            
            # RMSE
            rmse = np.sqrt(np.mean((orig_band - calib_band) ** 2))
            per_band_rmse[band_idx] = rmse
    
    mean_correlation = np.mean(per_band_correlation)
    mean_rmse = np.mean(per_band_rmse)
    
    # Validation passes if correlation is high (calibration preserves spectral structure)
    validation_passed = mean_correlation > 0.9
    
    results = {
        "per_band_correlation": per_band_correlation,
        "per_band_rmse": per_band_rmse,
        "mean_correlation": mean_correlation,
        "mean_rmse": mean_rmse,
        "validation_passed": validation_passed
    }
    
    logger.info(f"Calibration validation: mean correlation = {mean_correlation:.4f}, mean RMSE = {mean_rmse:.4f}, passed = {validation_passed}")
    
    return results
