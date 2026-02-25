"""
Atmospheric Correction Module

Simulates atmospheric correction to convert at-sensor Radiance to Surface Reflectance.
Implements the full radiative transfer equation derived from physics.

Physics:
    ρ(λ) = π × (L_sensor(λ) - L_path(λ)) / (E_s(λ) × cos(θ_s) × T(λ))
    
    where:
    - L_sensor: At-sensor radiance
    - L_path: Path radiance (scattered light)
    - E_s: Solar irradiance at top of atmosphere
    - θ_s: Solar zenith angle
    - T: Two-way atmospheric transmittance
"""

import numpy as np
import xarray as xr
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def compute_solar_irradiance(
    wavelengths_nm: np.ndarray,
    day_of_year: int = 200
) -> np.ndarray:
    """
    Compute solar irradiance at top of atmosphere for each wavelength.
    
    Uses a simplified Kurucz solar spectrum approximation as a Gaussian
    centered around 550nm (visible peak). Applies Earth-Sun distance
    correction based on day of year.
    
    Physics:
        Solar spectrum peaks in visible range (~550nm). We approximate
        using a Gaussian: E_s(λ) = 1800 × exp(-((λ-0.55)²)/(2×0.3²))
        
        Earth-Sun distance correction:
        d_correction = (1 + 0.033 × cos(2π × (day_of_year - 3) / 365))
    
    Args:
        wavelengths_nm: Wavelength array in nanometers
        day_of_year: Day of year (1-365), default 200 (summer)
        
    Returns:
        Solar irradiance array in W/m²/μm (same shape as wavelengths_nm)
    """
    wavelengths_um = wavelengths_nm / 1000.0  # Convert to micrometers
    
    # Simplified Gaussian approximation of solar spectrum
    # Peak at ~550nm (0.55 μm) with standard deviation ~0.3 μm
    peak_wavelength_um = 0.55
    sigma_um = 0.3
    base_irradiance = 1800.0  # W/m²/μm at peak
    
    # Gaussian solar spectrum
    solar_spectrum = base_irradiance * np.exp(
        -((wavelengths_um - peak_wavelength_um) ** 2) / (2 * sigma_um ** 2)
    )
    
    # Earth-Sun distance correction
    # Maximum distance variation is ~3.3% (perihelion vs aphelion)
    day_angle = 2 * np.pi * (day_of_year - 3) / 365.0
    distance_correction = 1 + 0.033 * np.cos(day_angle)
    
    solar_irradiance = solar_spectrum * distance_correction
    
    return solar_irradiance


def compute_atmospheric_transmittance(
    wavelengths_nm: np.ndarray,
    solar_zenith_deg: float = 35.0,
    view_zenith_deg: float = 0.0,
    aod_550: float = 0.1,
    water_vapor_gcm2: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute atmospheric transmittance and path radiance using Beer-Lambert law.
    
    Physics:
        Beer-Lambert transmittance:
        T(λ) = exp(-τ(λ) / cos(θ_s)) × exp(-τ(λ) / cos(θ_v))
        
        where τ(λ) = τ_rayleigh(λ) + τ_aerosol(λ) + τ_water(λ)
        
        Rayleigh optical depth:
        τ_rayleigh(λ) = 0.0088 × λ^(-4.15)  [λ in μm]
        
        Aerosol optical depth (Angstrom model):
        τ_aerosol(λ) = AOD_550 × (λ/0.55)^(-1.3)
        
        Path radiance (Rayleigh scattering):
        L_path(λ) ∝ (1 - T_rayleigh(λ)) × E_s(λ) × cos(θ_s) / (4π)
    
    Args:
        wavelengths_nm: Wavelength array in nanometers
        solar_zenith_deg: Solar zenith angle in degrees (0 = overhead)
        view_zenith_deg: View zenith angle in degrees (0 = nadir)
        aod_550: Aerosol optical depth at 550nm
        water_vapor_gcm2: Precipitable water vapor in g/cm²
        
    Returns:
        Tuple of:
        - transmittance_array: Two-way transmittance (n_bands,)
        - path_radiance_array: Path radiance in W/m²/sr/μm (n_bands,)
        
    Raises:
        ValueError: If solar zenith angle > 85° (near-horizon, invalid)
    """
    if solar_zenith_deg > 85.0:
        raise ValueError(f"Solar zenith angle {solar_zenith_deg}° > 85° is invalid (near horizon)")
    
    wavelengths_um = wavelengths_nm / 1000.0  # Convert to micrometers
    
    # Convert angles to radians
    solar_zenith_rad = np.deg2rad(solar_zenith_deg)
    view_zenith_rad = np.deg2rad(view_zenith_deg)
    
    # Rayleigh optical depth: τ_R(λ) = 0.0088 × λ^(-4.15) [λ in μm]
    # Rayleigh scattering follows λ^-4 dependence (stronger at shorter wavelengths)
    tau_rayleigh = 0.0088 * (wavelengths_um ** (-4.15))
    
    # Aerosol optical depth: Angstrom exponent model
    # τ_aerosol(λ) = AOD_550 × (λ/0.55)^(-α), where α ≈ 1.3
    angstrom_exponent = 1.3
    tau_aerosol = aod_550 * ((wavelengths_um / 0.55) ** (-angstrom_exponent))
    
    # Water vapor absorption (simplified - strong bands at 1.4μm and 1.9μm)
    # Approximate as Gaussian absorption features
    water_bands_um = [1.4, 1.9]
    tau_water = np.zeros_like(wavelengths_um)
    for band_center in water_bands_um:
        absorption = water_vapor_gcm2 * 0.1 * np.exp(
            -((wavelengths_um - band_center) ** 2) / (2 * 0.1 ** 2)
        )
        tau_water += absorption
    
    # Total optical depth
    tau_total = tau_rayleigh + tau_aerosol + tau_water
    
    # Two-way transmittance: downwelling and upwelling
    # T(λ) = exp(-τ(λ) / cos(θ_s)) × exp(-τ(λ) / cos(θ_v))
    sec_solar = 1.0 / np.cos(solar_zenith_rad)
    sec_view = 1.0 / np.cos(view_zenith_rad)
    
    transmittance_down = np.exp(-tau_total * sec_solar)
    transmittance_up = np.exp(-tau_total * sec_view)
    transmittance = transmittance_down * transmittance_up
    
    # Path radiance (Rayleigh scattering dominates)
    # L_path ∝ (1 - T_rayleigh) × E_s × cos(θ_s) / (4π)
    solar_irradiance = compute_solar_irradiance(wavelengths_nm)
    
    # Rayleigh-only transmittance for path radiance
    transmittance_rayleigh_down = np.exp(-tau_rayleigh * sec_solar)
    
    # Path radiance coefficient (simplified model)
    path_coefficient = (1.0 - transmittance_rayleigh_down) * solar_irradiance * np.cos(solar_zenith_rad) / (4 * np.pi)
    
    # Scale by aerosol contribution (aerosols also scatter)
    aerosol_scattering_factor = 1.0 + 0.3 * (tau_aerosol / (tau_rayleigh + 1e-10))
    path_radiance = path_coefficient * aerosol_scattering_factor
    
    logger.info(f"Computed transmittance: mean = {np.mean(transmittance):.4f}, range = [{np.min(transmittance):.4f}, {np.max(transmittance):.4f}]")
    
    return transmittance, path_radiance


def apply_atmospheric_correction(
    radiance_cube: xr.DataArray,
    wavelengths_nm: np.ndarray,
    solar_zenith_deg: float = 35.0,
    aod_550: float = 0.1,
    water_vapor_gcm2: float = 2.0,
    clip_reflectance: bool = True
) -> xr.DataArray:
    """
    Apply atmospheric correction: convert Radiance to Surface Reflectance.
    
    Physics:
        Full radiative transfer equation:
        ρ(λ) = π × (L_sensor(λ) - L_path(λ)) / (E_s(λ) × cos(θ_s) × T(λ))
        
        The π factor comes from integrating over the Lambertian hemisphere
        (assuming a Lambertian surface). This is the standard equation for
        converting at-sensor radiance to surface reflectance.
    
    Args:
        radiance_cube: At-sensor radiance cube (band, y, x) in W/m²/sr/μm
        wavelengths_nm: Wavelength array in nanometers
        solar_zenith_deg: Solar zenith angle in degrees
        aod_550: Aerosol optical depth at 550nm
        water_vapor_gcm2: Precipitable water vapor in g/cm²
        clip_reflectance: Whether to clip reflectance to [0, 1] physical range
        
    Returns:
        Surface reflectance cube as xr.DataArray (band, y, x) with values in [0, 1]
        
    Raises:
        ValueError: If solar zenith angle > 85°
    """
    if solar_zenith_deg > 85.0:
        raise ValueError(f"Solar zenith angle {solar_zenith_deg}° > 85° is invalid")
    
    # Compute atmospheric parameters
    transmittance, path_radiance = compute_atmospheric_transmittance(
        wavelengths_nm,
        solar_zenith_deg=solar_zenith_deg,
        view_zenith_deg=0.0,  # Nadir view
        aod_550=aod_550,
        water_vapor_gcm2=water_vapor_gcm2
    )
    
    solar_irradiance = compute_solar_irradiance(wavelengths_nm)
    solar_zenith_rad = np.deg2rad(solar_zenith_deg)
    
    # Prepare arrays for broadcasting
    # transmittance, path_radiance, solar_irradiance: (n_bands,)
    # radiance_cube: (n_bands, y, x)
    
    transmittance_3d = transmittance[:, np.newaxis, np.newaxis]
    path_radiance_3d = path_radiance[:, np.newaxis, np.newaxis]
    solar_irradiance_3d = solar_irradiance[:, np.newaxis, np.newaxis]
    
    # Apply reflectance equation: ρ = π × (L - L_path) / (E_s × cos(θ_s) × T)
    numerator = np.pi * (radiance_cube.values - path_radiance_3d)
    denominator = solar_irradiance_3d * np.cos(solar_zenith_rad) * transmittance_3d
    
    # Guard against near-zero denominator (opaque atmosphere bands)
    denominator = np.where(denominator < 1e-10, np.nan, denominator)
    
    reflectance = numerator / denominator
    
    # Flag pixels with reflectance > 1.0 before clipping (unphysical)
    n_unphysical = np.sum(reflectance > 1.0)
    if n_unphysical > 0:
        logger.warning(f"Found {n_unphysical} pixels with reflectance > 1.0 (unphysical, will be clipped)")
    
    # Clip to physical range [0, 1]
    if clip_reflectance:
        reflectance = np.clip(reflectance, 0.0, 1.0)
    
    # Create xarray DataArray
    reflectance_cube = xr.DataArray(
        reflectance,
        dims=radiance_cube.dims,
        coords=radiance_cube.coords,
        attrs={
            **radiance_cube.attrs,
            "units": "reflectance",
            "atmospheric_correction_applied": True,
            "solar_zenith_deg": solar_zenith_deg,
            "aod_550": aod_550,
            "water_vapor_gcm2": water_vapor_gcm2,
            "description": "Surface reflectance (atmospherically corrected)"
        }
    )
    
    logger.info(f"Applied atmospheric correction: reflectance range [{np.nanmin(reflectance):.4f}, {np.nanmax(reflectance):.4f}]")
    
    return reflectance_cube


def compute_toa_reflectance(
    radiance_cube: xr.DataArray,
    wavelengths_nm: np.ndarray,
    solar_zenith_deg: float = 35.0
) -> xr.DataArray:
    """
    Compute Top-of-Atmosphere (TOA) reflectance without atmospheric correction.
    
    This is a quick approximation useful for comparison against surface reflectance.
    It assumes no atmospheric effects (no path radiance, no transmittance loss).
    
    Physics:
        ρ_TOA = π × L / (E_s × cos(θ_s))
        
        This is the reflectance that would be observed at the top of atmosphere
        if there were no atmospheric scattering or absorption.
    
    Args:
        radiance_cube: At-sensor radiance cube (band, y, x) in W/m²/sr/μm
        wavelengths_nm: Wavelength array in nanometers
        solar_zenith_deg: Solar zenith angle in degrees
        
    Returns:
        TOA reflectance cube as xr.DataArray (band, y, x)
    """
    solar_irradiance = compute_solar_irradiance(wavelengths_nm)
    solar_zenith_rad = np.deg2rad(solar_zenith_deg)
    
    # Prepare for broadcasting
    solar_irradiance_3d = solar_irradiance[:, np.newaxis, np.newaxis]
    
    # TOA reflectance: ρ_TOA = π × L / (E_s × cos(θ_s))
    toa_reflectance = np.pi * radiance_cube.values / (solar_irradiance_3d * np.cos(solar_zenith_rad))
    
    # Clip to reasonable range
    toa_reflectance = np.clip(toa_reflectance, 0.0, 2.0)  # TOA can exceed 1.0
    
    toa_cube = xr.DataArray(
        toa_reflectance,
        dims=radiance_cube.dims,
        coords=radiance_cube.coords,
        attrs={
            **radiance_cube.attrs,
            "units": "reflectance",
            "toa_reflectance": True,
            "solar_zenith_deg": solar_zenith_deg,
            "description": "Top-of-atmosphere reflectance (no atmospheric correction)"
        }
    )
    
    return toa_cube
