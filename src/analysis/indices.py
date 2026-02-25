"""
Vegetation Spectral Indices Module

Computes vegetation spectral indices from reflectance cube.
All indices return xr.DataArray with proper metadata.

Indices use nearest-wavelength band selection for robustness.
"""

import numpy as np
import xarray as xr
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def compute_ndvi(cube: xr.DataArray) -> xr.DataArray:
    """
    Compute Normalized Difference Vegetation Index (NDVI).
    
    Physics:
        NDVI = (NIR - Red) / (NIR + Red)
        
        NDVI measures photosynthetic activity and vegetation health.
        - Healthy vegetation: NDVI > 0.4
        - Sparse vegetation: 0.2 < NDVI < 0.4
        - Non-vegetation: NDVI < 0.2
        
        Red band (~670nm): Chlorophyll absorption
        NIR band (~865nm): High reflectance from healthy vegetation
    
    Args:
        cube: Reflectance cube (band, y, x) with wavelength coordinates
        
    Returns:
        NDVI map as xr.DataArray (y, x) with values in [-1, 1]
    """
    # Select bands using nearest wavelength
    red_band = cube.sel(band=670.0, method="nearest")
    nir_band = cube.sel(band=865.0, method="nearest")
    
    # Compute NDVI: (NIR - Red) / (NIR + Red)
    numerator = nir_band - red_band
    denominator = nir_band + red_band
    
    # Guard against division by zero → NaN
    ndvi = numerator / denominator
    ndvi = xr.where(denominator == 0, np.nan, ndvi)
    
    ndvi.attrs = {
        "index": "NDVI",
        "description": "Normalized Difference Vegetation Index",
        "formula": "(NIR - Red) / (NIR + Red)",
        "red_band_nm": float(red_band.coords["band"].values),
        "nir_band_nm": float(nir_band.coords["band"].values),
        "units": "dimensionless"
    }
    
    logger.info(f"Computed NDVI: range [{float(ndvi.min()):.4f}, {float(ndvi.max()):.4f}]")
    
    return ndvi


def compute_evi(cube: xr.DataArray) -> xr.DataArray:
    """
    Compute Enhanced Vegetation Index (EVI).
    
    Physics:
        EVI = 2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)
        
        EVI improves upon NDVI by:
        - Reducing atmospheric effects (aerosol scattering)
        - Reducing soil background effects
        - Better sensitivity in high biomass regions
        
        Blue band (~490nm): Used for atmospheric correction
        Red band (~670nm): Chlorophyll absorption
        NIR band (~865nm): High vegetation reflectance
    
    Args:
        cube: Reflectance cube (band, y, x) with wavelength coordinates
        
    Returns:
        EVI map as xr.DataArray (y, x)
    """
    # Select bands
    blue_band = cube.sel(band=490.0, method="nearest")
    red_band = cube.sel(band=670.0, method="nearest")
    nir_band = cube.sel(band=865.0, method="nearest")
    
    # Compute EVI: 2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)
    numerator = 2.5 * (nir_band - red_band)
    denominator = nir_band + 6 * red_band - 7.5 * blue_band + 1.0
    
    # Guard against division by zero
    evi = numerator / denominator
    evi = xr.where(denominator == 0, np.nan, evi)
    
    evi.attrs = {
        "index": "EVI",
        "description": "Enhanced Vegetation Index",
        "formula": "2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)",
        "blue_band_nm": float(blue_band.coords["band"].values),
        "red_band_nm": float(red_band.coords["band"].values),
        "nir_band_nm": float(nir_band.coords["band"].values),
        "units": "dimensionless"
    }
    
    logger.info(f"Computed EVI: range [{float(evi.min()):.4f}, {float(evi.max()):.4f}]")
    
    return evi


def compute_red_edge_position(cube: xr.DataArray) -> xr.DataArray:
    """
    Compute Red Edge Position (REP) - wavelength of maximum first derivative.
    
    Physics:
        The red edge is the steep slope in vegetation reflectance between
        680-750nm, caused by the transition from chlorophyll absorption
        (red) to high NIR reflectance.
        
        Red edge position is sensitive to:
        - Chlorophyll content (higher REP → more chlorophyll)
        - Vegetation stress (stressed vegetation has lower REP)
        - Leaf area index
        
        Method: Find wavelength of maximum first derivative in 680-750nm range.
    
    Args:
        cube: Reflectance cube (band, y, x) with wavelength coordinates
        
    Returns:
        Red edge position map as xr.DataArray (y, x) with values in nm
    """
    # Select red edge region: 680-750nm
    red_edge_region = cube.sel(band=slice(680, 750))
    
    if len(red_edge_region.band) < 2:
        logger.warning("Insufficient bands in red edge region, returning NaN")
        return xr.full_like(cube.isel(band=0), np.nan).drop("band")
    
    # Compute first derivative along wavelength dimension
    wavelengths = red_edge_region.coords["band"].values
    reflectance_values = red_edge_region.values  # (n_bands, y, x)
    
    # Compute derivative: dR/dλ
    # Use central difference: (R[i+1] - R[i-1]) / (λ[i+1] - λ[i-1])
    n_bands = len(wavelengths)
    derivative = np.zeros_like(reflectance_values)
    
    for i in range(1, n_bands - 1):
        dR = reflectance_values[i + 1, :, :] - reflectance_values[i - 1, :, :]
        dlambda = wavelengths[i + 1] - wavelengths[i - 1]
        derivative[i, :, :] = dR / dlambda
    
    # Find index of maximum derivative for each pixel
    max_deriv_idx = np.argmax(derivative, axis=0)  # (y, x)
    
    # Get corresponding wavelength
    red_edge_position = wavelengths[max_deriv_idx]
    
    # Create DataArray
    rep_da = xr.DataArray(
        red_edge_position,
        dims=["y", "x"],
        coords={"y": cube.coords["y"], "x": cube.coords["x"]},
        attrs={
            "index": "Red Edge Position",
            "description": "Wavelength of maximum first derivative in 680-750nm range",
            "units": "nm",
            "wavelength_range": "680-750 nm"
        }
    )
    
    logger.info(f"Computed Red Edge Position: range [{float(rep_da.min()):.1f}, {float(rep_da.max()):.1f}] nm")
    
    return rep_da


def compute_ndwi(cube: xr.DataArray) -> xr.DataArray:
    """
    Compute Normalized Difference Water Index (NDWI).
    
    Physics:
        NDWI = (Green - NIR) / (Green + NIR)
        
        NDWI detects water content in vegetation and open water bodies.
        - Water bodies: NDWI > 0.3
        - Vegetation with high water content: NDWI > 0
        - Dry vegetation/soil: NDWI < 0
        
        Green band (~550nm): High reflectance from water
        NIR band (~865nm): Strong absorption by water
    
    Args:
        cube: Reflectance cube (band, y, x) with wavelength coordinates
        
    Returns:
        NDWI map as xr.DataArray (y, x) with values in [-1, 1]
    """
    # Select bands
    green_band = cube.sel(band=550.0, method="nearest")
    nir_band = cube.sel(band=865.0, method="nearest")
    
    # Compute NDWI: (Green - NIR) / (Green + NIR)
    numerator = green_band - nir_band
    denominator = green_band + nir_band
    
    # Guard against division by zero
    ndwi = numerator / denominator
    ndwi = xr.where(denominator == 0, np.nan, ndwi)
    
    ndwi.attrs = {
        "index": "NDWI",
        "description": "Normalized Difference Water Index",
        "formula": "(Green - NIR) / (Green + NIR)",
        "green_band_nm": float(green_band.coords["band"].values),
        "nir_band_nm": float(nir_band.coords["band"].values),
        "units": "dimensionless"
    }
    
    logger.info(f"Computed NDWI: range [{float(ndwi.min()):.4f}, {float(ndwi.max()):.4f}]")
    
    return ndwi


def compute_nbr(cube: xr.DataArray) -> xr.DataArray:
    """
    Compute Normalized Burn Ratio (NBR).
    
    Physics:
        NBR = (NIR - SWIR) / (NIR + SWIR)
        
        NBR is sensitive to:
        - Vegetation moisture stress (lower NBR → more stress)
        - Burn severity (post-fire analysis)
        - Vegetation health
        
        NIR band (~865nm): High vegetation reflectance
        SWIR band (~2200nm): Strong absorption by water and cellulose
    
    Args:
        cube: Reflectance cube (band, y, x) with wavelength coordinates
        
    Returns:
        NBR map as xr.DataArray (y, x) with values in [-1, 1]
    """
    # Select bands
    nir_band = cube.sel(band=865.0, method="nearest")
    swir_band = cube.sel(band=2200.0, method="nearest")
    
    # Compute NBR: (NIR - SWIR) / (NIR + SWIR)
    numerator = nir_band - swir_band
    denominator = nir_band + swir_band
    
    # Guard against division by zero
    nbr = numerator / denominator
    nbr = xr.where(denominator == 0, np.nan, nbr)
    
    nbr.attrs = {
        "index": "NBR",
        "description": "Normalized Burn Ratio",
        "formula": "(NIR - SWIR) / (NIR + SWIR)",
        "nir_band_nm": float(nir_band.coords["band"].values),
        "swir_band_nm": float(swir_band.coords["band"].values),
        "units": "dimensionless"
    }
    
    logger.info(f"Computed NBR: range [{float(nbr.min()):.4f}, {float(nbr.max()):.4f}]")
    
    return nbr


def compute_all_indices(cube: xr.DataArray) -> xr.Dataset:
    """
    Compute all vegetation indices and return as xr.Dataset.
    
    Args:
        cube: Reflectance cube (band, y, x) with wavelength coordinates
        
    Returns:
        xr.Dataset with variables: ndvi, evi, red_edge, ndwi, nbr
    """
    logger.info("Computing all vegetation indices...")
    
    ndvi = compute_ndvi(cube)
    evi = compute_evi(cube)
    red_edge = compute_red_edge_position(cube)
    ndwi = compute_ndwi(cube)
    nbr = compute_nbr(cube)
    
    indices_dataset = xr.Dataset({
        "ndvi": ndvi,
        "evi": evi,
        "red_edge": red_edge,
        "ndwi": ndwi,
        "nbr": nbr
    })
    
    indices_dataset.attrs = {
        "description": "Vegetation spectral indices computed from hyperspectral reflectance",
        "indices": ["NDVI", "EVI", "Red Edge Position", "NDWI", "NBR"]
    }
    
    logger.info("Computed all vegetation indices")
    
    return indices_dataset
