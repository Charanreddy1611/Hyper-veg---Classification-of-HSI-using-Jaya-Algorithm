"""
Data Loading Module for Indian Pines Hyperspectral Dataset

Handles downloading, loading, and packaging the Indian Pines dataset into
a clean xarray DataArray with proper metadata.
"""

import numpy as np
import scipy.io
import xarray as xr
import requests
from pathlib import Path
from typing import Tuple, Dict
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def download_indian_pines(data_dir: str = "data/") -> None:
    """
    Download the Indian Pines dataset files if not already present.
    
    Downloads both the hyperspectral cube and ground truth labels from
    the public repository. Shows download progress with tqdm.
    
    Args:
        data_dir: Directory to store the downloaded files
        
    Raises:
        requests.RequestException: If download fails
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    urls = {
        "Indian_pines_corrected.mat": "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
        "Indian_pines_gt.mat": "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
    }
    
    for filename, url in urls.items():
        filepath = data_path / filename
        
        if filepath.exists():
            logger.info(f"{filename} already exists, skipping download")
            continue
        
        logger.info(f"Downloading {filename}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Verify file size
            downloaded_size = filepath.stat().st_size
            if total_size > 0 and downloaded_size != total_size:
                logger.warning(f"File size mismatch for {filename}: expected {total_size}, got {downloaded_size}")
            else:
                logger.info(f"Successfully downloaded {filename} ({downloaded_size / 1024 / 1024:.2f} MB)")
                
        except requests.RequestException as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise


def get_class_names() -> Dict[int, str]:
    """
    Get mapping of class IDs to class names for Indian Pines dataset.
    
    Returns:
        Dictionary mapping class integer (0-16) to class name string
    """
    return {
        0: "Unlabeled",
        1: "Alfalfa",
        2: "Corn-notill",
        3: "Corn-mintill",
        4: "Corn",
        5: "Grass-pasture",
        6: "Grass-trees",
        7: "Grass-pasture-mowed",
        8: "Hay-windrowed",
        9: "Oats",
        10: "Soybean-notill",
        11: "Soybean-mintill",
        12: "Soybean-clean",
        13: "Wheat",
        14: "Woods",
        15: "Buildings-Grass-Trees-Drives",
        16: "Stone-Steel-Towers"
    }


def load_indian_pines(data_dir: str = "data/") -> Tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """
    Load the Indian Pines dataset from .mat files and package into xarray.
    
    Loads the hyperspectral cube and ground truth labels, converts to proper
    format with dimensions [band, y, x], and creates an xarray DataArray with
    wavelength coordinates and metadata.
    
    Args:
        data_dir: Directory containing the .mat files
        
    Returns:
        Tuple of:
        - cube_dataarray: xr.DataArray with dims ["band", "y", "x"]
        - labels_array: Ground truth labels as numpy array (145, 145)
        - wavelengths_array: Wavelength centers in nm (200,)
        
    Raises:
        FileNotFoundError: If required .mat files are not found
        ValueError: If data shape is unexpected
    """
    data_path = Path(data_dir)
    
    cube_file = data_path / "Indian_pines_corrected.mat"
    gt_file = data_path / "Indian_pines_gt.mat"
    
    if not cube_file.exists():
        raise FileNotFoundError(f"Hyperspectral cube file not found: {cube_file}")
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    
    # Load .mat files
    logger.info("Loading hyperspectral cube...")
    cube_mat = scipy.io.loadmat(str(cube_file))
    gt_mat = scipy.io.loadmat(str(gt_file))
    
    # Extract data (keys are usually 'indian_pines_corrected' and 'indian_pines_gt')
    cube_key = None
    gt_key = None
    
    for key in cube_mat.keys():
        if not key.startswith('__') and 'indian' in key.lower():
            cube_key = key
            break
    
    for key in gt_mat.keys():
        if not key.startswith('__') and ('gt' in key.lower() or 'indian' in key.lower()):
            gt_key = key
            break
    
    if cube_key is None:
        # Try to find any non-meta key
        for key in cube_mat.keys():
            if not key.startswith('__'):
                cube_key = key
                break
    
    if gt_key is None:
        for key in gt_mat.keys():
            if not key.startswith('__'):
                gt_key = key
                break
    
    if cube_key is None or gt_key is None:
        raise ValueError("Could not find data keys in .mat files")
    
    cube_data = cube_mat[cube_key].astype(np.float32)
    labels_data = gt_mat[gt_key].astype(np.int32)
    
    # Indian Pines corrected has shape (145, 145, 200) - need to transpose to (200, 145, 145)
    if cube_data.ndim == 3:
        # Assume shape is (y, x, band) and convert to (band, y, x)
        if cube_data.shape[2] == 200:
            cube_data = np.transpose(cube_data, (2, 0, 1))
        elif cube_data.shape[0] == 200:
            # Already in (band, y, x) format
            pass
        else:
            raise ValueError(f"Unexpected cube shape: {cube_data.shape}")
    
    # Define wavelength centers: 200 bands from 400nm to 2500nm evenly spaced
    wavelengths = np.linspace(400, 2500, 200)
    
    # Create xarray DataArray
    cube_da = xr.DataArray(
        cube_data,
        dims=["band", "y", "x"],
        coords={
            "band": wavelengths,
            "y": np.arange(cube_data.shape[1]),
            "x": np.arange(cube_data.shape[2])
        },
        attrs={
            "sensor": "AVIRIS",
            "scene": "Indian Pines",
            "n_classes": 16,
            "wavelength_units": "nm",
            "description": "Indian Pines hyperspectral dataset (corrected, water vapor bands removed)",
            "spatial_shape": f"{cube_data.shape[1]}x{cube_data.shape[2]}",
            "n_bands": cube_data.shape[0]
        }
    )
    
    logger.info(f"Loaded cube: shape {cube_data.shape}, wavelength range {wavelengths.min():.1f}-{wavelengths.max():.1f} nm")
    
    return cube_da, labels_data, wavelengths


def get_dataset_info(cube: xr.DataArray, labels: np.ndarray) -> None:
    """
    Print a formatted summary of the dataset.
    
    Displays shape, wavelength range, number of labeled pixels, class
    distribution (count and percentage), and memory usage.
    
    Args:
        cube: Hyperspectral cube DataArray
        labels: Ground truth labels array
    """
    class_names = get_class_names()
    
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    
    print(f"\nSpatial Shape: {cube.shape[1]} × {cube.shape[2]} pixels")
    print(f"Number of Bands: {cube.shape[0]}")
    print(f"Wavelength Range: {cube.coords['band'].min().values:.1f} - {cube.coords['band'].max().values:.1f} nm")
    
    # Memory usage
    cube_memory_mb = cube.nbytes / (1024 * 1024)
    labels_memory_mb = labels.nbytes / (1024 * 1024)
    print(f"Memory Usage: {cube_memory_mb:.2f} MB (cube) + {labels_memory_mb:.2f} MB (labels) = {cube_memory_mb + labels_memory_mb:.2f} MB total")
    
    # Label statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_pixels = labels.size
    labeled_pixels = np.sum(labels > 0)
    
    print(f"\nLabeled Pixels: {labeled_pixels:,} / {total_pixels:,} ({100*labeled_pixels/total_pixels:.1f}%)")
    
    print("\nClass Distribution:")
    print("-" * 60)
    print(f"{'Class ID':<10} {'Class Name':<35} {'Count':<12} {'Percentage':<10}")
    print("-" * 60)
    
    for label_id, count in zip(unique_labels, counts):
        class_name = class_names.get(int(label_id), f"Unknown-{label_id}")
        percentage = 100 * count / total_pixels
        print(f"{label_id:<10} {class_name:<35} {count:<12,} {percentage:<10.2f}%")
    
    print("="*60 + "\n")
