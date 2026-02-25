"""
HyperVeg: End-to-End Hyperspectral Vegetation Analysis Pipeline

Runs the complete pipeline from data loading through classification.
"""

import time
import logging
from pathlib import Path
import numpy as np
import xarray as xr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
from src.data.loader import download_indian_pines, load_indian_pines, get_class_names, get_dataset_info
from src.pipeline.calibration import generate_synthetic_dn, apply_radiometric_calibration, validate_calibration
from src.pipeline.atmospheric import apply_atmospheric_correction
from src.pipeline.preprocessing import (
    remove_water_vapor_bands, remove_noisy_bands, normalize_reflectance, extract_pixel_matrix
)
from src.analysis.indices import compute_all_indices
from src.analysis.band_selection import run_jaya_with_ranking, apply_band_selection, plot_band_selection_results
from src.analysis.unmixing import extract_endmembers_manual, unmix_cube, compute_reconstruction_error
from src.models.svm_classifier import create_spatial_blocks, train_svm_classifier, run_spatial_cv
from src.models.evaluation import compute_metrics, plot_confusion_matrix, plot_classification_map
from src.visualization.plots import (
    plot_false_color_composite, plot_spectral_signatures, plot_spectral_indices_maps,
    plot_abundance_maps
)


def print_banner():
    """Print project banner."""
    print("\n" + "="*70)
    print(" " * 15 + "HyperVeg Pipeline")
    print(" " * 10 + "Hyperspectral Vegetation Analysis")
    print("="*70 + "\n")


def main():
    """Run the complete HyperVeg pipeline."""
    print_banner()
    
    # Setup paths
    data_dir = Path("data")
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    total_start_time = time.time()
    
    # ========================================================================
    # STAGE 1: Data Loading
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 1: DATA LOADING")
    print("="*70)
    stage_start = time.time()
    
    # Download data if needed
    print("\n[1.1] Downloading Indian Pines dataset...")
    download_indian_pines(str(data_dir))
    
    # Load dataset
    print("\n[1.2] Loading dataset...")
    cube, labels, wavelengths = load_indian_pines(str(data_dir))
    class_names = get_class_names()
    get_dataset_info(cube, labels)
    
    stage_time = time.time() - stage_start
    print(f"\n[OK] Stage 1 completed in {stage_time:.2f} seconds")
    
    # ========================================================================
    # STAGE 2: Simulate DN from Reflectance
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 2: SIMULATE DIGITAL NUMBERS")
    print("="*70)
    stage_start = time.time()
    
    print("\n[2.1] Generating synthetic DN from reflectance...")
    dn_cube, gain_array, offset_array = generate_synthetic_dn(cube, wavelengths, seed=42)
    print(f"  Generated DN cube: shape {dn_cube.shape}, range [{dn_cube.values.min()}, {dn_cube.values.max()}]")
    
    stage_time = time.time() - stage_start
    print(f"\n[OK] Stage 2 completed in {stage_time:.2f} seconds")
    
    # ========================================================================
    # STAGE 3: Radiometric Calibration (DN → Radiance)
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 3: RADIOMETRIC CALIBRATION (DN -> Radiance)")
    print("="*70)
    stage_start = time.time()
    
    print("\n[3.1] Applying radiometric calibration...")
    radiance_cube = apply_radiometric_calibration(dn_cube, gain_array, offset_array)
    
    print("\n[3.2] Validating calibration...")
    validation_results = validate_calibration(cube, radiance_cube, gain_array, offset_array)
    print(f"  Mean correlation: {validation_results['mean_correlation']:.4f}")
    print(f"  Validation passed: {validation_results['validation_passed']}")
    
    stage_time = time.time() - stage_start
    print(f"\n[OK] Stage 3 completed in {stage_time:.2f} seconds")
    
    # ========================================================================
    # STAGE 4: Atmospheric Correction (Radiance → Reflectance)
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 4: ATMOSPHERIC CORRECTION (Radiance -> Reflectance)")
    print("="*70)
    stage_start = time.time()
    
    print("\n[4.1] Applying atmospheric correction...")
    reflectance_cube = apply_atmospheric_correction(
        radiance_cube, wavelengths, solar_zenith_deg=35.0, aod_550=0.1
    )
    print(f"  Reflectance range: [{np.nanmin(reflectance_cube.values):.4f}, {np.nanmax(reflectance_cube.values):.4f}]")
    
    stage_time = time.time() - stage_start
    print(f"\n[OK] Stage 4 completed in {stage_time:.2f} seconds")
    
    # ========================================================================
    # ANALYSIS STAGE 1: Spectral Indices
    # ========================================================================
    print("\n" + "="*70)
    print("ANALYSIS STAGE 1: SPECTRAL INDICES")
    print("="*70)
    stage_start = time.time()
    
    print("\n[6.1] Computing vegetation indices...")
    # Compute indices on original reflectance cube (before band selection)
    indices_dataset = compute_all_indices(reflectance_cube)
    
    print("\n[6.2] Saving indices maps...")
    plot_spectral_indices_maps(indices_dataset, save_path=str(outputs_dir / "indices_maps.png"))
    
    stage_time = time.time() - stage_start
    print(f"\n[OK] Analysis Stage 1 completed in {stage_time:.2f} seconds")
    
    # ========================================================================
    # ANALYSIS STAGE 2: Jaya Band Selection
    # ========================================================================
    print("\n" + "="*70)
    print("ANALYSIS STAGE 2: JAYA BAND SELECTION")
    print("Reference: Patro et al. (2019) IET Image Processing")
    print("="*70)
    stage_start = time.time()
    
    print("\n[7.1] Running Jaya band selection...")
    selected_indices, band_frequencies, cost_histories = run_jaya_with_ranking(
        cube=reflectance_cube,
        n_clusters=17,
        population_size=10,
        max_iterations=50,
        bands_per_run=5,
        n_evaluations=50,
        final_n_bands=18,
        gaussian_sigma=1.0,
        seed=42
    )
    
    print("\n[7.2] Applying band selection to cube...")
    reduced_cube = apply_band_selection(reflectance_cube, selected_indices)
    
    print("\n[7.3] Plotting band selection results...")
    plot_band_selection_results(
        band_frequencies=band_frequencies,
        selected_indices=selected_indices,
        wavelengths=wavelengths,
        cost_histories=cost_histories,
        save_path=str(outputs_dir / "band_selection_results.png")
    )
    
    print(f"  Selected {len(selected_indices)} bands from {reflectance_cube.shape[0]}")
    print(f"  Selected wavelengths: {wavelengths[selected_indices].astype(int)} nm")
    
    # Update cube_clean and normalize the reduced cube
    cube_clean = reduced_cube
    cube_normalized, norm_params = normalize_reflectance(reduced_cube, method="standardize")
    
    stage_time = time.time() - stage_start
    print(f"\n[OK] Analysis Stage 2 completed in {stage_time:.2f} seconds")
    
    # ========================================================================
    # ANALYSIS STAGE 3: Spectral Unmixing
    # ========================================================================
    print("\n" + "="*70)
    print("ANALYSIS STAGE 3: SPECTRAL UNMIXING")
    print("="*70)
    stage_start = time.time()
    
    print("\n[8.1] Extracting endmembers...")
    endmember_classes = [2, 10, 5, 14]  # Corn, Soybean, Grass, Woods
    # Use original reflectance_cube for endmember extraction (before band selection)
    endmembers, endmember_names = extract_endmembers_manual(
        reflectance_cube, labels, endmember_classes, class_names
    )
    
    print("\n[8.2] Performing FCLS unmixing...")
    # Use original reflectance_cube for unmixing (before band selection)
    abundances = unmix_cube(reflectance_cube, endmembers, endmember_names)
    plot_abundance_maps(abundances, save_path=str(outputs_dir / "abundance_maps.png"))
    
    print("\n[8.3] Computing reconstruction error...")
    # Use original reflectance_cube for reconstruction error
    reconstruction_error = compute_reconstruction_error(reflectance_cube, abundances, endmembers)
    
    stage_time = time.time() - stage_start
    print(f"\n[OK] Analysis Stage 3 completed in {stage_time:.2f} seconds")
    
    # ========================================================================
    # MODEL STAGE 1: SVM Classification
    # ========================================================================
    print("\n" + "="*70)
    print("MODEL STAGE 1: SVM CLASSIFICATION")
    print("="*70)
    stage_start = time.time()
    
    print("\n[9.1] Extracting pixel matrix...")
    X, y = extract_pixel_matrix(cube_normalized, labels, exclude_unlabeled=True)
    
    print("\n[9.2] Creating spatial blocks...")
    block_ids = create_spatial_blocks(labels, n_blocks=5)
    
    print("\n[9.3] Running spatial cross-validation...")
    svm_results = run_spatial_cv(X, y, block_ids, n_blocks=5)
    
    print(f"\n  SVM Results:")
    print(f"    Overall Accuracy: {svm_results['overall_accuracy']['mean']:.4f} ± {svm_results['overall_accuracy']['std']:.4f}")
    print(f"    Kappa: {svm_results['kappa']['mean']:.4f} ± {svm_results['kappa']['std']:.4f}")
    
    # Train final model on all data for visualization
    print("\n[9.4] Training final SVM model...")
    spatial_positions = np.zeros((len(y), 2), dtype=int)
    n_y, n_x = labels.shape
    for i in range(len(y)):
        y_pos = i // n_x
        x_pos = i % n_x
        spatial_positions[i] = [y_pos, x_pos]
    
    # Use 80% for training, 20% for test
    split_idx = int(0.8 * len(X))
    X_train_svm = X[:split_idx]
    X_test_svm = X[split_idx:]
    y_train_svm = y[:split_idx]
    y_test_svm = y[split_idx:]
    
    svm_pipeline = train_svm_classifier(X_train_svm, y_train_svm, optimize=False)
    y_pred_svm = svm_pipeline.predict(X_test_svm)
    svm_metrics = compute_metrics(y_test_svm, y_pred_svm, class_names)
    
    # Plot classification map
    plot_classification_map(
        y_pred_svm, y_test_svm, class_names,
        title="SVM Classification Results",
        save_path=str(outputs_dir / "svm_classification_map.png")
    )
    
    stage_time = time.time() - stage_start
    print(f"\n[OK] Model Stage 1 completed in {stage_time:.2f} seconds")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\n[11.1] False color composite...")
    plot_false_color_composite(
        cube_clean, save_path=str(outputs_dir / "false_color_composite.png")
    )
    
    print("\n[11.2] Spectral signatures...")
    plot_spectral_signatures(
        reflectance_cube, labels, class_names,
        class_ids=[2, 10, 5, 14],  # Corn, Soybean, Grass, Woods
        wavelengths=wavelengths,
        save_path=str(outputs_dir / "spectral_signatures.png")
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    total_time = time.time() - total_start_time
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"\nOutputs saved to: {outputs_dir.absolute()}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
