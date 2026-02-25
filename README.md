# HyperVeg: Production-Quality Hyperspectral Vegetation Analysis Pipeline

A comprehensive end-to-end pipeline for analyzing hyperspectral remote sensing data with focus on vegetation classification and spectral analysis. Built to demonstrate mastery of remote sensing physics, geospatial engineering, and machine learning.

## Scientific Background

Hyperspectral remote sensing captures the Earth's surface reflectance across hundreds of narrow, contiguous spectral bands, typically spanning the visible through shortwave infrared (400-2500 nm). Unlike multispectral sensors that measure a few broad bands, hyperspectral sensors provide detailed spectral signatures that enable precise material identification and vegetation health assessment.

The spectral signature of vegetation exhibits characteristic features:
- **Chlorophyll absorption** at ~670 nm (red) and ~430 nm (blue)
- **High reflectance** in the near-infrared (700-1300 nm) due to internal leaf structure
- **Water absorption** features at ~1400 nm and ~1900 nm
- **Cellulose/lignin absorption** in the SWIR (2000-2500 nm)

These features enable computation of vegetation indices (NDVI, EVI) and spectral unmixing to decompose mixed pixels into pure endmember fractions. Machine learning models can leverage the full spectral information for accurate land cover classification.

## Dataset

**Indian Pines** - AVIRIS hyperspectral sensor, Northwestern Indiana, USA
- **Spatial dimensions**: 145 × 145 pixels
- **Spectral bands**: 200 bands (corrected version, water vapor bands removed)
- **Wavelength range**: 400–2500 nm (visible through SWIR)
- **Ground truth**: 16 vegetation and land cover classes
  - Agricultural crops: Corn (3 variants), Soybean (3 variants), Alfalfa, Oats, Wheat
  - Natural vegetation: Grass (3 variants), Hay, Woods
  - Other: Buildings, Stone-Steel-Towers

**Download URLs** (no login required):
- Data: http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat
- Labels: http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    HyperVeg Pipeline                            │
└─────────────────────────────────────────────────────────────────┘

1. DATA LOADING
   └─ Download & load Indian Pines dataset → xarray DataArray

2. RADIOMETRIC CALIBRATION
   └─ Simulate DN → Apply calibration → Radiance (W/m²/sr/μm)
      L(λ) = gain(λ) × DN(λ) + offset(λ)

3. ATMOSPHERIC CORRECTION
   └─ Radiance → Surface Reflectance
      ρ(λ) = π × (L_sensor - L_path) / (E_s × cos(θ_s) × T(λ))
      T(λ) = exp(-τ(λ) / cos(θ_s)) × exp(-τ(λ) / cos(θ_v))
      τ(λ) = τ_rayleigh(λ) + τ_aerosol(λ) + τ_water(λ)

4. PREPROCESSING
   └─ Remove water vapor bands → Remove noisy bands → Normalize

5. SPECTRAL ANALYSIS
   ├─ Vegetation Indices (NDVI, EVI, NBR, NDWI, Red Edge)
   └─ Spectral Unmixing (FCLS)

6. BAND SELECTION
   └─ Jaya Optimization (Patro et al. 2019) - Selects 18 optimal bands

7. MACHINE LEARNING
   └─ SVM with spatial cross-validation

8. EVALUATION & VISUALIZATION
   └─ Metrics, confusion matrices, classification maps
```

## Physics Equations

### Radiometric Calibration
```
L(λ) = gain(λ) × DN(λ) + offset(λ)
```
where `L` is at-sensor radiance (W/m²/sr/μm), `DN` is digital number, and `gain`/`offset` are sensor-specific calibration coefficients.

### Atmospheric Transmittance (Beer-Lambert Law)
```
T(λ) = exp(-τ(λ) / cos(θ_s)) × exp(-τ(λ) / cos(θ_v))
```
where `τ(λ)` is total optical depth:
```
τ(λ) = τ_rayleigh(λ) + τ_aerosol(λ) + τ_water(λ)
```

Rayleigh scattering optical depth:
```
τ_rayleigh(λ) = 0.0088 × λ^(-4.15)  [λ in μm]
```

Aerosol optical depth (Angstrom model):
```
τ_aerosol(λ) = AOD_550 × (λ/0.55)^(-1.3)
```

### Surface Reflectance
```
ρ(λ) = π × (L_sensor(λ) - L_path(λ)) / (E_s(λ) × cos(θ_s) × T(λ))
```
where `π` comes from Lambertian hemisphere integration, `L_path` is path radiance (scattered light), and `E_s` is solar irradiance at top of atmosphere.

### Vegetation Indices

**NDVI** (Normalized Difference Vegetation Index):
```
NDVI = (NIR - Red) / (NIR + Red)
```

**EVI** (Enhanced Vegetation Index):
```
EVI = 2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)
```

### Spectral Unmixing (Linear Mixing Model)
```
x = E × a + ε
```
subject to:
- `a_k >= 0` (non-negativity)
- `Σ a_k = 1` (sum-to-one)

where `x` is observed spectrum, `E` is endmember matrix, and `a` is abundance vector.

## Installation

### Requirements
- Python 3.8+
- See `requirements.txt` for full dependency list

### Setup
```bash
# Clone repository
git clone <repository-url>
cd hyperveg

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Complete Pipeline
```bash
python main.py
```

This will:
1. Download the Indian Pines dataset (if not present)
2. Run the full pipeline: calibration → atmospheric correction → preprocessing
3. Compute spectral indices and perform Jaya band selection
4. Train and evaluate SVM classifier
5. Generate all visualizations and save to `outputs/` directory

### Project Structure
```
hyperveg/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                        # Main pipeline script
├── data/                          # Dataset storage (gitignored)
│   └── .gitkeep
├── outputs/                       # Generated figures and results (gitignored)
│   └── .gitkeep
├── src/
│   ├── data/
│   │   └── loader.py             # Data loading and download
│   ├── pipeline/
│   │   ├── calibration.py        # Radiometric calibration
│   │   ├── atmospheric.py        # Atmospheric correction
│   │   └── preprocessing.py      # Data cleaning and normalization
│   ├── analysis/
│   │   ├── indices.py            # Vegetation indices
│   │   ├── band_selection.py     # Jaya band selection
│   │   └── unmixing.py           # Spectral unmixing
│   ├── models/
│   │   ├── svm_classifier.py     # SVM with spatial CV
│   │   └── evaluation.py         # Metrics and evaluation
│   └── visualization/
│       └── plots.py              # All visualization functions
├── notebooks/
│   ├── 01_exploration.ipynb     # Exploration notebook
│   └── 02_full_pipeline.ipynb  # Complete pipeline notebook
└── tests/
    ├── test_calibration.py
    ├── test_atmospheric.py
    ├── test_indices.py
    ├── test_band_selection.py
    └── test_models.py
```

## Results

### Classification Performance

**SVM (RBF kernel, spatial cross-validation, 18 selected bands)**:
- Overall Accuracy: ~XX% (varies by fold)
- Cohen's Kappa: ~XX
- Macro F1: ~XX

*Note: Actual numbers will be generated when running the pipeline. Band selection reduces from 200 to 18 bands while preserving physical interpretability.*

### Key Features

- **Physics-based processing**: Full radiometric calibration and atmospheric correction pipeline
- **Jaya band selection**: Feature selection (not extraction) that preserves physical meaning of spectral bands
- **Spatial cross-validation**: Scientifically correct train/test splitting that avoids spatial autocorrelation
- **Comprehensive analysis**: Vegetation indices, spectral unmixing, and optimal band selection
- **Production-quality code**: Type hints, docstrings, logging, unit tests
- **Publication-ready visualizations**: High-quality figures for presentations and papers

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## References

1. **Indian Pines Dataset**: 
   - Baumgardner, M. F., Biehl, L. L., & Landgrebe, D. A. (2015). "220 Band AVIRIS Hyperspectral Image Data Set: June 12, 1992 Indian Pine Test Site 3." Purdue University Research Repository. DOI: 10.4231/R7RX991C

2. **Hyperspectral Remote Sensing**:
   - Chang, C. I. (2003). "Hyperspectral Imaging: Techniques for Spectral Detection and Classification." Springer.

3. **Atmospheric Correction**:
   - Gao, B. C., et al. (1993). "Derivation of scaled surface reflectances from AVIRIS data." Remote Sensing of Environment, 44(2-3), 165-178.

4. **Spectral Unmixing**:
   - Keshava, N., & Mustard, J. F. (2002). "Spectral unmixing." IEEE Signal Processing Magazine, 19(1), 44-57.

5. **Spatial Cross-Validation**:
   - Ploton, P., et al. (2020). "Spatial validation reveals poor predictive performance of large-scale ecological mapping models." Nature Communications, 11(1), 1-11.

6. **Jaya Band Selection**:
   - Patro, R.N., Subudhi, S., Biswal, P.K. (2019). "Spectral clustering and spatial Frobenius norm-based Jaya optimisation for BS of hyperspectral images." IET Image Processing, 13(2), 307-315.

## License

This project is for demonstration purposes as part of a job interview portfolio.

## Author

Built for a Junior Geospatial Data Scientist position interview, demonstrating expertise in:
- Remote sensing physics and radiative transfer
- Geospatial data engineering
- Machine learning for hyperspectral classification
- Production-quality Python development
