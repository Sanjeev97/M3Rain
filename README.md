# M3Rain - Multi-Modal Meteorological Transformer for Rainfall Prediction

A deep learning framework that combines weather radar imagery and personal weather station (PWS) data for accurate rainfall prediction using a novel multi-modal transformer architecture.

## Overview

M3Rain integrates two complementary data sources:
- **Weather Radar Data**: NEXRAD Level-2 radar reflectivity images providing spatial precipitation patterns
- **Personal Weather Station Data**: Ground-truth meteorological measurements including temperature, humidity, pressure, and wind parameters

The model uses a multi-modal transformer architecture (M3) that processes radar images as visual tokens and PWS time series as contextual features to predict rainfall rates.

## Architecture

The M3 (Multi-Modal Meteorological) transformer consists of:
- **Vision Transformer Encoder**: Processes radar image patches
- **Time Series Encoder**: Encodes PWS meteorological features
- **Multi-Modal Attention**: Attention between visual and temporal modalities
- **Temporal Decoder**: Generates rainfall predictions

### Baseline Models

The repository also includes several baseline models for comparison:
- **DLinear**: Decomposition-Linear model for time series forecasting
- **PatchTST**: Patch-based Time Series Transformer
- **iTransformer**: Inverted Transformer for multivariate time series
- **M3T**: Basic multi-modal transformer baseline

Each baseline model has been adapted with M3Rain-specific configurations (DLinearnM3, PatchTSTnM3, iTnM3).

## Dataset

The project uses data from Lake Charles, Louisiana (KLCH radar station):
- **Radar Data**: NEXRAD Level-2 data from AWS S3 (`noaa-nexrad-level2`)
- **PWS Data**: Weather Underground personal weather station observations
- **Time Period**: 2022-2024
- **Spatial Coverage**: 100km radius around Lake Charles
- **Temporal Resolution**: 15-minute intervals

## Project Structure

```
M3Rain/
├── Scripts/
│   ├── PWS/                    # Personal Weather Station data processing
│   │   ├── convertwu-csv.py    # JSON to CSV conversion
│   │   └── downloadwu_modified.bash  # Weather Underground data download
│   └── Radar/                  # Radar data processing pipeline
│       ├── 1-download-radar.py         # Download NEXRAD data
│       ├── 2-convert-format.py         # Convert radar file formats
│       ├── 3-convert-cartesian.py      # Convert to Cartesian coordinates
│       ├── 4-extract-lakecharles.py    # Extract Lake Charles region
│       ├── 5-interpolate-composite4v1-radar-verify.py  # Create composite reflectivity
│       ├── 5-interpolate-pws.py        # Interpolate PWS data
│       ├── 6-filter-align-radar-pws-20.py  # Align radar and PWS data
│       ├── preprocessing_pws.ipynb     # PWS data preprocessing notebook
│       └── preprocessing_radar.ipynb   # Radar data preprocessing notebook
├── dataset/
│   ├── aligned_dataset.py      # HDF5 dataset loader
│   └── data_wrapper.py         # Data augmentation wrapper
├── layers/                     # Neural network layer components
│   ├── Embed.py               # Embedding layers
│   ├── SelfAttention_Family.py # Attention mechanisms
│   ├── Transformer_EncDec.py  # Transformer encoder/decoder
│   └── __init__.py
├── models/                     # Model architectures
│   ├── DLinear.py             # DLinear baseline model
│   ├── DLinearnM3.py          # DLinear adapted for M3Rain
│   ├── PatchTST.py            # PatchTST time series transformer
│   ├── PatchTSTnM3.py         # PatchTST adapted for M3Rain
│   ├── iTransformer.py        # iTransformer model
│   ├── iTnM3.py               # iTransformer adapted for M3Rain
│   ├── m3.py                  # M3 transformer architecture
│   ├── transformer.py        # Basic transformer model
│   └── __init__.py
├── modules/
│   └── attention.py           # Attention mechanism implementations
├── util/
│   ├── lr_sched.py            # Learning rate scheduling
│   ├── metrics.py             # Evaluation metrics
│   └── misc.py                # Utility functions
├── run_m3.py                  # M3 model training script
├── test_m3.py                 # M3 model testing/evaluation script
├── run_transformer.py         # Baseline models training script
└── test_transformer.py        # Baseline models testing script
```

## Installation

### Requirements
- Python 3.10+
- PyTorch
- NumPy
- Pandas
- h5py
- netCDF4
- einops
- timm
- scipy
- scikit-learn
- matplotlib
- tqdm

### Model-Specific Dependencies
- **Attention Mechanisms**: einops for tensor operations
- **Time Series Models**: layers from this repository
- **Baseline Models**: Custom implementations included

### Radar Data Processing Tools
- RadxConvert (from LROSE/Radx toolkit)
- Radx2Grid (from LROSE/Radx toolkit)

### AWS CLI
For downloading NEXRAD data:
```bash
pip install awscli
```

## Data Processing Pipeline

### 1. Radar Data Processing

```bash
# Download NEXRAD Level-2 data
python Scripts/Radar/1-download-radar.py

# Convert radar file formats
python Scripts/Radar/2-convert-format.py

# Convert to Cartesian grid
python Scripts/Radar/3-convert-cartesian.py

# Extract Lake Charles region
python Scripts/Radar/4-extract-lakecharles.py

# Create composite reflectivity with verification
python Scripts/Radar/5-interpolate-composite4v1-radar-verify.py
```

### 2. PWS Data Processing

```bash
# Download Weather Underground data
bash Scripts/PWS/downloadwu_modified.bash

# Convert JSON to CSV
python Scripts/PWS/convertwu-csv.py

# Interpolate PWS data to regular intervals
python Scripts/Radar/5-interpolate-pws.py
```

### 3. Data Alignment

```bash
# Align radar and PWS data, create HDF5 dataset
python Scripts/Radar/6-filter-align-radar-pws-20.py
```

## Model Training

### M3 Model Training
python run_m3.py \


### Baseline Models Training
python run_transformer.py \


All baseline models are configured for 4-step input and 4-step prediction with 20 PWS features.

## Model Evaluation

### M3 Model Evaluation
python test_m3.py \


### Baseline Models Evaluation
python test_transformer.py \


### Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **PCC**: Pearson Correlation Coefficient
- **CSI**: Critical Success Index (for rainfall detection)

Performance is evaluated using both regression metrics (RMSE, MAE, R², correlation) and precipitation detection metrics CSI.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{m3rain2024,
  title={M3Rain: Multi-Modal Meteorological Transformer for Rainfall Prediction},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024}
}
```


## Acknowledgments

- NOAA for providing NEXRAD radar data
- Weather Underground for personal weather station data
- LROSE team for radar data processing tools
- PyTorch and timm communities for deep learning frameworks

## Contact

For questions and support, please open an issue on GitHub or contact [contact information].

---
