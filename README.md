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
- **Multi-Modal Attention**: Cross-attention between visual and temporal modalities
- **Temporal Decoder**: Generates rainfall predictions

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
│       └── 6-filter-align-radar-pws-20.py  # Align radar and PWS data
├── dataset/
│   ├── aligned_dataset.py      # HDF5 dataset loader
│   └── data_wrapper.py         # Data augmentation wrapper
├── models/
│   └── m3.py                   # M3 transformer architecture
├── util/
│   ├── lr_sched.py            # Learning rate scheduling
│   ├── metrics.py             # Evaluation metrics
│   └── misc.py                # Utility functions
├── run_m3.py                  # Training script
└── test_m3.py                 # Testing/evaluation script
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

python run_m3.py 

### Key Training Parameters
- **Learning Rate**: 1e-3 with cosine annealing
- **Batch Size**: 64
- **Optimizer**: AdamW with weight decay
- **Loss Function**: MSE Loss
- **Data Split**: 85% train, 15% test

## Model Evaluation

python test_m3.py \

### Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **PCC**: Pearson Correlation Coefficient
- **CSI**: Critical Success Index (for rainfall detection)
- **Precision/Recall/F1**: Binary classification metrics

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

## License


## Acknowledgments

- NOAA for providing NEXRAD radar data
- Weather Underground for personal weather station data
- LROSE team for radar data processing tools
- PyTorch and timm communities for deep learning frameworks

## Contact

For questions and support, please open an issue on GitHub or contact [contact information].

---

**Note**: This project requires significant computational resources for data processing and model training. Consider using cloud computing resources for large-scale experiments.
