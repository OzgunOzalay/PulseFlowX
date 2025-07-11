<div align="center">
  <img src="PulseFlowX.png" alt="PulseFlowX Logo" width="400"/>
</div>

# Advanced FIR Analysis with AFNI in Unpredictable Threat Task

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AFNI](https://img.shields.io/badge/AFNI-23.0+-green.svg)](https://afni.nimh.nih.gov/)
[![FIR Analysis](https://img.shields.io/badge/Analysis-FIR-red.svg)](https://en.wikipedia.org/wiki/Finite_impulse_response)
[![fMRI](https://img.shields.io/badge/Analysis-fMRI-purple.svg)](https://en.wikipedia.org/wiki/Functional_magnetic_resonance_imaging)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04+-orange.svg)](https://ubuntu.com/)


</div>

**PulseFlowX** is a comprehensive AFNI-based pipeline for **Finite Impulse Response (FIR)** analysis of sustained and phasic threat responses in fMRI data, with advanced group comparisons between Alcohol Use Disorder (AUD) and Healthy Controls (HC). This pipeline leverages FIR deconvolution methods to capture the full temporal dynamics of neural responses to threat stimuli.

## Features

- **FIR deconvolution analysis** using AFNI's TENT basis functions
- **Complete preprocessing pipeline** with AFNI tools
- **Sustained vs Phasic analysis** for comprehensive threat response characterization
- **ROI analysis** with subject-specific masks
- **Group comparisons** with publication-quality statistics
- **Automated visualizations** and reporting

## Installation

### System Requirements

- **Operating System**: Ubuntu 20.04+ (recommended)
- **RAM**: Minimum 8GB, 16GB+ recommended for large datasets
- **Storage**: Sufficient space for raw and processed fMRI data
- **Processor**: Multi-core CPU recommended for parallel processing

### 1. Install AFNI

**PulseFlowX** requires AFNI version 23.0 or higher with FIR deconvolution capabilities.

#### Ubuntu/Debian Installation:
```bash
# Update system
sudo apt-get update

# Install AFNI dependencies
sudo apt-get install -y tcsh xfonts-base python3-matplotlib python3-numpy \
                       python3-flask python3-flask-cors python3-pil libgsl-dev \
                       netpbm gnome-tweak-tool libjpeg62 xvfb xterm vim curl \
                       gedit evince firefox

# Download and install AFNI
curl -O https://afni.nimh.nih.gov/pub/dist/bin/misc/@update.afni.binaries
tcsh @update.afni.binaries -package linux_ubuntu_16_64 -do_extras
```

#### Verify AFNI Installation:
```bash
# Add AFNI to PATH (add to ~/.bashrc)
export PATH=$PATH:$HOME/abin

# Verify installation
afni -ver
3dDeconvolve -help
```

### 2. Install Python Dependencies

```bash
# Install Python 3.8+ (if not already installed)
sudo apt-get install python3 python3-pip

# Install PulseFlowX dependencies
pip install -r requirements.txt
```

#### Python Dependencies (`requirements.txt`):
```
pandas>=1.3.0          # Data manipulation and analysis
numpy>=1.20.0          # Numerical computing
rich>=10.0.0           # Rich text and beautiful formatting
scipy>=1.7.0           # Scientific computing
```

### 3. Verify Installation

```bash
# Test AFNI functionality
3dinfo -help

# Test Python environment
python3 -c "import pandas, numpy, scipy; print('All Python dependencies installed successfully!')"
```

## Quick Start

1. **Organize your data:**
```
Data/
├── sub-ALC2158/
│   ├── anat/sub-ALC2158_T1w.nii
│   └── func/sub-ALC2158_task-unpredictablethreat_run-1_bold.nii
└── ...
```

3. **Configure groups:**
```bash
python pulseflow_00_setup.py --add sub-ALC2158 AUD
python pulseflow_00_setup.py --add sub-ALC2161 HC
```

4. **Run the PulseFlowX pipeline:**
```bash
# Step 1: Preprocessing
python pulseflow_01_preprocess.py

# Step 2: Sustained/Phasic Analysis (with NIfTI conversion)
python pulseflow_02_dynamics.py --convert_nifti

# Step 3: ROI Analysis (requires subject-specific masks)
python pulseflow_03_roi.py --roi amygdala --hemisphere left

# Step 4: Group Analysis & Publication Outputs
python pulseflow_04_stats.py

# Alternative: Convert to NIfTI only (if analysis already completed)
python pulseflow_02_dynamics.py --nifti_only
```

## Pipeline Components

### `pulseflow_00_setup.py`
- Group assignment management
- Subject organization utilities
- Initial pipeline configuration

### `pulseflow_01_preprocess.py`
- Slice timing correction, despiking, motion correction
- GLM analysis with contrast calculations
- Quality control generation

### `pulseflow_02_dynamics.py`
- **FIR deconvolution** to extract sustained (0-20s) and phasic (0-14s) responses
- TENT basis functions for temporal dynamics modeling
- Group-level statistical comparisons
- Contrast generation for threat processing conditions
- **NIfTI conversion** for brain map inspection with FSLeyes

### `pulseflow_03_roi.py`
- ROI-based activation extraction
- Subject-specific mask support (left/right hemisphere analysis)
- Individual and group-level statistics

### `pulseflow_04_stats.py`
- Publication-quality statistical tables
- Comprehensive visualizations
- Effect size calculations and significance testing

## ROI Analysis Setup

For ROI analysis, place subject-specific masks in:
```
processed_data/roi_analysis/subject_masks/
├── sub-2158_Amyg_L_DWI.nii.gz
├── sub-2158_Amyg_R_DWI.nii.gz
└── ...
```

## Output Structure

```
processed_data/
├── glm_results/           # GLM analysis outputs
├── sustained_phasic_analysis/  # Sustained/phasic responses
│   ├── nifti_results/     # NIfTI files for FSLeyes inspection
│   └── group_analysis/    # Group-level results
├── roi_analysis/          # ROI extraction results
└── publication_outputs/   # Publication-ready tables & figures
    ├── figures/           # High-quality visualizations
    ├── tables/            # Statistical results
    └── GROUP_ANALYSIS_REPORT.md
```

## NIfTI Conversion and Brain Map Inspection

### NIfTI Conversion Functionality

The `pulseflow_02_dynamics.py` script converts AFNI group analysis results to NIfTI format for easy inspection with neuroimaging viewers like FSLeyes:

#### **Command Line Options:**

```bash
# Run analysis and convert to NIfTI
python pulseflow_02_dynamics.py --convert_nifti

# Convert existing results to NIfTI only
python pulseflow_02_dynamics.py --nifti_only

# Run group analysis only (without individual subject analysis)
python pulseflow_02_dynamics.py --group_only --convert_nifti
```

#### **NIfTI Files Generated:**

The conversion creates 12 NIfTI files (t-stats + p-values) for each contrast:

**Sustained Response Contrasts:**
- `Fear_vs_Neutral_sustained_AUD_vs_HC_tstat.nii.gz` + `*_pval.nii.gz`
- `Predictable_vs_Unknown_sustained_AUD_vs_HC_tstat.nii.gz` + `*_pval.nii.gz`
- `Unknown_Fear_vs_Neutral_sustained_AUD_vs_HC_tstat.nii.gz` + `*_pval.nii.gz`

**Phasic Response Contrasts:**
- `Fear_vs_Neutral_phasic_AUD_vs_HC_tstat.nii.gz` + `*_pval.nii.gz`
- `Predictable_vs_Unknown_phasic_AUD_vs_HC_tstat.nii.gz` + `*_pval.nii.gz`
- `Unknown_Fear_vs_Neutral_phasic_AUD_vs_HC_tstat.nii.gz` + `*_pval.nii.gz`

#### **Inspecting Results with FSLeyes:**

```bash
# Install FSLeyes (if not already installed)
pip install fsleyes

# Open p-value map for significance testing (p < 0.05)
fsleyes processed_data/sustained_phasic_analysis/nifti_results/Fear_vs_Neutral_sustained_AUD_vs_HC_pval.nii.gz

# Open t-stat map for effect direction and magnitude
fsleyes processed_data/sustained_phasic_analysis/nifti_results/Fear_vs_Neutral_sustained_AUD_vs_HC_tstat.nii.gz

# Open both together for comprehensive analysis
fsleyes processed_data/sustained_phasic_analysis/nifti_results/Fear_vs_Neutral_sustained_AUD_vs_HC_pval.nii.gz processed_data/sustained_phasic_analysis/nifti_results/Fear_vs_Neutral_sustained_AUD_vs_HC_tstat.nii.gz
```

#### **File Specifications:**

- **Format**: NIfTI-1 compressed (.nii.gz)
- **Content**: Both t-statistics and p-values from group comparisons (AUD vs HC)
- **Space**: Same as original AFNI analysis space
- **Size**: ~2.7MB per file
- **Threshold**: No thresholding applied (raw statistics)

#### **Analysis Details:**

**T-statistic files (*_tstat.nii.gz):**
- **Positive values**: AUD > HC activation
- **Negative values**: HC > AUD activation  
- **Zero values**: No group difference
- **Use for**: Effect direction and magnitude

**P-value files (*_pval.nii.gz):**
- **Values**: Two-tailed p-values (0.0 to 1.0)
- **Significance**: p < 0.05 for statistical significance
- **Use for**: Thresholding significant regions
- **Workflow**: Threshold p-map at p < 0.05, then check t-map for direction/magnitude

## System Requirements Summary

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **Operating System** | Ubuntu 20.04+ | Other Linux distributions may work |
| **AFNI** | Version 23.0+ | Must include FIR deconvolution tools |
| **Python** | 3.8+ | With pip package manager |
| **RAM** | 8GB minimum, 16GB+ recommended | For processing large fMRI datasets |
| **Storage** | Variable | Depends on dataset size |
| **CPU** | Multi-core recommended | For parallel processing |
| **ROI Masks** | Subject-specific masks in ORIG space | Required for ROI analysis |

For detailed installation instructions, see the [Installation](#installation) section above.

## Citation

If you use PulseFlowX, please cite the relevant AFNI tools and consider referencing this repository.

## Author & Lab

**PulseFlowX** was developed by **Ozgun Ozalay** (oozalay@unmc.edu) at the **Blackford Emotional Neuroscience Lab**, Monroe - Meyer Institute (MMI), University of Nebraska Medical Center (UNMC).

### Lab Information
- **Lab**: Blackford Emotional Neuroscience Lab
- **Institution**: Monroe - Meyer Institute (MMI), University of Nebraska Medical Center (UNMC)
- **Contact**: oozalay@unmc.edu

## License

This project is licensed under the MIT License.
