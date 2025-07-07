# HRF-Deconvolution: FMRI Processing Pipeline

This pipeline processes fMRI data using AFNI tools, implementing a comprehensive preprocessing and analysis workflow for threat processing studies in Alcohol Use Disorder (AUD) vs Healthy Controls (HC).

## Features

- Complete preprocessing pipeline including:
  - Slice timing correction
  - Despiking
  - Motion correction
  - Spatial smoothing
  - Quality control (QC) generation
- Automated timing file creation from event TSVs
- AFNI processing script generation
- Detailed logging and progress tracking
- Quality control visualization
- **Sustained vs Phasic Response Analysis** using TENT basis functions
- **ROI Analysis** with atlas-based region extraction (amygdala, insula, prefrontal cortex, hippocampus)
- **Group Analysis** for AUD vs HC comparisons with statistical testing
- **Flexible Group Management** via JSON configuration
- **Automated Contrast Calculations** for threat processing conditions
- **Comprehensive Analysis Pipeline** with dependency checking and reporting

## Directory Structure

```
processed_data/
├── raw/                    # Original data copies
├── preprocessed/           # Preprocessed data
├── timing_files/          # Timing files for analysis
├── scripts/               # AFNI processing scripts
├── logs/                  # Processing logs
├── qc/                    # Quality control outputs
└── sustained_phasic_analysis/  # Sustained/phasic response analysis
    ├── sub-*/             # Individual subject results
    │   ├── sustained/     # Sustained response components
    │   └── phasic/        # Phasic response components
    └── group_analysis/    # Group-level results
└── roi_analysis/          # ROI-based analysis
    ├── atlas_masks/       # Atlas-based ROI masks
    ├── sub-*/             # Individual subject ROI results
    ├── plots/             # ROI analysis visualizations
    └── group_statistics/  # Statistical test results

plots/
├── hrf_analysis/          # HRF analysis visualizations
├── timeseries/            # Time series plots
└── sustained_phasic/      # Sustained/phasic analysis plots
```

## Requirements

- Python 3.6+
- AFNI tools (with Harvard-Oxford atlas)
- Python packages (see requirements.txt):
  - pandas >= 1.3.0
  - numpy >= 1.20.0
  - rich >= 10.0.0
  - matplotlib >= 3.5.0
  - seaborn >= 0.11.0
  - scipy >= 1.7.0
  - nibabel >= 3.2.0

## Installation

1. Install AFNI tools:
```bash
# Follow AFNI installation instructions for your system
# https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/main_toc.html
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Organize your data in the following structure:
```
Data/
├── sub-ALC2158/
│   ├── anat/
│   │   └── sub-ALC2158_T1w.nii
│   └── func/
│       ├── sub-ALC2158_task-unpredictablethreat_run-1_bold.nii
│       ├── sub-ALC2158_task-unpredictablethreat_run-1_events.tsv
│       └── ...
└── sub-ALC2161/
    └── ...
```

2. **Configure Group Assignments** (for sustained/phasic analysis):
```bash
# View current group assignments
python manage_groups.py --list

# List all available subjects
python manage_groups.py --subjects

# Add a subject to a group
python manage_groups.py --add sub-ALC2158 AUD

# Remove a subject from a group
python manage_groups.py --remove sub-ALC2158 HC
```

3. Run the preprocessing pipeline:
```bash
# Sequential processing (memory-efficient, recommended)
python process_fmri.py

# Or explicitly enable sequential processing
python process_fmri.py --sequential

# Parallel processing (use with caution - high memory usage)
python process_fmri.py --parallel --threads 4

# Process a single subject
python process_fmri.py --subject sub-ALC2158

# Skip cleanup of existing processed data
python process_fmri.py --no-cleanup


```

4. Run the sustained/phasic analysis:
```bash
python sustained_phasic_analysis.py
```

5. Run ROI analysis (e.g., amygdala):
```bash
python roi_analysis.py --roi amygdala --hemisphere bilateral
```

6. Generate visualizations:
```bash
python plot_sustained_phasic_results.py
```

7. Run ROI analysis:
```bash
python roi_analysis.py --roi amygdala
```

## Processing Steps

1. **Data Organization**
   - Copies raw data to processing directory
   - Creates necessary directory structure

2. **Preprocessing**
   - Slice timing correction (3dTcat)
   - Despiking (3dDespike) - **Never skipped, critical for data quality**
   - Motion correction (3dvolreg) - **Single-threaded (AFNI limitation)**
   - Spatial smoothing (3dmerge) - **Multi-threaded with OpenCL acceleration**

3. **Timing File Creation**
   - Processes event TSV files
   - Creates timing files for each condition:
     - phasic1 (FearCue, NeutralCue)
     - phasic2 (FearImage, NeutralImage)
     - sustained (UnknownCue, UnknownFear, UnknownNeutral)

4. **AFNI Processing**
   - Generates AFNI processing scripts
   - Includes motion parameters
   - Sets up GLM analysis with **8 parallel jobs**
   - **Multi-threaded execution with environment optimization**

5. **Contrast Calculation**
   - **Automated calculation of 10 standard contrasts**
   - **Fixed 3dcalc syntax for reliable contrast computation**
   - **JSON metadata tracking for all contrast files**

6. **Quality Control**
   - Generates mean and standard deviation maps
   - Creates QC visualizations

## Output Files

For each subject, the pipeline generates:
- **Preprocessed functional data** (despiked, motion-corrected, smoothed)
- **Timing files for each condition** (7 experimental conditions)
- **GLM results** (beta coefficients, t-statistics, fitted data, residuals)
- **Contrast maps** (10 standard contrasts with metadata)
- **AFNI processing script** (reproducible analysis commands)
- **QC visualizations** (motion parameters, mean/std maps)
- **Detailed processing logs** (comprehensive error tracking)

## Logging

- Console output with progress bars and color-coded status
- Detailed logs in the `logs` directory
- Error tracking and reporting

## Group Configuration

The pipeline uses a JSON file (`subject_groups.json`) to manage subject group assignments:

```json
{
  "groups": {
    "AUD": ["sub-ALC2158", "sub-ALC2118", ...],
    "HC": ["sub-ALC2161", ...]
  },
  "metadata": {
    "description": "Subject group assignments for AUD vs HC analysis",
    "total_subjects": 6,
    "aud_count": 5,
    "hc_count": 1
  }
}
```

You can edit this file directly or use the `manage_groups.py` utility script.

## ROI Analysis

The pipeline includes comprehensive ROI analysis using the Harvard-Oxford atlas:

### Available ROIs
- **Amygdala**: Key region for threat processing and fear responses
- **Insula**: Interoception and emotional processing  
- **Prefrontal Cortex**: Executive control and emotion regulation
- **Hippocampus**: Memory and contextual fear processing

### Analysis Features
- **Atlas-based masks**: Automatic download and preparation of Harvard-Oxford atlas masks
- **Hemisphere options**: Left, right, or bilateral analysis
- **Statistical testing**: Independent t-tests with effect size calculations (Cohen's d)
- **Group comparisons**: AUD vs HC comparisons within each ROI
- **Visualization**: Comprehensive plots showing group differences and statistical significance

### Example Usage
```bash
# Analyze amygdala responses
python roi_analysis.py --roi amygdala --hemisphere bilateral

# Analyze left insula
python roi_analysis.py --roi insula --hemisphere left

# Run complete analysis for all subjects
python roi_analysis.py --roi amygdala
```

### Output Files
- **ROI masks**: Atlas-based region masks in NIfTI format
- **Extracted values**: Mean activation values for each condition within ROI
- **Statistical results**: JSON files with t-tests, p-values, and effect sizes
- **Visualizations**: Publication-ready plots showing group differences

## Memory Management

The pipeline supports different processing modes to accommodate various system resources:

### Sequential Processing (Default - Memory Efficient)
- Processes one subject at a time
- Automatically clears memory after each subject
- Recommended for systems with limited RAM
- Default behavior when no flags are specified

### Parallel Processing (High Memory Usage)
- Processes multiple subjects simultaneously
- Can significantly speed up processing on high-end systems
- Requires substantial RAM (8GB+ recommended per thread)
- Use with caution on memory-constrained systems

### Memory Usage Guidelines
- **Sequential mode**: ~2-4GB RAM total
- **Parallel mode**: ~4-8GB RAM per thread
- **Large datasets**: Consider processing subjects in batches

### Processing Time Expectations
- **Motion correction**: ~5-15 minutes per run (single-threaded, AFNI limitation)
- **Other preprocessing steps**: ~1-3 minutes per run (multi-threaded)
- **GLM analysis**: ~10-30 minutes per subject (8 parallel jobs)
- **Contrast calculation**: ~1-2 minutes per subject (10 contrasts)
- **Total per subject**: ~30-60 minutes depending on data size

### Performance Optimization
- **OpenMP threading**: Automatically uses 8 threads for AFNI commands
- **AFNI CPU clock**: Enabled for optimal multi-threading performance
- **OpenCL acceleration**: Automatically enabled if available (speeds up many AFNI operations)
- **Multi-threaded steps**: Despiking, spatial smoothing, and GLM analysis benefit from threading
- **Single-threaded steps**: Motion correction (3dvolreg) runs on single core (AFNI limitation)
- **High-quality motion correction**: Uses Fourier interpolation and two-pass processing

### Data Management
- **Automatic cleanup**: Removes existing processed data before starting (ensures fresh analysis)
- **Log preservation**: Processing logs are preserved across runs
- **Skip cleanup**: Use `--no-cleanup` to preserve existing processed data

## Quality Control

QC outputs include:
- Mean activation maps
- Standard deviation maps
- Motion parameter plots
- Processing status reports

## Troubleshooting

### **Contrast Calculation Errors**
If you see errors like "Can't interpret symbol 'A-B'" in contrast calculations:
- **Cause**: Incorrect 3dcalc expression syntax (fixed in latest version)
- **Solution**: The pipeline now uses correct expression format without extra quotes
- **Verification**: Check that contrast files are created in `processed_data/contrasts/[subject]/`

### **Memory Issues**
- **Sequential processing**: Use `--sequential` flag for memory-efficient processing
- **Parallel processing**: Use `--parallel` only on systems with sufficient RAM (8GB+ per thread)
- **Monitor usage**: Check system resources during processing

### **AFNI Command Failures**
- **Despiking errors**: Ensure AFNI is properly installed and in PATH
- **Motion correction slowness**: This is normal - 3dvolreg is single-threaded by design
- **GLM convergence**: Check design matrix visualization (`X.jpg` files) for issues

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. # HRF-Deconvolution
