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
- **Group Analysis** for AUD vs HC comparisons
- **Flexible Group Management** via JSON configuration
- **Automated Contrast Calculations** for threat processing conditions

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

plots/
├── hrf_analysis/          # HRF analysis visualizations
├── timeseries/            # Time series plots
└── sustained_phasic/      # Sustained/phasic analysis plots
```

## Requirements

- Python 3.6+
- AFNI tools
- Python packages (see requirements.txt):
  - pandas >= 1.3.0
  - numpy >= 1.20.0
  - rich >= 10.0.0

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
python process_fmri.py
```

4. Run the sustained/phasic analysis:
```bash
python sustained_phasic_analysis.py
```

5. Generate visualizations:
```bash
python plot_sustained_phasic_results.py
```

## Processing Steps

1. **Data Organization**
   - Copies raw data to processing directory
   - Creates necessary directory structure

2. **Preprocessing**
   - Slice timing correction (3dTcat)
   - Despiking (3dDespike)
   - Motion correction (3dvolreg)
   - Spatial smoothing (3dmerge)

3. **Timing File Creation**
   - Processes event TSV files
   - Creates timing files for each condition:
     - phasic1 (FearCue, NeutralCue)
     - phasic2 (FearImage, NeutralImage)
     - sustained (UnknownCue, UnknownFear, UnknownNeutral)

4. **AFNI Processing**
   - Generates AFNI processing scripts
   - Includes motion parameters
   - Sets up GLM analysis

5. **Quality Control**
   - Generates mean and standard deviation maps
   - Creates QC visualizations

## Output Files

For each subject, the pipeline generates:
- Preprocessed functional data
- Timing files for each condition
- AFNI processing script
- QC visualizations
- Detailed processing logs

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

## Quality Control

QC outputs include:
- Mean activation maps
- Standard deviation maps
- Motion parameter plots
- Processing status reports

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. # HRF-Deconvolution
