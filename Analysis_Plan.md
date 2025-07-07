# fMRI Threat Processing Analysis Plan
**HRF Deconvolution Project - AUD vs HC Group Comparison**

---

## 📋 **Project Overview**

### **Research Question**
Compare patterns of threat processing between Alcohol Use Disorder (AUD) patients and Healthy Controls (HC) using fMRI data, focusing on fear vs neutral responses within different ROIs.

### **Experimental Design**
- **Task**: Unpredictable Threat Task
- **Groups**: AUD vs HC (target: 160 subjects total)
- **Current Test Data**: 2 subjects (sub-ALC2158, sub-ALC2161)
- **Runs**: 4 runs per subject (originally 5, processing first 4)
- **Conditions**: 7 experimental conditions
  - `FearCue`, `NeutralCue` (phasic threat - cues)
  - `FearImage`, `NeutralImage` (phasic threat - images)  
  - `UnknownCue`, `UnknownFear`, `UnknownNeutral` (sustained threat)

---

## 🎯 **Analysis Goals**

1. **Within-subject contrasts**: Fear vs Neutral processing
2. **Between-group comparisons**: AUD vs HC differences in threat processing
3. **ROI analysis**: Extract activation patterns from specific brain regions
4. **Pattern analysis**: Compare activation patterns across conditions and groups

---

## 🛠️ **Current Pipeline Status**

### **✅ Completed Components**

#### **1. Main Processing Pipeline** (`process_fmri.py`)
- **Preprocessing**: Slice timing, despiking, motion correction, spatial smoothing
- **GLM Analysis**: 3dDeconvolve with separate regressors for each condition (8 parallel jobs)
- **Parallel Processing**: Optimized for multi-core execution with environment variables
- **Automated Contrast Calculation**: Integrated contrast computation with **fixed 3dcalc syntax**
- **Quality Control**: Motion parameters and QC images
- **Memory Management**: Sequential/parallel processing options with automatic cleanup

#### **2. Contrast Calculator** (Integrated in `process_fmri.py`)
- **10 Standard Contrasts** predefined for experimental design
- **Fixed 3dcalc Syntax** - resolved expression parsing errors
- **Custom Contrast Creation** capability
- **Batch Processing** for multiple subjects
- **JSON Metadata Tracking** for all contrast files

#### **3. Directory Structure**
```
processed_data/
├── raw/                    # Copied original data
├── preprocessed/          # Intermediate preprocessing files
├── glm_results/          # Statistical analysis outputs
├── contrasts/            # Contrast maps for each subject
│   └── [subject]/
│       ├── [contrast_name]+orig.*  # Individual contrast files
│       └── [subject]_contrasts.json # Contrast metadata
├── timing_files/         # Experimental condition timing
├── qc/                   # Quality control outputs
├── scripts/              # Generated AFNI scripts
└── logs/                 # Processing logs
```

---

## 📊 **Standard Contrasts Available**

### **Fear vs Neutral Comparisons**
1. **`FearCue_vs_NeutralCue`** - Fear cues > Neutral cues
2. **`FearImage_vs_NeutralImage`** - Fear images > Neutral images
3. **`Fear_vs_Neutral`** - Overall Fear > Neutral

### **Phasic vs Sustained Threat**
4. **`Phasic_vs_Sustained`** - Predictable > Unpredictable threat
5. **`Sustained_vs_Phasic`** - Unpredictable > Predictable threat

### **Context-Specific Contrasts**
6. **`UnknownFear_vs_UnknownNeutral`** - Unknown fear > Unknown neutral
7. **`Cues_vs_Images`** - Cue processing > Image processing
8. **`Images_vs_Cues`** - Image processing > Cue processing

### **Threat Sensitivity**
9. **`Cue_ThreatSensitivity`** - (FearCue-NeutralCue) > (FearImage-NeutralImage)
10. **`Image_ThreatSensitivity`** - (FearImage-NeutralImage) > (FearCue-NeutralCue)

---

## 🗂️ **Output Files Reference**

### **GLM Results** (`glm_results/[subject]/`)
- **`[subject]_glm+orig.*`** - Main statistical bucket with beta coefficients and t-statistics
- **`[subject]_fitts+orig.*`** - Model-fitted data
- **`[subject]_errts+orig.*`** - Residual errors
- **`[subject]_X.xmat.1D`** - Design matrix
- **`[subject]_X.jpg`** - Visual design matrix
- **`[subject]_*_combined.1D`** - Combined timing files per condition

### **Contrast Maps** (`contrasts/[subject]/`)
- **`[subject]_[contrast_name]+orig.*`** - Individual contrast maps
- **`[subject]_contrasts.json`** - Contrast metadata and file paths

### **Preprocessed Data** (`preprocessed/[subject]/`)
- **`*_smoothed.nii`** - Final preprocessed functional data
- **`motion_run*.1D`** - Motion parameters (6 parameters per run)

### **Quality Control** (`qc/[subject]/`)
- **`mean_run*.nii`** - Mean intensity images per run
- **`std_run*.nii`** - Standard deviation images per run

---

## 💻 **Usage Instructions**

### **Run Full Pipeline**
```bash
python process_fmri.py
```

### **Calculate Contrasts Only** (for existing GLM results)
```bash
# Contrasts are automatically calculated during pipeline execution
# To recalculate contrasts for existing GLM results:
python3 -c "
from process_fmri import ContrastCalculator
import logging
logger = logging.getLogger('test')
calc = ContrastCalculator('processed_data', logger)
result = calc.calculate_all_contrasts('sub-ALC2158', 'processed_data/glm_results/sub-ALC2158/sub-ALC2158_glm+orig')
print(f'Completed {len(result)} contrasts')
"
```

### **Create Custom Contrasts**
```python
from process_fmri import ContrastCalculator
import logging

logger = logging.getLogger('test')
calculator = ContrastCalculator("processed_data", logger)
calculator.create_custom_contrast(
    subject_id="sub-ALC2158",
    glm_file="processed_data/glm_results/sub-ALC2158/sub-ALC2158_glm+orig",
    contrast_name="MyCustomContrast",
    formula="a-b",
    conditions=["FearCue", "NeutralCue"],
    description="Custom contrast description"
)
```

---

## 🔍 **Current Test Results**

### **Subjects Processed**: sub-ALC2158, sub-ALC2161
### **Available Data**:
- ✅ Preprocessed functional data (4 runs each)
- ✅ GLM statistical maps (7 conditions)
- ✅ **10 contrast maps per subject** (all working correctly)
- ✅ Motion parameters and QC metrics
- ✅ **Contrast metadata in JSON format**

### **Key Files to Examine**:
- `processed_data/glm_results/sub-ALC2158/sub-ALC2158_glm+orig.*`
- `processed_data/contrasts/sub-ALC2158/sub-ALC2158_Fear_vs_Neutral+orig.*`
- `processed_data/contrasts/batch_contrast_summary.json`

---

## 🚀 **Next Steps**

### **Immediate Tasks**
1. **Quality Assessment**
   - [ ] Check motion parameters (should be <3mm translation, <3° rotation)
   - [ ] Examine design matrix visualization (`X.jpg` files)
   - [ ] Review contrast maps for expected activation patterns

2. **ROI Analysis**
   - [ ] Define ROI masks (amygdala, insula, prefrontal cortex, etc.)
   - [ ] Extract mean activation values from each ROI for each contrast
   - [ ] Create ROI extraction script

3. **Group-Level Analysis**
   - [ ] Implement group-level statistical testing
   - [ ] Create AUD vs HC comparison scripts
   - [ ] Generate group activation maps

### **Medium-Term Goals**
1. **Expand to Full Dataset**
   - [ ] Process all 160 subjects
   - [ ] Implement batch processing for large datasets
   - [ ] Add automated quality control checks

2. **Advanced Analyses**
   - [ ] Multivariate pattern analysis (MVPA)
   - [ ] Connectivity analysis
   - [ ] Machine learning classification

3. **Visualization & Reporting**
   - [ ] Create automated reporting pipeline
   - [ ] Generate publication-ready figures
   - [ ] Statistical summary tables

---

## 🔧 **Technical Notes**

### **Processing Parameters**
- **Spatial Smoothing**: 4mm FWHM
- **Motion Correction**: 3dvolreg with Fourier interpolation
- **GLM Basis Function**: GAM (Gamma function)
- **Polynomial Detrending**: 3rd order
- **Parallel Processing**: CPU cores - 1

### **Design Improvements Made**
1. **Fixed GLM Design**: Changed from 3 combined regressors to 7 separate regressors
2. **Added Parallel Processing**: Speeds up motion correction and preprocessing
3. **Automated Contrast Calculation**: Reduces manual work and errors
4. **Comprehensive Logging**: Detailed processing logs for troubleshooting

### **Known Issues & Solutions**
- **Progress Bar Conflicts**: Resolved by using simplified logging in parallel processes
- **Pickling Errors**: Fixed by using standalone functions for multiprocessing
- **Combined Conditions**: Resolved by creating separate regressors for proper contrasts
- **Contrast Calculation Errors**: **FIXED** - Resolved 3dcalc expression syntax issues
- **Memory Management**: Added sequential/parallel processing options with automatic cleanup

---

## 📚 **Dependencies**

### **Required Software**
- **AFNI**: 3dDeconvolve, 3dcalc, 3dTcat, etc.
- **Python 3.8+**
- **Required Python packages**: `pandas`, `numpy`, `rich`, `pathlib`

### **File Structure Requirements**
```
Data/
├── sub-[ID]/
│   ├── anat/
│   │   └── sub-[ID]_T1w.nii
│   └── func/
│       ├── sub-[ID]_task-unpredictablethreat_run-[1-4]_bold.nii
│       └── sub-[ID]_task-unpredictablethreat_run-[1-4]_events.tsv
```

---

## 📈 **Success Metrics**

### **Pipeline Validation**
- [ ] All preprocessing steps complete without errors
- [ ] GLM converges successfully for all subjects
- [ ] Contrast maps show expected activation patterns
- [ ] Motion parameters within acceptable limits

### **Analysis Validation**
- [ ] Fear > Neutral contrasts show threat-related brain activation
- [ ] Group differences emerge in predicted regions
- [ ] Results are reproducible across processing runs

---

## 🔄 **Version Control**

**Current Repository**: `https://github.com/OzgunOzalay/HRF-Deconvolution.git`

**Key Commits**:
- Latest: **Fixed contrast calculation syntax** - resolved 3dcalc expression errors
- Previous: Comprehensive contrast calculation tools
- Previous: Improved GLM design with separate regressors
- Previous: Parallelized preprocessing pipeline

**Backup Strategy**: All code and configurations are version controlled and pushed to GitHub.

---

## 📞 **Contact & Notes**

**Project Status**: ✅ **Pipeline Fully Operational - All Components Working**

**Current Focus**: Moving from single-subject GLM analysis to group-level comparisons and ROI extraction.

**Latest Fix**: **Contrast calculation errors resolved** - all 10 standard contrasts now compute successfully.

**Next Session**: Start with ROI definition and extraction tools to enable AUD vs HC group comparisons.

---

*Last Updated: July 6, 2025*
*Pipeline Version: 1.1 - Fixed Contrast Calculation Syntax* 