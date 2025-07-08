#!/usr/bin/env python3
"""
ROI Analysis Module for fMRI Threat Processing
Integrates atlas-based ROI extraction and statistical testing for group comparisons.
"""

import json
import argparse
from pathlib import Path
import subprocess
import logging
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class ROIAnalyzer:
    """Class for ROI-based analysis using atlas masks."""

    def __init__(self, output_dir, logger=None):
        self.output_dir = Path(output_dir)
        self.roi_dir = self.output_dir / "roi_analysis"
        self.roi_dir.mkdir(exist_ok=True)
        self.logger = logger or self._setup_logger()

        # Define available ROIs and their atlas specifications
        self.available_rois = {
            'amygdala': {
                'atlas': 'CA_N27_ML',
                'subcortical': True,
                'hemispheres': ['left', 'right', 'bilateral'],
                'description': 'Amygdala - key region for threat processing and fear responses'
            },
            'insula': {
                'atlas': 'CA_N27_ML',
                'cortical': True,
                'hemispheres': ['left', 'right', 'bilateral'],
                'description': 'Insula - interoception and emotional processing'
            },
            'prefrontal_cortex': {
                'atlas': 'CA_N27_ML',
                'cortical': True,
                'hemispheres': ['left', 'right', 'bilateral'],
                'description': 'Prefrontal cortex - executive control and emotion regulation'
            },
            'hippocampus': {
                'atlas': 'CA_N27_ML',
                'subcortical': True,
                'hemispheres': ['left', 'right', 'bilateral'],
                'description': 'Hippocampus - memory and contextual fear processing'
            }
        }

        # Load group assignments
        self.groups = self._load_group_assignments()

        # Define conditions for analysis
        self.conditions = [
            'FearCue', 'NeutralCue',
            'FearImage', 'NeutralImage',
            'UnknownCue', 'UnknownFear', 'UnknownNeutral'
        ]

        self.response_types = ['sustained', 'phasic']

    def _setup_logger(self):
        """Setup logging for standalone use."""
        logger = logging.getLogger("ROIAnalyzer")
        logger.setLevel(logging.INFO)

        console_handler = RichHandler(rich_tracebacks=True)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        return logger

    def _load_group_assignments(self):
        """Load subject group assignments from JSON file."""
        group_file = Path("subject_groups.json")

        if not group_file.exists():
            self.logger.warning(f"Group assignment file {group_file} not found.")
            return {}

        try:
            with open(group_file, 'r') as f:
                data = json.load(f)
            return data.get('groups', {})
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error loading group assignments: {e}")
            return {}

    def download_atlas_masks(self, roi_name, hemisphere='bilateral'):
        """Download and prepare atlas masks for ROI analysis."""

        self.logger.info(f"Preparing atlas masks for {roi_name} ({hemisphere})")

        # Create atlas directory
        atlas_dir = self.roi_dir / "atlas_masks"
        atlas_dir.mkdir(exist_ok=True)

        # Define CA_N27_ML atlas specifications (available in AFNI installation)
        atlas_specs = {
            'amygdala': {
                'left': 'CA_N27_ML:left:Left_Amygdala',
                'right': 'CA_N27_ML:right:Right_Amygdala',
                'bilateral': 'bilateral_combine'  # Special flag for bilateral combination
            },
            'insula': {
                'left': 'CA_N27_ML:left:Left_Insula',
                'right': 'CA_N27_ML:right:Right_Insula',
                'bilateral': 'bilateral_combine'
            },
            'prefrontal_cortex': {
                'left': 'CA_N27_ML:left:Left_Frontal_Pole',
                'right': 'CA_N27_ML:right:Right_Frontal_Pole',
                'bilateral': 'bilateral_combine'
            },
            'hippocampus': {
                'left': 'CA_N27_ML:left:Left_Hippocampus',
                'right': 'CA_N27_ML:right:Right_Hippocampus',
                'bilateral': 'bilateral_combine'
            }
        }

        if roi_name not in atlas_specs:
            self.logger.error(f"ROI {roi_name} not found in atlas specifications")
            return None

        if hemisphere not in atlas_specs[roi_name]:
            self.logger.error(f"Hemisphere {hemisphere} not available for {roi_name}")
            return None

        atlas_label = atlas_specs[roi_name][hemisphere]
        output_file = atlas_dir / f"{roi_name}_{hemisphere}_mask_orig+orig"

        if atlas_label == 'bilateral_combine':
            # Create bilateral mask by combining left and right
            left_file = atlas_dir / f"{roi_name}_left_mask"
            right_file = atlas_dir / f"{roi_name}_right_mask"
            bilateral_tlrc = atlas_dir / f"{roi_name}_bilateral_mask_tlrc"
            
            # Create left and right masks first
            left_cmd = f"whereami -mask_atlas_region '{atlas_specs[roi_name]['left']}' -prefix {left_file}"
            right_cmd = f"whereami -mask_atlas_region '{atlas_specs[roi_name]['right']}' -prefix {right_file}"
            
            try:
                subprocess.run(left_cmd, shell=True, check=True, capture_output=True)
                subprocess.run(right_cmd, shell=True, check=True, capture_output=True)
                
                # Combine left and right masks
                combine_cmd = f"3dcalc -a {left_file}+tlrc -b {right_file}+tlrc -expr 'step(a)+step(b)' -prefix {bilateral_tlrc}"
                subprocess.run(combine_cmd, shell=True, check=True, capture_output=True)
                
                # Transform to ORIG space to match the data
                resample_cmd = f"3dresample -master processed_data/sustained_phasic_analysis/sub-ALC2158/sustained/sub-ALC2158_FearCue_sustained_mean+orig -input {bilateral_tlrc}+tlrc -prefix {output_file.with_suffix('')}"
                subprocess.run(resample_cmd, shell=True, check=True, capture_output=True)
                
                self.logger.info(f"✓ Created bilateral atlas mask in ORIG space: {output_file}")
                return output_file
                
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error creating bilateral atlas mask: {e}")
                return None
        else:
            # Create individual hemisphere mask
            tlrc_file = atlas_dir / f"{roi_name}_{hemisphere}_mask"
            cmd = f"whereami -mask_atlas_region '{atlas_label}' -prefix {tlrc_file}"
            
            try:
                subprocess.run(cmd, shell=True, check=True, capture_output=True)
                
                # Transform to ORIG space
                resample_cmd = f"3dresample -master processed_data/sustained_phasic_analysis/sub-ALC2158/sustained/sub-ALC2158_FearCue_sustained_mean+orig -input {tlrc_file}+tlrc -prefix {output_file.with_suffix('')}"
                subprocess.run(resample_cmd, shell=True, check=True, capture_output=True)
                
                self.logger.info(f"✓ Created atlas mask in ORIG space: {output_file}")
                return output_file
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error creating atlas mask: {e}")
                self.logger.error(f"Command: {cmd}")
                return None

    def load_subject_specific_mask(self, subject_id, roi_name, hemisphere):
        """Load subject-specific ROI mask in ORIG space."""
        
        # Look for subject-specific mask files (converted from NIFTI)
        # Priority: resampled masks first, then original masks
        mask_patterns = [
            f"{subject_id}_{roi_name}_{hemisphere}_mask_resampled+orig",  # Resampled version
            f"{subject_id}_{roi_name}_{hemisphere}_mask+orig",           # Original version
            f"{subject_id}_{roi_name}_{hemisphere}+orig",
            f"{roi_name}_{hemisphere}_{subject_id}+orig"
        ]
        
        # Check subject_masks directory
        search_dir = self.roi_dir / "subject_masks"
        
        if not search_dir.exists():
            self.logger.error(f"Subject masks directory not found: {search_dir}")
            return None
                
        for pattern in mask_patterns:
            mask_file = search_dir / pattern
            if mask_file.with_suffix('.HEAD').exists():
                if "_resampled" in pattern:
                    self.logger.info(f"✓ Found resampled subject-specific mask: {mask_file}")
                else:
                    self.logger.info(f"✓ Found subject-specific mask: {mask_file}")
                return mask_file
        
        # If not found, suggest running conversion and resampling
        self.logger.error(f"Subject-specific mask not found for {subject_id} - {roi_name} ({hemisphere})")
        self.logger.info(f"Expected file: {search_dir}/{subject_id}_{roi_name}_{hemisphere}_mask_resampled+orig.HEAD")
        self.logger.info("Run: python convert_nifti_to_afni.py then python resample_masks_to_func.py")
        return None

    def extract_roi_values(self, subject_id, data_file, roi_mask, roi_name, hemisphere):
        """Extract mean activation values from ROI."""

        self.logger.info(f"Extracting ROI values for {subject_id} from {roi_name} ({hemisphere})")

        # Create ROI extraction directory
        roi_extract_dir = self.roi_dir / subject_id / roi_name
        roi_extract_dir.mkdir(parents=True, exist_ok=True)

        # Use 3dmaskave to extract mean values from ROI
        output_file = roi_extract_dir / f"{subject_id}_{roi_name}_{hemisphere}_values.1D"

        cmd = f"3dmaskave -mask {roi_mask} -quiet {data_file} > {output_file}"

        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)

            # Read the extracted values (3dmaskave outputs a single mean value)
            mean_value = np.loadtxt(output_file)
            
            # Get number of voxels from mask info
            mask_info_cmd = f"3dmaskave -mask {roi_mask} -quiet {data_file}"
            mask_result = subprocess.run(mask_info_cmd, shell=True, capture_output=True, text=True)
            
            # Parse voxel count from output (e.g., "+++ 337 voxels survive the mask")
            n_voxels = 0
            for line in mask_result.stdout.split('\n'):
                if 'voxels survive the mask' in line:
                    n_voxels = int(line.split()[1])
                    break

            # Calculate summary statistics
            summary = {
                'mean': float(mean_value),
                'std': 0.0,  # 3dmaskave doesn't provide std, only mean
                'min': float(mean_value),
                'max': float(mean_value),
                'n_voxels': n_voxels,
                'file': str(output_file)
            }

            self.logger.info(f"✓ Extracted ROI values: mean={summary['mean']:.4f}, n_voxels={n_voxels}")
            return summary

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error extracting ROI values: {e}")
            return None

    def analyze_subject_roi(self, subject_id, roi_name='amygdala', hemisphere='bilateral'):
        """Complete ROI analysis for a single subject using subject-specific masks."""

        self.logger.info(f"Starting ROI analysis for {subject_id} - {roi_name} ({hemisphere})")

        # Load subject-specific mask (skip atlas mask creation)
        roi_mask = self.load_subject_specific_mask(subject_id, roi_name, hemisphere)
        
        if roi_mask is None:
            self.logger.error(f"Cannot proceed without subject-specific mask for {subject_id}")
            return None

        # Rest of analysis proceeds as normal...
        sustained_phasic_dir = self.output_dir / "sustained_phasic_analysis"
        subject_results = {}

        for response_type in self.response_types:
            subject_results[response_type] = {}

            # Extract values for each condition
            for condition in self.conditions:
                response_file = sustained_phasic_dir / subject_id / response_type / f"{subject_id}_{condition}_{response_type}_mean+orig"

                if response_file.with_suffix('.HEAD').exists():
                    roi_values = self.extract_roi_values(
                        subject_id, str(response_file), roi_mask, roi_name, hemisphere
                    )
                    if roi_values:
                        subject_results[response_type][condition] = roi_values
                else:
                    self.logger.warning(f"Response file not found: {response_file}")

            # Extract values for contrasts
            contrast_dir = sustained_phasic_dir / subject_id / response_type / "contrasts"
            if contrast_dir.exists():
                subject_results[response_type]['contrasts'] = {}

                for contrast_file in contrast_dir.glob(f"{subject_id}_{response_type}_*+orig.HEAD"):
                    contrast_name = contrast_file.stem.replace(f"{subject_id}_", "")
                    roi_values = self.extract_roi_values(
                        subject_id, str(contrast_file), roi_mask, roi_name, hemisphere
                    )
                    if roi_values:
                        subject_results[response_type]['contrasts'][contrast_name] = roi_values

        # Save results
        results_file = self.roi_dir / subject_id / f"{subject_id}_{roi_name}_{hemisphere}_roi_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(subject_results, f, indent=2)

        self.logger.info(f"✓ Completed ROI analysis for {subject_id}")
        return subject_results

    def run_group_statistics(self, roi_name='amygdala', hemisphere='bilateral'):
        """Run statistical tests for group comparisons within ROI."""

        self.logger.info(f"Running group statistics for {roi_name} ({hemisphere})")

        if not self.groups:
            self.logger.error("No group assignments found")
            return None

        # Collect data for each group
        group_data = {}
        all_results = {}

        for group_name, subjects in self.groups.items():
            group_data[group_name] = {}

            for response_type in self.response_types:
                group_data[group_name][response_type] = {}

                for condition in self.conditions:
                    condition_values = []

                    for subject in subjects:
                        results_file = self.roi_dir / subject / f"{subject}_{roi_name}_{hemisphere}_roi_results.json"

                        if results_file.exists():
                            with open(results_file, 'r') as f:
                                subject_results = json.load(f)

                            if (response_type in subject_results and
                                condition in subject_results[response_type]):
                                condition_values.append(
                                    subject_results[response_type][condition]['mean']
                                )

                    if condition_values:
                        group_data[group_name][response_type][condition] = condition_values

        # Run statistical tests
        statistical_results = {}

        for response_type in self.response_types:
            statistical_results[response_type] = {}

            for condition in self.conditions:
                aud_values = group_data.get('AUD', {}).get(response_type, {}).get(condition, [])
                hc_values = group_data.get('HC', {}).get(response_type, {}).get(condition, [])

                if aud_values and hc_values:
                    # Independent t-test
                    t_stat, p_value = stats.ttest_ind(aud_values, hc_values)

                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(aud_values) - 1) * np.var(aud_values, ddof=1) +
                                        (len(hc_values) - 1) * np.var(hc_values, ddof=1)) /
                                       (len(aud_values) + len(hc_values) - 2))
                    cohens_d = (np.mean(aud_values) - np.mean(hc_values)) / pooled_std

                    statistical_results[response_type][condition] = {
                        'aud_mean': float(np.mean(aud_values)),
                        'aud_std': float(np.std(aud_values)),
                        'hc_mean': float(np.mean(hc_values)),
                        'hc_std': float(np.std(hc_values)),
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'cohens_d': float(cohens_d),
                        'aud_n': len(aud_values),
                        'hc_n': len(hc_values),
                        'significant': bool(p_value < 0.05)
                    }

        # Save statistical results
        stats_file = self.roi_dir / f"{roi_name}_{hemisphere}_group_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(statistical_results, f, indent=2)

        self.logger.info(f"✓ Completed group statistics for {roi_name}")
        return statistical_results

    def create_roi_visualizations(self, roi_name='amygdala', hemisphere='bilateral'):
        """Create visualizations for ROI analysis results."""

        self.logger.info(f"Creating visualizations for {roi_name} ({hemisphere})")

        # Create plots directory
        plots_dir = self.roi_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Load statistical results
        stats_file = self.roi_dir / f"{roi_name}_{hemisphere}_group_statistics.json"
        if not stats_file.exists():
            self.logger.error(f"Statistical results not found: {stats_file}")
            return

        with open(stats_file, 'r') as f:
            stats_results = json.load(f)

        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'ROI Analysis: {roi_name.capitalize()} ({hemisphere}) - AUD vs HC', fontsize=16)

        # Plot 1: Sustained responses
        ax1 = axes[0, 0]
        sustained_data = []
        condition_names = []

        for condition in self.conditions:
            if condition in stats_results.get('sustained', {}):
                result = stats_results['sustained'][condition]
                sustained_data.append([result['aud_mean'], result['hc_mean']])
                condition_names.append(condition)

        if sustained_data:
            sustained_data = np.array(sustained_data)
            x = np.arange(len(condition_names))
            width = 0.35

            ax1.bar(x - width/2, sustained_data[:, 0], width, label='AUD', alpha=0.7, color='red')
            ax1.bar(x + width/2, sustained_data[:, 1], width, label='HC', alpha=0.7, color='blue')

            ax1.set_xlabel('Conditions')
            ax1.set_ylabel('Mean Response')
            ax1.set_title('Sustained Responses')
            ax1.set_xticks(x)
            ax1.set_xticklabels(condition_names, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Phasic responses
        ax2 = axes[0, 1]
        phasic_data = []

        for condition in self.conditions:
            if condition in stats_results.get('phasic', {}):
                result = stats_results['phasic'][condition]
                phasic_data.append([result['aud_mean'], result['hc_mean']])

        if phasic_data:
            phasic_data = np.array(phasic_data)
            x = np.arange(len(condition_names))

            ax2.bar(x - width/2, phasic_data[:, 0], width, label='AUD', alpha=0.7, color='red')
            ax2.bar(x + width/2, phasic_data[:, 1], width, label='HC', alpha=0.7, color='blue')

            ax2.set_xlabel('Conditions')
            ax2.set_ylabel('Mean Response')
            ax2.set_title('Phasic Responses')
            ax2.set_xticks(x)
            ax2.set_xticklabels(condition_names, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Plot 3: Statistical significance
        ax3 = axes[1, 0]
        p_values_sustained = []
        p_values_phasic = []

        for condition in self.conditions:
            if condition in stats_results.get('sustained', {}):
                p_values_sustained.append(stats_results['sustained'][condition]['p_value'])
            if condition in stats_results.get('phasic', {}):
                p_values_phasic.append(stats_results['phasic'][condition]['p_value'])

        if p_values_sustained and p_values_phasic:
            x = np.arange(len(condition_names))

            ax3.semilogy(x, p_values_sustained, 'o-', label='Sustained', color='red', alpha=0.7)
            ax3.semilogy(x, p_values_phasic, 's-', label='Phasic', color='blue', alpha=0.7)
            ax3.axhline(y=0.05, color='black', linestyle='--', alpha=0.5, label='p=0.05')

            ax3.set_xlabel('Conditions')
            ax3.set_ylabel('p-value (log scale)')
            ax3.set_title('Statistical Significance')
            ax3.set_xticks(x)
            ax3.set_xticklabels(condition_names, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: Effect sizes
        ax4 = axes[1, 1]
        effect_sizes_sustained = []
        effect_sizes_phasic = []

        for condition in self.conditions:
            if condition in stats_results.get('sustained', {}):
                effect_sizes_sustained.append(stats_results['sustained'][condition]['cohens_d'])
            if condition in stats_results.get('phasic', {}):
                effect_sizes_phasic.append(stats_results['phasic'][condition]['cohens_d'])

        if effect_sizes_sustained and effect_sizes_phasic:
            x = np.arange(len(condition_names))

            ax4.bar(x - width/2, effect_sizes_sustained, width, label='Sustained', alpha=0.7, color='red')
            ax4.bar(x + width/2, effect_sizes_phasic, width, label='Phasic', alpha=0.7, color='blue')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)

            ax4.set_xlabel('Conditions')
            ax4.set_ylabel("Cohen's d")
            ax4.set_title('Effect Sizes')
            ax4.set_xticks(x)
            ax4.set_xticklabels(condition_names, rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_file = plots_dir / f"{roi_name}_{hemisphere}_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ Created visualization: {plot_file}")

    def run_complete_roi_analysis(self, roi_name='amygdala', hemisphere='bilateral'):
        """Run complete ROI analysis pipeline."""

        self.logger.info(f"Starting complete ROI analysis for {roi_name} ({hemisphere})")

        # Analyze each subject
        for group, subjects in self.groups.items():
            for subject in subjects:
                self.analyze_subject_roi(subject, roi_name, hemisphere)

        # Run group statistics
        stats_results = self.run_group_statistics(roi_name, hemisphere)

        # Create visualizations
        self.create_roi_visualizations(roi_name, hemisphere)

        self.logger.info(f"✓ Completed ROI analysis for {roi_name}")
        return stats_results

def main():
    parser = argparse.ArgumentParser(description="ROI Analysis for fMRI Threat Processing")
    parser.add_argument("--output_dir", default="processed_data", help="Output directory")
    parser.add_argument("--roi", default="amygdala", choices=["amygdala", "insula", "prefrontal_cortex", "hippocampus"],
                       help="ROI to analyze")
    parser.add_argument("--hemisphere", default="bilateral", choices=["left", "right", "bilateral"],
                       help="Hemisphere to analyze")
    parser.add_argument("--subject", help="Process specific subject only")
    parser.add_argument("--group_only", action="store_true", help="Run group analysis only")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ROIAnalyzer(args.output_dir)

    if args.subject:
        # Process single subject
        analyzer.analyze_subject_roi(args.subject, args.roi, args.hemisphere)
    elif args.group_only:
        # Run group analysis only
        analyzer.run_group_statistics(args.roi, args.hemisphere)
        analyzer.create_roi_visualizations(args.roi, args.hemisphere)
    else:
        # Run complete analysis
        analyzer.run_complete_roi_analysis(args.roi, args.hemisphere)

if __name__ == "__main__":
    main() 