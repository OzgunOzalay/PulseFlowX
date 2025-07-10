#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
import subprocess
import logging
from rich.console import Console
from rich.logging import RichHandler


class SustainedPhasicAnalyzer:
    """Class for analyzing sustained vs phasic responses using TENT functions."""

    def __init__(self, output_dir, logger=None):
        self.output_dir = Path(output_dir)
        self.sustained_phasic_dir = self.output_dir / "sustained_phasic_analysis"
        self.sustained_phasic_dir.mkdir(exist_ok=True)
        self.logger = logger or self._setup_logger()

        # Define response types and their TENT ranges
        self.response_types = {
            'sustained': {
                'tent_range': list(range(11)),  # All 11 TENT functions (0-10)
                'time_window': '0-20s',
                'description': 'Full trial response (all 11 TENT functions)'
            },
            'phasic': {
                'tent_range': list(range(8)),   # First 8 TENT functions (0-7)
                'time_window': '0-14s',
                'description': 'Initial response (first 8 TENT functions)'
            }
        }

        # Define conditions
        self.conditions = [
            'FearCue', 'NeutralCue',
            'FearImage', 'NeutralImage',
            'UnknownCue', 'UnknownFear', 'UnknownNeutral'
        ]

        # Load group assignments from JSON file
        self.test_groups = self._load_group_assignments()

    def _setup_logger(self):
        """Setup logging for standalone use."""
        logger = logging.getLogger("SustainedPhasicAnalyzer")
        logger.setLevel(logging.INFO)

        console_handler = RichHandler(rich_tracebacks=True)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        return logger

    def _detect_available_subjects(self):
        """Detect available subjects from processed_data/glm_results directory."""
        glm_dir = self.output_dir / "glm_results"
        
        if not glm_dir.exists():
            self.logger.warning(f"GLM results directory not found: {glm_dir}")
            return []
        
        available_subjects = []
        for item in glm_dir.iterdir():
            if item.is_dir() and item.name.startswith('sub-'):
                # Check if GLM files exist for this subject
                glm_file = item / f"{item.name}_glm+orig"
                glm_head = glm_file.with_suffix('.HEAD')
                glm_brik = glm_file.with_suffix('.BRIK.gz')
                
                if glm_head.exists() and glm_brik.exists():
                    available_subjects.append(item.name)
                else:
                    self.logger.warning(f"GLM files missing for {item.name}")
        
        self.logger.info(f"Detected {len(available_subjects)} subjects with GLM files: {available_subjects}")
        return available_subjects

    def _load_group_assignments(self):
        """Load subject group assignments from JSON file."""
        group_file = Path("subject_groups.json")

        if not group_file.exists():
            self.logger.warning(f"Group assignment file {group_file} not found. Using default assignments.")
            return self._create_default_groups()

        try:
            with open(group_file, 'r') as f:
                data = json.load(f)

            groups = data.get('groups', {})

            # Validate that we have both AUD and HC groups
            if 'AUD' not in groups or 'HC' not in groups:
                raise ValueError("Group file must contain both 'AUD' and 'HC' groups")

            # Check if the subjects in the JSON file actually exist
            available_subjects = self._detect_available_subjects()
            
            if not available_subjects:
                self.logger.error("No subjects with GLM files found!")
                return self._create_default_groups()
            
            # Filter groups to only include available subjects
            filtered_groups = {}
            for group_name, subjects in groups.items():
                filtered_subjects = [s for s in subjects if s in available_subjects]
                if filtered_subjects:
                    filtered_groups[group_name] = filtered_subjects
                    self.logger.info(f"Group {group_name}: {len(filtered_subjects)} subjects available out of {len(subjects)}")
                else:
                    self.logger.warning(f"No subjects from group {group_name} found in available data")
            
            # If we don't have both groups, create default groups from available subjects
            if len(filtered_groups) < 2:
                self.logger.warning("Not enough groups found in JSON file, creating default groups from available subjects")
                return self._create_default_groups(available_subjects)
            
            # Log group information
            aud_count = len(filtered_groups['AUD'])
            hc_count = len(filtered_groups['HC'])
            self.logger.info(f"Using group assignments: {aud_count} AUD subjects, {hc_count} HC subjects")

            return filtered_groups

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error loading group assignments: {e}")
            self.logger.warning("Using default group assignments")
            return self._create_default_groups()

    def _create_default_groups(self, available_subjects=None):
        """Create default group assignments from available subjects."""
        if available_subjects is None:
            available_subjects = self._detect_available_subjects()
        
        if not available_subjects:
            self.logger.error("No subjects available for group assignment!")
            return {'AUD': [], 'HC': []}
        
        # Sort subjects to ensure consistent assignment
        available_subjects.sort()
        
        # Split subjects into two groups
        mid_point = len(available_subjects) // 2
        aud_subjects = available_subjects[:mid_point]
        hc_subjects = available_subjects[mid_point:]
        
        self.logger.info(f"Created default groups: {len(aud_subjects)} AUD, {len(hc_subjects)} HC")
        self.logger.info(f"AUD subjects: {aud_subjects}")
        self.logger.info(f"HC subjects: {hc_subjects}")
        
        return {
            'AUD': aud_subjects,
            'HC': hc_subjects
        }

    def extract_response_components(self, subject_id, glm_file, response_type):
        """Extract sustained or phasic response components from GLM results."""

        tent_range = self.response_types[response_type]['tent_range']
        self.logger.info(f"Extracting {response_type} response for {subject_id} using TENTs {tent_range}")

        # Create output directory
        output_dir = self.sustained_phasic_dir / subject_id / response_type
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for condition in self.conditions:
            # Extract TENT coefficients for this condition
            tent_coefficients = []

            for tent_idx in tent_range:
                subbrick = f'{condition}#{tent_idx}_Coef'

                # Use 3dcalc to extract the coefficient
                temp_file = output_dir / f"temp_{condition}_tent{tent_idx}"
                cmd = f"3dcalc -a '{glm_file}[{subbrick}]' -expr a -prefix {temp_file}"

                try:
                    subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                    tent_coefficients.append(str(temp_file) + "+orig")
                    self.logger.info(f"✓ Extracted {subbrick}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error extracting {subbrick}: {e}")
                    self.logger.error(f"Command: {cmd}")
                    self.logger.error(f"STDOUT: {e.stdout}")
                    self.logger.error(f"STDERR: {e.stderr}")
                    continue

            if tent_coefficients:
                # Combine TENT coefficients into a single file
                combined_file = output_dir / f"{subject_id}_{condition}_{response_type}_response"
                
                # Remove any existing combined files to avoid conflicts (before script creation)
                combined_head = combined_file.with_suffix('.HEAD')
                combined_brik = combined_file.with_suffix('.BRIK.gz')
                combined_head.unlink(missing_ok=True)
                combined_brik.unlink(missing_ok=True)
                if combined_file.exists():
                    combined_file.unlink()

                # Run 3dTcat directly instead of creating a script
                relative_tent_files = [Path(f).name for f in tent_coefficients]
                
                # Build the 3dTcat command
                cmd = ["3dTcat", "-prefix", combined_file.name] + relative_tent_files
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(output_dir))
                    if result.returncode != 0:
                        self.logger.error(f"3dTcat failed with return code {result.returncode}")
                        self.logger.error(f"STDOUT: {result.stdout}")
                        self.logger.error(f"STDERR: {result.stderr}")
                        continue

                    # Check if the combined file was created successfully
                    # 3dTcat creates files with +orig suffix
                    combined_head = Path(str(combined_file) + "+orig.HEAD")
                    combined_brik = Path(str(combined_file) + "+orig.BRIK.gz")
                    
                    if not (combined_head.exists() and combined_brik.exists()):
                        self.logger.error(f"Combined file not created for {condition}")
                        continue
                    
                    # Verify file integrity by checking file size (should be ~32MB for sustained, ~23MB for phasic)
                    expected_size = 32 if response_type == 'sustained' else 23  # MB
                    actual_size = combined_brik.stat().st_size / (1024 * 1024)
                    
                    if actual_size < expected_size * 0.8:  # Allow 20% tolerance
                        self.logger.warning(f"Combined file for {condition} seems corrupted (size: {actual_size:.1f}MB, expected: ~{expected_size}MB)")
                        # Remove corrupted file and skip
                        combined_head.unlink(missing_ok=True)
                        combined_brik.unlink(missing_ok=True)
                        continue

                    # Calculate mean across all TENT functions to get a single response value
                    mean_file = output_dir / f"{subject_id}_{condition}_{response_type}_mean"
                    mean_cmd = f"3dTstat -mean -prefix {mean_file.name} '{combined_file.name}+orig'"

                    try:
                        subprocess.run(mean_cmd, shell=True, check=True, capture_output=True, text=True, cwd=str(output_dir))
                        results[condition] = str(mean_file) + "+orig"
                        self.logger.info(f"✓ Created {response_type} response for {condition}")
                    except subprocess.CalledProcessError as e:
                        self.logger.error(f"Error calculating mean for {condition}: {e}")
                        continue

                    # Clean up temporary files only after successful combination
                    for temp_file in tent_coefficients:
                        temp_path = Path(temp_file.replace("+orig", ""))
                        # Remove the .HEAD file
                        head_path = temp_path.with_suffix('.HEAD')
                        if head_path.exists():
                            head_path.unlink()
                        # Remove the .BRIK.gz file
                        brik_path = temp_path.with_suffix('.BRIK.gz')
                        if brik_path.exists():
                            brik_path.unlink()

                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error combining TENT coefficients for {condition}: {e}")
                    self.logger.error(f"Script: {script_file}")
                    self.logger.error(f"STDOUT: {e.stdout}")
                    self.logger.error(f"STDERR: {e.stderr}")

                    # Clean up temporary files even on error
                    for temp_file in tent_coefficients:
                        temp_path = Path(temp_file.replace("+orig", ""))
                        # Remove the .HEAD file
                        head_path = temp_path.with_suffix('.HEAD')
                        if head_path.exists():
                            head_path.unlink()
                        # Remove the .BRIK.gz file
                        brik_path = temp_path.with_suffix('.BRIK.gz')
                        if brik_path.exists():
                            brik_path.unlink()

        return results

    def calculate_response_contrasts(self, subject_id, response_type, response_files):
        """Calculate contrasts for a specific response type."""

        self.logger.info(f"Calculating contrasts for {response_type} responses in {subject_id}")
        
        output_dir = self.sustained_phasic_dir / subject_id / response_type
        contrast_dir = output_dir / "contrasts"
        contrast_dir.mkdir(exist_ok=True)
        
        # Define contrasts for this response type
        contrasts = {
            f'{response_type}_Fear_vs_Neutral': {
                'formula': '(a+b)-(c+d)',
                'conditions': ['FearCue', 'FearImage', 'NeutralCue', 'NeutralImage'],
                'description': f'{response_type.capitalize()} Fear > Neutral response'
            },
            f'{response_type}_Phasic_vs_Sustained_Threat': {
                'formula': '(a+b+c+d)-(e+f+g)',
                'conditions': ['FearCue', 'NeutralCue', 'FearImage', 'NeutralImage', 'UnknownCue', 'UnknownFear', 'UnknownNeutral'],
                'description': f'{response_type.capitalize()} phasic > sustained threat response'
            },
            f'{response_type}_UnknownFear_vs_UnknownNeutral': {
                'formula': 'a-b',
                'conditions': ['UnknownFear', 'UnknownNeutral'],
                'description': f'{response_type.capitalize()} unknown fear > unknown neutral'
            }
        }
        
        contrast_results = {}
        
        for contrast_name, contrast_info in contrasts.items():
            # Check if all required conditions are available
            required_conditions = contrast_info['conditions']
            available_files = []
            
            for condition in required_conditions:
                if condition in response_files:
                    available_files.append(response_files[condition])
                else:
                    self.logger.warning(f"Missing {condition} for contrast {contrast_name}")
                    break
            else:
                # All conditions available, calculate contrast
                output_file = contrast_dir / f"{subject_id}_{contrast_name}"
                
                # Build 3dcalc command with absolute paths
                input_args = []
                for i, condition in enumerate(required_conditions):
                    var = chr(ord('a') + i)
                    input_args.extend([f"-{var}", response_files[condition]])
                
                cmd = [
                    "3dcalc",
                    *input_args,
                    "-expr", contrast_info['formula'],
                    "-prefix", str(output_file)
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    contrast_results[contrast_name] = {
                        'file': str(output_file) + "+orig",
                        'description': contrast_info['description']
                    }
                    self.logger.info(f"✓ Calculated contrast: {contrast_name}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error calculating contrast {contrast_name}: {e}")
        
        return contrast_results
    
    def run_group_analysis(self, response_type, contrast_name):
        """Run group-level analysis for a specific response type and contrast."""
        
        self.logger.info(f"Running group analysis for {response_type} {contrast_name}")
        
        # Create output directory
        group_output_dir = self.sustained_phasic_dir / "group_analysis" / response_type
        group_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect contrast files for each group
        aud_files = []
        hc_files = []
        
        for subject in self.test_groups['AUD']:
            contrast_file = self.sustained_phasic_dir / subject / response_type / "contrasts" / f"{subject}_{contrast_name}+orig"
            contrast_head = contrast_file.with_suffix('.HEAD')
            contrast_brik = contrast_file.with_suffix('.BRIK.gz')
            if contrast_head.exists() and contrast_brik.exists():
                aud_files.append(str(contrast_file))
            else:
                self.logger.warning(f"Contrast file not found for AUD subject {subject}: {contrast_file}")
        
        for subject in self.test_groups['HC']:
            contrast_file = self.sustained_phasic_dir / subject / response_type / "contrasts" / f"{subject}_{contrast_name}+orig"
            contrast_head = contrast_file.with_suffix('.HEAD')
            contrast_brik = contrast_file.with_suffix('.BRIK.gz')
            if contrast_head.exists() and contrast_brik.exists():
                hc_files.append(str(contrast_file))
            else:
                self.logger.warning(f"Contrast file not found for HC subject {subject}: {contrast_file}")
        
        if not aud_files or not hc_files:
            self.logger.error(f"No valid contrast files found for {contrast_name}")
            return None
        
        # Run 3dttest++ for group comparison
        output_prefix = group_output_dir / f"{contrast_name}_AUD_vs_HC"
        
        cmd = [
            "3dttest++",
            "-setA", *aud_files,
            "-setB", *hc_files,
            "-prefix", str(output_prefix)
            # Note: -Clustsim requires >=4 subjects per group, we have 3 per group
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            self.logger.info(f"✓ Group comparison completed: {output_prefix}+orig")
            return str(output_prefix) + "+orig"
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error in group comparison: {e}")
            return None
    
    def analyze_subject(self, subject_id):
        """Complete analysis for a single subject."""
        
        self.logger.info(f"Starting sustained/phasic analysis for {subject_id}")
        
        # Check if GLM results exist
        glm_file = self.output_dir / "glm_results" / subject_id / f"{subject_id}_glm+orig"
        glm_head = glm_file.with_suffix('.HEAD')
        glm_brik = glm_file.with_suffix('.BRIK.gz')
        
        if not (glm_head.exists() and glm_brik.exists()):
            self.logger.error(f"GLM files not found: {glm_file}")
            self.logger.error(f"  HEAD: {glm_head} - {'exists' if glm_head.exists() else 'missing'}")
            self.logger.error(f"  BRIK: {glm_brik} - {'exists' if glm_brik.exists() else 'missing'}")
            return None
        
        subject_results = {}
        
        # Analyze each response type
        for response_type in self.response_types.keys():
            self.logger.info(f"Processing {response_type} response for {subject_id}")
            
            # Extract response components
            response_files = self.extract_response_components(subject_id, glm_file, response_type)
            
            if response_files:
                # Calculate contrasts
                contrast_results = self.calculate_response_contrasts(subject_id, response_type, response_files)
                
                subject_results[response_type] = {
                    'response_files': response_files,
                    'contrasts': contrast_results
                }
        
        # Save results
        results_file = self.sustained_phasic_dir / subject_id / f"{subject_id}_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(subject_results, f, indent=2)
        
        self.logger.info(f"✓ Completed analysis for {subject_id}")
        return subject_results
    
    def run_group_analyses(self):
        """Run group-level analyses for all contrasts."""
        
        self.logger.info("Running group-level analyses")
        
        group_results = {}
        
        # Run group analysis for each response type and contrast
        for response_type in self.response_types.keys():
            group_results[response_type] = {}
            
            # Get contrast names from first subject
            first_subject = self.test_groups['AUD'][0]
            results_file = self.sustained_phasic_dir / first_subject / f"{first_subject}_analysis_results.json"
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    subject_results = json.load(f)
                
                if response_type in subject_results:
                    contrasts = subject_results[response_type]['contrasts']
                    
                    for contrast_name in contrasts.keys():
                        result = self.run_group_analysis(response_type, contrast_name)
                        if result:
                            group_results[response_type][contrast_name] = {
                                'file': result,
                                'description': contrasts[contrast_name]['description']
                            }
        
        # Save group results
        group_results_file = self.sustained_phasic_dir / "group_analysis" / "group_analysis_results.json"
        with open(group_results_file, 'w') as f:
            json.dump(group_results, f, indent=2)
        
        self.logger.info("✓ Completed group-level analyses")
        return group_results
    
    def run_complete_analysis(self):
        """Run complete sustained/phasic analysis pipeline."""
        
        self.logger.info("Starting complete sustained/phasic analysis pipeline")
        
        # Analyze each subject
        for group, subjects in self.test_groups.items():
            for subject in subjects:
                self.analyze_subject(subject)
        
        # Skip group analyses for now (can be run separately later)
        self.logger.info("✓ Completed individual subject analyses")
        self.logger.info("Note: Group analysis skipped - run with --group_only flag after verifying individual results")
        
        return None

    def convert_results_to_nifti(self):
        """Convert all AFNI group analysis results to NIfTI format for inspection."""
        
        self.logger.info("Converting AFNI group analysis results to NIfTI format")
        
        # Create NIfTI directory
        nifti_dir = self.sustained_phasic_dir / "nifti_results"
        nifti_dir.mkdir(exist_ok=True)
        
        # Get group analysis results
        group_results_file = self.sustained_phasic_dir / "group_analysis" / "group_analysis_results.json"
        
        if not group_results_file.exists():
            self.logger.error("Group analysis results not found. Run group analysis first.")
            return
        
        with open(group_results_file, 'r') as f:
            group_results = json.load(f)
        
        # Convert each result file to NIfTI
        converted_files = []
        for response_type, contrasts in group_results.items():
            for contrast_name, contrast_info in contrasts.items():
                nifti_file = self._convert_afni_to_nifti(
                    contrast_info['file'], 
                    contrast_name, 
                    response_type, 
                    nifti_dir
                )
                if nifti_file:
                    converted_files.append(nifti_file)
        
        self.logger.info(f"✓ Converted {len(converted_files)} files to NIfTI format")
        self.logger.info(f"✓ NIfTI files saved to: {nifti_dir}")
        self.logger.info("You can now inspect these files with FSLeyes or other NIfTI viewers")
        
        return converted_files



    def _convert_afni_to_nifti(self, afni_file, contrast_name, response_type, output_dir):
        """Convert AFNI file to NIfTI format, extracting the t-stat sub-brick."""
        
        # Clean up the contrast name for filename
        clean_contrast_name = self._clean_contrast_name(contrast_name)
        
        # Create meaningful filename
        nifti_filename = f"{clean_contrast_name}_{response_type}_AUD_vs_HC.nii.gz"
        nifti_file = output_dir / nifti_filename
        
        # Extract the t-statistic sub-brick (usually sub-brick 1) for meaningful brain maps
        cmd = f"3dAFNItoNIFTI -prefix {nifti_file} '{afni_file}[1]'"
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0 and nifti_file.exists():
                self.logger.info(f"✓ Converted to NIfTI (t-stat): {nifti_filename}")
                return nifti_file
            else:
                # Try with sub-brick 0 as fallback
                cmd_fallback = f"3dAFNItoNIFTI -prefix {nifti_file} '{afni_file}[0]'"
                result = subprocess.run(cmd_fallback, shell=True, capture_output=True, text=True)
                if result.returncode == 0 and nifti_file.exists():
                    self.logger.info(f"✓ Converted to NIfTI (fallback): {nifti_filename}")
                    return nifti_file
                else:
                    self.logger.warning(f"Failed to convert {afni_file} to NIfTI")
                    return None
        except Exception as e:
            self.logger.warning(f"Error converting {afni_file} to NIfTI: {e}")
            return None





    def _clean_contrast_name(self, contrast_name):
        """Clean contrast name for better filenames."""
        # Remove response type prefix
        clean_name = contrast_name.replace('sustained_', '').replace('phasic_', '')
        
        # Replace underscores with spaces and capitalize
        clean_name = clean_name.replace('_', ' ')
        
        # Make it more readable
        name_mapping = {
            'Fear vs Neutral': 'Fear_vs_Neutral',
            'Phasic vs Sustained Threat': 'Predictable_vs_Unknown',
            'UnknownFear vs UnknownNeutral': 'Unknown_Fear_vs_Neutral'
        }
        
        return name_mapping.get(clean_name, clean_name)



def main():
    parser = argparse.ArgumentParser(description="Sustained vs Phasic Response Analysis")
    parser.add_argument("--output_dir", default="processed_data", help="Output directory")
    parser.add_argument("--subject", help="Process specific subject only")
    parser.add_argument("--group_only", action="store_true", help="Run group analysis only")
    parser.add_argument("--convert_nifti", action="store_true", help="Convert AFNI results to NIfTI format")
    parser.add_argument("--nifti_only", action="store_true", help="Convert to NIfTI only (skip analysis)")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SustainedPhasicAnalyzer(args.output_dir)
    
    if args.nifti_only:
        # Convert to NIfTI only
        analyzer.convert_results_to_nifti()
    elif args.subject:
        # Process single subject
        analyzer.analyze_subject(args.subject)
        if args.convert_nifti:
            analyzer.convert_results_to_nifti()
    elif args.group_only:
        # Run group analysis only
        analyzer.run_group_analyses()
        if args.convert_nifti:
            analyzer.convert_results_to_nifti()
    else:
        # Run complete analysis
        analyzer.run_complete_analysis()
        if args.convert_nifti:
            analyzer.convert_results_to_nifti()

if __name__ == "__main__":
    main() 