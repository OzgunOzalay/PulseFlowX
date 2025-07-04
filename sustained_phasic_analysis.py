#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path
import subprocess
import logging
import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

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
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        return logger
    
    def _load_group_assignments(self):
        """Load subject group assignments from JSON file."""
        group_file = Path("subject_groups.json")
        
        if not group_file.exists():
            self.logger.warning(f"Group assignment file {group_file} not found. Using default assignments.")
            return {
                'AUD': ['sub-ALC2158'],
                'HC': ['sub-ALC2161']
            }
        
        try:
            with open(group_file, 'r') as f:
                data = json.load(f)
            
            groups = data.get('groups', {})
            
            # Validate that we have both AUD and HC groups
            if 'AUD' not in groups or 'HC' not in groups:
                raise ValueError("Group file must contain both 'AUD' and 'HC' groups")
            
            # Log group information
            aud_count = len(groups['AUD'])
            hc_count = len(groups['HC'])
            self.logger.info(f"Loaded group assignments: {aud_count} AUD subjects, {hc_count} HC subjects")
            
            return groups
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error loading group assignments: {e}")
            self.logger.warning("Using default group assignments")
            return {
                'AUD': ['sub-ALC2158'],
                'HC': ['sub-ALC2161']
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
                cmd = f"3dcalc -a '{glm_file}[{subbrick}]' -expr 'a' -prefix {temp_file}"
                
                try:
                    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
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
                
                # Create a script to combine the TENT coefficients
                script_content = f"""#!/bin/bash
# Combine TENT coefficients for {condition} {response_type} response
3dTcat -prefix {combined_file} {' '.join(tent_coefficients)}
"""
                
                script_file = output_dir / f"combine_{condition}_{response_type}.sh"
                with open(script_file, 'w') as f:
                    f.write(script_content)
                
                # Make script executable and run it
                os.chmod(script_file, 0o755)
                try:
                    result = subprocess.run(str(script_file), shell=True, check=True, capture_output=True, text=True)
                    
                    # Calculate mean across all TENT functions to get a single response value
                    mean_file = output_dir / f"{subject_id}_{condition}_{response_type}_mean"
                    mean_cmd = f"3dTstat -mean -prefix {mean_file} '{combined_file}+orig'"
                    
                    try:
                        subprocess.run(mean_cmd, shell=True, check=True, capture_output=True, text=True)
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
                
                # Build 3dcalc command
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
            "-prefix", str(output_prefix),
            "-Clustsim"  # Add cluster simulation for multiple comparisons
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
        
        # Run group analyses
        group_results = self.run_group_analyses()
        
        self.logger.info("✓ Completed full analysis pipeline")
        return group_results

def main():
    parser = argparse.ArgumentParser(description="Sustained vs Phasic Response Analysis")
    parser.add_argument("--output_dir", default="processed_data", help="Output directory")
    parser.add_argument("--subject", help="Process specific subject only")
    parser.add_argument("--group_only", action="store_true", help="Run group analysis only")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SustainedPhasicAnalyzer(args.output_dir)
    
    if args.subject:
        # Process single subject
        analyzer.analyze_subject(args.subject)
    elif args.group_only:
        # Run group analysis only
        analyzer.run_group_analyses()
    else:
        # Run complete analysis
        analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 