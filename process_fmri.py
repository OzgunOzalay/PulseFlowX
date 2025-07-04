#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import json
import shutil
from datetime import datetime
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.logging import RichHandler
import multiprocessing
from functools import partial
import queue
import threading
import sys

def process_single_run(args):
    """Process a single run of fMRI data (static function for multiprocessing)."""
    run, subject_id, raw_dir, preproc_dir, qc_dir = args
    
    # Setup logging for this process - simpler approach without Rich
    logger = logging.getLogger(f"FMRIProcessor_Run{run}")
    logger.setLevel(logging.INFO)
    
    # Create console handler without Rich to avoid conflicts
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(f'%(asctime)s - Run {run} - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    
    input_file = Path(raw_dir) / subject_id / "func" / f"{subject_id}_task-unpredictablethreat_run-{run}_bold.nii"
    if not input_file.exists():
        logger.warning(f"Input file not found: {input_file}")
        return
    
    logger.info(f"Starting preprocessing for run {run}")
    
    # 1. Slice timing correction
    slice_timing_file = Path(preproc_dir) / subject_id / f"{subject_id}_task-unpredictablethreat_run-{run}_tcat.nii"
    cmd = f"3dTcat -prefix {slice_timing_file} {input_file}"
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        logger.info(f"Slice timing correction completed for run {run}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in slice timing correction for run {run}: {e}")
        return
    
    # 2. Despiking
    despiked_file = Path(preproc_dir) / subject_id / f"{subject_id}_task-unpredictablethreat_run-{run}_despiked.nii"
    # Try NEW method first, fall back to standard if it fails
    cmd = f"3dDespike -NEW -prefix {despiked_file} {slice_timing_file}"
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        logger.info(f"Despiking completed for run {run}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"NEW despiking failed for run {run}, trying standard method: {e}")
        # Fall back to standard despiking
        cmd = f"3dDespike -prefix {despiked_file} {slice_timing_file}"
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            logger.info(f"Standard despiking completed for run {run}")
        except subprocess.CalledProcessError as e2:
            logger.error(f"Error in despiking for run {run}: {e2}")
            return
    
    # 3. Motion correction
    motion_file = Path(preproc_dir) / subject_id / f"{subject_id}_task-unpredictablethreat_run-{run}_motion.nii"
    cmd = f"3dvolreg -prefix {motion_file} -1Dfile {Path(preproc_dir)}/{subject_id}/motion_run{run}.1D -Fourier -twopass -zpad 4 {despiked_file}"
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        logger.info(f"Motion correction completed for run {run}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in motion correction for run {run}: {e}")
        return
    
    # 4. Spatial smoothing
    smoothed_file = Path(preproc_dir) / subject_id / f"{subject_id}_task-unpredictablethreat_run-{run}_smoothed.nii"
    cmd = f"3dmerge -1blur_fwhm 4.0 -doall -prefix {smoothed_file} {motion_file}"
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        logger.info(f"Spatial smoothing completed for run {run}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in spatial smoothing for run {run}: {e}")
        return
    
    # 5. Generate QC plots
    qc_file = Path(qc_dir) / subject_id / f"qc_run{run}.png"
    cmd = f"3dTstat -mean -prefix {Path(qc_dir)}/{subject_id}/mean_run{run}.nii {smoothed_file} && 3dTstat -stdev -prefix {Path(qc_dir)}/{subject_id}/std_run{run}.nii {smoothed_file}"
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        logger.info(f"QC plots generated for run {run}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating QC plots for run {run}: {e}")
    
    logger.info(f"Completed preprocessing for run {run}")

class ContrastCalculator:
    """Class for calculating and managing statistical contrasts."""
    
    def __init__(self, output_dir, logger):
        self.output_dir = Path(output_dir)
        self.contrast_dir = self.output_dir / "contrasts"
        self.contrast_dir.mkdir(exist_ok=True)
        self.logger = logger
        
        # Define standard contrasts for the experimental design
        self.standard_contrasts = {
            # Simple contrasts
            'FearCue_vs_NeutralCue': {
                'formula': 'a-b',
                'conditions': ['FearCue', 'NeutralCue'],
                'description': 'Fear cues > Neutral cues'
            },
            'FearImage_vs_NeutralImage': {
                'formula': 'a-b', 
                'conditions': ['FearImage', 'NeutralImage'],
                'description': 'Fear images > Neutral images'
            },
            'Fear_vs_Neutral': {
                'formula': 'a+b-c-d',
                'conditions': ['FearCue', 'FearImage', 'NeutralCue', 'NeutralImage'],
                'description': 'Overall Fear > Neutral'
            },
            
            # Phasic vs Sustained threat
            'Phasic_vs_Sustained': {
                'formula': 'a+b+c+d-e-f-g',
                'conditions': ['FearCue', 'NeutralCue', 'FearImage', 'NeutralImage', 'UnknownCue', 'UnknownFear', 'UnknownNeutral'],
                'description': 'Phasic threat > Sustained threat'
            },
            'Sustained_vs_Phasic': {
                'formula': 'e+f+g-a-b-c-d',
                'conditions': ['UnknownCue', 'UnknownFear', 'UnknownNeutral', 'FearCue', 'NeutralCue', 'FearImage', 'NeutralImage'],
                'description': 'Sustained threat > Phasic threat'
            },
            
            # Unknown context contrasts
            'UnknownFear_vs_UnknownNeutral': {
                'formula': 'a-b',
                'conditions': ['UnknownFear', 'UnknownNeutral'],
                'description': 'Unknown fear outcomes > Unknown neutral outcomes'
            },
            
            # Cue vs Image contrasts
            'Cues_vs_Images': {
                'formula': 'a+b-c-d',
                'conditions': ['FearCue', 'NeutralCue', 'FearImage', 'NeutralImage'], 
                'description': 'Cue processing > Image processing'
            },
            'Images_vs_Cues': {
                'formula': 'c+d-a-b',
                'conditions': ['FearImage', 'NeutralImage', 'FearCue', 'NeutralCue'],
                'description': 'Image processing > Cue processing'
            },
            
            # Threat sensitivity contrasts
            'Cue_ThreatSensitivity': {
                'formula': 'a-b-c+d',
                'conditions': ['FearCue', 'NeutralCue', 'FearImage', 'NeutralImage'],
                'description': 'Threat sensitivity for cues > images'
            },
            'Image_ThreatSensitivity': {
                'formula': 'c-d-a+b', 
                'conditions': ['FearImage', 'NeutralImage', 'FearCue', 'NeutralCue'],
                'description': 'Threat sensitivity for images > cues'
            }
        }

    def get_available_conditions(self, glm_file):
        """Extract available conditions from GLM bucket file."""
        try:
            # Run 3dinfo to get brick labels
            result = subprocess.run(
                f"3dinfo -label {glm_file}",
                shell=True, capture_output=True, text=True, check=True
            )
            
            # Parse condition names from brick labels
            labels = result.stdout.strip().split('|')
            conditions = []
            
            for label in labels:
                if '#0_Coef' in label:  # Beta coefficient bricks
                    condition = label.replace('#0_Coef', '')
                    conditions.append(condition)
            
            return conditions
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error getting conditions from {glm_file}: {e}")
            return []

    def calculate_contrast(self, subject_id, glm_file, contrast_name, contrast_info):
        """Calculate a single contrast for a subject."""
        
        # Get available conditions
        available_conditions = self.get_available_conditions(glm_file)
        required_conditions = contrast_info['conditions']
        
        # Check if all required conditions are available
        missing_conditions = [c for c in required_conditions if c not in available_conditions]
        if missing_conditions:
            self.logger.warning(f"Missing conditions for contrast '{contrast_name}': {missing_conditions}")
            return None
        
        # Create output directory for this subject's contrasts
        subject_contrast_dir = self.contrast_dir / subject_id
        subject_contrast_dir.mkdir(exist_ok=True)
        
        # Build 3dcalc command
        output_file = subject_contrast_dir / f"{subject_id}_{contrast_name}"
        
        # Map conditions to input variables (a, b, c, d, etc.)
        input_args = []
        variable_map = {}
        
        for i, condition in enumerate(required_conditions):
            var = chr(ord('a') + i)  # a, b, c, d, ...
            variable_map[condition] = var
            input_args.extend([f"-{var}", f"{glm_file}[{condition}#0_Coef]"])
        
        # Build the command
        cmd = [
            "3dcalc",
            *input_args,
            "-expr", f"'{contrast_info['formula']}'",
            "-prefix", str(output_file)
        ]
        
        # Execute command
        try:
            self.logger.info(f"Calculating contrast '{contrast_name}' for {subject_id}")
            subprocess.run(cmd, check=True, capture_output=True)
            self.logger.info(f"✓ Contrast '{contrast_name}' completed: {output_file}+orig")
            return f"{output_file}+orig"
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error calculating contrast '{contrast_name}': {e}")
            return None

    def calculate_all_contrasts(self, subject_id, glm_file):
        """Calculate all standard contrasts for a subject."""
        
        # Check for the actual AFNI file with .BRIK.gz extension
        glm_file_brik = f"{glm_file}.BRIK.gz"
        if not os.path.exists(glm_file_brik):
            self.logger.error(f"GLM file not found: {glm_file_brik}")
            return {}
        
        self.logger.info(f"Calculating contrasts for {subject_id}")
        
        contrast_results = {}
        
        for contrast_name, contrast_info in self.standard_contrasts.items():
            result = self.calculate_contrast(subject_id, glm_file, contrast_name, contrast_info)
            if result:
                contrast_results[contrast_name] = {
                    'file': result,
                    'description': contrast_info['description']
                }
        
        # Save contrast information
        contrast_info_file = self.contrast_dir / subject_id / f"{subject_id}_contrasts.json"
        with open(contrast_info_file, 'w') as f:
            json.dump(contrast_results, f, indent=2)
        
        self.logger.info(f"✓ Completed {len(contrast_results)} contrasts for {subject_id}")
        return contrast_results

    def create_custom_contrast(self, subject_id, glm_file, contrast_name, formula, conditions, description="Custom contrast"):
        """Create a custom contrast with user-defined formula."""
        
        custom_contrast_info = {
            'formula': formula,
            'conditions': conditions,
            'description': description
        }
        
        return self.calculate_contrast(subject_id, glm_file, contrast_name, custom_contrast_info)

    def list_available_contrasts(self):
        """List all available standard contrasts."""
        self.logger.info("Available standard contrasts:")
        for name, info in self.standard_contrasts.items():
            self.logger.info(f"  {name}: {info['description']}")
            self.logger.info(f"    Formula: {info['formula']}")
            self.logger.info(f"    Conditions: {', '.join(info['conditions'])}")
            self.logger.info("")

class FMRIProcessor:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.console = Console()
        
        # Setup logging
        self.logger = logging.getLogger("FMRIProcessor")
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        console_handler = RichHandler(rich_tracebacks=True)
        console_handler.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        
        # Create directory structure
        self._setup_directories()
        
        # Define condition mappings - separate regressor for each condition
        self.condition_mapping = {
            'FearCue': ['FearCue'],
            'NeutralCue': ['NeutralCue'], 
            'FearImage': ['FearImage'],
            'NeutralImage': ['NeutralImage'],
            'UnknownCue': ['UnknownCue'],
            'UnknownFear': ['UnknownFear'],
            'UnknownNeutral': ['UnknownNeutral']
        }
        
        # Set number of processes for parallel processing
        # Use fewer processes to avoid resource conflicts with AFNI
        self.n_processes = min(2, max(1, multiprocessing.cpu_count() // 2))  # Use at most 2 processes
        
        # Initialize contrast calculator
        self.contrast_calculator = ContrastCalculator(self.output_dir, self.logger)

    def _setup_directories(self):
        """Setup the directory structure for processing."""
        self.console.print("[bold blue]Setting up directory structure...[/bold blue]")
        
        # Main directories
        self.raw_dir = self.output_dir / "raw"
        self.preprocessed_dir = self.output_dir / "preprocessed"
        self.timing_dir = self.output_dir / "timing_files"
        self.scripts_dir = self.output_dir / "scripts"
        self.logs_dir = self.output_dir / "logs"
        self.qc_dir = self.output_dir / "qc"
        self.glm_dir = self.output_dir / "glm_results"
        
        # Create directories
        for directory in [self.raw_dir, self.preprocessed_dir, self.timing_dir, 
                         self.scripts_dir, self.logs_dir, self.qc_dir, self.glm_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Setup log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"processing_{timestamp}.log"
        
        # Add file handler to logger
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

    def _copy_raw_data(self, subject_id):
        """Copy raw data to processing directory."""
        self.logger.info(f"Copying raw data for subject {subject_id}")
        
        subject_raw_dir = self.raw_dir / subject_id
        subject_raw_dir.mkdir(exist_ok=True)
        
        # Copy anatomical data
        anat_src = self.data_dir / subject_id / "anat" / f"{subject_id}_T1w.nii"
        anat_dest = subject_raw_dir / "anat" / f"{subject_id}_T1w.nii"
        anat_dest.parent.mkdir(exist_ok=True)
        
        if anat_src.exists():
            shutil.copy2(anat_src, anat_dest)
            self.logger.info(f"Copied anatomical data to {anat_dest}")
        else:
            self.logger.warning(f"Anatomical data not found at {anat_src}")
        
        # Copy functional data (only first 4 runs)
        func_dest_dir = subject_raw_dir / "func"
        func_dest_dir.mkdir(exist_ok=True)
        
        for run in range(1, 5):  # Changed from range(1, 6) to range(1, 5)
            # Copy BOLD data
            bold_src = self.data_dir / subject_id / "func" / f"{subject_id}_task-unpredictablethreat_run-{run}_bold.nii"
            bold_dest = func_dest_dir / f"{subject_id}_task-unpredictablethreat_run-{run}_bold.nii"
            
            if bold_src.exists():
                shutil.copy2(bold_src, bold_dest)
                self.logger.info(f"Copied BOLD data for run {run} to {bold_dest}")
            else:
                self.logger.warning(f"BOLD data not found for run {run} at {bold_src}")
            
            # Copy event files
            events_src = self.data_dir / subject_id / "func" / f"{subject_id}_task-unpredictablethreat_run-{run}_events.tsv"
            events_dest = func_dest_dir / f"{subject_id}_task-unpredictablethreat_run-{run}_events.tsv"
            
            if events_src.exists():
                shutil.copy2(events_src, events_dest)
                self.logger.info(f"Copied events file for run {run} to {events_dest}")
            else:
                self.logger.warning(f"Events file not found for run {run} at {events_src}")

    def _preprocess_data(self, subject_id):
        """Run initial preprocessing steps."""
        self.logger.info(f"Starting preprocessing for subject {subject_id}")
        
        subject_preproc_dir = self.preprocessed_dir / subject_id
        subject_preproc_dir.mkdir(exist_ok=True)
        
        # Create QC directory for this subject
        subject_qc_dir = self.qc_dir / subject_id
        subject_qc_dir.mkdir(exist_ok=True)
        
        # Prepare arguments for parallel processing (only first 4 runs)
        args_list = [
            (run, subject_id, str(self.raw_dir), str(self.preprocessed_dir), str(self.qc_dir))
            for run in range(1, 5)  # Changed from range(1, 6) to range(1, 5)
        ]
        
        self.console.print(f"[cyan]Processing 4 runs in parallel for {subject_id}...[/cyan]")
        
        # Create a pool of workers
        with multiprocessing.Pool(processes=self.n_processes) as pool:
            # Process runs in parallel - simplified without Rich progress bar
            # Add small delay between processes to avoid conflicts
            import time
            results = []
            for args in args_list:
                results.append(pool.apply_async(process_single_run, (args,)))
                time.sleep(0.5)  # Small delay between processes
            
            # Wait for all processes to complete
            results = [r.get() for r in results]
        
        self.console.print(f"[green]✓[/green] Completed preprocessing for {subject_id}")

    def process_timing_files(self, subject_id):
        """Process timing files for a single subject."""
        self.logger.info(f"Processing timing files for subject {subject_id}")
        subject_dir = self.raw_dir / subject_id
        timing_files = {}
        
        # Process each run (only first 4 runs)
        for run in range(1, 5):  # Changed from range(1, 6) to range(1, 5)
            tsv_file = subject_dir / "func" / f"{subject_id}_task-unpredictablethreat_run-{run}_events.tsv"
            if not tsv_file.exists():
                self.logger.warning(f"Warning: {tsv_file} not found")
                continue
                
            # Read TSV file
            df = pd.read_csv(tsv_file, sep='\t')
            
            # Create timing files for each condition
            for condition, trial_types in self.condition_mapping.items():
                # Filter events for this condition
                mask = df['trial_type'].isin(trial_types)
                onsets = df[mask]['onset'].values
                
                # Create timing file
                timing_file = self.timing_dir / f"{subject_id}_run{run}_{condition}.txt"
                np.savetxt(timing_file, onsets, fmt='%.3f')
                
                if condition not in timing_files:
                    timing_files[condition] = []
                timing_files[condition].append(str(timing_file))
        
        self.logger.info(f"Created timing files for {len(timing_files)} conditions")
        return timing_files

    def run_3ddeconvolve(self, subject_id, timing_files):
        """Run 3dDeconvolve for GLM analysis."""
        self.logger.info(f"Running 3dDeconvolve for subject {subject_id}")
        
        # Setup output directory
        subject_glm_dir = self.glm_dir / subject_id
        subject_glm_dir.mkdir(exist_ok=True)
        
        # Get preprocessed functional data (only first 4 runs)
        func_files = []
        for run in range(1, 5):  # Changed from range(1, 6) to range(1, 5)
            func_file = self.preprocessed_dir / subject_id / f"{subject_id}_task-unpredictablethreat_run-{run}_smoothed.nii"
            if func_file.exists():
                func_files.append(str(func_file))
            else:
                self.logger.warning(f"Preprocessed file not found: {func_file}")
        
        if not func_files:
            self.logger.error(f"No preprocessed functional files found for {subject_id}")
            return None
        
        # Build 3dDeconvolve command
        output_prefix = subject_glm_dir / f"{subject_id}_glm"
        
        cmd = [
            "3dDeconvolve",
            "-input", " ".join(func_files),
            f"-polort 3",  # Polynomial detrending
            f"-num_stimts {len(timing_files)}",  # Number of stimulus files
        ]
        
        # Add stimulus timing files
        stim_index = 1
        for condition, files in timing_files.items():
            # Combine timing files across runs for each condition
            combined_timing = subject_glm_dir / f"{subject_id}_{condition}_combined.1D"
            
            # Create combined timing file
            with open(combined_timing, 'w') as f:
                for i, timing_file in enumerate(files):
                    if os.path.exists(timing_file):
                        with open(timing_file, 'r') as tf:
                            onsets = tf.read().strip()
                            if onsets:  # Only write if there are onsets
                                f.write(onsets + " ")
                    if i < len(files) - 1:
                        f.write("\n")
            
            cmd.extend([
                f"-stim_times {stim_index} {combined_timing} 'TENT(0,20,11)'",
                f"-stim_label {stim_index} {condition}"
            ])
            stim_index += 1
        
        # Add output options
        cmd.extend([
            f"-fout -tout -x1D {subject_glm_dir}/{subject_id}_X.xmat.1D",
            f"-xjpeg {subject_glm_dir}/{subject_id}_X.jpg",
            f"-x1D_uncensored {subject_glm_dir}/{subject_id}_X.nocensor.xmat.1D",
            f"-fitts {subject_glm_dir}/{subject_id}_fitts",
            f"-errts {subject_glm_dir}/{subject_id}_errts",
            f"-bucket {output_prefix}"
        ])
        
        # Execute command
        full_cmd = " ".join(cmd)
        self.logger.info(f"Running: {full_cmd}")
        
        try:
            # Run 3dDeconvolve with timeout and progress monitoring
            self.logger.info(f"Starting 3dDeconvolve for {subject_id} (this may take 15-30 minutes)...")
            
            # Use timeout to prevent infinite hanging
            result = subprocess.run(
                full_cmd, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            self.logger.info(f"3dDeconvolve completed successfully for {subject_id}")
            
            # Log some output info
            if result.stdout:
                self.logger.info(f"3dDeconvolve output: {result.stdout[:200]}...")
            
            # Return the path without +orig suffix since AFNI will add it
            return str(output_prefix)
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"3dDeconvolve timed out for {subject_id} after 1 hour")
            return None
        except subprocess.CalledProcessError as e:
            self.logger.error(f"3dDeconvolve failed for {subject_id}: {e}")
            if e.stderr:
                self.logger.error(f"Error details: {e.stderr}")
            return None

    def create_afni_proc_script(self, subject_id, timing_files):
        """Create AFNI processing script for a subject."""
        self.logger.info(f"Creating AFNI processing script for subject {subject_id}")
        script_path = self.scripts_dir / f"proc_{subject_id}.sh"
        
        # Get anatomical and functional data paths (only first 4 runs)
        anat_file = self.raw_dir / subject_id / "anat" / f"{subject_id}_T1w.nii"
        func_files = [str(self.preprocessed_dir / subject_id / f"{subject_id}_task-unpredictablethreat_run-{run}_smoothed.nii")
                     for run in range(1, 5)]  # Changed from range(1, 6) to range(1, 5)
        
        # Create AFNI proc command
        cmd = [
            "afni_proc.py",
            f"-subj_id {subject_id}",
            f"-script {script_path}",
            "-scr_overwrite",
            "-blocks tshift align volreg blur mask scale regress",
            f"-copy_anat {anat_file}",
            f"-dsets {' '.join(func_files)}",
            "-tcat_remove_first_trs 2",
            "-align_opts_aea -giant_move -cost lpc+ZZ",
            "-volreg_align_to first",
            "-volreg_tlrc_warp",
            "-blur_size 4.0",
            "-regress_motion_per_run"
        ]
        
        # Add timing files
        for condition, files in timing_files.items():
            cmd.extend([
                f"-regress_stim_times {' '.join(files)}",
                f"-regress_stim_labels {condition}"
            ])
        
        # Add remaining options
        cmd.extend([
            "-regress_basis 'TENT(0,20,11)'",
            "-regress_opts_3dD -jobs 8",
            "-regress_make_ideal_sum sum_ideal.1D",
            "-regress_est_blur_epits",
            "-regress_est_blur_errts"
        ])
        
        # Write script
        with open(script_path, 'w') as f:
            f.write('#!/bin/bash\n\n')
            f.write(' '.join(cmd))
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        self.logger.info(f"Created processing script: {script_path}")
        return script_path

    def process_subject(self, subject_id):
        """Process a single subject's data."""
        self.console.print(f"\n[bold green]Processing subject {subject_id}[/bold green]")
        
        # Copy raw data
        self._copy_raw_data(subject_id)
        
        # Preprocess data
        self._preprocess_data(subject_id)
        
        # Process timing files
        timing_files = self.process_timing_files(subject_id)
        
        # Run 3dDeconvolve
        glm_result = self.run_3ddeconvolve(subject_id, timing_files)
        
        # Calculate contrasts
        contrast_results = {}
        if glm_result:
            # Check for GLM file with proper AFNI extension
            glm_file_with_ext = f"{glm_result}+orig"
            glm_file_brik = f"{glm_result}+orig.BRIK.gz"
            
            # Check if the actual AFNI file exists
            if os.path.exists(glm_file_brik):
                contrast_results = self.contrast_calculator.calculate_all_contrasts(subject_id, glm_file_with_ext)
            else:
                self.logger.warning(f"GLM file not found: {glm_file_brik}")
        
        # Create AFNI processing script
        script_path = self.create_afni_proc_script(subject_id, timing_files)
        
        self.console.print(f"[bold green]✓[/bold green] Completed processing for subject {subject_id}")
        
        return {
            'script_path': script_path,
            'glm_result': glm_result,
            'timing_files': timing_files,
            'contrasts': contrast_results
        }

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FMRI Processing Pipeline")
    parser.add_argument("--subject", type=str, help="Process specific subject only")
    parser.add_argument("--data_dir", type=str, default="Data", help="Input data directory")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Output directory")
    parser.add_argument("--no_parallel", action="store_true", help="Disable parallel processing")
    args = parser.parse_args()
    
    console = Console()
    console.print("[bold blue]Starting FMRI Processing Pipeline[/bold blue]")
    console.print("[yellow]Processing first 4 runs only[/yellow]")
    
    # Initialize processor
    processor = FMRIProcessor(args.data_dir, args.output_dir)
    
    # Override parallel processing if requested
    if args.no_parallel:
        processor.n_processes = 1
        console.print("[yellow]Parallel processing disabled[/yellow]")
    
    # Determine subjects to process
    if args.subject:
        # Process only the specified subject
        test_subjects = [args.subject]
        console.print(f"[cyan]Processing single subject: {args.subject}[/cyan]")
    else:
        # Process all test subjects
        test_subjects = ["sub-ALC2158", "sub-ALC2161", "sub-ALC2118", "sub-ALC2131", "sub-ALC2132", "sub-ALC2134"]
        console.print(f"[cyan]Processing all subjects: {', '.join(test_subjects)}[/cyan]")
    
    results = {}
    for subject in test_subjects:
        results[subject] = processor.process_subject(subject)
    
    console.print("\n[bold green]Pipeline completed successfully![/bold green]")
    console.print(f"Check the logs directory for detailed processing information.")
    console.print(f"GLM results are in the glm_results directory.")

if __name__ == "__main__":
    main() 