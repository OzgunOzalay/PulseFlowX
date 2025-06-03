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
    
    # Setup logging for this process
    logger = logging.getLogger(f"FMRIProcessor_{run}")
    logger.setLevel(logging.INFO)
    
    # Create handlers
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    
    input_file = Path(raw_dir) / subject_id / "func" / f"{subject_id}_task-unpredictablethreat_run-{run}_bold.nii"
    if not input_file.exists():
        logger.warning(f"Input file not found: {input_file}")
        return
    
    # 1. Slice timing correction
    slice_timing_file = Path(preproc_dir) / subject_id / f"{subject_id}_task-unpredictablethreat_run-{run}_tcat.nii"
    cmd = f"3dTcat -prefix {slice_timing_file} {input_file}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"Slice timing correction completed for run {run}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in slice timing correction for run {run}: {e}")
        return
    
    # 2. Despiking
    despiked_file = Path(preproc_dir) / subject_id / f"{subject_id}_task-unpredictablethreat_run-{run}_despiked.nii"
    cmd = f"3dDespike -NEW -prefix {despiked_file} {slice_timing_file}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"Despiking completed for run {run}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in despiking for run {run}: {e}")
        return
    
    # 3. Motion correction
    motion_file = Path(preproc_dir) / subject_id / f"{subject_id}_task-unpredictablethreat_run-{run}_motion.nii"
    cmd = f"3dvolreg -prefix {motion_file} -1Dfile {Path(preproc_dir)}/{subject_id}/motion_run{run}.1D -Fourier -twopass -zpad 4 {despiked_file}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"Motion correction completed for run {run}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in motion correction for run {run}: {e}")
        return
    
    # 4. Spatial smoothing
    smoothed_file = Path(preproc_dir) / subject_id / f"{subject_id}_task-unpredictablethreat_run-{run}_smoothed.nii"
    cmd = f"3dmerge -1blur_fwhm 4.0 -doall -prefix {smoothed_file} {motion_file}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"Spatial smoothing completed for run {run}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in spatial smoothing for run {run}: {e}")
        return
    
    # 5. Generate QC plots
    qc_file = Path(qc_dir) / subject_id / f"qc_run{run}.png"
    cmd = f"3dTstat -mean -prefix {Path(qc_dir)}/{subject_id}/mean_run{run}.nii {smoothed_file} && 3dTstat -stdev -prefix {Path(qc_dir)}/{subject_id}/std_run{run}.nii {smoothed_file}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"QC plots generated for run {run}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating QC plots for run {run}: {e}")

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
        
        # Define condition mappings
        self.condition_mapping = {
            'phasic1': ['FearCue', 'NeutralCue'],
            'phasic2': ['FearImage', 'NeutralImage'],
            'sustained': ['UnknownCue', 'UnknownFear', 'UnknownNeutral']
        }
        
        # Set number of processes for parallel processing
        self.n_processes = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free

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
        
        # Create directories
        for directory in [self.raw_dir, self.preprocessed_dir, self.timing_dir, 
                         self.scripts_dir, self.logs_dir, self.qc_dir]:
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
        
        # Copy functional data
        func_dest_dir = subject_raw_dir / "func"
        func_dest_dir.mkdir(exist_ok=True)
        
        for run in range(1, 6):
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
        
        # Prepare arguments for parallel processing
        args_list = [
            (run, subject_id, str(self.raw_dir), str(self.preprocessed_dir), str(self.qc_dir))
            for run in range(1, 6)
        ]
        
        # Create a pool of workers
        with multiprocessing.Pool(processes=self.n_processes) as pool:
            # Process runs in parallel
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("[cyan]Processing runs...", total=5)
                
                # Map the runs to the process pool
                for _ in pool.imap_unordered(process_single_run, args_list):
                    progress.update(task, advance=1)

    def process_timing_files(self, subject_id):
        """Process timing files for a single subject."""
        self.logger.info(f"Processing timing files for subject {subject_id}")
        subject_dir = self.raw_dir / subject_id
        timing_files = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Processing timing files...", total=5)
            
            # Process each run
            for run in range(1, 6):  # 5 runs
                tsv_file = subject_dir / "func" / f"{subject_id}_task-unpredictablethreat_run-{run}_events.tsv"
                if not tsv_file.exists():
                    self.logger.warning(f"Warning: {tsv_file} not found")
                    progress.update(task, advance=1)
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
                
                progress.update(task, advance=1)
        
        return timing_files

    def create_afni_proc_script(self, subject_id, timing_files):
        """Create AFNI processing script for a subject."""
        self.logger.info(f"Creating AFNI processing script for subject {subject_id}")
        script_path = self.scripts_dir / f"proc_{subject_id}.sh"
        
        # Get anatomical and functional data paths
        anat_file = self.raw_dir / subject_id / "anat" / f"{subject_id}_T1w.nii"
        func_files = [str(self.preprocessed_dir / subject_id / f"{subject_id}_task-unpredictablethreat_run-{run}_smoothed.nii")
                     for run in range(1, 6)]
        
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
            "-regress_basis 'GAM'",
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
        
        # Create AFNI processing script
        script_path = self.create_afni_proc_script(subject_id, timing_files)
        
        self.console.print(f"[bold green]✓[/bold green] Completed processing for subject {subject_id}")
        return script_path

def main():
    console = Console()
    console.print("[bold blue]Starting FMRI Processing Pipeline[/bold blue]")
    
    # Initialize processor
    processor = FMRIProcessor("Data", "processed_data")
    
    # Process test subjects
    test_subjects = ["sub-ALC2158", "sub-ALC2161"]
    
    for subject in test_subjects:
        processor.process_subject(subject)
    
    console.print("\n[bold green]Pipeline completed successfully![/bold green]")
    console.print(f"Check the logs directory for detailed processing information.")

if __name__ == "__main__":
    main() 