#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path
import subprocess
import logging
from rich.console import Console
from rich.logging import RichHandler

class ContrastCalculator:
    """Standalone class for calculating and managing statistical contrasts."""
    
    def __init__(self, output_dir, logger=None):
        self.output_dir = Path(output_dir)
        self.contrast_dir = self.output_dir / "contrasts"
        self.contrast_dir.mkdir(exist_ok=True)
        self.logger = logger or self._setup_logger()
        
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
                'formula': '(a+b)-(c+d)',
                'conditions': ['FearCue', 'FearImage', 'NeutralCue', 'NeutralImage'],
                'description': 'Overall Fear > Neutral'
            },
            
            # Phasic vs Sustained threat
            'Phasic_vs_Sustained': {
                'formula': '(a+b+c+d)-(e+f+g)',
                'conditions': ['FearCue', 'NeutralCue', 'FearImage', 'NeutralImage', 'UnknownCue', 'UnknownFear', 'UnknownNeutral'],
                'description': 'Phasic threat > Sustained threat'
            },
            'Sustained_vs_Phasic': {
                'formula': '(e+f+g)-(a+b+c+d)',
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
                'formula': '(a+b)-(c+d)',
                'conditions': ['FearCue', 'NeutralCue', 'FearImage', 'NeutralImage'], 
                'description': 'Cue processing > Image processing'
            },
            'Images_vs_Cues': {
                'formula': '(c+d)-(a+b)',
                'conditions': ['FearImage', 'NeutralImage', 'FearCue', 'NeutralCue'],
                'description': 'Image processing > Cue processing'
            },
            
            # Threat sensitivity contrasts
            'Cue_ThreatSensitivity': {
                'formula': '(a-b)-(c-d)',
                'conditions': ['FearCue', 'NeutralCue', 'FearImage', 'NeutralImage'],
                'description': 'Threat sensitivity for cues > images'
            },
            'Image_ThreatSensitivity': {
                'formula': '(c-d)-(a-b)', 
                'conditions': ['FearImage', 'NeutralImage', 'FearCue', 'NeutralCue'],
                'description': 'Threat sensitivity for images > cues'
            }
        }

    def _setup_logger(self):
        """Setup logging for standalone use."""
        logger = logging.getLogger("ContrastCalculator")
        logger.setLevel(logging.INFO)
        
        console_handler = RichHandler(rich_tracebacks=True)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        return logger

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
        
        if not os.path.exists(glm_file):
            self.logger.error(f"GLM file not found: {glm_file}")
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
        console = Console()
        console.print("\n[bold blue]Available Standard Contrasts:[/bold blue]")
        
        for name, info in self.standard_contrasts.items():
            console.print(f"\n[green]🎯 {name}[/green]")
            console.print(f"   📝 {info['description']}")
            console.print(f"   🧮 Formula: {info['formula']}")
            console.print(f"   📊 Conditions: {', '.join(info['conditions'])}")

    def batch_calculate_contrasts(self, glm_results_dir, subjects=None):
        """Calculate contrasts for multiple subjects."""
        
        glm_dir = Path(glm_results_dir)
        if not glm_dir.exists():
            self.logger.error(f"GLM results directory not found: {glm_dir}")
            return
        
        # Find all subjects if not specified
        if subjects is None:
            subjects = [d.name for d in glm_dir.iterdir() if d.is_dir()]
        
        all_results = {}
        
        for subject_id in subjects:
            subject_glm_dir = glm_dir / subject_id
            glm_file = subject_glm_dir / f"{subject_id}_glm+orig.HEAD"
            
            if glm_file.exists():
                results = self.calculate_all_contrasts(subject_id, str(glm_file).replace('.HEAD', ''))
                all_results[subject_id] = results
            else:
                self.logger.warning(f"GLM file not found for {subject_id}: {glm_file}")
        
        # Save batch results summary
        batch_summary = self.contrast_dir / "batch_contrast_summary.json"
        with open(batch_summary, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        self.logger.info(f"✓ Batch processing completed for {len(all_results)} subjects")
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Calculate statistical contrasts from fMRI GLM results")
    parser.add_argument("--glm_dir", required=True, help="Directory containing GLM results")
    parser.add_argument("--output_dir", required=True, help="Output directory for contrasts")
    parser.add_argument("--subjects", nargs="+", help="Specific subjects to process (default: all)")
    parser.add_argument("--list_contrasts", action="store_true", help="List available contrasts")
    parser.add_argument("--contrast", help="Calculate specific contrast only")
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = ContrastCalculator(args.output_dir)
    
    if args.list_contrasts:
        calculator.list_available_contrasts()
        return
    
    # Run batch processing
    calculator.batch_calculate_contrasts(args.glm_dir, args.subjects)

if __name__ == "__main__":
    main() 