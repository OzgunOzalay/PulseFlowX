#!/usr/bin/env python3
"""
Visualization script for sustained/phasic analysis results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import nibabel as nib
from scipy import stats

def load_afni_data(file_path):
    """Load AFNI data using nibabel."""
    try:
        # Try loading as NIfTI first
        img = nib.load(str(file_path))
        return img.get_fdata()
    except:
        # If that fails, try with .HEAD extension
        try:
            img = nib.load(str(file_path.with_suffix('.HEAD')))
            return img.get_fdata()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

def plot_response_comparison(subject_id, response_type, conditions, output_dir):
    """Plot sustained vs phasic responses for a subject."""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'{subject_id} - {response_type.capitalize()} Responses', fontsize=16)
    
    # Define colors for conditions
    colors = {
        'FearCue': 'red',
        'NeutralCue': 'blue', 
        'FearImage': 'darkred',
        'NeutralImage': 'darkblue',
        'UnknownCue': 'orange',
        'UnknownFear': 'darkorange',
        'UnknownNeutral': 'lightblue'
    }
    
    for i, condition in enumerate(conditions):
        row = i // 3
        col = i % 3
        
        # Hide the last subplot if we have fewer than 9 conditions
        if i >= len(conditions):
            axes[row, col].set_visible(False)
            continue
        
        # Load response data
        response_file = output_dir / subject_id / response_type / f"{subject_id}_{condition}_{response_type}_response+orig"
        data = load_afni_data(response_file)
        
        if data is not None:
            # Calculate mean across voxels (excluding zeros)
            mask = data != 0
            if mask.any():
                mean_response = np.mean(data[mask])
                std_response = np.std(data[mask])
                
                # Create a simple bar plot
                axes[row, col].bar([0], [mean_response], yerr=[std_response], 
                                 color=colors.get(condition, 'gray'), alpha=0.7)
                axes[row, col].set_title(f'{condition}')
                axes[row, col].set_ylabel('Response Magnitude')
                axes[row, col].set_xticks([])
            else:
                axes[row, col].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[row, col].set_title(f'{condition}')
        else:
            axes[row, col].text(0.5, 0.5, 'File not found', ha='center', va='center')
            axes[row, col].set_title(f'{condition}')
    
    plt.tight_layout()
    return fig

def plot_contrast_comparison(subject_id, response_type, output_dir):
    """Plot contrast results for a subject."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{subject_id} - {response_type.capitalize()} Contrasts', fontsize=16)
    
    contrasts = [
        f'{response_type}_Fear_vs_Neutral',
        f'{response_type}_Phasic_vs_Sustained_Threat', 
        f'{response_type}_UnknownFear_vs_UnknownNeutral'
    ]
    
    contrast_names = ['Fear vs Neutral', 'Phasic vs Sustained Threat', 'Unknown Fear vs Neutral']
    
    for i, (contrast, name) in enumerate(zip(contrasts, contrast_names)):
        contrast_file = output_dir / subject_id / response_type / "contrasts" / f"{subject_id}_{contrast}+orig"
        data = load_afni_data(contrast_file)
        
        if data is not None:
            # Create histogram of contrast values
            mask = data != 0
            if mask.any():
                contrast_values = data[mask]
                axes[i].hist(contrast_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].axvline(np.mean(contrast_values), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(contrast_values):.3f}')
                axes[i].set_title(name)
                axes[i].set_xlabel('Contrast Value')
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
            else:
                axes[i].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[i].set_title(name)
        else:
            axes[i].text(0.5, 0.5, 'File not found', ha='center', va='center')
            axes[i].set_title(name)
    
    plt.tight_layout()
    return fig

def plot_group_comparison(output_dir):
    """Plot comparison between subjects (AUD vs HC)."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Group Comparison: AUD vs HC', fontsize=16)
    
    response_types = ['sustained', 'phasic']
    conditions = ['FearCue', 'NeutralCue', 'FearImage']
    
    subjects = {
        'AUD': 'sub-ALC2158',
        'HC': 'sub-ALC2161'
    }
    
    for i, response_type in enumerate(response_types):
        for j, condition in enumerate(conditions):
            ax = axes[i, j]
            
            group_means = []
            group_names = []
            
            for group, subject in subjects.items():
                response_file = output_dir / subject / response_type / f"{subject}_{condition}_{response_type}_mean+orig"
                data = load_afni_data(response_file)
                
                if data is not None:
                    mask = data != 0
                    if mask.any():
                        mean_val = np.mean(data[mask])
                        group_means.append(mean_val)
                        group_names.append(group)
            
            if len(group_means) == 2:
                bars = ax.bar(group_names, group_means, color=['red', 'blue'], alpha=0.7)
                ax.set_title(f'{response_type.capitalize()} - {condition}')
                ax.set_ylabel('Mean Response')
                
                # Add value labels on bars
                for bar, val in zip(bars, group_means):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{val:.3f}', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                ax.set_title(f'{response_type.capitalize()} - {condition}')
    
    plt.tight_layout()
    return fig

def main():
    """Main function to generate all visualizations."""
    
    output_dir = Path("processed_data/sustained_phasic_analysis")
    plots_dir = Path("plots/sustained_phasic")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    subjects = ['sub-ALC2158', 'sub-ALC2161']
    response_types = ['sustained', 'phasic']
    conditions = ['FearCue', 'NeutralCue', 'FearImage', 'NeutralImage', 'UnknownCue', 'UnknownFear', 'UnknownNeutral']
    
    print("Generating visualizations...")
    
    # Generate individual subject plots
    for subject in subjects:
        for response_type in response_types:
            print(f"  Plotting {subject} {response_type} responses...")
            
            # Response comparison plot
            fig = plot_response_comparison(subject, response_type, conditions, output_dir)
            fig.savefig(plots_dir / f"{subject}_{response_type}_responses.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Contrast comparison plot
            fig = plot_contrast_comparison(subject, response_type, output_dir)
            fig.savefig(plots_dir / f"{subject}_{response_type}_contrasts.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    # Generate group comparison plot
    print("  Plotting group comparison...")
    fig = plot_group_comparison(output_dir)
    fig.savefig(plots_dir / "group_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Generate summary statistics
    print("  Generating summary statistics...")
    summary_stats = {}
    
    for subject in subjects:
        summary_stats[subject] = {}
        for response_type in response_types:
            summary_stats[subject][response_type] = {}
            
            for condition in conditions:
                response_file = output_dir / subject / response_type / f"{subject}_{condition}_{response_type}_mean+orig"
                data = load_afni_data(response_file)
                
                if data is not None:
                    mask = data != 0
                    if mask.any():
                        summary_stats[subject][response_type][condition] = {
                            'mean': float(np.mean(data[mask])),
                            'std': float(np.std(data[mask])),
                            'min': float(np.min(data[mask])),
                            'max': float(np.max(data[mask])),
                            'n_voxels': int(np.sum(mask))
                        }
    
    # Save summary statistics
    with open(plots_dir / "summary_statistics.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"✓ All visualizations saved to {plots_dir}")
    print(f"✓ Summary statistics saved to {plots_dir}/summary_statistics.json")

if __name__ == "__main__":
    main() 