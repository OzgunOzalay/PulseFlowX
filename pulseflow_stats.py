#!/usr/bin/env python3
"""
Publication-Quality Group Analysis for fMRI Threat Processing Study
Generates comprehensive statistical reports, tables, and visualizations for AUD vs HC comparisons.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from scipy import stats
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class PublicationGroupAnalyzer:
    """Advanced group analysis with publication-quality outputs."""
    
    def __init__(self, output_dir="processed_data"):
        self.output_dir = Path(output_dir)
        self.roi_dir = self.output_dir / "roi_analysis"
        self.console = Console()
        self.logger = self._setup_logger()
        
        # Publication directories
        self.pub_dir = self.output_dir / "publication_outputs"
        self.pub_dir.mkdir(exist_ok=True)
        (self.pub_dir / "figures").mkdir(exist_ok=True)
        (self.pub_dir / "tables").mkdir(exist_ok=True)
        (self.pub_dir / "statistics").mkdir(exist_ok=True)
        
        # Load group assignments
        self.groups = self._load_groups()
        
        # Define analysis parameters
        self.rois = ['amygdala']
        self.hemispheres = ['left', 'right']
        self.response_types = ['sustained', 'phasic']
        self.conditions = [
            'FearCue', 'NeutralCue', 'FearImage', 'NeutralImage',
            'UnknownCue', 'UnknownFear', 'UnknownNeutral'
        ]
        
        # Effect size thresholds (Cohen's conventions)
        self.effect_size_thresholds = {
            'small': 0.2,
            'medium': 0.5,
            'large': 0.8
        }
    
    def _setup_logger(self):
        """Setup logger."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def _load_groups(self):
        """Load group assignments."""
        group_file = Path("subject_groups.json")
        if group_file.exists():
            with open(group_file, 'r') as f:
                data = json.load(f)
            return data.get('groups', {})
        return {}
    
    def load_roi_data(self):
        """Load all ROI data into a structured format."""
        
        self.console.print("[bold blue]Loading ROI data...[/bold blue]")
        
        all_data = []
        
        for roi in self.rois:
            for hemisphere in self.hemispheres:
                # Load individual subject data
                for group_name, subjects in self.groups.items():
                    for subject in subjects:
                        results_file = self.roi_dir / subject / f"{subject}_{roi}_{hemisphere}_roi_results.json"
                        
                        if results_file.exists():
                            with open(results_file, 'r') as f:
                                subject_data = json.load(f)
                            
                            for response_type in self.response_types:
                                if response_type in subject_data:
                                    for condition in self.conditions:
                                        if condition in subject_data[response_type]:
                                            activation = subject_data[response_type][condition]['mean']
                                            
                                            all_data.append({
                                                'subject': subject,
                                                'group': group_name,
                                                'roi': roi,
                                                'hemisphere': hemisphere,
                                                'response_type': response_type,
                                                'condition': condition,
                                                'activation': activation
                                            })
        
        self.df = pd.DataFrame(all_data)
        self.console.print(f"[green]✓ Loaded {len(all_data)} data points from {len(self.df['subject'].unique())} subjects[/green]")
        
        return self.df
    
    def run_statistical_tests(self):
        """Run comprehensive statistical tests."""
        
        self.console.print("[bold blue]Running statistical tests...[/bold blue]")
        
        results = []
        
        for roi in self.rois:
            for hemisphere in self.hemispheres:
                for response_type in self.response_types:
                    for condition in self.conditions:
                        # Filter data
                        subset = self.df[
                            (self.df['roi'] == roi) &
                            (self.df['hemisphere'] == hemisphere) &
                            (self.df['response_type'] == response_type) &
                            (self.df['condition'] == condition)
                        ]
                        
                        if len(subset) == 0:
                            continue
                        
                        # Separate groups
                        aud_data = subset[subset['group'] == 'AUD']['activation'].values
                        hc_data = subset[subset['group'] == 'HC']['activation'].values
                        
                        if len(aud_data) == 0 or len(hc_data) == 0:
                            continue
                        
                        # Statistical tests
                        t_stat, p_val = stats.ttest_ind(aud_data, hc_data)
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(aud_data) - 1) * np.var(aud_data, ddof=1) +
                                            (len(hc_data) - 1) * np.var(hc_data, ddof=1)) /
                                           (len(aud_data) + len(hc_data) - 2))
                        cohens_d = (np.mean(aud_data) - np.mean(hc_data)) / pooled_std
                        
                        # Confidence intervals for means
                        aud_ci = stats.t.interval(0.95, len(aud_data)-1, 
                                                loc=np.mean(aud_data), 
                                                scale=stats.sem(aud_data))
                        hc_ci = stats.t.interval(0.95, len(hc_data)-1,
                                               loc=np.mean(hc_data),
                                               scale=stats.sem(hc_data))
                        
                        # Effect size interpretation
                        if abs(cohens_d) >= self.effect_size_thresholds['large']:
                            effect_magnitude = 'Large'
                        elif abs(cohens_d) >= self.effect_size_thresholds['medium']:
                            effect_magnitude = 'Medium'
                        elif abs(cohens_d) >= self.effect_size_thresholds['small']:
                            effect_magnitude = 'Small'
                        else:
                            effect_magnitude = 'Negligible'
                        
                        results.append({
                            'ROI': roi.title(),
                            'Hemisphere': hemisphere.title(),
                            'Response_Type': response_type.title(),
                            'Condition': condition,
                            'AUD_Mean': np.mean(aud_data),
                            'AUD_SD': np.std(aud_data, ddof=1),
                            'AUD_CI_Lower': aud_ci[0],
                            'AUD_CI_Upper': aud_ci[1],
                            'AUD_N': len(aud_data),
                            'HC_Mean': np.mean(hc_data),
                            'HC_SD': np.std(hc_data, ddof=1),
                            'HC_CI_Lower': hc_ci[0],
                            'HC_CI_Upper': hc_ci[1],
                            'HC_N': len(hc_data),
                            'T_Statistic': t_stat,
                            'P_Value': p_val,
                            'Cohens_D': cohens_d,
                            'Effect_Magnitude': effect_magnitude,
                            'Significant': p_val < 0.05,
                            'Significant_Bonferroni': p_val < (0.05 / len(self.conditions))
                        })
        
        self.stats_df = pd.DataFrame(results)
        self.console.print(f"[green]✓ Completed {len(results)} statistical tests[/green]")
        
        return self.stats_df
    
    def create_publication_table(self):
        """Create publication-ready statistical table."""
        
        self.console.print("[bold blue]Creating publication table...[/bold blue]")
        
        # Format table for publication
        pub_table = self.stats_df.copy()
        
        # Round numerical values appropriately
        pub_table['AUD_Mean'] = pub_table['AUD_Mean'].round(3)
        pub_table['AUD_SD'] = pub_table['AUD_SD'].round(3)
        pub_table['HC_Mean'] = pub_table['HC_Mean'].round(3)
        pub_table['HC_SD'] = pub_table['HC_SD'].round(3)
        pub_table['T_Statistic'] = pub_table['T_Statistic'].round(3)
        pub_table['Cohens_D'] = pub_table['Cohens_D'].round(3)
        
        # Format p-values
        pub_table['P_Value_Formatted'] = pub_table['P_Value'].apply(
            lambda x: f"{x:.3f}" if x >= 0.001 else "<0.001"
        )
        
        # Create mean ± SD columns
        pub_table['AUD_Mean_SD'] = pub_table.apply(
            lambda row: f"{row['AUD_Mean']:.3f} ± {row['AUD_SD']:.3f}", axis=1
        )
        pub_table['HC_Mean_SD'] = pub_table.apply(
            lambda row: f"{row['HC_Mean']:.3f} ± {row['HC_SD']:.3f}", axis=1
        )
        
        # Select and rename columns for publication
        final_table = pub_table[[
            'ROI', 'Hemisphere', 'Response_Type', 'Condition',
            'AUD_Mean_SD', 'HC_Mean_SD', 'T_Statistic', 
            'P_Value_Formatted', 'Cohens_D', 'Effect_Magnitude'
        ]].rename(columns={
            'AUD_Mean_SD': 'AUD (M ± SD)',
            'HC_Mean_SD': 'HC (M ± SD)',
            'T_Statistic': 't',
            'P_Value_Formatted': 'p',
            'Cohens_D': "Cohen's d",
            'Effect_Magnitude': 'Effect Size'
        })
        
        # Save table
        table_file = self.pub_dir / "tables" / "group_comparison_results.csv"
        final_table.to_csv(table_file, index=False)
        
        # Save Excel version with formatting
        excel_file = self.pub_dir / "tables" / "group_comparison_results.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            final_table.to_excel(writer, sheet_name='Group_Comparisons', index=False)
        
        self.console.print(f"[green]✓ Publication table saved to {table_file}[/green]")
        
        return final_table
    
    def create_comprehensive_visualizations(self):
        """Create publication-quality visualizations."""
        
        self.console.print("[bold blue]Creating publication visualizations...[/bold blue]")
        
        # Set seaborn style
        sns.set_style("whitegrid")
        
        # 1. Summary heatmap of effect sizes
        self._create_effect_size_heatmap()
        
        # 2. Group comparison plots by condition
        self._create_condition_comparison_plots()
        
        # 3. Statistical significance overview
        self._create_significance_overview()
        
        # 4. Individual condition deep-dive plots
        self._create_individual_condition_plots()
        
        # 5. Forest plot of effect sizes
        self._create_forest_plot()
        
        self.console.print("[green]✓ All publication visualizations created[/green]")
    
    def _create_effect_size_heatmap(self):
        """Create heatmap showing effect sizes across conditions."""
        
        # Prepare data for heatmap
        heatmap_data = self.stats_df.pivot_table(
            index=['Response_Type', 'Condition'],
            columns=['ROI', 'Hemisphere'],
            values='Cohens_D'
        )
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, annot=True, cmap='RdBu_r', center=0,
                   fmt='.2f', cbar_kws={'label': "Cohen's d"})
        plt.title('Effect Sizes (AUD vs HC) Across Conditions and ROIs', pad=20)
        plt.xlabel('ROI and Hemisphere')
        plt.ylabel('Response Type and Condition')
        plt.tight_layout()
        
        plt.savefig(self.pub_dir / "figures" / "effect_size_heatmap.png")
        plt.close()
    
    def _create_condition_comparison_plots(self):
        """Create group comparison plots for each condition."""
        
        # Get unique combinations
        combinations = self.stats_df[['ROI', 'Hemisphere', 'Response_Type']].drop_duplicates()
        
        for _, combo in combinations.iterrows():
            roi = combo['ROI']
            hemisphere = combo['Hemisphere']
            response_type = combo['Response_Type']
            
            # Filter data
            subset = self.df[
                (self.df['roi'] == roi.lower()) &
                (self.df['hemisphere'] == hemisphere.lower()) &
                (self.df['response_type'] == response_type.lower())
            ]
            
            if len(subset) == 0:
                continue
            
            # Create subplot for this combination
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f'{roi} ({hemisphere}) - {response_type} Responses: AUD vs HC', fontsize=16)
            
            for i, condition in enumerate(self.conditions):
                row = i // 4
                col = i % 4
                
                if i >= len(self.conditions):
                    axes[row, col].set_visible(False)
                    continue
                
                cond_data = subset[subset['condition'] == condition]
                
                if len(cond_data) > 0:
                    # Box plot with individual points
                    sns.boxplot(data=cond_data, x='group', y='activation', ax=axes[row, col])
                    sns.stripplot(data=cond_data, x='group', y='activation', ax=axes[row, col],
                                size=8, alpha=0.7, color='black')
                    
                    # Get statistics for this condition
                    stats_row = self.stats_df[
                        (self.stats_df['ROI'] == roi) &
                        (self.stats_df['Hemisphere'] == hemisphere) &
                        (self.stats_df['Response_Type'] == response_type) &
                        (self.stats_df['Condition'] == condition)
                    ]
                    
                    if len(stats_row) > 0:
                        p_val = stats_row.iloc[0]['P_Value']
                        cohens_d = stats_row.iloc[0]['Cohens_D']
                        
                        # Add statistical annotations
                        if p_val < 0.001:
                            sig_text = "***"
                        elif p_val < 0.01:
                            sig_text = "**"
                        elif p_val < 0.05:
                            sig_text = "*"
                        else:
                            sig_text = "ns"
                        
                        axes[row, col].set_title(f'{condition}\np={p_val:.3f} {sig_text}\nd={cohens_d:.2f}')
                    else:
                        axes[row, col].set_title(condition)
                    
                    axes[row, col].set_xlabel('')
                    axes[row, col].set_ylabel('Activation')
                else:
                    axes[row, col].text(0.5, 0.5, 'No data', ha='center', va='center')
                    axes[row, col].set_title(condition)
            
            plt.tight_layout()
            filename = f"{roi.lower()}_{hemisphere.lower()}_{response_type.lower()}_comparison.png"
            plt.savefig(self.pub_dir / "figures" / filename)
            plt.close()
    
    def _create_significance_overview(self):
        """Create overview of statistical significance."""
        
        # Count significant results
        sig_counts = self.stats_df.groupby(['Response_Type', 'ROI', 'Hemisphere']).agg({
            'Significant': 'sum',
            'P_Value': 'count'
        }).rename(columns={'P_Value': 'Total'})
        
        sig_counts['Proportion'] = sig_counts['Significant'] / sig_counts['Total']
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Number of significant results
        sig_plot_data = sig_counts.reset_index()
        sig_plot_data['Label'] = (sig_plot_data['ROI'] + ' ' + 
                                 sig_plot_data['Hemisphere'] + ' ' +
                                 sig_plot_data['Response_Type'])
        
        sns.barplot(data=sig_plot_data, x='Label', y='Significant', ax=ax1)
        ax1.set_title('Number of Significant Group Differences (p < 0.05)')
        ax1.set_xlabel('ROI, Hemisphere, Response Type')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: P-value distribution
        ax2.hist(self.stats_df['P_Value'], bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
        ax2.set_xlabel('P-values')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of P-values')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.pub_dir / "figures" / "significance_overview.png")
        plt.close()
    
    def _create_individual_condition_plots(self):
        """Create detailed plots for key conditions."""
        
        key_conditions = ['FearCue', 'FearImage', 'NeutralCue']
        
        for condition in key_conditions:
            condition_data = self.df[self.df['condition'] == condition]
            
            if len(condition_data) == 0:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Detailed Analysis: {condition} Condition', fontsize=16)
            
            # Plot by hemisphere and response type
            for i, (hemisphere, response_type) in enumerate([
                ('left', 'sustained'), ('right', 'sustained'),
                ('left', 'phasic'), ('right', 'phasic')
            ]):
                row = i // 2
                col = i % 2
                
                subset = condition_data[
                    (condition_data['hemisphere'] == hemisphere) &
                    (condition_data['response_type'] == response_type)
                ]
                
                if len(subset) > 0:
                    sns.violinplot(data=subset, x='group', y='activation', ax=axes[row, col])
                    sns.stripplot(data=subset, x='group', y='activation', ax=axes[row, col],
                                size=8, alpha=0.8, color='white', edgecolor='black', linewidth=1)
                    
                    axes[row, col].set_title(f'{hemisphere.title()} {response_type.title()}')
                    axes[row, col].set_xlabel('Group')
                    axes[row, col].set_ylabel('Activation')
                else:
                    axes[row, col].text(0.5, 0.5, 'No data', ha='center', va='center')
                    axes[row, col].set_title(f'{hemisphere.title()} {response_type.title()}')
            
            plt.tight_layout()
            plt.savefig(self.pub_dir / "figures" / f"{condition.lower()}_detailed.png")
            plt.close()
    
    def _create_forest_plot(self):
        """Create forest plot of effect sizes."""
        
        # Prepare data
        forest_data = self.stats_df.copy()
        forest_data['Label'] = (forest_data['Hemisphere'] + ' ' +
                               forest_data['Response_Type'] + ' ' +
                               forest_data['Condition'])
        
        # Sort by effect size
        forest_data = forest_data.sort_values('Cohens_D')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, len(forest_data) * 0.3 + 2))
        
        y_pos = range(len(forest_data))
        
        # Plot effect sizes
        colors = ['red' if sig else 'gray' for sig in forest_data['Significant']]
        ax.scatter(forest_data['Cohens_D'], y_pos, c=colors, s=50, alpha=0.7)
        
        # Add vertical lines for effect size thresholds
        for threshold, label in [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
            ax.axvline(x=threshold, color='lightgray', linestyle='--', alpha=0.5)
            ax.axvline(x=-threshold, color='lightgray', linestyle='--', alpha=0.5)
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(forest_data['Label'])
        ax.set_xlabel("Cohen's d (Effect Size)")
        ax.set_title('Effect Sizes: AUD vs HC Group Differences')
        
        # Add legend
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8)
        gray_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8)
        ax.legend([red_patch, gray_patch], ['Significant (p < 0.05)', 'Not significant'])
        
        plt.tight_layout()
        plt.savefig(self.pub_dir / "figures" / "forest_plot.png")
        plt.close()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        
        self.console.print("[bold blue]Generating summary report...[/bold blue]")
        
        # Calculate summary statistics
        total_tests = len(self.stats_df)
        significant_tests = self.stats_df['Significant'].sum()
        large_effects = (abs(self.stats_df['Cohens_D']) >= 0.8).sum()
        medium_effects = ((abs(self.stats_df['Cohens_D']) >= 0.5) & 
                         (abs(self.stats_df['Cohens_D']) < 0.8)).sum()
        
        # Most significant findings
        top_findings = self.stats_df.nsmallest(5, 'P_Value')[
            ['ROI', 'Hemisphere', 'Response_Type', 'Condition', 'P_Value', 'Cohens_D']
        ]
        
        # Largest effect sizes
        self.stats_df['Abs_Cohens_D'] = abs(self.stats_df['Cohens_D'])
        largest_effects = self.stats_df.nlargest(5, 'Abs_Cohens_D')[
            ['ROI', 'Hemisphere', 'Response_Type', 'Condition', 'P_Value', 'Cohens_D']
        ]
        
        # Generate report
        report = f"""
# fMRI Threat Processing Study: Group Analysis Report

## Study Overview
- **Groups**: AUD (n={len(self.groups.get('AUD', []))}) vs HC (n={len(self.groups.get('HC', []))})
- **ROIs**: {', '.join([roi.title() for roi in self.rois])}
- **Hemispheres**: {', '.join([h.title() for h in self.hemispheres])}
- **Response Types**: {', '.join([rt.title() for rt in self.response_types])}
- **Conditions**: {len(self.conditions)} threat processing conditions

## Statistical Summary
- **Total Comparisons**: {total_tests}
- **Significant Results**: {significant_tests} ({significant_tests/total_tests*100:.1f}%)
- **Large Effect Sizes**: {large_effects} (|d| ≥ 0.8)
- **Medium Effect Sizes**: {medium_effects} (0.5 ≤ |d| < 0.8)

## Top 5 Most Significant Findings
{top_findings.to_string(index=False)}

## Top 5 Largest Effect Sizes
{largest_effects.to_string(index=False)}

## Clinical Interpretation
{'[Analysis suggests significant group differences in threat processing]' if significant_tests > 0 else '[No significant group differences detected]'}

## Files Generated
- Statistical tables: publication_outputs/tables/
- Figures: publication_outputs/figures/
- Full statistics: publication_outputs/statistics/
"""
        
        # Save report
        report_file = self.pub_dir / "GROUP_ANALYSIS_REPORT.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save detailed statistics
        stats_file = self.pub_dir / "statistics" / "complete_statistics.json"
        stats_dict = self.stats_df.to_dict('records')
        with open(stats_file, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        self.console.print(f"[green]✓ Summary report saved to {report_file}[/green]")
        
        # Display key findings in console
        self._display_console_summary()
    
    def _display_console_summary(self):
        """Display summary in console with rich formatting."""
        
        # Key statistics table
        table = Table(title="Group Analysis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        total_tests = len(self.stats_df)
        significant_tests = self.stats_df['Significant'].sum()
        
        table.add_row("Total Comparisons", str(total_tests))
        table.add_row("Significant Results", f"{significant_tests} ({significant_tests/total_tests*100:.1f}%)")
        table.add_row("Mean Effect Size", f"{abs(self.stats_df['Cohens_D']).mean():.3f}")
        table.add_row("Largest Effect Size", f"{abs(self.stats_df['Cohens_D']).max():.3f}")
        
        self.console.print(table)
        
        # Most significant finding
        if significant_tests > 0:
            top_result = self.stats_df.loc[self.stats_df['P_Value'].idxmin()]
            
            panel = Panel(
                f"[bold]Most Significant Finding:[/bold]\n"
                f"ROI: {top_result['ROI']} ({top_result['Hemisphere']})\n"
                f"Condition: {top_result['Condition']} ({top_result['Response_Type']})\n"
                f"p = {top_result['P_Value']:.4f}, d = {top_result['Cohens_D']:.3f}\n"
                f"AUD: {top_result['AUD_Mean']:.3f} ± {top_result['AUD_SD']:.3f}\n"
                f"HC: {top_result['HC_Mean']:.3f} ± {top_result['HC_SD']:.3f}",
                title="Key Finding",
                border_style="green"
            )
            self.console.print(panel)
    
    def run_complete_analysis(self):
        """Run complete publication-quality analysis."""
        
        self.console.print(Panel.fit(
            "[bold blue]fMRI Threat Processing: Publication-Quality Group Analysis[/bold blue]",
            border_style="blue"
        ))
        
        # Load and process data
        self.load_roi_data()
        
        # Run statistical tests
        self.run_statistical_tests()
        
        # Create publication outputs
        self.create_publication_table()
        self.create_comprehensive_visualizations()
        self.generate_summary_report()
        
        self.console.print(Panel.fit(
            "[bold green]✓ Publication-quality analysis complete![/bold green]\n"
            f"Check outputs in: {self.pub_dir}",
            border_style="green"
        ))

def main():
    """Main function."""
    analyzer = PublicationGroupAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 