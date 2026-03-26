"""
Tool for generating visualization plots from DEG analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Union, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False

try:
    from matplotlib_venn import venn2
    VENN_AVAILABLE = True
except ImportError:
    VENN_AVAILABLE = False

from .base_tool import BaseTool
from ..exceptions import AnalysisError, ValidationError


class DEGPlotterTool(BaseTool):
    """Tool for generating visualization plots from DEG analysis results."""
    
    @property
    def name(self) -> str:
        return "DEGPlotter"
    
    @property
    def description(self) -> str:
        return "Generate visualization plots from DEG analysis results"
    
    def execute(self, deg_file: Union[str, Path], counts_file: Optional[Union[str, Path]] = None,
                metadata_file: Optional[Union[str, Path]] = None,
                output_dir: Union[str, Path] = None, **kwargs) -> Dict:
        """
        Generate all plots from DEG results.
        
        Args:
            deg_file: Path to DEG results CSV file
            counts_file: Optional path to counts file for expression plots
            metadata_file: Optional path to metadata file for expression plots
            output_dir: Directory to save plots (default: same directory as deg_file)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with plot file paths and summary
            
        Raises:
            AnalysisError: If plotting fails
        """
        deg_file = Path(deg_file).resolve()
        if not deg_file.exists():
            raise AnalysisError(f"DEG file not found: {deg_file}")
        
        # Set output directory
        if output_dir is None:
            output_dir = deg_file.parent / "plots"
        else:
            output_dir = Path(output_dir) / "plots"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load DEG data
        try:
            deg_data = pd.read_csv(deg_file)
            self.logger.info(f"📊 Loaded DEG data: {len(deg_data)} genes")
        except Exception as e:
            raise AnalysisError(f"Failed to load DEG file: {e}")
        
        # Load optional expression data
        expression_data = None
        sample_metadata = None
        if counts_file and Path(counts_file).exists():
            try:
                expression_data = pd.read_csv(counts_file, index_col=0)
                self.logger.info(f"📊 Loaded expression data: {expression_data.shape}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load counts file: {e}")
        
        if metadata_file and Path(metadata_file).exists():
            try:
                sample_metadata = pd.read_csv(metadata_file)
                self.logger.info(f"📋 Loaded metadata: {len(sample_metadata)} samples")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load metadata file: {e}")
        
        plots_created = []
        
        # Generate all plots
        try:
            # 1. Significant genes plot
            self.logger.info("📈 Generating significant genes plot...")
            self._sig_genes_plot(deg_data, output_dir)
            plots_created.append("sig_genes_plot.png")
            
            # 2. Venn diagram
            if VENN_AVAILABLE:
                self.logger.info("📈 Generating Venn diagram...")
                self._plot_venn(deg_data, output_dir)
                plots_created.append("venn_diagram.png")
            else:
                self.logger.warning("⚠️ matplotlib_venn not available, skipping Venn diagram")
            
            # 3. Upset plot
            self.logger.info("📈 Generating upset plot...")
            self._plot_upset(deg_data, output_dir)
            plots_created.append("upset_plot.png")
            
            # 4. Volcano plot
            self.logger.info("📈 Generating volcano plot...")
            self._plot_volcano(deg_data, output_dir)
            plots_created.append("volcano_plot.png")
            
            # 5. MA plot
            self.logger.info("📈 Generating MA plot...")
            self._plot_ma(deg_data, output_dir)
            plots_created.append("ma_plot.png")
            
            # 6. Genes scatter plot (if expression data available)
            if expression_data is not None and sample_metadata is not None:
                self.logger.info("📈 Generating genes scatter plot...")
                try:
                    self._plot_genes_scatter(deg_data, expression_data, sample_metadata, output_dir)
                    plots_created.append("genes_scatter_plot.png")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to generate scatter plot: {e}")
            
            # 7. Heatmap (if expression data available)
            if expression_data is not None:
                self.logger.info("📈 Generating heatmap...")
                try:
                    self._heatmap_plot(deg_data, expression_data, output_dir)
                    plots_created.append("heatmap_plot.png")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to generate heatmap: {e}")
            
            self.logger.info(f"✅ Generated {len(plots_created)} plots in {output_dir}")
            
            return {
                "plots_created": plots_created,
                "output_dir": str(output_dir),
                "n_plots": len(plots_created),
                "summary": f"Successfully generated {len(plots_created)} plots"
            }
            
        except Exception as e:
            raise AnalysisError(f"Plot generation failed: {e}")
    
    def _sig_genes_plot(self, deg_data: pd.DataFrame, output_dir: Path) -> None:
        """Generate significant genes bar plot."""
        plot_colors = ['salmon', 'skyblue']
        comparison = deg_data['Comparison'].unique()
        n_comparisons = len(comparison)
        fig, ax = plt.subplots()
        
        bar_width = 0.35
        index = np.arange(n_comparisons)
        data = {}
        
        for i, comp in enumerate(comparison):
            comp_data = deg_data[deg_data['Comparison'] == comp]
            up_count = (comp_data['log2FoldChange'] > 0).sum()
            down_count = (comp_data['log2FoldChange'] < 0).sum()
            counts = [down_count, up_count]
            
            ax.barh(index[i], counts[1], bar_width, color=plot_colors[0], label='Up' if i == 0 else '')
            ax.barh(index[i] + bar_width, counts[0], bar_width, color=plot_colors[1], label='Down' if i == 0 else '')
            
            for j, count in enumerate(counts):
                ax.text(count, index[i] + j * bar_width, str(count), va='center', ha='right')
            data[comp] = {'down_regulated': counts[0], 'up_regulated': counts[1]}
        
        ax.set_xlabel('Count')
        ax.set_ylabel('Comparison')
        ax.set_title('Number of differentially expressed genes')
        ax.set_yticks(index + bar_width / 2)
        ax.set_yticklabels(comparison)
        ax.legend(title='Regulation', loc='upper right')
        
        fig.savefig(output_dir / 'sig_genes_plot.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_venn(self, deg_data: pd.DataFrame, output_dir: Path) -> None:
        """Generate Venn diagram."""
        if not VENN_AVAILABLE:
            return
        
        comp = deg_data['Comparison'].unique()
        deg_data_comp = deg_data[deg_data['Comparison'] == comp[0]]
        
        up_counts = (deg_data_comp['log2FoldChange'] > 0).sum()
        down_counts = (deg_data_comp['log2FoldChange'] < 0).sum()
        intersection = 0
        whole_complement = (deg_data_comp['log2FoldChange'] == 0).sum()
        
        color_palette = plt.cm.get_cmap('Paired', 2)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        venn2(
            subsets=(up_counts, down_counts, intersection),
            set_labels=('Up Regulated', 'Down Regulated'),
            set_colors=(color_palette(0), color_palette(1)),
            alpha=0.5,
            ax=ax
        )
        
        plt.text(0.45, 0.45, f'No Regulation\n{whole_complement}', 
                ha='center', va='center', fontsize=8, color='black')
        plt.title('Venn Diagram of Differentially Expressed Genes')
        plt.savefig(output_dir / 'venn_diagram.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_upset(self, deg_data: pd.DataFrame, output_dir: Path) -> None:
        """Generate upset plot."""
        comp = deg_data['Comparison'].unique()
        results_df = deg_data[deg_data['Comparison'] == comp[0]]
        
        up_counts = (results_df['log2FoldChange'] > 0).sum()
        down_counts = (results_df['log2FoldChange'] < 0).sum()
        
        plot_data = pd.DataFrame({'Group': ['Up', 'Down'], 'Count': [up_counts, down_counts]})
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Group', y='Count', data=plot_data, palette='gray')
        
        for index, row in plot_data.iterrows():
            plt.text(row.name, row['Count'], str(row['Count']), ha='center', va='bottom')
        
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title('Upset Plot')
        plt.savefig(output_dir / 'upset_plot.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_volcano(self, deg_data: pd.DataFrame, output_dir: Path) -> None:
        """Generate volcano plot."""
        comp = deg_data['Comparison'].unique()
        results_df = deg_data[deg_data['Comparison'] == comp[0]].copy()
        
        data = pd.DataFrame()
        data['Fold'] = results_df['log2FoldChange']
        
        if 'padj' in results_df.columns:
            data['FDR'] = results_df['padj']
        elif 'pvalue' in results_df.columns:
            data['FDR'] = results_df['pvalue']
        else:
            raise ValueError("Neither 'padj' nor 'pvalue' column found in data")
        
        if 'Gene' in results_df.columns:
            data['Gene'] = results_df['Gene']
        else:
            data['Gene'] = results_df.index
        
        data['upOrDownorNone'] = data['Fold'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        data.loc[:, 'FDR'] = data['FDR'].replace(0, np.nan)
        data.dropna(subset=['FDR'], inplace=True)
        data.loc[:, 'FDR'] = -np.log10(data['FDR'])
        data.loc[:, ['Fold', 'FDR']] = data[['Fold', 'FDR']].round(3)
        
        num_total = len(data[data['upOrDownorNone'].isin([1, -1])])
        num_up = sum(data['upOrDownorNone'] == 1)
        num_down = sum(data['upOrDownorNone'] == -1)
        
        top_up = data[data['upOrDownorNone'] == 1].nlargest(10, 'Fold')['Gene'].tolist()
        top_down = data[data['upOrDownorNone'] == -1].nsmallest(10, 'Fold')['Gene'].tolist()
        example_genes = (top_up[:2] + top_down[:2])[:4]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = sns.scatterplot(data=data, x='Fold', y='FDR', hue='upOrDownorNone', 
                                 palette=['green', 'red'], ax=ax)
        
        legend_labels = {-1: 'Down', 0: 'None', 1: 'Up'}
        handles, labels = scatter.get_legend_handles_labels()
        ax.legend(handles=handles, labels=[legend_labels[int(float(label))] for label in labels], 
                 title='Regulated', loc='best')
        
        ax.set_xlabel('Log2 Fold Change')
        ax.set_ylabel('-log10(Adjusted p-Val)')
        ax.set_title('Fold Change vs. Adjusted p-Value')
        ax.grid(True)
        ax.axhline(-np.log10(0.05), color='black', linestyle='--')
        ax.axvline(1, color='black', linestyle='--')
        ax.axvline(-1, color='black', linestyle='--')
        
        fdr_threshold = -np.log10(0.05)
        fc_threshold = 1.0
        
        outliers = data[
            (data['FDR'] > fdr_threshold) & 
            (abs(data['Fold']) > fc_threshold)
        ].copy()
        
        outliers['score'] = outliers['FDR'] * abs(outliers['Fold'])
        
        max_labels = 30 if ADJUST_TEXT_AVAILABLE else 15
        if len(outliers) > max_labels:
            outliers = outliers.nlargest(max_labels, 'score')
        
        texts = []
        for idx, row in outliers.iterrows():
            text = ax.text(
                row['Fold'], row['FDR'], str(row['Gene']),
                fontsize=8, ha='left' if row['Fold'] > 0 else 'right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
            )
            texts.append(text)
        
        if ADJUST_TEXT_AVAILABLE and len(texts) > 0:
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                expand_points=(1.2, 1.2), expand_text=(1.2, 1.2),
                force_text=(0.5, 0.5), force_points=(0.5, 0.5),
                avoid_text=True, avoid_points=True
            )
        
        fig.savefig(output_dir / 'volcano_plot.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ma(self, deg_data: pd.DataFrame, output_dir: Path) -> None:
        """Generate MA plot."""
        comp = deg_data['Comparison'].unique()
        results_df = deg_data[deg_data['Comparison'] == comp[0]].copy()
        
        data = pd.DataFrame()
        data['Fold'] = results_df['log2FoldChange']
        
        if 'baseMean' in results_df.columns:
            baseMean_values = results_df['baseMean']
            if baseMean_values.notna().sum() == 0:
                data['Mean'] = np.abs(results_df['log2FoldChange']) + 1
            else:
                data['Mean'] = baseMean_values
        else:
            alt_names = ['AveExpr', 'mean', 'Mean', 'logCPM']
            found = False
            for alt_name in alt_names:
                if alt_name in results_df.columns:
                    alt_values = results_df[alt_name]
                    if alt_values.notna().sum() > 0:
                        data['Mean'] = alt_values
                        found = True
                        break
            if not found:
                data['Mean'] = np.abs(results_df['log2FoldChange']) + 1
        
        if 'Gene' in results_df.columns:
            data['Gene'] = results_df['Gene']
        else:
            data['Gene'] = results_df.index
        
        data = data.dropna(subset=['Fold'])
        if data['Mean'].isna().sum() > 0:
            data['Mean'] = data['Mean'].fillna(data['Mean'].median() if data['Mean'].notna().sum() > 0 else 1.0)
        
        data = data[~np.isinf(data['Fold'])]
        data = data[~np.isinf(data['Mean'])]
        
        if data['Mean'].min() >= 0 and data['Mean'].max() <= 20:
            data.loc[:, 'Mean'] = np.log1p(data['Mean'])
        
        data['upOrDownorNone'] = data['Fold'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        data.loc[:, ['Mean', 'Fold']] = data[['Mean', 'Fold']].round(3)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        legend_labels = {-1: 'Down', 0: 'None', 1: 'Up'}
        colors = {-1: 'blue', 0: 'gray', 1: 'red'}
        
        for reg_type in [-1, 0, 1]:
            subset = data[data['upOrDownorNone'] == reg_type]
            if len(subset) > 0:
                ax.scatter(subset['Mean'], subset['Fold'], c=colors[reg_type], 
                          label=legend_labels[reg_type], alpha=0.6, s=20)
        
        ax.legend(title='Regulated', loc='best')
        ax.set_xlabel('Log(1 + Mean Expression)')
        ax.set_ylabel('Log2 Fold Change')
        ax.set_title('Fold Change vs. Mean Expression (MA Plot)')
        ax.grid(True)
        
        fig.savefig(output_dir / 'ma_plot.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_genes_scatter(self, deg_data: pd.DataFrame, expression_data: pd.DataFrame,
                           sample_metadata: pd.DataFrame, output_dir: Path) -> None:
        """Generate genes scatter plot."""
        comp = deg_data['Comparison'].unique()
        results_df = deg_data[deg_data['Comparison'] == comp[0]].copy()
        
        comp_str = comp[0]
        if '_vs_' in comp_str:
            groups = comp_str.split('_vs_')
        elif '-' in comp_str:
            groups = comp_str.split('-')
        else:
            raise ValueError(f"Could not parse comparison string: {comp_str}")
        
        if len(groups) != 2:
            raise ValueError(f"Comparison should have exactly 2 groups, found: {groups}")
        
        group1 = groups[0].strip()
        group2 = groups[1].strip()
        
        expr_df = expression_data.copy()
        if 'Gene' in expr_df.columns:
            expr_df = expr_df.set_index('Gene')
        
        if 'Gene' in results_df.columns:
            genes = results_df['Gene'].tolist()
        else:
            genes = results_df.index.tolist()
        
        available_genes = [g for g in genes if g in expr_df.index]
        if len(available_genes) == 0:
            raise ValueError("No matching genes found between DEG results and expression data")
        
        expr_df = expr_df.loc[available_genes]
        
        if sample_metadata is not None:
            if 'condition' in sample_metadata.columns:
                group_col = 'condition'
            elif 'Group' in sample_metadata.columns:
                group_col = 'Group'
            elif 'group' in sample_metadata.columns:
                group_col = 'group'
            else:
                raise ValueError("sample_metadata must have a 'group' or 'Group' column")
            
            sample_to_group = dict(zip(sample_metadata.iloc[:, 0], sample_metadata[group_col]))
            group1_cols = [col for col in expr_df.columns if sample_to_group.get(col, '').strip() == group1]
            group2_cols = [col for col in expr_df.columns if sample_to_group.get(col, '').strip() == group2]
        else:
            group1_cols = [col for col in expr_df.columns if group1.lower() in col.lower()]
            group2_cols = [col for col in expr_df.columns if group2.lower() in col.lower()]
        
        if len(group1_cols) == 0 or len(group2_cols) == 0:
            raise ValueError(f"Could not find samples for groups: {group1}, {group2}")
        
        avg_group1_expression = expr_df[group1_cols].mean(axis=1)
        avg_group2_expression = expr_df[group2_cols].mean(axis=1)
        
        if avg_group1_expression.max() > 20 or avg_group2_expression.max() > 20:
            avg_group1_expression = np.log1p(avg_group1_expression)
            avg_group2_expression = np.log1p(avg_group2_expression)
        
        merged_df = results_df.copy()
        if 'Gene' in merged_df.columns:
            merged_df = merged_df.set_index('Gene')
        
        merged_df['regulation'] = merged_df['log2FoldChange'].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        
        common_genes = [g for g in avg_group1_expression.index if g in merged_df.index]
        avg_group1_expression = avg_group1_expression.loc[common_genes]
        avg_group2_expression = avg_group2_expression.loc[common_genes]
        regulation = merged_df.loc[common_genes, 'regulation']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = {-1: 'blue', 0: 'gray', 1: 'red'}
        legend_labels = {-1: 'Down', 0: 'None', 1: 'Up'}
        
        for reg_type in [-1, 0, 1]:
            mask = regulation == reg_type
            if mask.sum() > 0:
                ax.scatter(avg_group1_expression[mask], avg_group2_expression[mask],
                          c=colors[reg_type], label=legend_labels[reg_type],
                          alpha=0.6, s=30)
        
        ax.set_xlabel(group1)
        ax.set_ylabel(group2)
        ax.set_title(f'Scatter Plot of Genes: {group1} vs {group2}')
        ax.legend(title='Regulation', loc='best')
        ax.grid(True, alpha=0.3)
        
        min_val = min(avg_group1_expression.min(), avg_group2_expression.min())
        max_val = max(avg_group1_expression.max(), avg_group2_expression.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1, label='y=x')
        
        fig.savefig(output_dir / 'genes_scatter_plot.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _heatmap_plot(self, deg_data: pd.DataFrame, expression_data: pd.DataFrame, output_dir: Path) -> None:
        """Generate heatmap of significantly expressed genes."""
        comp = deg_data['Comparison'].unique()
        results_df = deg_data[deg_data['Comparison'] == comp[0]].copy()
        
        if 'padj' in results_df.columns:
            p_col = 'padj'
            p_threshold = 0.05
        elif 'pvalue' in results_df.columns:
            p_col = 'pvalue'
            p_threshold = 0.05
        else:
            raise ValueError("Neither 'padj' nor 'pvalue' column found in data")
        
        sigs = results_df[
            (results_df[p_col] < p_threshold) & 
            (abs(results_df['log2FoldChange']) > 0.5)
        ].copy()
        
        if len(sigs) == 0:
            raise ValueError("No significantly expressed genes found with the specified thresholds")
        
        if 'Gene' in sigs.columns:
            sig_genes = sigs['Gene'].tolist()
        else:
            sig_genes = sigs.index.tolist()
        
        expr_df = expression_data.copy()
        if 'Gene' in expr_df.columns:
            expr_df = expr_df.set_index('Gene')
        
        available_genes = [g for g in sig_genes if g in expr_df.index]
        if len(available_genes) == 0:
            raise ValueError("No matching genes found between DEG results and expression data")
        
        grapher = expr_df.loc[available_genes]
        
        if grapher.max().max() > 20:
            grapher = np.log1p(grapher)
        
        grapher = grapher.T
        
        if grapher.shape[1] > 100:
            sigs_sorted = sigs.sort_values('log2FoldChange', key=abs, ascending=False).head(100)
            if 'Gene' in sigs_sorted.columns:
                top_genes = sigs_sorted['Gene'].tolist()
            else:
                top_genes = sigs_sorted.index.tolist()
            available_top = [g for g in top_genes if g in grapher.columns]
            grapher = grapher[available_top]
        
        figsize = (max(10, grapher.shape[1] * 0.2), max(8, grapher.shape[0] * 0.3))
        clustermap = sns.clustermap(grapher, z_score=0, cmap='RdYlBu_r', 
                                   figsize=figsize,
                                   cbar_kws={'label': 'Z-score'})
        
        plt.suptitle('Heatmap of Significantly Expressed Genes', y=1.02)
        clustermap.ax_row_dendrogram.remove()
        
        clustermap.savefig(output_dir / 'heatmap_plot.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def validate_input(self, deg_file: Union[str, Path], **kwargs) -> None:
        """Validate input parameters."""
        if not Path(deg_file).exists():
            raise ValidationError(f"DEG file does not exist: {deg_file}")
    
    def validate_output(self, result: Dict) -> Dict:
        """Validate output result."""
        required_fields = ["plots_created", "output_dir", "n_plots", "summary"]
        for field in required_fields:
            if field not in result:
                raise ValidationError(f"Missing required field in result: {field}")
        return result

