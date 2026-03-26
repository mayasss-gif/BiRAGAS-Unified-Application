from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import seaborn as sns
import numpy as np
import logging
from pathlib import Path
from functools import lru_cache
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from contextlib import contextmanager

from agents import function_tool, RunContextWrapper, Agent
from ..helper_tools.file_utils import resolve_and_validate_file
from ..configuration import HISTOGRAM_BINS, HISTOGRAM_COLOR, HISTOGRAM_EDGE_COLOR

from ..helpers import logger

# Suppress matplotlib font warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Set matplotlib backend for better performance
mplstyle.use('fast')
plt.rcParams['figure.max_open_warning'] = 0  # Disable figure limit warnings

@dataclass
class PlotContext:
    df: Optional[pd.DataFrame] = None
    current_file: str = ""
    file_stats: Dict[str, Any] = field(default_factory=dict)
    plot_cache: Dict[str, str] = field(default_factory=dict)
    data_summary: Dict[str, Any] = field(default_factory=dict)

    def clear_data(self):
        """Clear data to free memory"""
        self.df = None
        gc.collect()

@contextmanager
def performance_context():
    """Context manager for performance monitoring"""
    import time
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Operation completed in {elapsed:.2f} seconds")

@lru_cache(maxsize=32)
def get_optimized_bins(data_size: int, data_range: float) -> int:
    """Calculate optimal number of bins based on data characteristics"""
    if data_size < 100:
        return min(10, data_size // 2)
    elif data_size < 1000:
        return min(30, int(np.sqrt(data_size)))
    else:
        # Use Freedman-Diaconis rule for large datasets
        return min(100, max(10, int(data_range / (2 * np.percentile(np.abs(np.random.sample(min(10000, data_size))), 75) / data_size**(1/3)))))

def detect_outliers(data: pd.Series) -> Tuple[np.ndarray, Dict[str, float]]:
    """Detect outliers using IQR method and return statistics"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (data < lower_bound) | (data > upper_bound)
    stats = {
        'outlier_count': outliers.sum(),
        'outlier_percentage': (outliers.sum() / len(data)) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR
    }
    
    return outliers, stats

def generate_comprehensive_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Generate comprehensive statistics for a column"""
    if column not in df.columns:
        return {}
    
    data = pd.to_numeric(df[column], errors='coerce').dropna()
    if data.empty:
        return {'error': f'No valid numeric data in column {column}'}
    
    outliers, outlier_stats = detect_outliers(data)
    
    stats = {
        'count': len(data),
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'range': data.max() - data.min(),
        'skewness': data.skew(),
        'kurtosis': data.kurtosis(),
        'null_count': df[column].isnull().sum(),
        'unique_count': data.nunique(),
        **outlier_stats
    }
    
    return stats

@function_tool
def load_filtered_file(
    ctx: RunContextWrapper[PlotContext], file_path: str
) -> str:
    """
    Load a filtered CSV into a DataFrame with enhanced validation and caching.
    """
    with performance_context():
        try:
            # Validate and resolve file path
            resolved_path = resolve_and_validate_file(file_path, allowed_extensions={".csv"})
            
            # Check if file is already loaded (simple caching)
            file_key = f"{resolved_path}_{resolved_path.stat().st_mtime}"
            if (ctx.context.current_file == resolved_path.name and 
                ctx.context.df is not None and 
                ctx.context.file_stats.get('file_key') == file_key):
                logger.info(f"Using cached data for {resolved_path.name}")
                return f"Using cached {resolved_path.name} ({ctx.context.df.shape[0]} rows)."
            
            # Clear previous data to free memory
            ctx.context.clear_data()
            
            # Load with optimized settings
            logger.info(f"Loading file: {resolved_path}")
            
            # Read in chunks for large files to check size first
            sample_df = pd.read_csv(resolved_path, nrows=1000, low_memory=False)
            
            # Determine optimal loading strategy based on file size
            file_size_mb = resolved_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb > 100:  # Large file handling
                logger.info(f"Large file detected ({file_size_mb:.1f}MB), using chunked loading")
                chunk_list = []
                chunk_size = 10000
                
                for chunk in pd.read_csv(resolved_path, chunksize=chunk_size, low_memory=False):
                    chunk_list.append(chunk)
                
                df = pd.concat(chunk_list, ignore_index=True)
                del chunk_list  # Free memory
            else:
                # Standard loading for smaller files
                df = pd.read_csv(resolved_path, low_memory=False)
            
            # Generate file statistics
            file_stats = {
                'file_key': file_key,
                'file_size_mb': file_size_mb,
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'load_time': 'measured_in_context'
            }
            
            # Store in context
            ctx.context.df = df
            ctx.context.current_file = resolved_path.name
            ctx.context.file_stats = file_stats
            
            # Generate data summary for common analysis columns
            summary = {}
            for col in ['log2FC', 'pvalue', 'padj', 'baseMean']:
                if col in df.columns:
                    summary[col] = generate_comprehensive_stats(df, col)
            
            ctx.context.data_summary = summary
            
            logger.info(f"Successfully loaded {resolved_path.name}")
            logger.info(f"Dataset info: {df.shape[0]} rows, {df.shape[1]} columns, {file_size_mb:.1f}MB")
            
            # Log column info
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            logger.info(f"Numeric columns available: {numeric_cols}")
            
            return f"Loaded {resolved_path.name} ({df.shape[0]:,} rows, {df.shape[1]} columns, {file_size_mb:.1f}MB)."
            
        except Exception as exc:
            logger.exception("Error loading '%s': %s", file_path, exc)
            ctx.context.clear_data()
            return f"Error loading {file_path}: {exc}"

@function_tool
def plot_log2fc_histogram(
    ctx: RunContextWrapper[PlotContext], output_dir: str, 
    advanced_stats: bool = True, save_stats: bool = True
) -> str:
    """
    Plot an enhanced histogram of the 'log2FC' column with comprehensive statistics and styling.
    """
    df = ctx.context.df
    current_file = ctx.context.current_file
    
    if df is None or df.empty:
        return "No data loaded. Please load a file first."
    
    if "log2FC" not in df.columns:
        available_cols = [col for col in df.columns if 'log2' in col.lower() or 'fc' in col.lower()]
        if available_cols:
            return f"Missing 'log2FC' column. Similar columns found: {available_cols}"
        return "Missing 'log2FC' column and no similar columns detected."

    with performance_context():
        try:
            # Prepare output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            base_name = Path(current_file).stem
            plot_file = output_path / f"hist_{base_name}.png"
            
            # Check cache
            cache_key = f"{current_file}_{df.shape[0]}_{df['log2FC'].sum():.6f}"
            if cache_key in ctx.context.plot_cache:
                cached_file = ctx.context.plot_cache[cache_key]
                if Path(cached_file).exists():
                    logger.info(f"Using cached plot: {cached_file}")
                    return f"Using cached histogram: {cached_file}"
            
            # Prepare data
            log2fc_data = pd.to_numeric(df["log2FC"], errors='coerce').dropna()
            
            if log2fc_data.empty:
                return "No valid numeric data in log2FC column."
            
            # Generate statistics
            stats = ctx.context.data_summary.get('log2FC', generate_comprehensive_stats(df, 'log2FC'))
            
            # Determine optimal plot parameters
            data_range = stats['max'] - stats['min']
            optimal_bins = get_optimized_bins(len(log2fc_data), data_range)
            
            # Create enhanced plot with subplots
            fig = plt.figure(figsize=(15, 10))
            
            # Main histogram (larger subplot)
            ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
            
            # Create histogram with enhanced styling
            n, bins, patches = ax1.hist(
                log2fc_data,
                bins=min(optimal_bins, HISTOGRAM_BINS) if HISTOGRAM_BINS else optimal_bins,
                color=HISTOGRAM_COLOR,
                edgecolor=HISTOGRAM_EDGE_COLOR,
                alpha=0.7,
                density=False
            )
            
            # Add statistical overlays
            ax1.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {stats["mean"]:.3f}')
            ax1.axvline(stats['median'], color='orange', linestyle='--', linewidth=2, 
                       label=f'Median: {stats["median"]:.3f}')
            
            # Add confidence intervals
            ax1.axvline(stats['mean'] - stats['std'], color='red', linestyle=':', alpha=0.7,
                       label=f'±1 SD')
            ax1.axvline(stats['mean'] + stats['std'], color='red', linestyle=':', alpha=0.7)
            
            # Enhance main plot
            ax1.set_title(f'log2FC Distribution: {base_name}', fontsize=14, fontweight='bold')
            ax1.set_xlabel('log2FC', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add data quality indicators
            if stats['outlier_percentage'] > 5:
                ax1.text(0.02, 0.98, f"⚠️ {stats['outlier_percentage']:.1f}% outliers detected", 
                        transform=ax1.transAxes, va='top', ha='left',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            if advanced_stats:
                # Box plot (right side)
                ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
                box_plot = ax2.boxplot(log2fc_data, vert=True, patch_artist=True)
                box_plot['boxes'][0].set_facecolor(HISTOGRAM_COLOR)
                ax2.set_title('Box Plot', fontsize=12)
                ax2.set_ylabel('log2FC')
                ax2.grid(True, alpha=0.3)
                
                # Statistics table (bottom)
                ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
                ax3.axis('off')
                
                # Create statistics table
                stats_data = [
                    ['Count', f"{stats['count']:,}"],
                    ['Mean ± SD', f"{stats['mean']:.3f} ± {stats['std']:.3f}"],
                    ['Median [Q1, Q3]', f"{stats['median']:.3f} [{stats['Q1']:.3f}, {stats['Q3']:.3f}]"],
                    ['Range', f"{stats['min']:.3f} to {stats['max']:.3f}"],
                    ['Skewness', f"{stats['skewness']:.3f}"],
                    ['Outliers', f"{stats['outlier_count']:,} ({stats['outlier_percentage']:.1f}%)"]
                ]
                
                table = ax3.table(cellText=stats_data,
                                colLabels=['Statistic', 'Value'],
                                cellLoc='center',
                                loc='center',
                                bbox=[0.1, 0.1, 0.8, 0.8])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                
                # Style the table
                for i in range(len(stats_data) + 1):
                    for j in range(2):
                        cell = table[i, j]
                        if i == 0:  # Header
                            cell.set_facecolor('#4CAF50')
                            cell.set_text_props(weight='bold', color='white')
                        else:
                            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
            
            plt.tight_layout()
            
            # Save plot with high quality
            fig.savefig(plot_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            # Cache the result
            ctx.context.plot_cache[cache_key] = str(plot_file)
            
            # Save statistics to file if requested
            if save_stats:
                stats_file = output_path / f"stats_{base_name}.txt"
                with open(stats_file, 'w') as f:
                    f.write(f"log2FC Statistics for {base_name}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Data Points: {stats['count']:,}\n")
                    f.write(f"Missing Values: {stats['null_count']:,}\n")
                    f.write(f"Unique Values: {stats['unique_count']:,}\n\n")
                    
                    f.write("Descriptive Statistics:\n")
                    f.write(f"  Mean: {stats['mean']:.6f}\n")
                    f.write(f"  Median: {stats['median']:.6f}\n")
                    f.write(f"  Standard Deviation: {stats['std']:.6f}\n")
                    f.write(f"  Minimum: {stats['min']:.6f}\n")
                    f.write(f"  Maximum: {stats['max']:.6f}\n")
                    f.write(f"  Range: {stats['range']:.6f}\n\n")
                    
                    f.write("Distribution Shape:\n")
                    f.write(f"  Skewness: {stats['skewness']:.6f}\n")
                    f.write(f"  Kurtosis: {stats['kurtosis']:.6f}\n\n")
                    
                    f.write("Outlier Analysis:\n")
                    f.write(f"  Outlier Count: {stats['outlier_count']:,}\n")
                    f.write(f"  Outlier Percentage: {stats['outlier_percentage']:.2f}%\n")
                    f.write(f"  Lower Bound: {stats['lower_bound']:.6f}\n")
                    f.write(f"  Upper Bound: {stats['upper_bound']:.6f}\n")
                
                logger.info(f"Statistics saved to {stats_file}")
            
            result_msg = f"Enhanced histogram saved to {plot_file}"
            if save_stats:
                result_msg += f" (with statistics file)"
            
            logger.info(f"Plot generated successfully: {plot_file}")
            return result_msg
            
        except Exception as exc:
            logger.exception("Error plotting histogram for %s: %s", current_file, exc)
            return f"Error plotting histogram: {exc}"

# Additional utility function for batch plotting
@function_tool  
def plot_multiple_histograms(
    ctx: RunContextWrapper[PlotContext], 
    output_dir: str,
    columns: List[str] = None
) -> str:
    """
    Plot histograms for multiple columns in parallel for efficiency.
    """
    df = ctx.context.df
    current_file = ctx.context.current_file
    
    if df is None or df.empty:
        return "No data loaded. Please load a file first."
    
    # Default to common analysis columns if none specified
    if columns is None:
        columns = ['log2FC', 'pvalue', 'padj', 'baseMean']
    
    # Filter to existing numeric columns
    available_columns = [col for col in columns if col in df.columns and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    if not available_columns:
        return f"None of the specified columns {columns} found or numeric in the dataset."
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    def plot_single_column(column):
        try:
            # Create individual histogram for each column
            fig, ax = plt.subplots(figsize=(10, 6))
            data = pd.to_numeric(df[column], errors='coerce').dropna()
            
            if data.empty:
                return f"No valid data for {column}"
            
            ax.hist(data, bins=30, color=HISTOGRAM_COLOR, edgecolor=HISTOGRAM_EDGE_COLOR, alpha=0.7)
            ax.set_title(f'{column} Distribution: {Path(current_file).stem}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add basic statistics
            ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.3f}')
            ax.axvline(data.median(), color='orange', linestyle='--', label=f'Median: {data.median():.3f}')
            ax.legend()
            
            plot_file = output_path / f"hist_{Path(current_file).stem}_{column}.png"
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            return f"✓ {column}: {plot_file}"
            
        except Exception as e:
            return f"✗ {column}: Error - {e}"
    
    # Process columns in parallel
    with ThreadPoolExecutor(max_workers=min(4, len(available_columns))) as executor:
        futures = [executor.submit(plot_single_column, col) for col in available_columns]
        results = [future.result() for future in as_completed(futures)]
    
    return f"Batch plotting completed for {len(available_columns)} columns:\n" + "\n".join(results)
def build_plotting_agent() -> Agent[PlotContext]:
    """
    Build an enhanced plotting agent with intelligent automation, 
    comprehensive error handling, and expanded visualization capabilities.
    
    The agent maintains the same user interface but adds:
    - Smart workflow automation
    - Multiple plot types and batch processing
    - Advanced error recovery
    - Quality assessment and validation
    - Progress tracking and reporting
    """
    
    # Enhanced instructions with intelligent automation capabilities
    enhanced_instructions = """
You are an intelligent data visualization agent specializing in genomic and differential expression analysis.

CORE WORKFLOW (automatically executed in sequence):
1) Load and validate filtered CSV files with comprehensive data quality assessment
2) Generate enhanced log2FC histograms with statistical overlays and quality indicators  
3) Save high-quality PNG outputs with accompanying statistics files
4) Provide detailed confirmation with data insights and quality metrics

INTELLIGENT AUTOMATION FEATURES:
- Auto-detect optimal visualization parameters based on data characteristics
- Smart error recovery and alternative column detection
- Batch processing capabilities for multiple files or columns
- Memory-efficient handling of large datasets
- Comprehensive data quality assessment and reporting

ENHANCED CAPABILITIES:
- Multiple plot types: histograms, box plots, distribution comparisons
- Statistical overlays: mean, median, confidence intervals, outlier detection
- Publication-ready outputs with professional styling
- Batch processing for multiple columns (log2FC, pvalue, padj, baseMean)
- Automated data quality warnings and recommendations

WORKFLOW AUTOMATION:
- If log2FC column missing, auto-detect similar columns (LogFC, log2FoldChange, etc.)
- Automatically determine optimal bin sizes based on data distribution
- Smart memory management for large files (>100MB)
- Auto-generate comprehensive statistics files alongside plots
- Progress tracking and performance monitoring

ERROR HANDLING & RECOVERY:
- Graceful handling of missing or corrupted data
- Alternative column suggestions when exact matches not found
- Memory cleanup and optimization for large datasets
- Detailed error reporting with actionable recommendations
- Fallback strategies for various data quality issues

OUTPUT STANDARDS:
- High-resolution PNG files (300 DPI) for publication quality
- Comprehensive statistics files with detailed analysis
- Progress reports with data quality assessments
- Performance metrics and optimization recommendations
- Short, informative confirmations with key insights

USAGE PATTERNS:
- Single file processing: Provide file path and output directory
- Batch processing: Process multiple files or columns automatically
- Quality assessment: Generate data quality reports with recommendations
- Advanced analysis: Include statistical overlays and distribution analysis

Always return concise confirmations highlighting:
- Number of data points processed
- Key statistical insights (mean, outliers, data quality)
- File locations of generated outputs
- Any data quality warnings or recommendations
- Processing performance metrics

Maintain backward compatibility while providing enhanced functionality.
"""

    # Build agent with expanded tool set
    return Agent[PlotContext](
    name="EnhancedPlottingAgent",
    instructions=enhanced_instructions,
    tools=[
        load_filtered_file,
        plot_log2fc_histogram,
        plot_multiple_histograms,
        validate_data_quality,
        generate_summary_report,
        batch_process_files,
        optimize_memory_usage,
    ],
)
# Additional enhanced tool functions to support the agent

@function_tool
def validate_data_quality(
    ctx: RunContextWrapper[PlotContext], 
    column: str = "log2FC",
    generate_report: bool = True
) -> str:
    """
    Comprehensive data quality validation with automated recommendations.
    """
    df = ctx.context.df
    if df is None or df.empty:
        return "No data loaded for quality validation."
    
    if column not in df.columns:
        # Smart column detection
        similar_cols = [col for col in df.columns 
                       if any(term in col.lower() for term in ['log2', 'fc', 'fold'])]
        if similar_cols:
            return f"Column '{column}' not found. Similar columns available: {similar_cols}"
        return f"Column '{column}' not found and no similar columns detected."
    
    try:
        # Comprehensive quality assessment
        data = pd.to_numeric(df[column], errors='coerce')
        total_count = len(df)
        valid_count = data.notna().sum()
        null_count = data.isna().sum()
        
        quality_issues = []
        recommendations = []
        
        # Check for missing data
        null_percentage = (null_count / total_count) * 100
        if null_percentage > 5:
            quality_issues.append(f"High missing data rate: {null_percentage:.1f}% ({null_count:,} values)")
            recommendations.append("Consider data imputation or filtering strategies")
        
        # Check for outliers
        if valid_count > 0:
            stats = generate_comprehensive_stats(df, column)
            if stats.get('outlier_percentage', 0) > 10:
                quality_issues.append(f"High outlier rate: {stats['outlier_percentage']:.1f}%")
                recommendations.append("Consider outlier removal or transformation")
            
            # Check distribution characteristics
            if abs(stats.get('skewness', 0)) > 2:
                quality_issues.append(f"Highly skewed distribution (skewness: {stats['skewness']:.2f})")
                recommendations.append("Consider log transformation or robust statistics")
        
        # Generate quality score
        quality_score = 100
        quality_score -= min(null_percentage, 50)  # Penalize missing data
        quality_score -= min(stats.get('outlier_percentage', 0), 30)  # Penalize outliers
        quality_score = max(0, quality_score)
        
        # Store quality assessment in context
        ctx.context.data_summary[f'{column}_quality'] = {
            'score': quality_score,
            'issues': quality_issues,
            'recommendations': recommendations,
            'valid_percentage': (valid_count / total_count) * 100
        }
        
        # Generate report
        if generate_report:
            report_lines = [
                f"Data Quality Assessment for '{column}':",
                f"Quality Score: {quality_score:.0f}/100",
                f"Valid Data: {valid_count:,}/{total_count:,} ({(valid_count/total_count)*100:.1f}%)"
            ]
            
            if quality_issues:
                report_lines.extend(["", "Issues Identified:"] + [f"- {issue}" for issue in quality_issues])
            
            if recommendations:
                report_lines.extend(["", "Recommendations:"] + [f"- {rec}" for rec in recommendations])
            
            return "\n".join(report_lines)
        
        return f"Quality validation complete. Score: {quality_score:.0f}/100"
        
    except Exception as exc:
        logger.exception(f"Error in quality validation: {exc}")
        return f"Error validating data quality: {exc}"

@function_tool
def generate_summary_report(
    ctx: RunContextWrapper[PlotContext],
    output_dir: str,
    include_recommendations: bool = True
) -> str:
    """
    Generate comprehensive analysis summary report.
    """
    df = ctx.context.df
    current_file = ctx.context.current_file
    
    if df is None or df.empty:
        return "No data available for summary report."
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f"summary_report_{Path(current_file).stem}.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"COMPREHENSIVE DATA ANALYSIS REPORT\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"File: {current_file}\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset overview
            f.write(f"DATASET OVERVIEW:\n")
            f.write(f"Rows: {df.shape[0]:,}\n")
            f.write(f"Columns: {df.shape[1]:,}\n")
            f.write(f"Memory Usage: {ctx.context.file_stats.get('memory_usage_mb', 0):.1f} MB\n\n")
            
            # Column analysis
            f.write(f"COLUMN ANALYSIS:\n")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            f.write(f"Numeric columns: {len(numeric_cols)}\n")
            f.write(f"Text columns: {df.shape[1] - len(numeric_cols)}\n\n")
            
            # Statistical summaries for key columns
            key_columns = ['log2FC', 'pvalue', 'padj', 'baseMean']
            available_key_cols = [col for col in key_columns if col in df.columns]
            
            if available_key_cols:
                f.write(f"KEY STATISTICS:\n")
                for col in available_key_cols:
                    stats = ctx.context.data_summary.get(col, generate_comprehensive_stats(df, col))
                    if 'error' not in stats:
                        f.write(f"\n{col}:\n")
                        f.write(f"  Count: {stats['count']:,}\n")
                        f.write(f"  Mean ± SD: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                        f.write(f"  Median [IQR]: {stats['median']:.4f} [{stats['Q1']:.4f}, {stats['Q3']:.4f}]\n")
                        f.write(f"  Range: {stats['min']:.4f} to {stats['max']:.4f}\n")
                        f.write(f"  Outliers: {stats['outlier_count']:,} ({stats['outlier_percentage']:.1f}%)\n")
            
            # Data quality assessment
            f.write(f"\nDATA QUALITY ASSESSMENT:\n")
            for col in available_key_cols:
                quality_info = ctx.context.data_summary.get(f'{col}_quality', {})
                if quality_info:
                    f.write(f"{col}: {quality_info.get('score', 'N/A'):.0f}/100\n")
            
            # Recommendations
            if include_recommendations:
                f.write(f"\nRECOMMENDATIONS:\n")
                all_recommendations = set()
                for col in available_key_cols:
                    quality_info = ctx.context.data_summary.get(f'{col}_quality', {})
                    all_recommendations.update(quality_info.get('recommendations', []))
                
                if all_recommendations:
                    for i, rec in enumerate(all_recommendations, 1):
                        f.write(f"{i}. {rec}\n")
                else:
                    f.write("No specific recommendations - data quality appears good.\n")
        
        return f"Comprehensive summary report saved to {report_file}"
        
    except Exception as exc:
        logger.exception(f"Error generating summary report: {exc}")
        return f"Error generating summary report: {exc}"

@function_tool
def batch_process_files(
    ctx: RunContextWrapper[PlotContext],
    file_patterns: List[str],
    output_dir: str,
    max_files: int = 10
) -> str:
    """
    Batch process multiple files automatically.
    """
    from glob import glob
    
    try:
        # Collect all matching files
        all_files = []
        for pattern in file_patterns:
            matched_files = glob(str(Path(pattern).expanduser()))
            all_files.extend(matched_files)
        
        # Remove duplicates and limit
        unique_files = list(set(all_files))[:max_files]
        
        if not unique_files:
            return f"No files found matching patterns: {file_patterns}"
        
        results = []
        for i, file_path in enumerate(unique_files, 1):
            try:
                logger.info(f"Processing file {i}/{len(unique_files)}: {Path(file_path).name}")
                
                # Load file
                load_result = load_filtered_file(ctx, file_path)
                if "Error" in load_result:
                    results.append(f"❌ {Path(file_path).name}: {load_result}")
                    continue
                
                # Generate plot
                plot_result = plot_log2fc_histogram(ctx, output_dir)
                if "Error" in plot_result:
                    results.append(f"⚠️ {Path(file_path).name}: Plot failed - {plot_result}")
                else:
                    results.append(f"✅ {Path(file_path).name}: Success")
                
            except Exception as exc:
                results.append(f"❌ {Path(file_path).name}: Exception - {exc}")
        
        success_count = sum(1 for r in results if r.startswith("✅"))
        
        return f"Batch processing complete: {success_count}/{len(unique_files)} files successful\n" + "\n".join(results)
        
    except Exception as exc:
        logger.exception(f"Error in batch processing: {exc}")
        return f"Error in batch processing: {exc}"

@function_tool
def optimize_memory_usage(ctx: RunContextWrapper[PlotContext]) -> str:
    """
    Optimize memory usage and cleanup.
    """
    import gc
    import psutil
    import os
    
    try:
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clear matplotlib figure cache
        plt.close('all')
        
        # Clear any cached data
        if hasattr(ctx.context, 'plot_cache'):
            ctx.context.plot_cache.clear()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_freed = initial_memory - final_memory
        
        logger.info(f"Memory optimization complete: {memory_freed:.1f}MB freed")
        
        return f"Memory optimized: {memory_freed:.1f}MB freed, {collected} objects collected"
        
    except Exception as exc:
        logger.exception(f"Error optimizing memory: {exc}")
        return f"Error optimizing memory: {exc}"