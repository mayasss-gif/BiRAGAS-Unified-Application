import pandas as pd
import datetime
import os
from typing import Dict, List, Tuple
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import warnings
warnings.filterwarnings('ignore')


class PathwayReportContent:
    """
    Content generator for pathway analysis reports.
    This class generates story elements that can be used to build PDF reports.
    """
    
    def __init__(self):
        """Initialize the content generator with styles."""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles for the report."""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        # Section header style
        self.section_style = ParagraphStyle(
            'CustomSection',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        )
        
        # Subsection style
        self.subsection_style = ParagraphStyle(
            'CustomSubsection',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.darkgreen
        )
        
        # Body text style
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        )
        
        # Bullet style
        self.bullet_style = ParagraphStyle(
            'CustomBullet',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=3,
            leftIndent=20
        )
    
    def load_and_analyze_pathway_data(self, pathway_file_path: str, disease: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load pathway data and perform analysis to prepare for story generation.
        
        Args:
            pathway_file_path: Path to the pathway analysis results CSV file
            disease: Disease name for analysis
            
        Returns:
            Tuple of (DataFrame, analysis_dict)
        """
        # Load data
        df = pd.read_csv(pathway_file_path)
        
        # Analyze data
        analysis = self._analyze_pathway_data(df, disease)

        return df, analysis
    
    def _analyze_pathway_data(self, df: pd.DataFrame, disease: str) -> Dict:
        """
        Analyze the pathway data and extract key statistics and findings.
        
        Args:
            df: DataFrame containing pathway analysis results
            disease: Disease name for context
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'disease': disease,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_pathways': len(df),
            'upregulated_pathways': 0,
            'downregulated_pathways': 0,
            'pathway_sources': {},
            'top_significant': [],
            'top_upregulated': [],
            'top_downregulated': [],
            'clinical_relevance_summary': {},
            'functional_relevance_summary': {},
            'statistics': {},
            'df': df  # Include the DataFrame for later use
        }
        
        # Basic statistics
        if 'Regulation' in df.columns:
            analysis['upregulated_pathways'] = len(df[df['Regulation'] == 'Up'])
            analysis['downregulated_pathways'] = len(df[df['Regulation'] == 'Down'])
        
        # Pathway sources distribution
        if 'Pathway source' in df.columns:
            analysis['pathway_sources'] = df['Pathway source'].value_counts().to_dict()
        
        # Top significant pathways by p-value
        if 'p_value' in df.columns:
            top_sig = df.nsmallest(10, 'p_value')[['Pathway', 'p_value', 'fdr', 'Regulation', 'number_of_genes']]
            analysis['top_significant'] = top_sig.to_dict('records')
        
        # Top upregulated pathways
        if 'Regulation' in df.columns and 'p_value' in df.columns:
            up_df = df[df['Regulation'] == 'Up']
            if not up_df.empty:
                top_up = up_df.nsmallest(10, 'p_value')[['Pathway', 'p_value', 'fdr', 'number_of_genes']]
                analysis['top_upregulated'] = top_up.to_dict('records')
        
        # Top downregulated pathways
        if 'Regulation' in df.columns and 'p_value' in df.columns:
            down_df = df[df['Regulation'] == 'Down']
            if not down_df.empty:
                top_down = down_df.nsmallest(10, 'p_value')[['Pathway', 'p_value', 'fdr', 'number_of_genes']]
                analysis['top_downregulated'] = top_down.to_dict('records')
        
        # Statistical overview
        if 'p_value' in df.columns:
            analysis['statistics'] = {
                'mean_p_value': df['p_value'].mean(),
                'median_p_value': df['p_value'].median(),
                'min_p_value': df['p_value'].min(),
                'significant_pathways': len(df[df['p_value'] < 0.05]),
                'highly_significant': len(df[df['p_value'] < 0.01]),
                'mean_genes_per_pathway': df['number_of_genes'].mean() if 'number_of_genes' in df.columns else 0
            }
        
        # Analyze clinical and functional relevance
        if 'clinical_relevance' in df.columns:
            analysis['clinical_relevance_summary'] = self._analyze_text_field(df, 'clinical_relevance')
        
        if 'functional_relevance' in df.columns:
            analysis['functional_relevance_summary'] = self._analyze_text_field(df, 'functional_relevance')
        
        return analysis
    
    def _analyze_text_field(self, df: pd.DataFrame, field: str) -> Dict:
        """Analyze text fields for common themes and patterns."""
        summary = {
            'total_entries': len(df[df[field].notna()]),
            'common_themes': [],
            'key_terms': []
        }
        
        # Extract common themes from text
        if field in df.columns:
            text_data = df[field].dropna()
            if not text_data.empty:
                # Simple keyword analysis
                all_text = ' '.join(text_data.astype(str)).lower()
                common_terms = ['biomarker', 'prognosis', 'therapeutic', 'diagnosis', 'treatment', 
                              'survival', 'metastasis', 'progression', 'resistance', 'target']
                
                summary['key_terms'] = [term for term in common_terms if term in all_text]
        
        return summary
    
    def generate_title_page(self, disease: str, timestamp: str) -> List:
        """
        Generate title page story elements.
        
        Args:
            disease: Disease name
            timestamp: Analysis timestamp
            
        Returns:
            List of story elements for title page
        """
        story = []
        story.append(Paragraph(f"Pathway Analysis Report", self.title_style))
        story.append(Spacer(1, 10))
        # story.append(Paragraph(f"Disease: {disease}", self.body_style))
        # story.append(Spacer(1, 5))
        # story.append(Paragraph(f"Generated on: {timestamp}", self.body_style))
        # story.append(PageBreak())
        return story
    
    def generate_executive_summary(self, analysis: Dict) -> List:
        """
        Generate executive summary story elements.
        
        Args:
            analysis: Analysis results dictionary
            
        Returns:
            List of story elements for executive summary
        """
        story = []
        story.append(Paragraph("Executive Summary", self.section_style))
        story.append(Paragraph(self._generate_executive_summary_text(analysis), self.body_style))
        story.append(Spacer(1, 12))
        return story
    
    def _generate_executive_summary_text(self, analysis: Dict) -> str:
        """Generate executive summary text."""
        total = analysis['total_pathways']
        up = analysis['upregulated_pathways']
        down = analysis['downregulated_pathways']
        
        summary = f"""
        <b>Bottom Line Up Front:</b> This pathway enrichment analysis reveals {total:,} significantly 
        enriched biological pathways in {analysis['disease']} samples, with {up:,} upregulated and {down:,} 
        downregulated pathways. The analysis provides critical insights into the biological mechanisms 
        underlying {analysis['disease']} pathogenesis and identifies potential therapeutic targets.
        """
        
        if analysis.get('statistics'):
            stats = analysis['statistics']
            if 'highly_significant' in stats:
                summary += f" Among these, {stats['highly_significant']} pathways show highly significant enrichment (p < 0.01), "
                summary += f"indicating robust biological relevance to {analysis['disease']}."
        
        return summary
    
    def generate_key_findings(self, analysis: Dict) -> List:
        """
        Generate key findings story elements.
        
        Args:
            analysis: Analysis results dictionary
            
        Returns:
            List of story elements for key findings
        """
        story = []
        story.append(Paragraph("Key Findings", self.section_style))
        
        total = analysis['total_pathways']
        up = analysis['upregulated_pathways']
        down = analysis['downregulated_pathways']
        
        story.append(Paragraph(f"• Total pathways analyzed: {total:,} pathways", self.bullet_style))
        story.append(Paragraph(f"• Upregulated pathways: {up:,} ({up/total*100:.1f}% of total)", self.bullet_style))
        story.append(Paragraph(f"• Downregulated pathways: {down:,} ({down/total*100:.1f}% of total)", self.bullet_style))
        
        if analysis.get('statistics'):
            stats = analysis['statistics']
            if 'significant_pathways' in stats:
                story.append(Paragraph(f"• Statistically significant pathways: {stats['significant_pathways']} (p < 0.05)", self.bullet_style))
            if 'highly_significant' in stats:
                story.append(Paragraph(f"• Highly significant pathways: {stats['highly_significant']} (p < 0.01)", self.bullet_style))
            if 'mean_genes_per_pathway' in stats:
                story.append(Paragraph(f"• Average genes per pathway: {stats['mean_genes_per_pathway']:.1f} genes", self.bullet_style))
        
        story.append(Spacer(1, 12))
        return story
    
    def generate_pathway_source_analysis(self, analysis: Dict) -> List:
        """
        Generate pathway source analysis story elements.
        
        Args:
            analysis: Analysis results dictionary
            
        Returns:
            List of story elements for pathway source analysis
        """
        story = []
        story.append(Paragraph("Pathway Database Analysis", self.section_style))
        
        if analysis.get('pathway_sources'):
            story.append(Paragraph("<b>Pathway Sources Distribution:</b>", self.body_style))
            for source, count in analysis['pathway_sources'].items():
                percentage = count / analysis['total_pathways'] * 100
                story.append(Paragraph(f"• {source}: {count} pathways ({percentage:.1f}%)", self.bullet_style))
            story.append(Spacer(1, 6))
        
        story.append(Spacer(1, 12))
        return story
    
    def generate_top_pathways_analysis(self, analysis: Dict) -> List:
        """
        Generate top pathways analysis story elements.
        
        Args:
            analysis: Analysis results dictionary
            
        Returns:
            List of story elements for top pathways analysis
        """
        story = []
        story.append(Paragraph("Top Enriched Pathways", self.section_style))
        
        # Most significant pathways overall
        if analysis.get('top_significant'):
            story.append(Paragraph("Most Significant Pathways (All)", self.subsection_style))
            
            # Create table data
            table_data = [['Rank', 'Pathway', 'P-value', 'FDR', 'Regulation', 'Genes']]
            for i, pathway in enumerate(analysis['top_significant'][:10], 1):
                table_data.append([
                    str(i),
                    pathway['Pathway'][:40] + '...' if len(pathway['Pathway']) > 40 else pathway['Pathway'],
                    f"{pathway['p_value']:.2e}",
                    f"{pathway['fdr']:.2e}",
                    pathway.get('Regulation', 'N/A'),
                    str(pathway.get('number_of_genes', 'N/A'))
                ])
            
            # Create table
            table = Table(table_data, colWidths=[0.5*inch, 2.5*inch, 1*inch, 1*inch, 1*inch, 0.8*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (1, 1), (1, -1), 'LEFT'),  # Left align pathway names
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            story.append(table)
            story.append(Spacer(1, 12))
        
        # Top upregulated pathways
        if analysis.get('top_upregulated'):
            story.append(Paragraph("Top Upregulated Pathways", self.subsection_style))
            
            # Create table data
            table_data = [['Rank', 'Pathway', 'P-value', 'FDR', 'Genes']]
            for i, pathway in enumerate(analysis['top_upregulated'][:10], 1):
                table_data.append([
                    str(i),
                    pathway['Pathway'][:40] + '...' if len(pathway['Pathway']) > 40 else pathway['Pathway'],
                    f"{pathway['p_value']:.2e}",
                    f"{pathway['fdr']:.2e}",
                    str(pathway.get('number_of_genes', 'N/A'))
                ])
            
            # Create table
            table = Table(table_data, colWidths=[0.5*inch, 3*inch, 1*inch, 1*inch, 0.8*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightcoral),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (1, 1), (1, -1), 'LEFT'),  # Left align pathway names
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            story.append(table)
            story.append(Spacer(1, 12))
        
        # Top downregulated pathways
        if analysis.get('top_downregulated'):
            story.append(Paragraph("Top Downregulated Pathways", self.subsection_style))
            
            # Create table data
            table_data = [['Rank', 'Pathway', 'P-value', 'FDR', 'Genes']]
            for i, pathway in enumerate(analysis['top_downregulated'][:10], 1):
                table_data.append([
                    str(i),
                    pathway['Pathway'][:40] + '...' if len(pathway['Pathway']) > 40 else pathway['Pathway'],
                    f"{pathway['p_value']:.2e}",
                    f"{pathway['fdr']:.2e}",
                    str(pathway.get('number_of_genes', 'N/A'))
                ])
            
            # Create table
            table = Table(table_data, colWidths=[0.5*inch, 3*inch, 1*inch, 1*inch, 0.8*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (1, 1), (1, -1), 'LEFT'),  # Left align pathway names
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            story.append(table)
        
        story.append(Spacer(1, 12))
        return story
    
    def generate_clinical_insights(self, analysis: Dict, df: pd.DataFrame) -> List:
        """
        Generate clinical insights story elements.
        
        Args:
            analysis: Analysis results dictionary
            df: DataFrame containing pathway data
            
        Returns:
            List of story elements for clinical insights
        """
        story = []
        story.append(Paragraph("Clinical and Functional Insights", self.section_style))
        
        # Clinical relevance summary
        if analysis.get('clinical_relevance_summary'):
            summary = analysis['clinical_relevance_summary']
            story.append(Paragraph("<b>Clinical Relevance Analysis:</b>", self.body_style))
            story.append(Paragraph(f"• Total pathways with clinical annotations: {summary['total_entries']}", self.bullet_style))
            if summary.get('key_terms'):
                story.append(Paragraph(f"• Key clinical themes: {', '.join(summary['key_terms'])}", self.bullet_style))
            story.append(Spacer(1, 6))
        
        # Functional relevance summary
        if analysis.get('functional_relevance_summary'):
            summary = analysis['functional_relevance_summary']
            story.append(Paragraph("<b>Functional Relevance Analysis:</b>", self.body_style))
            story.append(Paragraph(f"• Total pathways with functional annotations: {summary['total_entries']}", self.bullet_style))
            if summary.get('key_terms'):
                story.append(Paragraph(f"• Key functional themes: {', '.join(summary['key_terms'])}", self.bullet_style))
            story.append(Spacer(1, 6))
        
        # Sample clinical insights from top pathways
        if 'clinical_relevance' in df.columns and 'Pathway' in df.columns:
            story.append(Paragraph("<b>Sample Clinical Insights from Top Pathways:</b>", self.body_style))
            
            # Get top 5 pathways with clinical relevance
            top_pathways = df[df['clinical_relevance'].notna()].head(5)
            for i, (_, pathway) in enumerate(top_pathways.iterrows(), 1):
                pathway_name = pathway['Pathway'][:50] + '...' if len(pathway['Pathway']) > 50 else pathway['Pathway']
                clinical_text = pathway['clinical_relevance'][:200] + '...' if len(str(pathway['clinical_relevance'])) > 200 else pathway['clinical_relevance']
                story.append(Paragraph(f"<b>{i}. {pathway_name}</b>", self.body_style))
                story.append(Paragraph(f"Clinical relevance: {clinical_text}", self.bullet_style))
                story.append(Spacer(1, 3))
        
        story.append(Spacer(1, 12))
        return story
    
    def generate_distribution_analysis(self, analysis: Dict, df: pd.DataFrame, output_dir: str = "reports") -> List:
        """
        Generate distribution analysis story elements with visualizations.
        
        Args:
            analysis: Analysis results dictionary
            df: DataFrame containing pathway results
            output_dir: Directory to save plots
            
        Returns:
            List of story elements for distribution analysis
        """
        story = []
        story.append(Paragraph("Distribution Analysis", self.section_style))
        
        # P-value distribution
        if analysis.get('statistics'):
            stats = analysis['statistics']
            story.append(Paragraph("<b>Statistical Overview:</b>", self.body_style))
            story.append(Paragraph(f"• Mean p-value: {stats.get('mean_p_value', 0):.4f}", self.bullet_style))
            story.append(Paragraph(f"• Median p-value: {stats.get('median_p_value', 0):.4f}", self.bullet_style))
            story.append(Paragraph(f"• Minimum p-value: {stats.get('min_p_value', 0):.2e}", self.bullet_style))
            story.append(Spacer(1, 6))
        
        # Add visualizations
        plot_paths = self.create_visualizations(df, output_dir)
        
        # Add plots to the story
        if plot_paths:
            story.append(Spacer(1, 12))
            story.append(Paragraph("<b>Visualizations:</b>", self.body_style))
            story.append(Spacer(1, 6))
            
            for plot_path in plot_paths:
                if os.path.exists(plot_path):
                    # Get plot title based on filename
                    plot_name = os.path.basename(plot_path)
                    if 'p_value_distribution' in plot_name:
                        plot_title = "Distribution of P-values"
                        img_width = 6*inch
                        img_height = 3.6*inch
                    elif 'regulation_distribution' in plot_name:
                        plot_title = "Distribution of Pathway Regulation"
                        img_width = 4.8*inch
                        img_height = 3.6*inch
                    elif 'source_distribution' in plot_name:
                        plot_title = "Distribution of Pathway Sources"
                        img_width = 4.8*inch
                        img_height = 3.6*inch
                    else:
                        continue
                    
                    story.append(Paragraph(f"<b>{plot_title}:</b>", self.body_style))
                    story.append(Spacer(1, 6))
                    
                    # Add the image with proper aspect ratio
                    img = Image(plot_path, width=img_width, height=img_height)
                    story.append(img)
                    story.append(Spacer(1, 12))
        
        story.append(Spacer(1, 12))
        return story
    
    def generate_technical_notes(self, df: pd.DataFrame) -> List:
        """
        Generate technical notes story elements.
        
        Args:
            df: DataFrame containing pathway results
            
        Returns:
            List of story elements for technical notes
        """
        story = []
        story.append(Paragraph("Technical Notes", self.section_style))
        
        story.append(Paragraph("<b>Analysis Parameters:</b>", self.body_style))
        story.append(Paragraph(f"• Total pathways analyzed: {len(df):,}", self.bullet_style))
        
        # Column analysis
        key_columns = ['Pathway', 'p_value', 'fdr', 'Regulation', 'number_of_genes', 'clinical_relevance']
        available_columns = [col for col in key_columns if col in df.columns]
        story.append(Paragraph(f"• Available metrics: {', '.join(available_columns)}", self.bullet_style))
        
        # Significance thresholds
        story.append(Paragraph("• Significance threshold: p-value < 0.05", self.bullet_style))
        story.append(Paragraph("• FDR threshold: FDR < 0.05", self.bullet_style))
        
        # Pathway sources
        if 'Pathway source' in df.columns:
            sources = df['Pathway source'].unique()
            story.append(Paragraph(f"• Pathway databases: {', '.join(sources)}", self.bullet_style))
        
        return story
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str) -> List[str]:
        """
        Create visualizations for the pathway report.
        
        Args:
            df: DataFrame containing pathway results
            output_dir: Directory to save plots
            
        Returns:
            List of plot file paths
        """
        plot_paths = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. P-value distribution
        if 'p_value' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df['p_value'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
            plt.axvline(x=0.01, color='orange', linestyle='--', alpha=0.7, label='p=0.01')
            plt.xlabel('P-value')
            plt.ylabel('Number of Pathways')
            plt.title('Distribution of P-values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(output_dir, 'p_value_distribution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        # 2. Regulation distribution
        if 'Regulation' in df.columns:
            plt.figure(figsize=(8, 6))
            regulation_counts = df['Regulation'].value_counts()
            colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            plt.pie(regulation_counts.values, labels=regulation_counts.index, autopct='%1.1f%%', 
                   colors=colors_list[:len(regulation_counts)])
            plt.title('Distribution of Pathway Regulation')
            
            plot_path = os.path.join(output_dir, 'regulation_distribution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        # 3. Pathway source distribution
        if 'Pathway source' in df.columns:
            plt.figure(figsize=(8, 6))
            source_counts = df['Pathway source'].value_counts()
            colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', 
                   colors=colors_list[:len(source_counts)])
            plt.title('Distribution of Pathway Sources')
            
            plot_path = os.path.join(output_dir, 'source_distribution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        return plot_paths
    
    def generate_complete_report_story(self, analysis: Dict, df: pd.DataFrame, 
                                     output_dir: str = "reports") -> List:
        """
        Generate complete pathway report story elements.
        
        Args:
            analysis: Analysis results dictionary
            df: DataFrame containing pathway results
            output_dir: Directory to save plots
            
        Returns:
            List of all story elements for the complete report
        """
        story = []
        
        # Title page
        story.extend(self.generate_title_page(analysis['disease'], analysis['timestamp']))
        
        # Executive Summary
        story.extend(self.generate_executive_summary(analysis))
        
        # Key Findings
        story.extend(self.generate_key_findings(analysis))
        
        # Pathway Source Analysis
        story.extend(self.generate_pathway_source_analysis(analysis))
        
        # Top Pathways Analysis
        story.extend(self.generate_top_pathways_analysis(analysis))
        
        # Clinical Insights
        story.extend(self.generate_clinical_insights(analysis, df))
        
        # Distribution Analysis with visualizations
        story.extend(self.generate_distribution_analysis(analysis, df, output_dir))
        
        # Technical Notes
        story.extend(self.generate_technical_notes(df))
        
        return story
    

def pathway_report_content(pathway_file_path, disease, output_dir="reports"):
    """
    Generate pathway report content.
    
    Args:
        pathway_file_path: Path to the pathway file
        disease: Disease name
        output_dir: Directory to save plots
        
    Returns:
        List of story elements for the pathway report
    """
    content_gen = PathwayReportContent()
    df, analysis = content_gen.load_and_analyze_pathway_data(pathway_file_path, disease)
    story = content_gen.generate_complete_report_story(analysis, df, output_dir)
    return story 