import pandas as pd
import numpy as np
import datetime
import os
from typing import Dict, List, Tuple, Optional
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import warnings
warnings.filterwarnings('ignore')

from .config import REPORTS_DIR

class DEGReportContent:
    """
    Content generator for DEG analysis reports.
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
    
    def load_and_analyze_data(self, deg_file_path: str, disease: str) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
        """
        Load DEG data and perform analysis to prepare for story generation.
        
        Args:
            deg_file_path: Path to the DEG analysis results CSV file
            disease: Disease name for analysis
            
        Returns:
            Tuple of (DataFrame, analysis_dict, pathway_insights, clinical_implications)
        """
        # Load data
        df = pd.read_csv(deg_file_path)
        
        # Analyze data
        analysis = self._analyze_deg_data(df, disease)
        pathway_insights = self._generate_pathway_insights(analysis)
        clinical_implications = self._generate_clinical_implications(analysis, pathway_insights)

        return df, analysis, pathway_insights, clinical_implications
    
    def _analyze_deg_data(self, df: pd.DataFrame, disease: str) -> Dict:
        """
        Analyze the DEG data and extract key statistics and findings.
        
        Args:
            df: DataFrame containing DEG analysis results
            disease: Disease name for context
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'disease': disease,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_genes': len(df),
            'upregulated': 0,
            'downregulated': 0,
            'tier_distribution': {},
            'top_upregulated': [],
            'top_downregulated': [],
            'top_significant': [],
            'statistics': {},
            'pathway_insights': {},
            'clinical_implications': {},
            'df': df  # Include the DataFrame for later use
        }
        
        # Basic statistics
        if 'Patient_LFC_mean' in df.columns:
            lfc_col = 'Patient_LFC_mean'
        elif 'Cohort_LFC_mean' in df.columns:
            lfc_col = 'Cohort_LFC_mean'
        else:
            # Find any log2FC column
            lfc_cols = [col for col in df.columns if 'log2fc' in col.lower()]
            lfc_col = lfc_cols[0] if lfc_cols else None
        
        if lfc_col:
            analysis['upregulated'] = len(df[df[lfc_col] > 0])
            analysis['downregulated'] = len(df[df[lfc_col] < 0])
            
            # Statistical overview
            analysis['statistics'] = {
                'mean_lfc_up': df[df[lfc_col] > 0][lfc_col].mean(),
                'median_lfc_up': df[df[lfc_col] > 0][lfc_col].median(),
                'max_lfc_up': df[df[lfc_col] > 0][lfc_col].max(),
                'mean_lfc_down': df[df[lfc_col] < 0][lfc_col].mean(),
                'median_lfc_down': df[df[lfc_col] < 0][lfc_col].median(),
                'min_lfc_down': df[df[lfc_col] < 0][lfc_col].min(),
                'total_significant': len(df)
            }
        
        # Tier distribution
        if 'Tier' in df.columns:
            analysis['tier_distribution'] = df['Tier'].value_counts().to_dict()
        
        # Top genes by fold change
        if lfc_col:
            # Top upregulated
            top_up = df[df[lfc_col] > 0].nlargest(10, lfc_col)[['Gene', lfc_col]]
            analysis['top_upregulated'] = top_up.to_dict('records')
            
            # Top downregulated
            top_down = df[df[lfc_col] < 0].nsmallest(10, lfc_col)[['Gene', lfc_col]]
            analysis['top_downregulated'] = top_down.to_dict('records')
        
        # Top genes by significance (if p-value column exists)
        pval_cols = [col for col in df.columns if 'p-value' in col.lower() or 'padj' in col.lower()]
        if pval_cols:
            pval_col = pval_cols[0]
            top_sig = df.nsmallest(10, pval_col)[['Gene', pval_col, lfc_col if lfc_col else '']]
            analysis['top_significant'] = top_sig.to_dict('records')
        
        # Additional scores if available
        if 'Composite_Evidence_Score' in df.columns:
            analysis['top_evidence'] = df.nlargest(10, 'Composite_Evidence_Score')[['Gene', 'Composite_Evidence_Score']].to_dict('records')
        
        if 'GC_score' in df.columns:
            analysis['top_genecards'] = df[df['GC_score'].notna()].nlargest(10, 'GC_score')[['Gene', 'GC_score']].to_dict('records')
        
        return analysis
    
    def _generate_pathway_insights(self, analysis: Dict) -> Dict:
        """
        Generate pathway insights based on the DEG analysis.
        
        Args:
            analysis: Analysis results dictionary
            
        Returns:
            Dictionary containing pathway insights
        """
        insights = {
            'upregulated_pathways': [],
            'downregulated_pathways': [],
            'key_biological_processes': [],
            'therapeutic_targets': []
        }
        
        # Analyze top upregulated genes for pathway patterns
        up_genes = [gene['Gene'] for gene in analysis.get('top_upregulated', [])]
        
        # Common pathway keywords
        pathway_keywords = {
            'inflammation': ['IL', 'TNF', 'CXCL', 'CCL', 'IFN', 'NFKB', 'STAT'],
            'immune_response': ['CD', 'HLA', 'MHC', 'TCR', 'BCR', 'IG'],
            'extracellular_matrix': ['COL', 'FN', 'LAM', 'FMOD', 'SPRR'],
            'cell_proliferation': ['MKI', 'PCNA', 'MCM', 'CDK', 'CCND'],
            'apoptosis': ['BCL', 'CASP', 'BAX', 'BAK', 'FAS'],
            'metabolism': ['G6PD', 'HK', 'PFK', 'LDH', 'ACO'],
            'stress_response': ['HSP', 'HSPA', 'HSPB', 'HSPC', 'HSPD'],
            'dna_repair': ['BRCA', 'PARP', 'ATM', 'ATR', 'RPA']
        }
        
        # Analyze upregulated pathways
        for pathway, keywords in pathway_keywords.items():
            matches = [gene for gene in up_genes if any(keyword in gene.upper() for keyword in keywords)]
            if matches:
                insights['upregulated_pathways'].append({
                    'pathway': pathway.replace('_', ' ').title(),
                    'genes': matches,
                    'count': len(matches)
                })
        
        # Analyze downregulated pathways
        down_genes = [gene['Gene'] for gene in analysis.get('top_downregulated', [])]
        for pathway, keywords in pathway_keywords.items():
            matches = [gene for gene in down_genes if any(keyword in gene.upper() for keyword in keywords)]
            if matches:
                insights['downregulated_pathways'].append({
                    'pathway': pathway.replace('_', ' ').title(),
                    'genes': matches,
                    'count': len(matches)
                })
        
        return insights
    
    def _generate_clinical_implications(self, analysis: Dict, pathway_insights: Dict) -> Dict:
        """
        Generate clinical implications based on the analysis.
        
        Args:
            analysis: Analysis results dictionary
            pathway_insights: Pathway insights dictionary
            
        Returns:
            Dictionary containing clinical implications
        """
        implications = {
            'disease_mechanisms': [],
            'biomarkers': [],
            'therapeutic_targets': [],
            'monitoring_recommendations': [],
            'research_directions': []
        }
        
        # Disease mechanisms based on pathway analysis
        up_pathways = [p['pathway'] for p in pathway_insights.get('upregulated_pathways', [])]
        down_pathways = [p['pathway'] for p in pathway_insights.get('downregulated_pathways', [])]
        
        if 'Inflammation' in up_pathways:
            implications['disease_mechanisms'].append("Active inflammatory response with cytokine upregulation")
        if 'Immune Response' in up_pathways:
            implications['disease_mechanisms'].append("Enhanced immune cell activation and signaling")
        if 'Extracellular Matrix' in up_pathways:
            implications['disease_mechanisms'].append("Tissue remodeling and fibrosis development")
        
        if 'Immune Response' in down_pathways:
            implications['disease_mechanisms'].append("Immune suppression and dysfunction")
        if 'Dna Repair' in down_pathways:
            implications['disease_mechanisms'].append("Impaired DNA repair mechanisms")
        
        # Biomarkers from top genes
        top_genes = [gene['Gene'] for gene in analysis.get('top_upregulated', [])[:5]]
        implications['biomarkers'] = top_genes
        
        # Therapeutic targets
        if analysis.get('top_evidence'):
            evidence_genes = [gene['Gene'] for gene in analysis.get('top_evidence', [])[:5]]
            implications['therapeutic_targets'] = evidence_genes
        
        # Monitoring recommendations
        implications['monitoring_recommendations'] = [
            "Monitor inflammatory markers and cytokine levels",
            "Assess immune cell function and counts",
            "Track tissue remodeling markers",
            "Evaluate DNA repair capacity"
        ]
        
        # Research directions
        implications['research_directions'] = [
            "Functional validation of top hit genes",
            "Pathway enrichment analysis for systematic understanding",
            "Longitudinal studies to track expression changes",
            "Correlation with clinical parameters"
        ]
        
        return implications
    
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
        story.append(Paragraph(f"Comprehensive DEG Report", self.title_style))
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
        total = analysis['total_genes']
        up = analysis['upregulated']
        down = analysis['downregulated']
        ratio = up / down if down > 0 else float('inf')
        
        summary = f"""
        <b>Bottom Line Up Front:</b> This transcriptome analysis reveals {total:,} statistically significant 
        differentially expressed genes (DEGs) in {analysis['disease']} samples, with {up:,} upregulated and {down:,} 
        downregulated genes. The data demonstrates a pronounced biological signature with a {ratio:.2f}:1 ratio of 
        upregulated to downregulated genes.
        """
        
        if analysis.get('statistics'):
            stats = analysis['statistics']
            if 'max_lfc_up' in stats and not pd.isna(stats['max_lfc_up']):
                max_up = 2 ** stats['max_lfc_up']
                summary += f" The most dramatic changes include genes showing up to {max_up:.0f}-fold upregulation, "
                summary += f"indicating significant biological pathway alterations characteristic of {analysis['disease']}."
        
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
        
        total = analysis['total_genes']
        up = analysis['upregulated']
        down = analysis['downregulated']
        
        story.append(Paragraph(f"• Total DEGs identified: {total:,} genes", self.bullet_style))
        story.append(Paragraph(f"• Upregulated genes: {up:,} ({up/total*100:.1f}% of total DEGs)", self.bullet_style))
        story.append(Paragraph(f"• Downregulated genes: {down:,} ({down/total*100:.1f}% of total DEGs)", self.bullet_style))
        
        if analysis.get('statistics'):
            stats = analysis['statistics']
            if 'max_lfc_up' in stats and not pd.isna(stats['max_lfc_up']):
                max_up = 2 ** stats['max_lfc_up']
                min_down = 2 ** abs(stats['min_lfc_down']) if 'min_lfc_down' in stats and not pd.isna(stats['min_lfc_down']) else 0
                story.append(Paragraph(f"• Expression range: 1.0 to {max_up:.0f}-fold upregulation; 1.0 to {min_down:.0f}-fold downregulation", self.bullet_style))
        
        story.append(Spacer(1, 12))
        return story
    
    def generate_detailed_analysis(self, analysis: Dict) -> List:
        """
        Generate detailed analysis story elements.
        
        Args:
            analysis: Analysis results dictionary
            
        Returns:
            List of story elements for detailed analysis
        """
        story = []
        story.append(Paragraph("Detailed Analysis", self.section_style))
        
        # Upregulated genes
        story.append(Paragraph("Upregulated Genes", self.subsection_style))
        story.extend(self._generate_gene_analysis(analysis, 'upregulated'))
        story.append(Spacer(1, 12))
        
        # Downregulated genes
        story.append(Paragraph("Downregulated Genes", self.subsection_style))
        story.extend(self._generate_gene_analysis(analysis, 'downregulated'))
        story.append(Spacer(1, 12))
        
        return story
    
    def _generate_gene_analysis(self, analysis: Dict, direction: str) -> List:
        """Generate gene analysis section for upregulated or downregulated genes."""
        elements = []
        
        if direction == 'upregulated':
            genes = analysis.get('top_upregulated', [])
            stats = analysis.get('statistics', {})
            mean_lfc = stats.get('mean_lfc_up', 0)
            median_lfc = stats.get('median_lfc_up', 0)
            max_lfc = stats.get('max_lfc_up', 0)
        else:
            genes = analysis.get('top_downregulated', [])
            stats = analysis.get('statistics', {})
            mean_lfc = stats.get('mean_lfc_down', 0)
            median_lfc = stats.get('median_lfc_down', 0)
            max_lfc = stats.get('min_lfc_down', 0)
        
        # Statistical overview
        elements.append(Paragraph(f"<b>Statistical Overview:</b>", self.body_style))
        elements.append(Paragraph(f"• Mean log2 fold change: {mean_lfc:.2f} (approximately {2**mean_lfc:.1f}-fold average change)", self.bullet_style))
        elements.append(Paragraph(f"• Median log2 fold change: {median_lfc:.2f} (approximately {2**median_lfc:.1f}-fold median change)", self.bullet_style))
        elements.append(Paragraph(f"• Range: {max_lfc:.2f} log2 fold change ({2**abs(max_lfc):.0f}-fold change)", self.bullet_style))
        elements.append(Spacer(1, 6))
        
        # Top genes table
        if genes:
            elements.append(Paragraph(f"<b>Top 5 Most {direction.title()} Genes:</b>", self.body_style))
            
            # Create table data
            table_data = [['Rank', 'Gene', 'Log2FC', 'Fold Change']]
            for i, gene in enumerate(genes[:5], 1):
                fold_change = 2 ** abs(gene.get('Patient_LFC_mean', gene.get('Cohort_LFC_mean', 0)))
                table_data.append([
                    str(i),
                    gene['Gene'],
                    f"{gene.get('Patient_LFC_mean', gene.get('Cohort_LFC_mean', 0)):.2f}",
                    f"{fold_change:.0f}-fold"
                ])
            
            # Create table
            table = Table(table_data, colWidths=[0.5*inch, 1.5*inch, 1*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))
        
        # Most Significant Genes table (only for upregulated)
        if direction == 'upregulated' and analysis.get('df') is not None:
            elements.append(Paragraph(f"<b>Most Significant Upregulated Genes:</b>", self.body_style))
            elements.append(Paragraph("Genes with high fold change and strong evidence scores (GC_score, JL_score, PPI_Degree)", self.body_style))
            elements.append(Spacer(1, 6))
            
            # Get the most significant upregulated genes
            significant_genes = self._get_most_significant_genes(analysis['df'], direction)
            
            if significant_genes:
                # Create table data
                table_data = [['Rank', 'Gene', 'Log2FC', 'GC_Score', 'JL_Score', 'PPI_Degree']]
                for i, gene in enumerate(significant_genes[:5], 1):
                    table_data.append([
                        str(i),
                        gene['Gene'],
                        f"{gene.get('Patient_LFC_mean', gene.get('Cohort_LFC_mean', 0)):.2f}",
                        f"{gene.get('GC_score', 'N/A')}" if pd.notna(gene.get('GC_score')) else 'N/A',
                        f"{gene.get('JL_score', 'N/A')}" if pd.notna(gene.get('JL_score')) else 'N/A',
                        f"{gene.get('PPI_Degree', 'N/A')}" if pd.notna(gene.get('PPI_Degree')) else 'N/A'
                    ])
                
                # Create table
                table = Table(table_data, colWidths=[0.5*inch, 1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(table)
            else:
                elements.append(Paragraph("No significant genes found with complete evidence scores.", self.bullet_style))
        
        # Most Significant Genes table (only for downregulated)
        if direction == 'downregulated' and analysis.get('df') is not None:
            elements.append(Paragraph(f"<b>Most Significant Downregulated Genes:</b>", self.body_style))
            elements.append(Paragraph("Genes with high negative fold change and strong evidence scores (GC_score, JL_score, PPI_Degree)", self.body_style))
            elements.append(Spacer(1, 6))
            
            # Get the most significant downregulated genes
            significant_genes = self._get_most_significant_genes(analysis['df'], direction)
            
            if significant_genes:
                # Create table data
                table_data = [['Rank', 'Gene', 'Log2FC', 'GC_Score', 'JL_Score', 'PPI_Degree']]
                for i, gene in enumerate(significant_genes[:5], 1):
                    table_data.append([
                        str(i),
                        gene['Gene'],
                        f"{gene.get('Patient_LFC_mean', gene.get('Cohort_LFC_mean', 0)):.2f}",
                        f"{gene.get('GC_score', 'N/A')}" if pd.notna(gene.get('GC_score')) else 'N/A',
                        f"{gene.get('JL_score', 'N/A')}" if pd.notna(gene.get('JL_score')) else 'N/A',
                        f"{gene.get('PPI_Degree', 'N/A')}" if pd.notna(gene.get('PPI_Degree')) else 'N/A'
                    ])
                
                # Create table
                table = Table(table_data, colWidths=[0.5*inch, 1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightcoral),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(table)
            else:
                elements.append(Paragraph("No significant genes found with complete evidence scores.", self.bullet_style))
        
        return elements
    
    def _get_most_significant_genes(self, df: pd.DataFrame, direction: str) -> List[Dict]:
        """
        Get the most significant genes based on LFC and evidence scores.
        
        Args:
            df: DataFrame containing gene data
            direction: 'upregulated' or 'downregulated'
            
        Returns:
            List of gene dictionaries sorted by significance
        """
        # Determine LFC column
        if 'Patient_LFC_mean' in df.columns:
            lfc_col = 'Patient_LFC_mean'
        elif 'Cohort_LFC_mean' in df.columns:
            lfc_col = 'Cohort_LFC_mean'
        else:
            return []
        
        # Filter by direction
        if direction == 'upregulated':
            filtered_df = df[df[lfc_col] > 0].copy()
        else:
            filtered_df = df[df[lfc_col] < 0].copy()
        
        if filtered_df.empty:
            return []
        
        # Create a composite score for ranking
        # Handle missing values by filling with median or 0
        filtered_df['GC_score_filled'] = filtered_df['GC_score'].fillna(filtered_df['GC_score'].median() if pd.notna(filtered_df['GC_score'].median()) else 0)
        filtered_df['JL_score_filled'] = filtered_df['JL_score'].fillna(filtered_df['JL_score'].median() if pd.notna(filtered_df['JL_score'].median()) else 0)
        filtered_df['PPI_Degree_filled'] = filtered_df['PPI_Degree'].fillna(filtered_df['PPI_Degree'].median() if pd.notna(filtered_df['PPI_Degree'].median()) else 0)
        
        # Normalize scores to 0-1 range for fair comparison
        if filtered_df['GC_score_filled'].max() > 0:
            filtered_df['GC_score_norm'] = filtered_df['GC_score_filled'] / filtered_df['GC_score_filled'].max()
        else:
            filtered_df['GC_score_norm'] = 0
            
        if filtered_df['JL_score_filled'].max() > 0:
            filtered_df['JL_score_norm'] = filtered_df['JL_score_filled'] / filtered_df['JL_score_filled'].max()
        else:
            filtered_df['JL_score_norm'] = 0
            
        if filtered_df['PPI_Degree_filled'].max() > 0:
            filtered_df['PPI_Degree_norm'] = filtered_df['PPI_Degree_filled'] / filtered_df['PPI_Degree_filled'].max()
        else:
            filtered_df['PPI_Degree_norm'] = 0
        
        # Create composite significance score
        # Weight: LFC (40%), GC_score (25%), JL_score (20%), PPI_Degree (15%)
        filtered_df['significance_score'] = (
            0.4 * abs(filtered_df[lfc_col]) / abs(filtered_df[lfc_col]).max() +
            0.25 * filtered_df['GC_score_norm'] +
            0.20 * filtered_df['JL_score_norm'] +
            0.15 * filtered_df['PPI_Degree_norm']
        )
        
        # Sort by significance score and return top genes
        significant_genes = filtered_df.nlargest(10, 'significance_score')[['Gene', lfc_col, 'GC_score', 'JL_score', 'PPI_Degree']].to_dict('records')
        
        return significant_genes
    
    def generate_pathway_analysis(self, pathway_insights: Dict) -> List:
        """
        Generate pathway analysis story elements.
        
        Args:
            pathway_insights: Pathway insights dictionary
            
        Returns:
            List of story elements for pathway analysis
        """
        story = []
        story.append(Paragraph("Biological Pathways and Clinical Implications", self.section_style))
        
        # Upregulated pathways
        if pathway_insights.get('upregulated_pathways'):
            story.append(Paragraph("<b>Major Upregulated Pathways:</b>", self.body_style))
            for pathway in pathway_insights['upregulated_pathways']:
                story.append(Paragraph(f"• {pathway['pathway']} ({pathway['count']} genes: {', '.join(pathway['genes'][:3])})", self.bullet_style))
            story.append(Spacer(1, 6))
        
        # Downregulated pathways
        if pathway_insights.get('downregulated_pathways'):
            story.append(Paragraph("<b>Major Downregulated Pathways:</b>", self.body_style))
            for pathway in pathway_insights['downregulated_pathways']:
                story.append(Paragraph(f"• {pathway['pathway']} ({pathway['count']} genes: {', '.join(pathway['genes'][:3])})", self.bullet_style))
            story.append(Spacer(1, 6))
        
        story.append(Spacer(1, 12))
        return story
    
    def generate_clinical_significance(self, clinical_implications: Dict) -> List:
        """
        Generate clinical significance story elements.
        
        Args:
            clinical_implications: Clinical implications dictionary
            
        Returns:
            List of story elements for clinical significance
        """
        story = []
        story.append(Paragraph("Clinical Significance", self.section_style))
        
        # Disease mechanisms
        if clinical_implications.get('disease_mechanisms'):
            story.append(Paragraph("<b>Disease Mechanisms:</b>", self.body_style))
            for mechanism in clinical_implications['disease_mechanisms']:
                story.append(Paragraph(f"• {mechanism}", self.bullet_style))
            story.append(Spacer(1, 6))
        
        # Biomarkers
        if clinical_implications.get('biomarkers'):
            story.append(Paragraph("<b>Potential Biomarkers:</b>", self.body_style))
            story.append(Paragraph(f"• {', '.join(clinical_implications['biomarkers'])}", self.bullet_style))
            story.append(Spacer(1, 6))
        
        # Therapeutic targets
        if clinical_implications.get('therapeutic_targets'):
            story.append(Paragraph("<b>Therapeutic Targets:</b>", self.body_style))
            story.append(Paragraph(f"• {', '.join(clinical_implications['therapeutic_targets'])}", self.body_style))
        
        story.append(Spacer(1, 12))
        return story
    
    def generate_distribution_analysis(self, analysis: Dict, df: pd.DataFrame, output_dir: str = "reports") -> List:
        """
        Generate distribution analysis story elements with visualizations.
        
        Args:
            analysis: Analysis results dictionary
            df: DataFrame containing DEG results
            output_dir: Directory to save plots
            
        Returns:
            List of story elements for distribution analysis
        """
        story = []
        story.append(Paragraph("Distribution Analysis", self.section_style))
        
        # Tier distribution
        if analysis.get('tier_distribution'):
            story.append(Paragraph("<b>Gene Tier Distribution:</b>", self.body_style))
            for tier, count in analysis['tier_distribution'].items():
                percentage = count / analysis['total_genes'] * 100
                story.append(Paragraph(f"• {tier}: {count} genes ({percentage:.1f}%)", self.bullet_style))
            story.append(Spacer(1, 6))
        
        # Fold change categories
        if analysis.get('statistics'):
            stats = analysis['statistics']
            story.append(Paragraph("<b>Fold Change Categories:</b>", self.body_style))
            story.append(Paragraph("• Analysis of gene expression magnitude distribution", self.bullet_style))
        
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
                    if 'fold_change_distribution' in plot_name:
                        plot_title = "Distribution of Log2 Fold Changes"
                        # figsize was (10, 6) - maintain 5:3 ratio
                        img_width = 6*inch
                        img_height = 3.6*inch
                    elif 'tier_distribution' in plot_name:
                        plot_title = "Distribution of Gene Tiers"
                        # figsize was (8, 6) - maintain 4:3 ratio
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
    
    def generate_recommendations(self, clinical_implications: Dict) -> List:
        """
        Generate recommendations story elements.
        
        Args:
            clinical_implications: Clinical implications dictionary
            
        Returns:
            List of story elements for recommendations
        """
        story = []
        story.append(Paragraph("Recommendations", self.section_style))
        
        # Immediate clinical focus
        if clinical_implications.get('monitoring_recommendations'):
            story.append(Paragraph("<b>Immediate Clinical Focus:</b>", self.body_style))
            for i, rec in enumerate(clinical_implications['monitoring_recommendations'], 1):
                story.append(Paragraph(f"{i}. {rec}", self.bullet_style))
            story.append(Spacer(1, 6))
        
        # Future research directions
        if clinical_implications.get('research_directions'):
            story.append(Paragraph("<b>Future Research Directions:</b>", self.body_style))
            for i, direction in enumerate(clinical_implications['research_directions'], 1):
                story.append(Paragraph(f"{i}. {direction}", self.bullet_style))
        
        story.append(Spacer(1, 12))
        return story
    
    def generate_technical_notes(self, df: pd.DataFrame) -> List:
        """
        Generate technical notes story elements.
        
        Args:
            df: DataFrame containing DEG results
            
        Returns:
            List of story elements for technical notes
        """
        story = []
        story.append(Paragraph("Technical Notes", self.section_style))
        
        story.append(Paragraph("<b>Analysis Parameters:</b>", self.body_style))
        story.append(Paragraph(f"• Total genes analyzed: {len(df):,}", self.bullet_style))
        
        # Column analysis
        key_columns = ['Gene', 'Patient_LFC_mean', 'Cohort_LFC_mean', 'Tier', 'Composite_Evidence_Score']
        available_columns = [col for col in key_columns if col in df.columns]
        story.append(Paragraph(f"• Available metrics: {', '.join(available_columns)}", self.bullet_style))
        
        # Significance thresholds
        story.append(Paragraph("• Significance threshold: p-value < 0.05", self.bullet_style))
        story.append(Paragraph("• Effect size threshold: |log2FC| ≥ 1.0 (≥2-fold change)", self.bullet_style))
        
        return story
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str) -> List[str]:
        """
        Create visualizations for the report.
        
        Args:
            df: DataFrame containing DEG results
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
        
        # 1. Fold change distribution
        if 'Patient_LFC_mean' in df.columns or 'Cohort_LFC_mean' in df.columns:
            lfc_col = 'Patient_LFC_mean' if 'Patient_LFC_mean' in df.columns else 'Cohort_LFC_mean'
            
            plt.figure(figsize=(10, 6))
            plt.hist(df[lfc_col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            plt.xlabel('Log2 Fold Change')
            plt.ylabel('Number of Genes')
            plt.title('Distribution of Log2 Fold Changes')
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(output_dir, 'fold_change_distribution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        # 2. Tier distribution
        if 'Tier' in df.columns:
            plt.figure(figsize=(8, 6))
            tier_counts = df['Tier'].value_counts()
            colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            plt.pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%', colors=colors_list[:len(tier_counts)])
            plt.title('Distribution of Gene Tiers')
            
            plot_path = os.path.join(output_dir, 'tier_distribution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        return plot_paths
    
    def generate_complete_report_story(self, analysis: Dict, pathway_insights: Dict, 
                                     clinical_implications: Dict, df: pd.DataFrame, 
                                     output_dir: str = REPORTS_DIR) -> List:
        """
        Generate complete report story elements.
        
        Args:
            analysis: Analysis results dictionary
            pathway_insights: Pathway insights dictionary
            clinical_implications: Clinical implications dictionary
            df: DataFrame containing DEG results
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
        
        # Detailed Analysis
        story.extend(self.generate_detailed_analysis(analysis))
        
        # Biological Pathways
        story.extend(self.generate_pathway_analysis(pathway_insights))
        
        # Clinical Significance
        story.extend(self.generate_clinical_significance(clinical_implications))
        
        # Distribution Analysis with visualizations
        story.extend(self.generate_distribution_analysis(analysis, df, output_dir))
        
        # Recommendations
        story.extend(self.generate_recommendations(clinical_implications))
        
        # Technical Notes
        story.extend(self.generate_technical_notes(df))
        
        return story
    

def deg_report_content(deg_file_path, disease, output_dir="reports"):
    """
    Generate DEG report content.
    
    Args:
        deg_file_path: Path to the DEG file
        disease: Disease name
        output_dir: Directory to save plots
        
    Returns:
        List of story elements for the DEG report
    """
    content_gen = DEGReportContent()
    df, analysis, pathway_insights, clinical_implications = content_gen.load_and_analyze_data(
        deg_file_path, disease
    )
    story = content_gen.generate_complete_report_story(
        analysis, pathway_insights, clinical_implications, df, output_dir
    )
    return story