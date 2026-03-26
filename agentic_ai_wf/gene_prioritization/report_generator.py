import pandas as pd
import numpy as np
import os
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import warnings
warnings.filterwarnings('ignore')

# Import the new DEG report content generator
from .DEG_report import DEGReportContent
# from agentic_ai_wf.config.global_config import REPORT_DIR

class DEGReportGenerator:
    """
    Comprehensive PDF report generator for Differential Expression Gene (DEG) analysis results.
    """
    
    def __init__(self):
        """
        Initialize the report generator.
        
        """
        
        
        # Initialize the content generator
        self.content_generator = DEGReportContent()
        
        # Set up styles
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
    
    def add_header(self, canvas, doc):
        """Draws the header with the logo at the top right corner on the first page."""
        logo_image_file = r"C:\Ayass Bio Work\Agentic_AI_ABS\GenePrioritization\agentic_ai_abs\gene_prioritization\datasets\logo.jpg"
        if os.path.exists(logo_image_file):
            canvas.drawImage(logo_image_file, x=A4[0] - 2.1*inch - 10, y=A4[1] - 0.7*inch - 10, 
                            width=2.1*inch, height=0.7*inch, preserveAspectRatio=True)
    
    def _generate_patient_info(self, patient_name: str, disease: str) -> List:
        """
        Generate patient information section for the report.
        
        Args:
            patient_name: Patient name (optional)
            disease: Disease name
            timestamp: Report generation timestamp
            
        Returns:
            List of story elements for patient information
        """
        patient_info = []
        
        # Patient information header
        patient_info.append(Paragraph("Patient Information", self.section_style))
        patient_info.append(Spacer(1, 12))
        
        # Create patient info table
        table_data = []
        
        if patient_name:
            table_data.append(["Patient Name:", patient_name])
        
        table_data.extend([
            ["Disease:", disease],
            ["Report Date:", datetime.datetime.now().strftime("%B %d, %Y")],
            ["Report Time:", datetime.datetime.now().strftime("%I:%M %p")],
        ])
        
        # Create table
        table = Table(table_data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        patient_info.append(table)
        patient_info.append(Spacer(1, 20))
        
        return patient_info
    
    def generate_pdf_report(self, story_lists: Union[List, List[List]], 
                          disease: str, output_dir: str = "reports", 
                          patient_name: str = None) -> str:
        """
        Generate a PDF report using provided story elements.
        
        Args:
            story_lists: Story elements to include in the report. Can be:
                        - List: Single list of story elements
                        - List[List]: Multiple lists of story elements to combine
            disease: Disease name for the report
            output_dir: Directory to save the report
            patient_name: Optional patient name for the report
            
        Returns:
            Path to the generated PDF report
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate PDF
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(output_dir, f"{disease.replace(' ', '_')}_Transcriptome_Report.pdf")
        
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=30, leftMargin=30, bottomMargin=18)
        
        # Determine story elements
        if isinstance(story_lists[0], list):
            # Multiple lists provided - combine them
            story = []
            for story_list in story_lists:
                story.extend(story_list)
        else:
            # Single list provided
            story = story_lists
        
        # Add title page
        title = []
        title.append(Paragraph(f"{disease} Transcriptome Analysis<br/>", self.title_style))

        # Add patient information at the beginning
        patient_info = self._generate_patient_info(patient_name, disease)
        full_story = title + patient_info + story
        
        # Build PDF
        doc.build(full_story, onFirstPage=lambda canvas, doc: self.add_header(canvas, doc))
        
        return pdf_path
    
    

def generate_report(story_lists: Union[List, List[List]], 
                    disease: str, output_dir: str = "REPORT_DIR", 
                    patient_name: str = None) -> str:
    """
    Generate a DEG report using provided story elements.
    
    Args:
        story_lists: Story elements to include in the report. Can be:
                    - List: Single list of story elements
                    - List[List]: Multiple lists of story elements to combine
        disease: Disease name for the report
        output_dir: Directory to save the report
        patient_name: Optional patient name for the report
        
    Returns:
        Path to the generated PDF report
    """
    generator = DEGReportGenerator()
    return generator.generate_pdf_report(story_lists, disease, output_dir, patient_name)
