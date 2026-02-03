"""
Report generation module.
Generates CSV and PDF reports from analysis results.
"""

import pandas as pd
from typing import Optional
from profiling import get_cluster_name


def generate_csv_report(features_df: pd.DataFrame, output_path: str = "user_behavior_report.csv", cluster_names: dict = None) -> None:
    """
    Generate CSV report with user behavior analysis.
    
    Args:
        features_df: DataFrame with user features, clusters, and profiles
        output_path: Path to save CSV file
        cluster_names: Optional mapping of cluster_id -> name for custom cluster names
    """
    # Select columns for report
    report_columns = [
        'user', 'cluster', 'total_messages', 'messages_per_day',
        'influence_score', 'behavior_profile'
    ]
    
    # Filter to available columns
    available_columns = [col for col in report_columns if col in features_df.columns]
    
    report_df = features_df[available_columns].copy()
    
    # Map cluster IDs to names
    report_df['cluster'] = report_df['cluster'].apply(lambda x: get_cluster_name(x, cluster_names))
    report_df.rename(columns={'cluster': 'Cluster'}, inplace=True)
    
    # Round numeric columns
    if 'messages_per_day' in report_df.columns:
        report_df['messages_per_day'] = report_df['messages_per_day'].round(2)
    if 'influence_score' in report_df.columns:
        report_df['influence_score'] = report_df['influence_score'].round(2)
    
    # Rename columns for readability
    report_df.rename(columns={
        'user': 'User',
        'total_messages': 'Total Messages',
        'messages_per_day': 'Messages/Day',
        'influence_score': 'Influence Score',
        'behavior_profile': 'Behavior Profile'
    }, inplace=True)
    
    # Sort by influence score (descending)
    if 'Influence Score' in report_df.columns:
        report_df = report_df.sort_values('Influence Score', ascending=False)
    
    # Save to CSV
    report_df.to_csv(output_path, index=False)
    print(f"CSV report saved to: {output_path}")


def generate_pdf_report(features_df: pd.DataFrame, output_path: str = "whatsapp_user_behavior_report.pdf", cluster_names: dict = None) -> None:
    """
    Generate PDF report with user behavior analysis.
    
    Args:
        features_df: DataFrame with user features, clusters, and profiles
        output_path: Path to save PDF file
        cluster_names: Optional mapping of cluster_id -> name for custom cluster names
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        
    except ImportError:
        print("reportlab not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "reportlab"])
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
    
    # Create PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#333333'),
        spaceAfter=6,
        leading=14
    )
    
    # Title
    story.append(Paragraph("WhatsApp User Behavior Analysis Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Process each user
    # Sort by influence score
    sorted_df = features_df.sort_values('influence_score', ascending=False)
    
    for idx, row in sorted_df.iterrows():
        # User header
        user_name = str(row['user'])
        cluster_name = get_cluster_name(int(row['cluster']), cluster_names)
        
        story.append(Paragraph(f"<b>User:</b> {user_name}", heading_style))
        story.append(Paragraph(f"<b>Cluster:</b> {cluster_name}", body_style))
        story.append(Paragraph(f"<b>Total Messages:</b> {int(row.get('total_messages', 0))}", body_style))
        story.append(Paragraph(f"<b>Messages/Day:</b> {row.get('messages_per_day', 0):.2f}", body_style))
        story.append(Paragraph(f"<b>Influence Score:</b> {row.get('influence_score', 0):.2f}", body_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Behavior profile
        profile = row.get('behavior_profile', 'No profile available.')
        story.append(Paragraph(f"<b>Behavior Profile:</b> {profile}", body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Add separator line
        story.append(Paragraph("---", body_style))
        story.append(Spacer(1, 0.2*inch))
    
    # Build PDF
    doc.build(story)
    print(f"PDF report saved to: {output_path}")


def generate_reports(features_df: pd.DataFrame, csv_path: Optional[str] = None, pdf_path: Optional[str] = None, cluster_names: dict = None) -> None:
    """
    Generate both CSV and PDF reports.
    
    Args:
        features_df: DataFrame with user features, clusters, and profiles
        csv_path: Optional path for CSV output (default: user_behavior_report.csv)
        pdf_path: Optional path for PDF output (default: whatsapp_user_behavior_report.pdf)
        cluster_names: Optional mapping of cluster_id -> name for custom cluster names
    """
    if csv_path is None:
        csv_path = "user_behavior_report.csv"
    if pdf_path is None:
        pdf_path = "whatsapp_user_behavior_report.pdf"
    
    generate_csv_report(features_df, csv_path, cluster_names)
    generate_pdf_report(features_df, pdf_path, cluster_names)
