import os
import json
import csv
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from io import BytesIO
import base64
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Optional import for seaborn - gracefully handle if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

class ResultExporter:
    """
    Advanced result export system supporting multiple formats
    
    Features:
    - PDF reports with detailed analysis and visualizations
    - Excel files with multiple sheets and formatting
    - CSV exports for data analysis
    - HTML reports for web viewing
    - Customizable templates and styling
    - Batch export capabilities
    - Quality visualization and charts
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # PDF styling
        self.styles = getSampleStyleSheet()
        self.custom_styles = {
            'title': ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                textColor=colors.darkblue,
                spaceAfter=30,
                alignment=1  # Center alignment
            ),
            'heading': ParagraphStyle(
                'CustomHeading',
                parent=self.styles['Heading2'],
                fontSize=16,
                textColor=colors.darkgreen,
                spaceBefore=20,
                spaceAfter=12
            ),
            'subheading': ParagraphStyle(
                'CustomSubHeading',
                parent=self.styles['Heading3'],
                fontSize=14,
                textColor=colors.blue,
                spaceBefore=15,
                spaceAfter=8
            )
        }
        
        # Color schemes for visualizations
        self.color_schemes = {
            'default': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#005F73'],
            'professional': ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087'],
            'academic': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        }
    
    def export_individual_result(self, result_data: Dict, output_path: str, 
                               format_type: str = 'pdf', template: str = 'detailed') -> Dict:
        """
        Export individual OMR result in specified format
        
        Args:
            result_data: Complete result data including OMR results and quality reports
            output_path: Path where the exported file should be saved
            format_type: Export format ('pdf', 'excel', 'csv', 'html')
            template: Template style ('detailed', 'summary', 'academic')
            
        Returns:
            Export operation result with file paths and metadata
        """
        try:
            self.logger.info(f"Exporting individual result to {format_type} format")
            
            if format_type.lower() == 'pdf':
                return self._export_individual_pdf(result_data, output_path, template)
            elif format_type.lower() == 'excel':
                return self._export_individual_excel(result_data, output_path, template)
            elif format_type.lower() == 'csv':
                return self._export_individual_csv(result_data, output_path)
            elif format_type.lower() == 'html':
                return self._export_individual_html(result_data, output_path, template)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            self.logger.error(f"Error exporting individual result: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'format': format_type
            }
    
    def export_batch_results(self, batch_data: Dict, output_folder: str, 
                           formats: List[str] = ['pdf'], template: str = 'comprehensive') -> Dict:
        """
        Export batch processing results in multiple formats
        
        Args:
            batch_data: Complete batch processing data
            output_folder: Directory to save exported files
            formats: List of export formats
            template: Template style for exports
            
        Returns:
            Export operation results with file paths for each format
        """
        try:
            self.logger.info(f"Exporting batch results in formats: {formats}")
            
            export_results = {
                'success': True,
                'exported_files': {},
                'summary': {
                    'total_results': len(batch_data.get('individual_results', [])),
                    'export_timestamp': datetime.now().isoformat()
                }
            }
            
            # Create export directory
            os.makedirs(output_folder, exist_ok=True)
            
            # Export in each requested format
            for format_type in formats:
                try:
                    if format_type.lower() == 'pdf':
                        result = self._export_batch_pdf(batch_data, output_folder, template)
                    elif format_type.lower() == 'excel':
                        result = self._export_batch_excel(batch_data, output_folder, template)
                    elif format_type.lower() == 'csv':
                        result = self._export_batch_csv(batch_data, output_folder)
                    elif format_type.lower() == 'html':
                        result = self._export_batch_html(batch_data, output_folder, template)
                    else:
                        self.logger.warning(f"Unsupported batch export format: {format_type}")
                        continue
                    
                    if result['success']:
                        export_results['exported_files'][format_type] = result['file_path']
                    else:
                        export_results['success'] = False
                        self.logger.error(f"Failed to export {format_type}: {result.get('error')}")
                        
                except Exception as e:
                    self.logger.error(f"Error exporting batch in {format_type} format: {str(e)}")
                    export_results['success'] = False
            
            return export_results
            
        except Exception as e:
            self.logger.error(f"Error in batch export: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_quality_dashboard(self, quality_data: List[Dict], output_path: str) -> Dict:
        """
        Create visual quality dashboard with charts and analysis
        
        Args:
            quality_data: List of quality reports
            output_path: Path to save the dashboard
            
        Returns:
            Dashboard creation result
        """
        try:
            self.logger.info("Creating quality dashboard")
            
            # Set up matplotlib style
            plt.style.use('seaborn-v0_8')
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('OMR Processing Quality Dashboard', fontsize=20, fontweight='bold')
            
            # Extract quality metrics
            quality_scores = [item.get('overall_score', 0.0) for item in quality_data]
            image_qualities = []
            confidence_scores = []
            
            for item in quality_data:
                img_quality = item.get('image_analysis', {}).get('overall_quality', 0.0)
                confidence = item.get('confidence_metrics', {}).get('overall_confidence', 0.0)
                image_qualities.append(img_quality)
                confidence_scores.append(confidence)
            
            # 1. Quality Score Distribution
            axes[0, 0].hist(quality_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Quality Score Distribution', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Quality Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(np.mean(quality_scores), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(quality_scores):.3f}')
            axes[0, 0].legend()
            
            # 2. Quality vs Confidence Scatter
            axes[0, 1].scatter(quality_scores, confidence_scores, alpha=0.6, s=50)
            axes[0, 1].set_title('Quality vs Confidence Correlation', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Overall Quality Score')
            axes[0, 1].set_ylabel('Confidence Score')
            
            # Add trend line
            if len(quality_scores) > 1:
                z = np.polyfit(quality_scores, confidence_scores, 1)
                p = np.poly1d(z)
                axes[0, 1].plot(quality_scores, p(quality_scores), "r--", alpha=0.8)
            
            # 3. Quality Grade Distribution (Pie Chart)
            grade_counts = self._calculate_quality_grades(quality_scores)
            labels = list(grade_counts.keys())
            sizes = list(grade_counts.values())
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'][:len(labels)]
            
            axes[0, 2].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
            axes[0, 2].set_title('Quality Grade Distribution', fontsize=14, fontweight='bold')
            
            # 4. Image Quality Trends
            file_indices = range(len(image_qualities))
            axes[1, 0].plot(file_indices, image_qualities, marker='o', linewidth=2, markersize=4)
            axes[1, 0].set_title('Image Quality Trends', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('File Index')
            axes[1, 0].set_ylabel('Image Quality Score')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Confidence Score Box Plot
            data_for_box = [quality_scores, image_qualities, confidence_scores]
            labels_for_box = ['Overall Quality', 'Image Quality', 'Confidence']
            
            box_plot = axes[1, 1].boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
            axes[1, 1].set_title('Quality Metrics Comparison', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('Score')
            
            # Color the boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
            
            # 6. Processing Summary Statistics
            axes[1, 2].axis('off')
            
            # Create summary statistics text
            stats_text = f"""
            Processing Summary Statistics
            
            Total Files Processed: {len(quality_data)}
            
            Quality Scores:
            • Average: {np.mean(quality_scores):.3f}
            • Median: {np.median(quality_scores):.3f}
            • Std Dev: {np.std(quality_scores):.3f}
            • Min: {np.min(quality_scores):.3f}
            • Max: {np.max(quality_scores):.3f}
            
            Files Requiring Review: {sum(1 for item in quality_data if item.get('requires_review', False))}
            
            Review Rate: {(sum(1 for item in quality_data if item.get('requires_review', False)) / len(quality_data) * 100):.1f}%
            """
            
            axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                           fontsize=12, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Quality dashboard saved to: {output_path}")
            
            return {
                'success': True,
                'file_path': output_path,
                'statistics': {
                    'total_files': len(quality_data),
                    'average_quality': np.mean(quality_scores),
                    'files_requiring_review': sum(1 for item in quality_data if item.get('requires_review', False))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error creating quality dashboard: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _export_individual_pdf(self, result_data: Dict, output_path: str, template: str) -> Dict:
        """Export individual result as PDF"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Title
            omr_result = result_data.get('omr_result', {})
            filename = result_data.get('filename', 'Unknown File')
            
            title = Paragraph(f"OMR Evaluation Report: {filename}", self.custom_styles['title'])
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Summary section
            story.append(Paragraph("EVALUATION SUMMARY", self.custom_styles['heading']))
            
            summary_data = [
                ['Metric', 'Value'],
                ['Total Questions', str(omr_result.get('total_questions', 'N/A'))],
                ['Correct Answers', str(omr_result.get('correct_answers', 'N/A'))],
                ['Percentage Score', f"{omr_result.get('percentage_score', 0):.1f}%"],
                ['Processing Date', result_data.get('timestamp', 'Unknown')],
                ['Quality Score', f"{result_data.get('quality_score', 0):.3f}"],
                ['Requires Review', 'Yes' if result_data.get('requires_review', False) else 'No']
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Detailed results section (if template is detailed)
            if template == 'detailed':
                story.append(Paragraph("DETAILED ANSWER ANALYSIS", self.custom_styles['heading']))
                
                detailed_results = omr_result.get('detailed_results', {})
                if detailed_results:
                    # Create answer table
                    answer_data = [['Question', 'Detected Answer', 'Correct Answer', 'Result', 'Confidence']]
                    
                    for q_num, result in detailed_results.items():
                        detected = result.get('detected_answer', 'None')
                        correct = result.get('correct_answer', 'N/A')
                        is_correct = '✓' if result.get('is_correct', False) else '✗'
                        confidence = f"{result.get('confidence', 0.0):.3f}"
                        
                        answer_data.append([str(q_num), str(detected), str(correct), is_correct, confidence])
                    
                    answer_table = Table(answer_data)
                    answer_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    
                    story.append(answer_table)
                    story.append(Spacer(1, 20))
            
            # Quality analysis section
            quality_report = result_data.get('quality_report', {})
            if quality_report:
                story.append(Paragraph("QUALITY ANALYSIS", self.custom_styles['heading']))
                
                # Image quality metrics
                image_analysis = quality_report.get('image_analysis', {})
                if image_analysis and 'quality_scores' in image_analysis:
                    story.append(Paragraph("Image Quality Metrics", self.custom_styles['subheading']))
                    
                    quality_scores = image_analysis['quality_scores']
                    quality_data = [
                        ['Metric', 'Score', 'Assessment'],
                        ['Brightness', f"{quality_scores.get('brightness_score', 0):.3f}", 
                         self._get_quality_assessment(quality_scores.get('brightness_score', 0))],
                        ['Contrast', f"{quality_scores.get('contrast_score', 0):.3f}", 
                         self._get_quality_assessment(quality_scores.get('contrast_score', 0))],
                        ['Sharpness', f"{quality_scores.get('sharpness_score', 0):.3f}", 
                         self._get_quality_assessment(quality_scores.get('sharpness_score', 0))],
                        ['Noise Level', f"{quality_scores.get('noise_score', 0):.3f}", 
                         self._get_quality_assessment(quality_scores.get('noise_score', 0))],
                        ['Uniformity', f"{quality_scores.get('uniformity_score', 0):.3f}", 
                         self._get_quality_assessment(quality_scores.get('uniformity_score', 0))]
                    ]
                    
                    quality_table = Table(quality_data)
                    quality_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.green),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 11),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    
                    story.append(quality_table)
                    story.append(Spacer(1, 15))
                
                # Recommendations
                recommendations = quality_report.get('recommendations', [])
                if recommendations:
                    story.append(Paragraph("Recommendations", self.custom_styles['subheading']))
                    
                    for rec in recommendations:
                        story.append(Paragraph(f"• {rec}", self.styles['Normal']))
                    
                    story.append(Spacer(1, 15))
            
            # Build PDF
            doc.build(story)
            
            return {
                'success': True,
                'file_path': output_path,
                'format': 'pdf'
            }
            
        except Exception as e:
            self.logger.error(f"Error creating PDF: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'format': 'pdf'
            }
    
    def _export_individual_excel(self, result_data: Dict, output_path: str, template: str) -> Dict:
        """Export individual result as Excel file"""
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Summary sheet
                omr_result = result_data.get('omr_result', {})
                summary_data = {
                    'Metric': [
                        'Filename', 'Total Questions', 'Correct Answers', 'Percentage Score',
                        'Processing Date', 'Quality Score', 'Requires Review'
                    ],
                    'Value': [
                        result_data.get('filename', 'Unknown'),
                        omr_result.get('total_questions', 'N/A'),
                        omr_result.get('correct_answers', 'N/A'),
                        f"{omr_result.get('percentage_score', 0):.1f}%",
                        result_data.get('timestamp', 'Unknown'),
                        f"{result_data.get('quality_score', 0):.3f}",
                        'Yes' if result_data.get('requires_review', False) else 'No'
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Detailed results sheet
                detailed_results = omr_result.get('detailed_results', {})
                if detailed_results:
                    details_data = []
                    for q_num, result in detailed_results.items():
                        details_data.append({
                            'Question': q_num,
                            'Detected_Answer': result.get('detected_answer', 'None'),
                            'Correct_Answer': result.get('correct_answer', 'N/A'),
                            'Is_Correct': result.get('is_correct', False),
                            'Confidence': result.get('confidence', 0.0)
                        })
                    
                    details_df = pd.DataFrame(details_data)
                    details_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
                
                # Quality metrics sheet
                quality_report = result_data.get('quality_report', {})
                if quality_report:
                    quality_data = []
                    
                    # Image quality metrics
                    image_analysis = quality_report.get('image_analysis', {})
                    if 'quality_scores' in image_analysis:
                        for metric, score in image_analysis['quality_scores'].items():
                            quality_data.append({
                                'Category': 'Image Quality',
                                'Metric': metric.replace('_', ' ').title(),
                                'Score': score,
                                'Assessment': self._get_quality_assessment(score)
                            })
                    
                    # Confidence metrics
                    confidence_metrics = quality_report.get('confidence_metrics', {})
                    for metric, score in confidence_metrics.items():
                        quality_data.append({
                            'Category': 'Confidence',
                            'Metric': metric.replace('_', ' ').title(),
                            'Score': score,
                            'Assessment': self._get_quality_assessment(score)
                        })
                    
                    if quality_data:
                        quality_df = pd.DataFrame(quality_data)
                        quality_df.to_excel(writer, sheet_name='Quality_Metrics', index=False)
            
            return {
                'success': True,
                'file_path': output_path,
                'format': 'excel'
            }
            
        except Exception as e:
            self.logger.error(f"Error creating Excel file: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'format': 'excel'
            }
    
    def _export_individual_csv(self, result_data: Dict, output_path: str) -> Dict:
        """Export individual result as CSV file"""
        try:
            # Create comprehensive CSV with all data
            omr_result = result_data.get('omr_result', {})
            detailed_results = omr_result.get('detailed_results', {})
            
            csv_data = []
            for q_num, result in detailed_results.items():
                csv_data.append({
                    'filename': result_data.get('filename', 'Unknown'),
                    'question': q_num,
                    'detected_answer': result.get('detected_answer', 'None'),
                    'correct_answer': result.get('correct_answer', 'N/A'),
                    'is_correct': result.get('is_correct', False),
                    'confidence': result.get('confidence', 0.0),
                    'quality_score': result_data.get('quality_score', 0.0),
                    'requires_review': result_data.get('requires_review', False),
                    'processing_date': result_data.get('timestamp', 'Unknown')
                })
            
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False)
            
            return {
                'success': True,
                'file_path': output_path,
                'format': 'csv'
            }
            
        except Exception as e:
            self.logger.error(f"Error creating CSV file: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'format': 'csv'
            }
    
    def _export_individual_html(self, result_data: Dict, output_path: str, template: str) -> Dict:
        """Export individual result as HTML file"""
        try:
            omr_result = result_data.get('omr_result', {})
            quality_report = result_data.get('quality_report', {})
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>OMR Evaluation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ text-align: center; color: #2c3e50; }}
                    .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                    .section {{ margin: 20px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                    th {{ background-color: #3498db; color: white; }}
                    .correct {{ color: green; font-weight: bold; }}
                    .incorrect {{ color: red; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>OMR Evaluation Report</h1>
                    <h2>{result_data.get('filename', 'Unknown File')}</h2>
                </div>
                
                <div class="summary">
                    <h3>Summary</h3>
                    <p><strong>Total Questions:</strong> {omr_result.get('total_questions', 'N/A')}</p>
                    <p><strong>Correct Answers:</strong> {omr_result.get('correct_answers', 'N/A')}</p>
                    <p><strong>Percentage Score:</strong> {omr_result.get('percentage_score', 0):.1f}%</p>
                    <p><strong>Quality Score:</strong> {result_data.get('quality_score', 0):.3f}</p>
                    <p><strong>Requires Review:</strong> {'Yes' if result_data.get('requires_review', False) else 'No'}</p>
                </div>
            """
            
            # Add detailed results table if available
            detailed_results = omr_result.get('detailed_results', {})
            if detailed_results:
                html_content += """
                <div class="section">
                    <h3>Detailed Results</h3>
                    <table>
                        <tr>
                            <th>Question</th>
                            <th>Detected Answer</th>
                            <th>Correct Answer</th>
                            <th>Result</th>
                            <th>Confidence</th>
                        </tr>
                """
                
                for q_num, result in detailed_results.items():
                    result_class = 'correct' if result.get('is_correct', False) else 'incorrect'
                    result_symbol = '✓' if result.get('is_correct', False) else '✗'
                    
                    html_content += f"""
                        <tr>
                            <td>{q_num}</td>
                            <td>{result.get('detected_answer', 'None')}</td>
                            <td>{result.get('correct_answer', 'N/A')}</td>
                            <td class="{result_class}">{result_symbol}</td>
                            <td>{result.get('confidence', 0.0):.3f}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
            
            html_content += """
            </body>
            </html>
            """
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return {
                'success': True,
                'file_path': output_path,
                'format': 'html'
            }
            
        except Exception as e:
            self.logger.error(f"Error creating HTML file: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'format': 'html'
            }
    
    def _export_batch_pdf(self, batch_data: Dict, output_folder: str, template: str) -> Dict:
        """Export batch results as comprehensive PDF report"""
        try:
            output_path = os.path.join(output_folder, f"batch_report_{batch_data.get('batch_id', 'unknown')}.pdf")
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Title page
            title = Paragraph(f"OMR Batch Processing Report", self.custom_styles['title'])
            story.append(title)
            story.append(Spacer(1, 20))
            
            batch_name = batch_data.get('batch_name', 'Unnamed Batch')
            subtitle = Paragraph(f"Batch: {batch_name}", self.custom_styles['heading'])
            story.append(subtitle)
            story.append(Spacer(1, 30))
            
            # Batch summary
            summary = batch_data.get('summary', {})
            summary_data = [
                ['Metric', 'Value'],
                ['Batch ID', batch_data.get('batch_id', 'Unknown')],
                ['Processing Date', batch_data.get('timestamp', 'Unknown')],
                ['Total Files', str(summary.get('total_files', 0))],
                ['Successful Files', str(summary.get('successful_files', 0))],
                ['Failed Files', str(summary.get('failed_files', 0))],
                ['Success Rate', f"{summary.get('success_rate', 0):.1f}%"],
                ['Total Processing Time', f"{summary.get('total_processing_time', 0):.1f} seconds"]
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(PageBreak())
            
            # Individual results summary
            story.append(Paragraph("INDIVIDUAL RESULTS SUMMARY", self.custom_styles['heading']))
            
            individual_results = batch_data.get('individual_results', [])
            if individual_results:
                results_data = [['Filename', 'Success', 'Quality Score', 'Requires Review']]
                
                for result in individual_results:
                    success_text = '✓' if result.get('success', False) else '✗'
                    quality_score = f"{result.get('quality_score', 0.0):.3f}"
                    review_text = 'Yes' if result.get('requires_review', False) else 'No'
                    
                    results_data.append([
                        result.get('filename', 'Unknown'),
                        success_text,
                        quality_score,
                        review_text
                    ])
                
                results_table = Table(results_data)
                results_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.green),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 10)
                ]))
                
                story.append(results_table)
            
            # Build PDF
            doc.build(story)
            
            return {
                'success': True,
                'file_path': output_path,
                'format': 'pdf'
            }
            
        except Exception as e:
            self.logger.error(f"Error creating batch PDF: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'format': 'pdf'
            }
    
    def _export_batch_excel(self, batch_data: Dict, output_folder: str, template: str) -> Dict:
        """Export batch results as Excel file with multiple sheets"""
        try:
            output_path = os.path.join(output_folder, f"batch_results_{batch_data.get('batch_id', 'unknown')}.xlsx")
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Batch summary sheet
                summary = batch_data.get('summary', {})
                summary_data = pd.DataFrame([
                    ['Batch ID', batch_data.get('batch_id', 'Unknown')],
                    ['Batch Name', batch_data.get('batch_name', 'Unnamed')],
                    ['Processing Date', batch_data.get('timestamp', 'Unknown')],
                    ['Total Files', summary.get('total_files', 0)],
                    ['Successful Files', summary.get('successful_files', 0)],
                    ['Failed Files', summary.get('failed_files', 0)],
                    ['Success Rate (%)', f"{summary.get('success_rate', 0):.1f}"],
                    ['Total Processing Time (s)', f"{summary.get('total_processing_time', 0):.1f}"]
                ], columns=['Metric', 'Value'])
                
                summary_data.to_excel(writer, sheet_name='Batch_Summary', index=False)
                
                # Individual results sheet
                individual_results = batch_data.get('individual_results', [])
                if individual_results:
                    results_df = pd.DataFrame(individual_results)
                    results_df.to_excel(writer, sheet_name='Individual_Results', index=False)
                
                # Quality metrics sheet
                quality_metrics = batch_data.get('quality_metrics', {})
                if quality_metrics:
                    quality_data = []
                    for key, value in quality_metrics.items():
                        quality_data.append({'Metric': key.replace('_', ' ').title(), 'Value': value})
                    
                    quality_df = pd.DataFrame(quality_data)
                    quality_df.to_excel(writer, sheet_name='Quality_Metrics', index=False)
                
                # Failed files sheet
                failed_files = batch_data.get('failed_files', [])
                if failed_files:
                    failed_df = pd.DataFrame(failed_files)
                    failed_df.to_excel(writer, sheet_name='Failed_Files', index=False)
            
            return {
                'success': True,
                'file_path': output_path,
                'format': 'excel'
            }
            
        except Exception as e:
            self.logger.error(f"Error creating batch Excel: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'format': 'excel'
            }
    
    def _export_batch_csv(self, batch_data: Dict, output_folder: str) -> Dict:
        """Export batch results as CSV files"""
        try:
            batch_id = batch_data.get('batch_id', 'unknown')
            
            # Create consolidated CSV with all results
            all_results = []
            individual_results = batch_data.get('individual_results', [])
            
            for result in individual_results:
                if result.get('success', False):
                    # This would need the detailed OMR results to be meaningful
                    all_results.append({
                        'batch_id': batch_id,
                        'filename': result.get('filename', 'Unknown'),
                        'success': result.get('success', False),
                        'quality_score': result.get('quality_score', 0.0),
                        'requires_review': result.get('requires_review', False),
                        'error': result.get('error', '') if not result.get('success', False) else ''
                    })
            
            if all_results:
                output_path = os.path.join(output_folder, f"batch_results_{batch_id}.csv")
                df = pd.DataFrame(all_results)
                df.to_csv(output_path, index=False)
                
                return {
                    'success': True,
                    'file_path': output_path,
                    'format': 'csv'
                }
            else:
                return {
                    'success': False,
                    'error': 'No successful results to export',
                    'format': 'csv'
                }
            
        except Exception as e:
            self.logger.error(f"Error creating batch CSV: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'format': 'csv'
            }
    
    def _export_batch_html(self, batch_data: Dict, output_folder: str, template: str) -> Dict:
        """Export batch results as HTML report"""
        try:
            output_path = os.path.join(output_folder, f"batch_report_{batch_data.get('batch_id', 'unknown')}.html")
            
            # Generate HTML content
            summary = batch_data.get('summary', {})
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Batch Processing Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; }}
                    .header {{ text-align: center; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 20px; }}
                    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                    .metric-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                    .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
                    th {{ background-color: #3498db; color: white; }}
                    .success {{ color: green; font-weight: bold; }}
                    .failed {{ color: red; font-weight: bold; }}
                    .review {{ background-color: #fff3cd; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>OMR Batch Processing Report</h1>
                        <h2>{batch_data.get('batch_name', 'Unnamed Batch')}</h2>
                        <p>Batch ID: {batch_data.get('batch_id', 'Unknown')} | Date: {batch_data.get('timestamp', 'Unknown')}</p>
                    </div>
                    
                    <div class="summary">
                        <div class="metric-card">
                            <div class="metric-value">{summary.get('total_files', 0)}</div>
                            <div class="metric-label">Total Files</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{summary.get('successful_files', 0)}</div>
                            <div class="metric-label">Successful</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{summary.get('success_rate', 0):.1f}%</div>
                            <div class="metric-label">Success Rate</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{summary.get('total_processing_time', 0):.1f}s</div>
                            <div class="metric-label">Processing Time</div>
                        </div>
                    </div>
            """
            
            # Add individual results table
            individual_results = batch_data.get('individual_results', [])
            if individual_results:
                html_content += """
                    <h3>Individual Results</h3>
                    <table>
                        <tr>
                            <th>Filename</th>
                            <th>Status</th>
                            <th>Quality Score</th>
                            <th>Requires Review</th>
                        </tr>
                """
                
                for result in individual_results:
                    status_class = 'success' if result.get('success', False) else 'failed'
                    status_text = 'Success' if result.get('success', False) else 'Failed'
                    review_class = 'review' if result.get('requires_review', False) else ''
                    
                    html_content += f"""
                        <tr class="{review_class}">
                            <td>{result.get('filename', 'Unknown')}</td>
                            <td class="{status_class}">{status_text}</td>
                            <td>{result.get('quality_score', 0.0):.3f}</td>
                            <td>{'Yes' if result.get('requires_review', False) else 'No'}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                """
            
            html_content += """
                </div>
            </body>
            </html>
            """
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return {
                'success': True,
                'file_path': output_path,
                'format': 'html'
            }
            
        except Exception as e:
            self.logger.error(f"Error creating batch HTML: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'format': 'html'
            }
    
    def _calculate_quality_grades(self, quality_scores: List[float]) -> Dict[str, int]:
        """Calculate distribution of quality grades"""
        grades = {'Excellent': 0, 'Good': 0, 'Acceptable': 0, 'Poor': 0, 'Critical': 0}
        
        for score in quality_scores:
            if score >= 0.9:
                grades['Excellent'] += 1
            elif score >= 0.8:
                grades['Good'] += 1
            elif score >= 0.7:
                grades['Acceptable'] += 1
            elif score >= 0.6:
                grades['Poor'] += 1
            else:
                grades['Critical'] += 1
        
        # Remove empty grades
        return {grade: count for grade, count in grades.items() if count > 0}
    
    def _get_quality_assessment(self, score: float) -> str:
        """Get text assessment for quality score"""
        if score >= 0.9:
            return 'Excellent'
        elif score >= 0.8:
            return 'Good'
        elif score >= 0.7:
            return 'Acceptable'
        elif score >= 0.6:
            return 'Poor'
        else:
            return 'Critical'