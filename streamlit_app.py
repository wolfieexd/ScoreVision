"""
ScoreVision Pro - Streamlit MVP Interface
Professional OMR Evaluation System for Evaluators
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import requests
import io
from PIL import Image
import time

# Configure Streamlit page
st.set_page_config(
    page_title="ScoreVision Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2c3e50 0%, #3498db 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
    }
    
    .success-metric {
        border-left-color: #27ae60;
    }
    
    .warning-metric {
        border-left-color: #f39c12;
    }
    
    .error-metric {
        border-left-color: #e74c3c;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #34495e 0%, #2c3e50 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class OMRProcessor:
    """Handles OMR processing operations"""
    
    def __init__(self):
        self.backend_url = "http://localhost:5000"
    
    def check_backend_status(self):
        """Check if backend is running"""
        try:
            response = requests.get(f"{self.backend_url}/api/health", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def upload_file(self, file_data, file_type="omr"):
        """Upload file to backend"""
        try:
            files = {'file': file_data}
            endpoint = "/api/upload-omr" if file_type == "omr" else "/api/upload-answer-key"
            response = requests.post(f"{self.backend_url}{endpoint}", files=files)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def process_omr(self, omr_file, answer_key_file):
        """Process OMR sheet with answer key"""
        try:
            data = {
                "omr_file": omr_file,
                "answer_key_file": answer_key_file
            }
            response = requests.post(f"{self.backend_url}/api/process-omr", json=data)
            return response.json() if response.status_code == 200 else None
        except:
            return None

def load_sample_data():
    """Load sample data for dashboard"""
    
    # Sample processing statistics
    stats = {
        'total_processed': 50247,
        'accuracy_rate': 99.7,
        'avg_processing_time': 8.2,
        'institutions_served': 147,
        'active_sessions': 23,
        'error_rate': 0.3
    }
    
    # Sample recent activities
    activities = [
        {
            'exam': 'Advanced Physics - Batch 001',
            'sheets': 125,
            'institution': 'Metropolitan High School',
            'accuracy': 99.7,
            'status': 'Completed',
            'time_ago': '12 minutes ago'
        },
        {
            'exam': 'Mathematics Excellence - Batch 002', 
            'sheets': 89,
            'institution': 'Science Academy',
            'accuracy': 99.9,
            'status': 'Completed',
            'time_ago': '1 hour ago'
        },
        {
            'exam': 'Computer Science Assessment',
            'sheets': 156,
            'institution': 'Tech University',
            'accuracy': 98.4,
            'status': 'Processing',
            'time_ago': 'In progress'
        }
    ]
    
    # Sample performance data
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    performance_data = {
        'days': days,
        'sheets_processed': [1240, 1356, 1189, 1456, 1523, 1334, 1401],
        'accuracy_rates': [99.5, 99.6, 99.7, 99.8, 99.7, 99.9, 99.7],
        'processing_times': [8.1, 7.9, 8.3, 7.8, 8.0, 8.2, 8.1]
    }
    
    return stats, activities, performance_data

def create_performance_chart(performance_data):
    """Create performance trend chart"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Processing Volume', 'Accuracy & Processing Time Trends'),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )
    
    # Processing volume chart
    fig.add_trace(
        go.Bar(
            x=performance_data['days'],
            y=performance_data['sheets_processed'],
            name='Sheets Processed',
            marker_color='#3498db',
            text=performance_data['sheets_processed'],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Accuracy trend
    fig.add_trace(
        go.Scatter(
            x=performance_data['days'],
            y=performance_data['accuracy_rates'],
            mode='lines+markers',
            name='Accuracy Rate (%)',
            line=dict(color='#27ae60', width=3),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    # Processing time trend
    fig.add_trace(
        go.Scatter(
            x=performance_data['days'],
            y=performance_data['processing_times'],
            mode='lines+markers',
            name='Avg Processing Time (s)',
            line=dict(color='#f39c12', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ),
        row=2, col=1, secondary_y=True
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="System Performance Analytics",
        title_x=0.5
    )
    
    fig.update_yaxes(title_text="Number of Sheets", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy Rate (%)", row=2, col=1)
    fig.update_yaxes(title_text="Processing Time (seconds)", secondary_y=True, row=2, col=1)
    
    return fig

def dashboard_page():
    """Main dashboard for evaluators"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 ScoreVision Pro - Evaluator Dashboard</h1>
        <p>Professional OMR Evaluation & Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load sample data
    stats, activities, performance_data = load_sample_data()
    
    # System status check
    processor = OMRProcessor()
    backend_status = processor.check_backend_status()
    
    if backend_status:
        st.success("🟢 System Status: OPERATIONAL - All services running normally")
    else:
        st.error("🔴 System Status: BACKEND UNAVAILABLE - Please start the Flask backend server")
    
    # Key metrics
    st.subheader("📊 System Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📋 Total Processed",
            value=f"{stats['total_processed']:,}",
            delta="↗️ +12.3% this month"
        )
    
    with col2:
        st.metric(
            label="🎯 Accuracy Rate",
            value=f"{stats['accuracy_rate']}%",
            delta="↗️ +0.2% this week"
        )
    
    with col3:
        st.metric(
            label="⚡ Avg Processing Time",
            value=f"{stats['avg_processing_time']}s",
            delta="↘️ -0.3s improvement"
        )
    
    with col4:
        st.metric(
            label="🏢 Active Institutions",
            value=f"{stats['institutions_served']}",
            delta="↗️ +5 new this month"
        )
    
    # Performance charts
    st.subheader("📈 Performance Analytics")
    performance_chart = create_performance_chart(performance_data)
    st.plotly_chart(performance_chart, use_container_width=True)
    
    # Recent activities
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🕒 Recent Processing Activities")
        for activity in activities:
            status_color = "🟢" if activity['status'] == 'Completed' else "🟡"
            with st.expander(f"{status_color} {activity['exam']} - {activity['status']}"):
                st.write(f"**Institution:** {activity['institution']}")
                st.write(f"**Sheets Processed:** {activity['sheets']}")
                st.write(f"**Accuracy:** {activity['accuracy']}%")
                st.write(f"**Time:** {activity['time_ago']}")
    
    with col2:
        st.subheader("🚀 Quick Actions")
        
        if st.button("📤 Upload OMR Sheets", use_container_width=True, key="dashboard_upload_btn"):
            st.session_state.page = "upload"
            st.rerun()
        
        if st.button("📊 Batch Processing", use_container_width=True, key="dashboard_batch_btn"):
            st.session_state.page = "batch"
            st.rerun()
        
        if st.button("🔍 Review Results", use_container_width=True, key="dashboard_review_btn"):
            st.session_state.page = "review"
            st.rerun()
        
        if st.button("⚙️ System Settings", use_container_width=True, key="dashboard_settings_btn"):
            st.session_state.page = "settings"
            st.rerun()
        
        # System health indicators
        st.subheader("🛡️ System Health")
        health_metrics = {
            "CPU Usage": ("23%", "success"),
            "Memory": ("34%", "success"), 
            "Storage": ("67%", "warning"),
            "Network": ("145ms", "success")
        }
        
        for metric, (value, status) in health_metrics.items():
            color = "🟢" if status == "success" else "🟡" if status == "warning" else "🔴"
            st.write(f"{color} {metric}: {value}")

def upload_page():
    """Upload page for OMR sheets and answer keys"""
    
    st.markdown("""
    <div class="main-header">
        <h1>📤 Upload & Process OMR Sheets</h1>
        <p>Secure upload and processing of examination answer sheets</p>
    </div>
    """, unsafe_allow_html=True)
    
    processor = OMRProcessor()
    
    # Check backend status
    if not processor.check_backend_status():
        st.error("🔴 Backend server is not running. Please start the Flask server first.")
        st.code("python app.py", language="bash")
        return
    
    # Upload form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 OMR Answer Sheets")
        uploaded_omr = st.file_uploader(
            "Choose OMR sheet images",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=True,
            help="Upload scanned OMR answer sheets (PNG, JPG, TIFF, BMP)"
        )
        
        if uploaded_omr:
            st.success(f"✅ {len(uploaded_omr)} OMR sheet(s) uploaded")
            
            # Preview uploaded images
            with st.expander("🔍 Preview Uploaded Sheets"):
                for i, file in enumerate(uploaded_omr[:3]):  # Show first 3
                    image = Image.open(file)
                    st.image(image, caption=f"Sheet {i+1}: {file.name}", width=200)
                
                if len(uploaded_omr) > 3:
                    st.info(f"... and {len(uploaded_omr) - 3} more sheets")
    
    with col2:
        st.subheader("🔑 Answer Key")
        uploaded_key = st.file_uploader(
            "Choose answer key file",
            type=['json', 'xlsx', 'xls'],
            help="Upload answer key in JSON or Excel format"
        )
        
        if uploaded_key:
            st.success("✅ Answer key uploaded")
            
            # Preview answer key
            with st.expander("🔍 Preview Answer Key"):
                if uploaded_key.name.endswith('.json'):
                    key_data = json.load(uploaded_key)
                    st.json(key_data)
                else:
                    df = pd.read_excel(uploaded_key)
                    st.dataframe(df.head())
    
    # Processing options
    st.subheader("⚙️ Processing Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        processor_type = st.selectbox(
            "Processing Engine",
            ["Standard", "Universal", "High Accuracy"],
            help="Choose processing algorithm"
        )
    
    with col2:
        quality_check = st.selectbox(
            "Quality Level",
            ["Standard", "High", "Maximum"],
            help="Quality assurance level"
        )
    
    with col3:
        export_format = st.selectbox(
            "Export Format",
            ["PDF Report", "Excel", "CSV", "JSON"],
            help="Result export format"
        )
    
    # Process button
    if st.button("🚀 Start Processing", type="primary", use_container_width=True, key="upload_process_btn"):
        if uploaded_omr and uploaded_key:
            
            with st.spinner("Processing OMR sheets..."):
                # Simulate processing
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    time.sleep(0.02)  # Simulate processing time
                    progress_bar.progress(i + 1)
                    status_text.text(f'Processing sheet {(i//10)+1}... {i+1}% complete')
                
                st.success("✅ Processing completed successfully!")
                
                # Display results summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sheets Processed", len(uploaded_omr))
                
                with col2:
                    st.metric("Accuracy Rate", "99.7%")
                
                with col3:
                    st.metric("Processing Time", "12.3s")
                
                # Sample results
                st.subheader("📊 Processing Results")
                results_data = {
                    'Sheet': [f'Sheet_{i+1}' for i in range(len(uploaded_omr))],
                    'Score': np.random.randint(70, 100, len(uploaded_omr)),
                    'Accuracy': np.random.uniform(95, 100, len(uploaded_omr)),
                    'Status': ['✅ Processed'] * len(uploaded_omr)
                }
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Download button
                if st.button("📥 Download Results", type="secondary", key="upload_download_btn"):
                    st.download_button(
                        label="Download PDF Report",
                        data="Sample PDF content",
                        file_name="omr_results.pdf",
                        mime="application/pdf"
                    )
        else:
            st.error("❌ Please upload both OMR sheets and answer key before processing")

def batch_processing_page():
    """Batch processing interface"""
    
    st.markdown("""
    <div class="main-header">
        <h1>📊 Batch Processing Center</h1>
        <p>Efficient bulk processing of multiple OMR examination batches</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Batch upload section
    st.subheader("📤 Batch Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        batch_files = st.file_uploader(
            "Upload multiple OMR sheets",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=True,
            help="Select multiple OMR sheet images for batch processing"
        )
        
        answer_key = st.file_uploader(
            "Upload answer key for batch",
            type=['json', 'xlsx', 'xls'],
            help="Single answer key will be used for all sheets in batch"
        )
    
    with col2:
        st.subheader("⚙️ Batch Settings")
        
        max_workers = st.slider("Parallel Workers", 1, 8, 4)
        st.write(f"Using {max_workers} parallel processing threads")
        
        auto_export = st.checkbox("Auto-export results", value=True)
        send_notification = st.checkbox("Send completion notification", value=False)
        
        if send_notification:
            email = st.text_input("Notification email")
    
    # Start batch processing
    if st.button("🚀 Start Batch Processing", type="primary", use_container_width=True, key="batch_process_btn"):
        if batch_files and answer_key:
            
            # Create batch session
            batch_id = f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            st.info(f"🆔 Batch ID: {batch_id}")
            
            # Processing simulation
            with st.spinner("Initializing batch processing..."):
                time.sleep(2)
            
            # Progress tracking
            st.subheader("📈 Batch Progress")
            
            total_files = len(batch_files)
            progress_container = st.container()
            
            with progress_container:
                overall_progress = st.progress(0)
                current_file = st.empty()
                stats_cols = st.columns(4)
                
                with stats_cols[0]:
                    processed_metric = st.empty()
                with stats_cols[1]:
                    success_metric = st.empty() 
                with stats_cols[2]:
                    error_metric = st.empty()
                with stats_cols[3]:
                    speed_metric = st.empty()
                
                # Simulate batch processing
                for i in range(total_files):
                    time.sleep(1)  # Simulate processing time
                    
                    progress = (i + 1) / total_files
                    overall_progress.progress(progress)
                    current_file.text(f"Processing: {batch_files[i].name}")
                    
                    processed_metric.metric("Processed", f"{i+1}/{total_files}")
                    success_metric.metric("Success Rate", f"{99.5 + np.random.uniform(-0.5, 0.5):.1f}%")
                    error_metric.metric("Errors", f"{np.random.randint(0, 2)}")
                    speed_metric.metric("Speed", f"{np.random.uniform(8, 12):.1f}s/sheet")
                
                st.success("✅ Batch processing completed successfully!")
                
                # Results summary
                st.subheader("📊 Batch Results Summary")
                
                summary_data = {
                    'Metric': ['Total Sheets', 'Successfully Processed', 'Average Score', 'Processing Time', 'Accuracy Rate'],
                    'Value': [total_files, total_files, '87.3%', '2m 15s', '99.6%'],
                    'Status': ['✅', '✅', '✅', '✅', '✅']
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        else:
            st.error("❌ Please upload batch files and answer key")
    
    # Active batches section
    st.subheader("🔄 Active Batch Sessions")
    
    # Sample active batches
    active_batches = [
        {
            'Batch ID': 'BATCH_20240921_143052',
            'Files': 125,
            'Progress': 78,
            'Status': 'Processing',
            'ETA': '3m 15s'
        },
        {
            'Batch ID': 'BATCH_20240921_142130', 
            'Files': 89,
            'Progress': 100,
            'Status': 'Completed',
            'ETA': 'Finished'
        }
    ]
    
    for batch in active_batches:
        with st.expander(f"📁 {batch['Batch ID']} - {batch['Status']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Files", batch['Files'])
                
            with col2:
                st.metric("Progress", f"{batch['Progress']}%")
                
            with col3:
                st.metric("ETA", batch['ETA'])
            
            # Progress bar for active batches
            if batch['Status'] == 'Processing':
                st.progress(batch['Progress'] / 100)

def review_page():
    """Results review and analysis page"""
    
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Results Review & Analysis</h1>
        <p>Comprehensive examination results analysis and quality review</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter options
    st.subheader("🔧 Filter & Search")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=7), datetime.now()),
            help="Filter results by date range"
        )
    
    with col2:
        institution_filter = st.selectbox(
            "Institution",
            ["All Institutions", "Metropolitan High School", "Science Academy", "Tech University"]
        )
    
    with col3:
        subject_filter = st.selectbox(
            "Subject",
            ["All Subjects", "Physics", "Mathematics", "Computer Science", "Chemistry"]
        )
    
    with col4:
        status_filter = st.selectbox(
            "Status",
            ["All", "Completed", "Processing", "Requires Review"]
        )
    
    # Sample results data
    results_data = {
        'Exam ID': ['PHY_001', 'MATH_002', 'CS_003', 'CHEM_004', 'PHY_005'],
        'Subject': ['Physics', 'Mathematics', 'Computer Science', 'Chemistry', 'Physics'],
        'Institution': ['Metro High', 'Science Academy', 'Tech University', 'Metro High', 'Science Academy'],
        'Students': [125, 89, 156, 98, 134],
        'Avg Score': [78.5, 82.3, 75.8, 80.1, 77.9],
        'Accuracy': [99.7, 99.9, 98.4, 99.2, 99.6],
        'Status': ['✅ Completed', '✅ Completed', '🔍 Review', '✅ Completed', '✅ Completed'],
        'Date': ['2024-09-21', '2024-09-20', '2024-09-19', '2024-09-18', '2024-09-17']
    }
    
    results_df = pd.DataFrame(results_data)
    
    # Results table
    st.subheader("📋 Examination Results")
    
    selected_rows = st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Avg Score": st.column_config.ProgressColumn(
                "Average Score",
                help="Average examination score",
                min_value=0,
                max_value=100,
            ),
            "Accuracy": st.column_config.ProgressColumn(
                "Processing Accuracy",
                help="OMR processing accuracy",
                min_value=90,
                max_value=100,
            )
        }
    )
    
    # Detailed analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Score Distribution Analysis")
        
        # Sample score distribution data
        score_ranges = ['0-40', '41-60', '61-80', '81-100']
        score_counts = [5, 18, 45, 57]
        
        fig_scores = px.bar(
            x=score_ranges,
            y=score_counts,
            title="Score Distribution Across All Exams",
            labels={'x': 'Score Range', 'y': 'Number of Students'},
            color=score_counts,
            color_continuous_scale='viridis'
        )
        
        st.plotly_chart(fig_scores, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Accuracy Trends")
        
        # Sample accuracy trend data
        dates = pd.date_range(start='2024-09-15', end='2024-09-21', freq='D')
        accuracy_values = [99.3, 99.5, 99.7, 99.2, 99.8, 99.6, 99.7]
        
        fig_accuracy = px.line(
            x=dates,
            y=accuracy_values,
            title="Processing Accuracy Trend",
            labels={'x': 'Date', 'y': 'Accuracy (%)'},
            markers=True
        )
        
        fig_accuracy.update_traces(line_color='#27ae60', line_width=3)
        st.plotly_chart(fig_accuracy, use_container_width=True)
    
    # Quality review section
    st.subheader("🛡️ Quality Review Dashboard")
    
    quality_metrics = st.columns(4)
    
    with quality_metrics[0]:
        st.metric(
            "Quality Score",
            "98.4%",
            delta="↗️ +0.3%",
            help="Overall quality assessment"
        )
    
    with quality_metrics[1]:
        st.metric(
            "Flagged Items",
            "3",
            delta="↘️ -2",
            help="Items requiring manual review"
        )
    
    with quality_metrics[2]:
        st.metric(
            "Auto-Resolved",
            "147",
            delta="↗️ +12",
            help="Automatically resolved ambiguities"
        )
    
    with quality_metrics[3]:
        st.metric(
            "Confidence Level",
            "99.2%",
            delta="↗️ +0.1%",
            help="Average detection confidence"
        )
    
    # Flagged items for review
    st.subheader("⚠️ Items Requiring Review")
    
    flagged_items = [
        {
            'Exam': 'CS_003',
            'Student': 'STU_156',
            'Question': 'Q25',
            'Issue': 'Ambiguous marking',
            'Confidence': '72%',
            'Action': 'Manual Review'
        },
        {
            'Exam': 'PHY_001', 
            'Student': 'STU_089',
            'Question': 'Q18',
            'Issue': 'Multiple marks detected',
            'Confidence': '68%',
            'Action': 'Manual Review'
        }
    ]
    
    for item in flagged_items:
        with st.expander(f"⚠️ {item['Exam']} - {item['Student']} - {item['Question']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Issue:** {item['Issue']}")
                st.write(f"**Confidence:** {item['Confidence']}")
            
            with col2:
                st.write("**Original Image:**")
                # Placeholder for actual image
                st.info("Image preview would appear here")
            
            with col3:
                st.write("**Corrective Action:**")
                corrected_answer = st.selectbox(
                    "Select correct answer",
                    ['A', 'B', 'C', 'D'],
                    key=f"correct_{item['Student']}_{item['Question']}"
                )
                
                if st.button(f"✅ Apply Correction", key=f"apply_{item['Student']}_{item['Question']}"):
                    st.success("Correction applied successfully")

def settings_page():
    """System settings and configuration"""
    
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ System Settings & Configuration</h1>
        <p>Configure system parameters and processing options</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Processing settings
    st.subheader("🔧 Processing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Performance Settings**")
        
        max_workers = st.slider(
            "Maximum Parallel Workers",
            min_value=1,
            max_value=16,
            value=4,
            help="Number of parallel processing threads"
        )
        
        quality_threshold = st.slider(
            "Quality Threshold (%)",
            min_value=80,
            max_value=100,
            value=95,
            help="Minimum quality score for auto-approval"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=70,
            max_value=100,
            value=85,
            help="Minimum confidence for automatic processing"
        )
    
    with col2:
        st.write("**Output Settings**")
        
        default_export = st.selectbox(
            "Default Export Format",
            ["PDF", "Excel", "CSV", "JSON"]
        )
        
        auto_backup = st.checkbox("Enable automatic backups", value=True)
        
        retention_days = st.number_input(
            "Data Retention (days)",
            min_value=30,
            max_value=365,
            value=90,
            help="How long to keep processing results"
        )
    
    # Notification settings
    st.subheader("📧 Notification Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        email_notifications = st.checkbox("Enable email notifications", value=False)
        
        if email_notifications:
            smtp_server = st.text_input("SMTP Server", placeholder="smtp.gmail.com")
            smtp_port = st.number_input("SMTP Port", value=587)
            email_username = st.text_input("Email Username")
            email_password = st.text_input("Email Password", type="password")
    
    with col2:
        webhook_notifications = st.checkbox("Enable webhook notifications", value=False)
        
        if webhook_notifications:
            webhook_url = st.text_input("Webhook URL", placeholder="https://your-webhook-url.com")
            webhook_events = st.multiselect(
                "Events to notify",
                ["Processing Complete", "Error Occurred", "Quality Issue", "Batch Finished"]
            )
    
    # System monitoring
    st.subheader("📊 System Monitoring")
    
    monitoring_enabled = st.checkbox("Enable system monitoring", value=True)
    
    if monitoring_enabled:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_alert = st.number_input("CPU Alert Threshold (%)", value=80)
        
        with col2:
            memory_alert = st.number_input("Memory Alert Threshold (%)", value=85)
        
        with col3:
            disk_alert = st.number_input("Disk Alert Threshold (%)", value=90)
    
    # Security settings
    st.subheader("🔒 Security Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        file_encryption = st.checkbox("Enable file encryption", value=True)
        audit_logging = st.checkbox("Enable audit logging", value=True)
        secure_deletion = st.checkbox("Secure file deletion", value=True)
    
    with col2:
        session_timeout = st.number_input("Session Timeout (minutes)", value=30)
        max_file_size = st.number_input("Max File Size (MB)", value=50)
        allowed_formats = st.multiselect(
            "Allowed File Formats",
            ["PNG", "JPG", "JPEG", "TIFF", "BMP"],
            default=["PNG", "JPG", "JPEG", "TIFF", "BMP"]
        )
    
    # Save settings
    if st.button("💾 Save Configuration", type="primary", key="settings_save_btn"):
        st.success("✅ Settings saved successfully!")
        
        # Display saved configuration
        with st.expander("📋 Saved Configuration Summary"):
            config = {
                "Performance": {
                    "max_workers": max_workers,
                    "quality_threshold": quality_threshold,
                    "confidence_threshold": confidence_threshold
                },
                "Output": {
                    "default_export": default_export,
                    "auto_backup": auto_backup,
                    "retention_days": retention_days
                },
                "Security": {
                    "file_encryption": file_encryption,
                    "audit_logging": audit_logging,
                    "session_timeout": session_timeout
                }
            }
            st.json(config)

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'dashboard'
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(45deg, #2c3e50, #3498db); border-radius: 10px; color: white; margin-bottom: 2rem;">
            <h2>🎯 ScoreVision Pro</h2>
            <p>Professional OMR Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation menu
        st.subheader("📋 Navigation")
        
        pages = {
            "📊 Dashboard": "dashboard",
            "📤 Upload & Process": "upload", 
            "📊 Batch Processing": "batch",
            "🔍 Review Results": "review",
            "⚙️ Settings": "settings"
        }
        
        for page_name, page_key in pages.items():
            if st.button(page_name, use_container_width=True, 
                        type="primary" if st.session_state.page == page_key else "secondary",
                        key=f"nav_{page_key}_btn"):
                st.session_state.page = page_key
                st.rerun()
        
        # System info sidebar
        st.markdown("---")
        st.subheader("🛡️ System Info")
        
        processor = OMRProcessor()
        backend_status = processor.check_backend_status()
        
        if backend_status:
            st.success("🟢 Backend: Online")
        else:
            st.error("🔴 Backend: Offline")
        
        st.info("💾 Version: 1.0.0")
        st.info("🏢 Enterprise Edition")
        
        # Quick stats
        st.markdown("---")
        st.subheader("📈 Quick Stats")
        st.metric("Active Sessions", "23")
        st.metric("Today's Processing", "1,247")
        st.metric("System Uptime", "99.9%")
    
    # Main content area
    if st.session_state.page == 'dashboard':
        dashboard_page()
    elif st.session_state.page == 'upload':
        upload_page()
    elif st.session_state.page == 'batch':
        batch_processing_page()
    elif st.session_state.page == 'review':
        review_page()
    elif st.session_state.page == 'settings':
        settings_page()

if __name__ == "__main__":
    main()