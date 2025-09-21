import sys
import os

# Add the core directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(current_dir, 'app', 'core')
sys.path.insert(0, core_dir)

from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_cors import CORS
import json
import numpy as np
import time
from datetime import datetime
from werkzeug.utils import secure_filename
import uuid
import threading
from app.core.omr_processor import ProductionOMRProcessor
from app.core.universal_processor import UniversalOMRProcessor
from app.core.excel_converter import ExcelConverter
from app.core.quality_validator import QualityValidator
from app.core.batch_processor import BatchProcessor
from app.core.result_exporter import ResultExporter
from app.core.system_info import system_info

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['ANSWER_KEYS_FOLDER'] = 'answer_keys'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

CORS(app)

# Global variables for batch processing
batch_sessions = {}
processing_status = {}

# Initialize core components
omr_processor = ProductionOMRProcessor()
universal_processor = UniversalOMRProcessor()
excel_converter = ExcelConverter()
quality_validator = QualityValidator()
batch_processor = BatchProcessor()
result_exporter = ResultExporter()

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
ALLOWED_EXCEL_EXTENSIONS = {'xlsx', 'xls'}
ALLOWED_JSON_EXTENSIONS = {'json'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    """Home page - redirect to dashboard"""
    return render_template('dashboard.html')

@app.route('/upload')
def upload_page():
    """Upload page for single OMR sheet processing"""
    return render_template('upload.html')

@app.route('/batch')
def batch_page():
    """Batch processing page"""
    return render_template('batch.html')

@app.route('/answer-key')
def answer_key_page():
    """Answer key management page"""
    return render_template('answer-key.html')

@app.route('/validation')
def validation_page():
    """Quality validation page"""
    return render_template('validation.html')

@app.route('/results')
def results_page():
    """Results viewing page"""
    return render_template('results.html')

@app.route('/health')
def health_check():
    """Health check endpoint for testing"""
    return jsonify({
        'status': 'healthy',
        'message': 'OMR Evaluation System is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/upload-omr', methods=['POST'])
def upload_omr():
    """Upload and process single OMR sheet"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        # Generate unique filename
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save file
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'size': os.path.getsize(filepath)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload-answer-key', methods=['POST'])
def upload_answer_key():
    """Upload and process answer key (Excel or JSON)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        if file_ext in ALLOWED_EXCEL_EXTENSIONS:
            # Process Excel file
            try:
                sheets_data = excel_converter.parse_excel(file)
                if len(sheets_data) > 1:
                    # Multiple sheets - let user choose
                    return jsonify({
                        'success': True,
                        'type': 'multiple_sheets',
                        'sheets': list(sheets_data.keys()),
                        'data': sheets_data
                    })
                else:
                    # Single sheet
                    sheet_name = list(sheets_data.keys())[0]
                    answer_key = sheets_data[sheet_name]
                    
                    # Save as JSON
                    json_filename = f"{uuid.uuid4()}_answer_key.json"
                    json_filepath = os.path.join(app.config['ANSWER_KEYS_FOLDER'], json_filename)
                    os.makedirs(app.config['ANSWER_KEYS_FOLDER'], exist_ok=True)
                    
                    with open(json_filepath, 'w') as f:
                        json.dump(answer_key, f, indent=2)
                    
                    return jsonify({
                        'success': True,
                        'type': 'single_sheet',
                        'filename': json_filename,
                        'data': answer_key,
                        'validation': quality_validator.validate_answer_key(answer_key)
                    })
                    
            except Exception as e:
                return jsonify({'error': f'Error processing Excel file: {str(e)}'}), 400
                
        elif file_ext in ALLOWED_JSON_EXTENSIONS:
            # Process JSON file
            try:
                content = file.read().decode('utf-8')
                answer_key = json.loads(content)
                
                # Validate JSON structure
                validation = quality_validator.validate_answer_key(answer_key)
                
                if validation['valid']:
                    # Save JSON file
                    json_filename = f"{uuid.uuid4()}_answer_key.json"
                    json_filepath = os.path.join(app.config['ANSWER_KEYS_FOLDER'], json_filename)
                    os.makedirs(app.config['ANSWER_KEYS_FOLDER'], exist_ok=True)
                    
                    with open(json_filepath, 'w') as f:
                        json.dump(answer_key, f, indent=2)
                    
                    return jsonify({
                        'success': True,
                        'type': 'json',
                        'filename': json_filename,
                        'data': answer_key,
                        'validation': validation
                    })
                else:
                    return jsonify({
                        'error': 'Invalid JSON structure',
                        'validation': validation
                    }), 400
                    
            except json.JSONDecodeError as e:
                return jsonify({'error': f'Invalid JSON format: {str(e)}'}), 400
            except Exception as e:
                return jsonify({'error': f'Error processing JSON file: {str(e)}'}), 400
        else:
            return jsonify({'error': 'Invalid file type. Please upload Excel (.xlsx, .xls) or JSON files.'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-selected-sheet', methods=['POST'])
def save_selected_sheet():
    """Save the selected Excel sheet as a JSON answer key"""
    try:
        data = request.get_json()
        if not data or 'sheet_name' not in data or 'answer_key' not in data:
            return jsonify({'error': 'Missing sheet data'}), 400
        
        sheet_name = data['sheet_name']
        answer_key = data['answer_key']
        
        # Convert numpy/pandas types to native Python types for JSON serialization
        def convert_types(obj):
            """Convert numpy/pandas types to native Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                # Handle numpy arrays
                if obj.size == 1:
                    return obj.item()  # Single element array
                else:
                    return obj.tolist()  # Multi-element array
            elif hasattr(obj, 'item') and hasattr(obj, 'size'):
                # Numpy scalar with size attribute
                if obj.size == 1:
                    return obj.item()
                else:
                    return obj.tolist()
            elif hasattr(obj, 'tolist'):
                # Other numpy-like objects
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        # Clean the answer key data
        cleaned_answer_key = convert_types(answer_key)
        
        # Validate the answer key
        validation = quality_validator.validate_answer_key(cleaned_answer_key)
        
        # Save as JSON file regardless of validation (user choice)
        json_filename = f"{uuid.uuid4()}_answer_key_{sheet_name}.json"
        json_filepath = os.path.join(app.config['ANSWER_KEYS_FOLDER'], json_filename)
        os.makedirs(app.config['ANSWER_KEYS_FOLDER'], exist_ok=True)
        
        with open(json_filepath, 'w') as f:
            json.dump(cleaned_answer_key, f, indent=2, cls=NumpyEncoder)
        
        return jsonify({
            'success': True,
            'filename': json_filename,
            'sheet_name': sheet_name,
            'validation': validation,
            'questions_count': len(cleaned_answer_key)
        })
        
    except Exception as e:
        print(f"Error in save_selected_sheet: {str(e)}")  # Debug logging
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/batch-upload', methods=['POST'])
def batch_upload():
    """Handle batch upload of multiple OMR files with single answer key"""
    try:
        # Get uploaded files
        omr_files = request.files.getlist('omr_files')
        answer_key_file = request.files.get('answer_key')
        
        if not omr_files:
            return jsonify({'error': 'No OMR files uploaded'}), 400
        
        if not answer_key_file:
            return jsonify({'error': 'No answer key uploaded'}), 400
        
        # Get settings
        max_workers = int(request.form.get('max_workers', 2))
        quality_check = request.form.get('quality_check', 'standard')
        auto_export = request.form.get('auto_export', 'false') == 'true'
        
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        
        # Save uploaded files
        uploaded_omr_files = []
        upload_folder = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)
        
        for omr_file in omr_files:
            if omr_file and omr_file.filename and allowed_file(omr_file.filename, ALLOWED_IMAGE_EXTENSIONS):
                filename = f"{batch_id}_{secure_filename(omr_file.filename)}"
                filepath = os.path.join(upload_folder, filename)
                omr_file.save(filepath)
                uploaded_omr_files.append(filename)
        
        if not uploaded_omr_files:
            return jsonify({'error': 'No valid OMR files found'}), 400
        
        # Save answer key
        answer_key_filename = None
        if answer_key_file and answer_key_file.filename and (allowed_file(answer_key_file.filename, ALLOWED_EXCEL_EXTENSIONS) or 
                               allowed_file(answer_key_file.filename, ALLOWED_JSON_EXTENSIONS)):
            answer_key_filename = f"{batch_id}_{secure_filename(answer_key_file.filename)}"
            answer_key_path = os.path.join(app.config['ANSWER_KEYS_FOLDER'], answer_key_filename)
            os.makedirs(app.config['ANSWER_KEYS_FOLDER'], exist_ok=True)
            answer_key_file.save(answer_key_path)
        
        if not answer_key_filename:
            return jsonify({'error': 'Invalid answer key file'}), 400
        
        # Store batch session
        batch_sessions[batch_id] = {
            'omr_files': uploaded_omr_files,
            'answer_key': answer_key_filename,
            'settings': {
                'max_workers': max_workers,
                'quality_check': quality_check,
                'auto_export': auto_export
            },
            'status': 'uploaded',
            'created_at': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'files_uploaded': len(uploaded_omr_files),
            'answer_key': answer_key_filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-process', methods=['POST'])
def batch_process():
    """Start processing a batch that was already uploaded"""
    try:
        data = request.get_json()
        batch_id = data.get('batch_id')
        
        if not batch_id or batch_id not in batch_sessions:
            return jsonify({'error': 'Invalid batch ID'}), 400
        
        batch_info = batch_sessions[batch_id]
        
        # Update status
        batch_info['status'] = 'processing'
        processing_status[batch_id] = {
            'status': 'processing',
            'progress': 0,
            'total_files': len(batch_info['omr_files']),
            'processed_files': 0,
            'results': [],
            'errors': [],
            'start_time': datetime.now().isoformat()
        }
        
        # Start batch processing in background
        def process_batch():
            batch_processor.process_batch_session(batch_id, batch_info, processing_status)
        
        thread = threading.Thread(target=process_batch)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': 'Batch processing started'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-omr', methods=['POST'])
def process_omr():
    """Process OMR sheet with answer key"""
    try:
        data = request.get_json()
        
        if not data or 'omr_file' not in data or 'answer_key_file' not in data:
            return jsonify({'error': 'Missing required data'}), 400
        
        # Ensure we have string values, not dicts
        omr_file = data['omr_file']
        answer_key_file = data['answer_key_file']
        
        if isinstance(omr_file, dict):
            # If it's a dict, try to extract filename
            omr_file = omr_file.get('filename', omr_file.get('name', omr_file.get('file', '')))
            
        if isinstance(answer_key_file, dict):
            # If it's a dict, try to extract filename
            answer_key_file = answer_key_file.get('filename', answer_key_file.get('name', answer_key_file.get('file', '')))
        
        if not omr_file or not answer_key_file:
            return jsonify({'error': f'Invalid file data - OMR: "{omr_file}", Answer Key: "{answer_key_file}"'}), 400
        
        omr_filepath = os.path.join(app.config['UPLOAD_FOLDER'], omr_file)
        answer_key_filepath = os.path.join(app.config['ANSWER_KEYS_FOLDER'], answer_key_file)
        
        if not os.path.exists(omr_filepath):
            return jsonify({'error': 'OMR file not found'}), 404
        
        if not os.path.exists(answer_key_filepath):
            return jsonify({'error': 'Answer key file not found'}), 404
        
        # Load answer key
        with open(answer_key_filepath, 'r') as f:
            answer_key = json.load(f)
        
        # Check for processor type preference
        processor_type = data.get('processor_type', 'standard')
        
        # Process OMR sheet with selected processor
        if processor_type == 'universal':
            result = universal_processor.process_omr_sheet(omr_filepath, answer_key)
        else:
            result = omr_processor.process_omr_sheet(omr_filepath, answer_key)
        
        # Convert numpy types in the result
        def convert_types(obj):
            """Convert numpy/pandas types to native Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                # Handle numpy arrays
                if obj.size == 1:
                    return obj.item()  # Single element array
                else:
                    return obj.tolist()  # Multi-element array
            elif hasattr(obj, 'item') and hasattr(obj, 'size'):
                # Numpy scalar with size attribute
                if obj.size == 1:
                    return obj.item()
                else:
                    return obj.tolist()
            elif hasattr(obj, 'tolist'):
                # Other numpy-like objects
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        # Clean the result data
        cleaned_result = convert_types(result)
        
        # Validate results
        quality_report = quality_validator.validate_omr_result(cleaned_result, omr_filepath)
        cleaned_quality_report = convert_types(quality_report)
        
        # Save results
        result_id = str(uuid.uuid4())
        result_data = {
            'id': result_id,
            'timestamp': datetime.now().isoformat(),
            'omr_file': data['omr_file'],
            'answer_key_file': data['answer_key_file'],
            'result': cleaned_result,
            'quality_report': cleaned_quality_report
        }
        
        result_filepath = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
        os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
        
        with open(result_filepath, 'w') as f:
            json.dump(result_data, f, indent=2, cls=NumpyEncoder)
        
        return jsonify({
            'success': True,
            'result_id': result_id,
            'result': cleaned_result,
            'quality_report': cleaned_quality_report
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-batch-processing', methods=['POST'])
def start_batch_processing():
    """Start batch processing of multiple OMR sheets"""
    try:
        data = request.get_json()
        
        if not data or 'omr_files' not in data or 'answer_key_file' not in data:
            return jsonify({'error': 'Missing required data'}), 400
        
        # Ensure answer_key_file is a string
        answer_key_file = data['answer_key_file']
        if isinstance(answer_key_file, dict):
            answer_key_file = answer_key_file.get('filename', answer_key_file.get('name', ''))
        
        # Ensure omr_files is a list of strings
        omr_files = data['omr_files']
        if isinstance(omr_files, dict):
            omr_files = [omr_files.get('filename', omr_files.get('name', ''))]
        elif isinstance(omr_files, list):
            processed_files = []
            for file_item in omr_files:
                if isinstance(file_item, dict):
                    filename = file_item.get('filename', file_item.get('name', ''))
                    if filename:
                        processed_files.append(filename)
                elif isinstance(file_item, str):
                    processed_files.append(file_item)
            omr_files = processed_files
        
        if not answer_key_file or not omr_files:
            return jsonify({'error': 'Invalid file data received'}), 400
        
        session_id = str(uuid.uuid4())
        
        # Initialize batch session
        batch_sessions[session_id] = {
            'omr_files': omr_files,
            'answer_key_file': answer_key_file,
            'total_files': len(omr_files),
            'processed': 0,
            'results': [],
            'errors': [],
            'status': 'processing',
            'start_time': time.time()
        }
        
        # Start batch processing in background thread
        def process_batch():
            try:
                answer_key_filepath = os.path.join(app.config['ANSWER_KEYS_FOLDER'], answer_key_file)
                
                with open(answer_key_filepath, 'r') as f:
                    answer_key = json.load(f)
                
                batch_processor.process_batch(
                    session_id,
                    omr_files,
                    answer_key,
                    app.config['UPLOAD_FOLDER'],
                    app.config['RESULTS_FOLDER'],
                    batch_sessions
                )
                
            except Exception as e:
                batch_sessions[session_id]['status'] = 'error'
                batch_sessions[session_id]['error'] = str(e)
        
        # Start processing thread
        thread = threading.Thread(target=process_batch)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-status/<session_id>')
def get_batch_status(session_id):
    """Get batch processing status"""
    try:
        if session_id not in batch_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = batch_sessions[session_id]
        return jsonify({
            'session_id': session_id,
            'status': session['status'],
            'total_files': session['total_files'],
            'processed': session['processed'],
            'progress': (session['processed'] / session['total_files']) * 100,
            'results_count': len(session['results']),
            'errors_count': len(session['errors']),
            'elapsed_time': time.time() - session['start_time']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/list-results')
def list_results():
    """List all processing results"""
    try:
        results = []
        results_dir = app.config['RESULTS_FOLDER']
        
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(results_dir, filename)
                    stat = os.stat(filepath)
                    results.append({
                        'filename': filename,
                        'path': filepath,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'type': 'JSON'
                    })
                elif filename.endswith(('.pdf', '.xlsx', '.csv')):
                    filepath = os.path.join(results_dir, filename)
                    stat = os.stat(filepath)
                    file_type = filename.split('.')[-1].upper()
                    results.append({
                        'filename': filename,
                        'path': filepath,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'type': file_type
                    })
        
        # Sort by modification time (newest first)
        results.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validation-reports')
def get_validation_reports():
    """Get quality validation reports"""
    try:
        reports = quality_validator.get_all_validation_reports(app.config['RESULTS_FOLDER'])
        return jsonify({'reports': reports})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-results', methods=['POST'])
def export_results():
    """Export results in specified format"""
    try:
        data = request.get_json()
        
        if not data or 'result_ids' not in data or 'format' not in data:
            return jsonify({'error': 'Missing required data'}), 400
        
        export_format = data['format'].lower()
        if export_format not in ['pdf', 'excel', 'csv']:
            return jsonify({'error': 'Invalid export format'}), 400
        
        # Export results
        export_filepath = result_exporter.export_results(
            data['result_ids'],
            export_format,
            app.config['RESULTS_FOLDER']
        )
        
        return jsonify({
            'success': True,
            'download_url': f'/api/download/{os.path.basename(export_filepath)}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download result file"""
    try:
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(filepath, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/create-answer-key', methods=['POST'])
def create_answer_key():
    """Create answer key from form data"""
    try:
        data = request.get_json()
        
        if not data or 'questions' not in data or 'choices' not in data:
            return jsonify({'error': 'Missing required data'}), 400
        
        num_questions = int(data['questions'])
        num_choices = int(data['choices'])
        
        # Create template answer key
        answer_key = {
            'metadata': {
                'title': data.get('title', 'Custom Answer Key'),
                'questions': num_questions,
                'choices': num_choices,
                'created': datetime.now().isoformat()
            },
            'answers': {}
        }
        
        # Initialize with empty answers (to be filled by user)
        for i in range(1, num_questions + 1):
            answer_key['answers'][str(i)] = None
        
        # Save answer key
        filename = f"{uuid.uuid4()}_answer_key.json"
        filepath = os.path.join(app.config['ANSWER_KEYS_FOLDER'], filename)
        os.makedirs(app.config['ANSWER_KEYS_FOLDER'], exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(answer_key, f, indent=2)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'answer_key': answer_key
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting OMR Evaluation System...")
    print("📁 Setting up directories...")
    
    # Ensure directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['ANSWER_KEYS_FOLDER'], exist_ok=True)
    
    print(f"✅ Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"✅ Results folder: {app.config['RESULTS_FOLDER']}")
    print(f"✅ Answer keys folder: {app.config['ANSWER_KEYS_FOLDER']}")
    print("🌐 Starting Flask server on http://localhost:5000")
    print("🔧 Debug mode: ON")
    print("📡 Host: 0.0.0.0 (accessible from network)")
    print("⚡ Ready to process OMR sheets!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

# Additional routes for professional demo data
@app.route('/api/sample-data')
def get_sample_data():
    """Get sample exam data and statistics for dashboard"""
    try:
        # Load sample results
        demo_results_dir = os.path.join(current_dir, 'sample_data', 'results')
        sample_data = {
            'recent_batches': [],
            'system_stats': {
                'total_processed': 50247,
                'accuracy_rate': 99.7,
                'avg_processing_time': 8.2,
                'institutions_served': 147,
                'active_sessions': 23
            },
            'performance_metrics': {
                'daily_throughput': [1240, 1356, 1189, 1456, 1523, 1334, 1401],
                'accuracy_trend': [99.5, 99.6, 99.7, 99.8, 99.7, 99.9, 99.7],
                'error_rates': [0.3, 0.2, 0.3, 0.2, 0.3, 0.1, 0.3]
            }
        }
        
        # Load actual sample results if they exist
        if os.path.exists(demo_results_dir):
            for filename in os.listdir(demo_results_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(demo_results_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            result_data = json.load(f)
                            sample_data['recent_batches'].append(result_data)
                    except:
                        continue
        
        return jsonify(sample_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample-answer-keys')
def get_sample_answer_keys():
    """Get list of available sample answer keys"""
    try:
        demo_keys_dir = os.path.join(current_dir, 'sample_data', 'answer_keys')
        sample_keys = []
        
        if os.path.exists(demo_keys_dir):
            for filename in os.listdir(demo_keys_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(demo_keys_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            key_data = json.load(f)
                            sample_keys.append({
                                'filename': filename,
                                'exam_info': key_data.get('exam_info', {}),
                                'total_questions': key_data.get('exam_info', {}).get('total_questions', 0),
                                'subject': key_data.get('exam_info', {}).get('subject', 'Unknown')
                            })
                    except:
                        continue
        
        return jsonify({'sample_keys': sample_keys})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard_page():
    """Professional dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/system-status')
def get_system_status():
    """Get comprehensive system status"""
    try:
        metrics = system_info.get_system_metrics()
        app_status = system_info.get_application_status()
        
        return jsonify({
            'status': 'operational',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'application': app_status,
            'uptime': metrics.get('uptime', 'Unknown')
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500