import os
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import threading
import logging

# Import our model
from train import NetworkIntrusionDetectionModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'pcap', 'pcapng', 'cap'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store analysis tasks and results
analysis_tasks = {}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def analyze_pcap(file_path, task_id):
    """Analyze pcap file using the trained model"""
    try:
        # Load the model
        model = NetworkIntrusionDetectionModel(model_path="./models")
        
        # Update task status
        analysis_tasks[task_id]['status'] = 'processing'
        analysis_tasks[task_id]['progress'] = 10
        
        # Perform prediction
        result = model.predict(file_path)
        
        # Update task status with results
        analysis_tasks[task_id]['status'] = 'completed'
        analysis_tasks[task_id]['progress'] = 100
        analysis_tasks[task_id]['result'] = result
        analysis_tasks[task_id]['completed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Analysis completed for task {task_id}")
    
    except Exception as e:
        # Update task status with error
        analysis_tasks[task_id]['status'] = 'failed'
        analysis_tasks[task_id]['progress'] = 100
        analysis_tasks[task_id]['error'] = str(e)
        logger.error(f"Analysis failed for task {task_id}: {e}")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Please upload a .pcap, .pcapng, or .cap file'}), 400
    
    try:
        # Generate a unique filename
        original_filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4()}_{original_filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(file_path)
        
        # Generate a task ID
        task_id = str(uuid.uuid4())
        
        # Create task
        analysis_tasks[task_id] = {
            'id': task_id,
            'filename': original_filename,
            'file_path': file_path,
            'status': 'queued',
            'progress': 0,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'completed_at': None,
            'result': None,
            'error': None
        }
        
        # Start analysis in a background thread
        thread = threading.Thread(target=analyze_pcap, args=(file_path, task_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully and analysis started',
            'task_id': task_id
        })
    
    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """Get the status of an analysis task"""
    if task_id not in analysis_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = analysis_tasks[task_id]
    
    return jsonify({
        'id': task['id'],
        'filename': task['filename'],
        'status': task['status'],
        'progress': task['progress'],
        'created_at': task['created_at'],
        'completed_at': task['completed_at'],
        'result': task['result'],
        'error': task['error']
    })

@app.route('/results/<task_id>', methods=['GET'])
def get_results(task_id):
    """Get the full results of a completed analysis"""
    if task_id not in analysis_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = analysis_tasks[task_id]
    
    if task['status'] != 'completed':
        return jsonify({'error': 'Analysis not yet completed', 'status': task['status']}), 400
    
    return jsonify({
        'id': task['id'],
        'filename': task['filename'],
        'status': task['status'],
        'created_at': task['created_at'],
        'completed_at': task['completed_at'],
        'result': task['result']
    })

@app.route('/history', methods=['GET'])
def get_history():
    """Get the history of analysis tasks"""
    # Return a list of tasks with basic information
    task_list = [
        {
            'id': task_id,
            'filename': task['filename'],
            'status': task['status'],
            'created_at': task['created_at'],
            'completed_at': task['completed_at']
        }
        for task_id, task in analysis_tasks.items()
    ]
    
    # Sort by creation time, most recent first
    task_list.sort(key=lambda x: x['created_at'], reverse=True)
    
    return jsonify(task_list)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear analysis history"""
    global analysis_tasks
    analysis_tasks = {}
    
    return jsonify({'success': True, 'message': 'Analysis history cleared'})

@app.route('/check_model', methods=['GET'])
def check_model():
    """Check if the model files exist"""
    model_path = "./models"
    required_files = [
        'autoencoder_model.h5',
        'encoder_model.h5',
        'xgboost_classifier.model',
        'scaler.pkl',
        'label_encoder.pkl',
        'feature_columns.pkl',
        'attack_mapping.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        return jsonify({
            'status': 'missing',
            'missing_files': missing_files,
            'message': 'Some model files are missing. Please train the model first.'
        })
    
    return jsonify({
        'status': 'ready',
        'message': 'Model is ready for analysis.'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)