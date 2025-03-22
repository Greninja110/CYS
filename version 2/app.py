import os
import uuid
import json
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import threading
import logging
from logging.handlers import RotatingFileHandler

# Import our model
from train import KitsuneNIDSModel

# Configure logging
os.makedirs('logs', exist_ok=True)
log_file = os.path.join('logs', f'app_{datetime.now().strftime("%Y%m%d")}.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Starting Kitsune NIDS application")

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'pcap', 'pcapng', 'cap'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store analysis tasks and results
analysis_tasks = {}

# PATCH: Monkey patch the KitsuneNIDSModel.load_models method to fix the loading issue
original_load_models = KitsuneNIDSModel.load_models

def patched_load_models(self):
    """Patched version of load_models that handles file paths correctly"""
    logger.info("Using patched load_models method")
    
    try:
        # First try loading directly from the main model directory
        model_dir = self.model_path
        logger.info(f"Attempting to load models from main directory: {model_dir}")
        
        # Load encoder directly from main directory
        encoder_path = os.path.join(model_dir, 'encoder_model.h5')
        if os.path.exists(encoder_path):
            try:
                from tensorflow.keras.models import load_model
                self.encoder = load_model(encoder_path)
                logger.info(f"Successfully loaded encoder from {encoder_path}")
            except Exception as e:
                logger.error(f"Error loading encoder: {e}")
                return False
        else:
            logger.error(f"Encoder model not found at {encoder_path}")
            return False
        
        # Load classifier
        xgb_path = os.path.join(model_dir, 'xgboost_classifier.model')
        lgb_path = os.path.join(model_dir, 'lightgbm_classifier.txt')
        
        if os.path.exists(xgb_path):
            try:
                import xgboost as xgb
                self.classifier = xgb.Booster()
                self.classifier.load_model(xgb_path)
                logger.info(f"Successfully loaded XGBoost classifier from {xgb_path}")
            except Exception as e:
                logger.error(f"Error loading XGBoost classifier: {e}")
                return False
        elif os.path.exists(lgb_path):
            try:
                import lightgbm as lgb
                self.classifier = lgb.Booster(model_file=lgb_path)
                logger.info(f"Successfully loaded LightGBM classifier from {lgb_path}")
            except Exception as e:
                logger.error(f"Error loading LightGBM classifier: {e}")
                return False
        else:
            logger.error("No classifier model found")
            return False
        
        # Load scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                import joblib
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Successfully loaded scaler from {scaler_path}")
            except Exception as e:
                logger.error(f"Error loading scaler: {e}")
                return False
        else:
            logger.error(f"Scaler not found at {scaler_path}")
            return False
        
        # Load label encoder
        label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        if os.path.exists(label_encoder_path):
            try:
                import joblib
                self.label_encoder = joblib.load(label_encoder_path)
                logger.info(f"Successfully loaded label encoder from {label_encoder_path}")
            except Exception as e:
                logger.error(f"Error loading label encoder: {e}")
                return False
        else:
            logger.error(f"Label encoder not found at {label_encoder_path}")
            return False
        
        # Load feature columns
        feature_columns_path = os.path.join(model_dir, 'feature_columns.pkl')
        if os.path.exists(feature_columns_path):
            try:
                import pickle
                with open(feature_columns_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)
                logger.info(f"Successfully loaded feature columns ({len(self.feature_columns)} features)")
            except Exception as e:
                logger.error(f"Error loading feature columns: {e}")
                return False
        else:
            logger.error(f"Feature columns not found at {feature_columns_path}")
            return False
        
        # Load attack mapping
        attack_mapping_path = os.path.join(model_dir, 'attack_mapping.pkl')
        if os.path.exists(attack_mapping_path):
            try:
                import pickle
                with open(attack_mapping_path, 'rb') as f:
                    self.attack_mapping = pickle.load(f)
                logger.info(f"Successfully loaded attack mapping")
            except Exception as e:
                logger.warning(f"Error loading attack mapping: {e}")
                # Not critical, continue with default mapping
        else:
            logger.warning(f"Attack mapping not found at {attack_mapping_path}")
            # Not critical, continue with default mapping
        
        logger.info("All models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Unhandled error in patched load_models: {e}")
        logger.error(traceback.format_exc())
        return False

# Apply the patch
KitsuneNIDSModel.load_models = patched_load_models

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def analyze_pcap(file_path, task_id):
    """Analyze pcap file using the trained model"""
    try:
        logger.info(f"Starting analysis for task {task_id}, file: {file_path}")
        
        # Update task status
        analysis_tasks[task_id]['status'] = 'processing'
        analysis_tasks[task_id]['progress'] = 5
        
        # Verify file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The uploaded file was not found at {file_path}")
        
        # Update progress
        analysis_tasks[task_id]['progress'] = 10
        
        # Create model instance
        logger.info(f"Creating model instance with model path: ./models")
        model = KitsuneNIDSModel(model_path="./models")
        
        # Load models
        logger.info("Loading model components")
        analysis_tasks[task_id]['progress'] = 20
        load_success = model.load_models()
        
        if not load_success:
            raise RuntimeError("Failed to load model components. Check logs for details.")
        
        # Update progress
        analysis_tasks[task_id]['progress'] = 40
        logger.info(f"Model loaded successfully, starting PCAP analysis for {file_path}")
        
        # Check file size before processing
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"PCAP file size: {file_size_mb:.2f} MB")
        
        # Perform prediction
        analysis_tasks[task_id]['progress'] = 50
        result = model.predict(file_path)
        
        # Check if prediction failed
        if 'error' in result:
            analysis_tasks[task_id]['status'] = 'failed'
            analysis_tasks[task_id]['progress'] = 100
            analysis_tasks[task_id]['error'] = result['error']
            logger.error(f"Analysis failed for task {task_id}: {result['error']}")
            return
        
        # Update task status with results
        analysis_tasks[task_id]['status'] = 'completed'
        analysis_tasks[task_id]['progress'] = 100
        analysis_tasks[task_id]['result'] = result
        analysis_tasks[task_id]['completed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Analysis completed for task {task_id}")
    
    except Exception as e:
        logger.error(f"Error during analysis for task {task_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update task status with error
        analysis_tasks[task_id]['status'] = 'failed'
        analysis_tasks[task_id]['progress'] = 100
        analysis_tasks[task_id]['error'] = str(e)
        analysis_tasks[task_id]['error_details'] = traceback.format_exc()

@app.route('/')
def index():
    """Render the main page"""
    logger.info("Serving index page")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start analysis"""
    logger.info("File upload requested")
    
    if 'file' not in request.files:
        logger.warning("No file part in the request")
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("No file selected")
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        logger.warning(f"File type not allowed: {file.filename}")
        return jsonify({'error': 'File type not allowed. Please upload a .pcap, .pcapng, or .cap file'}), 400
    
    try:
        # Generate a unique filename
        original_filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4()}_{original_filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        logger.info(f"Saving uploaded file to {file_path}")
        
        # Save the file
        file.save(file_path)
        
        # Verify file was saved correctly
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Failed to save file to {file_path}")
        
        # Generate a task ID
        task_id = str(uuid.uuid4())
        logger.info(f"Created task {task_id} for file {original_filename}")
        
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
        logger.info(f"Started analysis thread for task {task_id}")
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully and analysis started',
            'task_id': task_id
        })
    
    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """Get the status of an analysis task"""
    if task_id not in analysis_tasks:
        logger.warning(f"Task not found: {task_id}")
        return jsonify({'error': 'Task not found'}), 404
    
    task = analysis_tasks[task_id]
    logger.debug(f"Returning status for task {task_id}: {task['status']}")
    
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
        logger.warning(f"Task not found: {task_id}")
        return jsonify({'error': 'Task not found'}), 404
    
    task = analysis_tasks[task_id]
    
    if task['status'] != 'completed':
        logger.warning(f"Attempted to get results for incomplete task {task_id}: {task['status']}")
        return jsonify({'error': 'Analysis not yet completed', 'status': task['status']}), 400
    
    logger.info(f"Returning results for task {task_id}")
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
    logger.info("Retrieving analysis history")
    
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
    
    logger.info(f"Returning {len(task_list)} history items")
    return jsonify(task_list)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear analysis history"""
    logger.info("Clearing analysis history")
    global analysis_tasks
    analysis_tasks = {}
    
    return jsonify({'success': True, 'message': 'Analysis history cleared'})

@app.route('/check_model', methods=['GET'])
def check_model():
    """Check if the model files exist"""
    logger.info("Checking for model files")
    model_path = "./models"
    required_files = [
        'autoencoder_model.h5',
        'encoder_model.h5',
        'xgboost_classifier.model',  # Or lightgbm_classifier.txt
        'scaler.pkl',
        'label_encoder.pkl',
        'feature_columns.pkl',
        'attack_mapping.pkl'
    ]
    
    # Log all files in the model directory for debugging
    if os.path.exists(model_path):
        logger.info(f"Files in model directory {model_path}:")
        for file in os.listdir(model_path):
            file_path = os.path.join(model_path, file)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path) / 1024
                logger.info(f"  {file} - {file_size:.2f} KB")
    else:
        logger.error(f"Model directory {model_path} does not exist")
    
    missing_files = []
    for file in required_files:
        # Check for XGBoost or LightGBM classifier
        if file == 'xgboost_classifier.model':
            if os.path.exists(os.path.join(model_path, 'lightgbm_classifier.txt')):
                logger.info("Using LightGBM classifier instead of XGBoost")
                continue
        
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            logger.warning(f"Missing required model file: {file}")
            missing_files.append(file)
        else:
            logger.info(f"Found required model file: {file}")
    
    if missing_files:
        logger.warning(f"Model check failed, missing files: {missing_files}")
        return jsonify({
            'status': 'missing',
            'missing_files': missing_files,
            'message': 'Some model files are missing. Please train the model first.'
        })
    
    logger.info("Model check passed, all required files present")
    return jsonify({
        'status': 'ready',
        'message': 'Model is ready for analysis.'
    })

@app.route('/kitsune_info', methods=['GET'])
def kitsune_info():
    """Get information about the Kitsune dataset and model"""
    logger.info("Retrieving Kitsune model info")
    model = KitsuneNIDSModel(model_path="./models")
    
    # Try to load the model to get attack types
    try:
        logger.info("Attempting to load model for attack type information")
        load_success = model.load_models()
        
        if load_success:
            logger.info(f"Model loaded successfully, found {len(model.label_encoder.classes_)} attack types")
            attack_types = [
                {"name": attack_type, "description": model.attack_mapping.get(attack_type, attack_type)}
                for attack_type in model.label_encoder.classes_
            ]
        else:
            raise RuntimeError("Model load reported failure")
    except Exception as e:
        logger.warning(f"Could not load model for attack info: {e}")
        logger.info("Using default attack types")
        # Use default attack types if model not loaded
        attack_types = [
            {"name": "Normal", "description": "Normal Network Traffic"},
            {"name": "Active_Wiretap", "description": "Active Wiretap Attack"},
            {"name": "ARP_MitM", "description": "ARP Man-in-the-Middle Attack"},
            {"name": "Fuzzing", "description": "Protocol Fuzzing Attack"},
            {"name": "Mirai_Botnet", "description": "Mirai Botnet Activity"},
            {"name": "OS_Scan", "description": "Operating System Scanning"},
            {"name": "SSDP_Flood", "description": "SSDP Flooding Attack"},
            {"name": "SSL_Renegotiation", "description": "SSL Renegotiation Attack"},
            {"name": "SYN_DoS", "description": "SYN Denial of Service"},
            {"name": "Video_Injection", "description": "Video Injection Attack"}
        ]
    
    return jsonify({
        'dataset_name': 'Kitsune',
        'attack_types': attack_types,
        'model_type': 'Hybrid Autoencoder + XGBoost/LightGBM',
        'description': 'The Kitsune Network Intrusion Detection System analyzes network traffic to detect 9 different types of attacks and anomalies.'
    })

@app.route('/error_info/<task_id>', methods=['GET'])
def error_info(task_id):
    """Get detailed error information for a failed task"""
    logger.info(f"Retrieving error info for task {task_id}")
    
    if task_id not in analysis_tasks:
        logger.warning(f"Task not found: {task_id}")
        return jsonify({'error': 'Task not found'}), 404
    
    task = analysis_tasks[task_id]
    
    if task['status'] != 'failed':
        logger.warning(f"Task {task_id} did not fail: {task['status']}")
        return jsonify({'error': 'This task did not fail', 'status': task['status']}), 400
    
    error_message = task.get('error', 'Unknown error')
    error_details = task.get('error_details', '')
    
    # Get additional information about the failure
    error_info = {
        'error_message': error_message,
        'error_details': error_details,
        'file_name': task['filename'],
        'file_path': task['file_path'],
        'created_at': task['created_at'],
        'suggestions': [
            'Check if the pcap file is valid and not corrupted',
            'Verify that the file contains network traffic data',
            'Ensure the model is properly trained and all required files are present',
            'Try with a different pcap file to see if the issue persists',
            'Check the logs for more detailed error information'
        ]
    }
    
    return jsonify(error_info)

# Add a favicon route to prevent 404 errors
@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.errorhandler(404)
def not_found(e):
    logger.info(f"Route not found: {request.path}")
    return jsonify({'error': 'Route not found'}), 404

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({
        'error': 'An unexpected error occurred',
        'message': str(e)
    }), 500

if __name__ == '__main__':
    try:
        logger.info("Starting Flask application")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        logger.critical(traceback.format_exc())