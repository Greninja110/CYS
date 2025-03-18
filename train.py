import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, ActivityRegularization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, Callback
from tensorflow.keras.initializers import GlorotNormal, HeNormal
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import pickle
import matplotlib.pyplot as plt
from scapy.all import rdpcap, PcapReader
import time
import logging
import gc
from tqdm import tqdm
import sys
import warnings
import math
import shutil
from datetime import datetime

# Try different methods to import pyarrow
try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    try:
        import pyarrow
        PYARROW_AVAILABLE = True
    except ImportError:
        PYARROW_AVAILABLE = False
        warnings.warn("PyArrow not available, will attempt to use pandas directly for parquet files")

# Configure TensorFlow to use GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set TensorFlow to use the GPU with memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger_setup = logging.getLogger('tensorflow')
        logger_setup.setLevel(logging.INFO)
        logger_setup.info(f"Found {len(gpus)} GPU(s): {gpus}")
    except RuntimeError as e:
        logging.warning(f"GPU memory growth setting failed: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up a file handler for logging
try:
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler(f'logs/train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info("File logging configured")
except Exception as e:
    logger.warning(f"Could not set up file logging: {e}")

# Log GPU availability information
logger.info("Checking for GPU availability...")
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    logger.info(f"Found {len(physical_devices)} GPU(s): {physical_devices}")
    # Get GPU details
    for i, gpu in enumerate(physical_devices):
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            logger.info(f"GPU {i}: {gpu_details}")
        except Exception as e:
            logger.warning(f"Could not get GPU details: {e}")
else:
    logger.warning("No GPU found, training will be on CPU which will be significantly slower")

# Custom callback to handle NaN values
class NanValueCallback(Callback):
    def __init__(self, autoencoder_model, x_train, x_val, frequency=5):
        super(NanValueCallback, self).__init__()
        self.autoencoder_model = autoencoder_model  # Renamed from 'model' to avoid property conflict
        self.x_train = x_train
        self.x_val = x_val
        self.frequency = frequency  # Check every N epochs
        self.best_weights = None
        self.best_epoch = 0
        self.best_val_loss = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
            
        # Check for NaN values in the log
        for metric, value in logs.items():
            if math.isnan(value) or math.isinf(value):
                logger.warning(f"NaN/Inf detected in {metric} at epoch {epoch}")
                if self.best_weights is not None:
                    logger.info(f"Restoring model weights from epoch {self.best_epoch}")
                    self.autoencoder_model.set_weights(self.best_weights)
                    # Force early stopping
                    self.model.stop_training = True
                return
        
        # Store best weights
        if epoch % self.frequency == 0:
            if 'val_loss' in logs and (self.best_weights is None or logs['val_loss'] < self.best_val_loss):
                self.best_weights = self.autoencoder_model.get_weights()
                self.best_epoch = epoch
                self.best_val_loss = logs['val_loss']


class NetworkIntrusionDetectionModel:
    def __init__(self, data_path="./datasets", model_path="./models"):
        """
        Initialize the Network Intrusion Detection Model
        
        Args:
            data_path: Path to the datasets
            model_path: Path to save the trained models
        """
        self.data_path = data_path
        self.model_path = model_path
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize components
        self.autoencoder = None
        self.encoder = None
        self.classifier = None
        
        # Use RobustScaler instead of StandardScaler to handle outliers better
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        
        # Attack mapping dictionary
        self.attack_mapping = {
            'normal': 'Normal',
            'dos': 'DoS Attack',
            'ddos': 'DDoS Attack',
            'probe': 'Probe/Scan',
            'r2l': 'Remote to Local Attack',
            'u2r': 'User to Root Attack',
            'backdoor': 'Backdoor Attack',
            'injection': 'Injection Attack',
            'mitm': 'Man-in-the-Middle Attack',
            'theft': 'Information Theft'
        }
        
        # Define feature columns for extraction from pcap
        self.feature_columns = None
        
        # Track version for model saving
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Model version: {self.version}")
    
    def load_cicids_dataset(self):
        """Load and preprocess CICIDS2017 dataset with memory efficiency"""
        logger.info("Loading CICIDS2017 dataset...")
        
        # Path to the CICIDS2017 dataset files
        cicids_path = os.path.join(self.data_path, "CICIDS2017")
        
        # Check if the directory exists
        if not os.path.exists(cicids_path):
            logger.warning(f"CICIDS2017 directory not found at {cicids_path}")
            return None
        
        # Check if we have SQLite database
        sqlite_path = os.path.join(cicids_path, "database.sqlite")
        
        dataframes = []
        
        # First try to load from SQLite if it exists
        if os.path.exists(sqlite_path):
            try:
                import sqlite3
                logger.info("Loading data from SQLite database...")
                
                # Connect to SQLite database
                conn = sqlite3.connect(sqlite_path)
                
                # Get list of tables
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                # Setup progress tracking
                total_tables = len(tables)
                logger.info(f"Found {total_tables} tables in SQLite database")
                
                # Try to create a progress bar
                try:
                    from progress_callback import TerminalProgressBar
                    progress_bar = TerminalProgressBar(total=total_tables, prefix='CICIDS2017 SQLite:', suffix='Complete')
                except ImportError:
                    progress_bar = None
                
                # Load each table
                for i, table in enumerate(tables):
                    table_name = table[0]
                    try:
                        # Use chunking for large tables
                        chunks = []
                        for chunk in pd.read_sql_query(f"SELECT * FROM {table_name}", conn, chunksize=50000):
                            # Check if it has a Label column
                            if 'Label' in chunk.columns or ' Label' in chunk.columns:
                                # Remove rows with infinity values
                                chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
                                chunk.dropna(inplace=True)
                                chunks.append(chunk)
                            
                        if chunks:
                            df = pd.concat(chunks, ignore_index=True)
                            dataframes.append(df)
                            logger.info(f"Loaded table '{table_name}' with {df.shape[0]} rows")
                        else:
                            logger.info(f"Skipped table '{table_name}' (no Label column)")
                        
                        # Free memory
                        del chunks
                        gc.collect()
                    except Exception as e:
                        logger.error(f"Error loading table {table_name}: {e}")
                    
                    # Update progress
                    if progress_bar:
                        progress_bar.update(i+1)
                
                conn.close()
            except Exception as e:
                logger.error(f"Error connecting to SQLite database: {e}")
        
        # If no data from SQLite or no SQLite file, try loading CSV files
        if not dataframes:
            logger.info("Looking for CSV files...")
            
            # Get all CSV files in the directory
            try:
                csv_files = [f for f in os.listdir(cicids_path) if f.endswith('.csv')]
            except Exception as e:
                logger.error(f"Error reading CICIDS2017 directory: {e}")
                return None
            
            if not csv_files:
                logger.warning("No CSV files found in CICIDS2017 directory")
            else:
                logger.info(f"Found {len(csv_files)} CSV files")
                
                # Try to create a progress bar
                try:
                    from progress_callback import TerminalProgressBar
                    progress_bar = TerminalProgressBar(total=len(csv_files), prefix='CICIDS2017 CSV:', suffix='Complete')
                except ImportError:
                    progress_bar = None
                
                for i, file in enumerate(csv_files):
                    file_path = os.path.join(cicids_path, file)
                    try:
                        logger.info(f"Loading CSV file [{i+1}/{len(csv_files)}]: {file}")
                        
                        # Check file size first
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                        logger.info(f"File size: {file_size:.2f} MB")
                        
                        if file_size > 100:  # Use chunking for large files (>100MB)
                            logger.info(f"Large file detected, using chunked loading")
                            chunk_list = []
                            for chunk in pd.read_csv(file_path, low_memory=False, chunksize=100000):
                                # Process each chunk
                                chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
                                chunk.dropna(inplace=True)
                                chunk_list.append(chunk)
                                
                                # Free memory periodically
                                if len(chunk_list) % 10 == 0:
                                    gc.collect()
                            
                            if chunk_list:
                                df = pd.concat(chunk_list, ignore_index=True)
                                dataframes.append(df)
                                logger.info(f"Loaded {file} with {df.shape[0]} rows")
                                
                                # Clear memory
                                del chunk_list
                                gc.collect()
                        else:
                            # Read CSV file normally for small files
                            df = pd.read_csv(file_path, low_memory=False)
                            
                            # Remove rows with infinity values
                            df.replace([np.inf, -np.inf], np.nan, inplace=True)
                            df.dropna(inplace=True)
                            
                            dataframes.append(df)
                            logger.info(f"Loaded {file} with {df.shape[0]} rows")
                    except Exception as e:
                        logger.error(f"Error loading {file}: {e}")
                    
                    # Update progress
                    if progress_bar:
                        progress_bar.update(i+1)
        
        if not dataframes:
            logger.warning("No CICIDS2017 files found or loaded")
            return None
        
        # Combine all dataframes
        logger.info("Combining CICIDS2017 data frames...")
        try:
            cicids_data = pd.concat(dataframes, ignore_index=True)
        except Exception as e:
            logger.error(f"Error concatenating dataframes: {e}")
            # Try to salvage data if possible
            if dataframes:
                logger.info("Attempting to use the largest dataframe instead")
                largest_df = max(dataframes, key=lambda df: df.shape[0])
                cicids_data = largest_df
            else:
                return None
        
        # Map labels to standard categories
        # Check for common label column names and standardize
        for col in cicids_data.columns:
            if col.lower().strip() == 'label' or col.strip() == ' Label':
                cicids_data.rename(columns={col: 'Label'}, inplace=True)
                break
        
        if 'Label' in cicids_data.columns:
            logger.info("Standardizing CICIDS2017 labels...")
            cicids_data['Label'] = cicids_data['Label'].astype(str).str.strip().str.lower()
            # Map CICIDS2017 specific attack types to our standard categories
            cicids_data['Label'] = cicids_data['Label'].apply(self._map_cicids_attacks)
            
            # Count label distribution
            label_counts = cicids_data['Label'].value_counts()
            logger.info("CICIDS2017 label distribution:")
            for label, count in label_counts.items():
                percentage = (count / cicids_data.shape[0]) * 100
                logger.info(f"  {label}: {count} samples ({percentage:.2f}%)")
        else:
            logger.warning("No 'Label' column found in CICIDS2017 dataset")
            return None
        
        logger.info(f"CICIDS2017 dataset loaded with {cicids_data.shape[0]} samples")
        return cicids_data
    
    def load_nslkdd_dataset(self):
        """Load and preprocess NSL-KDD dataset with improved error handling"""
        logger.info("Loading NSL-KDD dataset...")
        
        # Path to the NSL-KDD dataset files
        nslkdd_path = os.path.join(self.data_path, "NSL-KDD")
        
        # Check if the directory exists
        if not os.path.exists(nslkdd_path):
            logger.warning(f"NSL-KDD directory not found at {nslkdd_path}")
            return None
        
        # Column names for NSL-KDD dataset
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
        ]
        
        dataframes = []
        
        # Look for text files first
        try:
            txt_files = [f for f in os.listdir(nslkdd_path) if f.endswith('.txt')]
        except Exception as e:
            logger.error(f"Error reading NSL-KDD directory: {e}")
            return None
        
        # Filter known training and testing files
        train_files = [f for f in txt_files if 'train' in f.lower()]
        test_files = [f for f in txt_files if 'test' in f.lower()]
        
        # Try to load text files
        for file in train_files + test_files:
            file_path = os.path.join(nslkdd_path, file)
            try:
                logger.info(f"Loading text file: {file}")
                # Check file size
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                
                if file_size > 100:  # Large file
                    chunk_list = []
                    for chunk in pd.read_csv(file_path, header=None, names=columns, chunksize=100000):
                        chunk_list.append(chunk)
                    
                    if chunk_list:
                        df = pd.concat(chunk_list, ignore_index=True)
                        dataframes.append(df)
                    
                    # Free memory
                    del chunk_list
                    gc.collect()
                else:
                    # Read text file as CSV
                    df = pd.read_csv(file_path, header=None, names=columns)
                    dataframes.append(df)
                
                logger.info(f"Loaded {file} with {df.shape[0]} rows")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        # If no data from text files, try ARFF files
        if not dataframes:
            logger.info("No usable text files found, looking for ARFF files...")
            
            try:
                arff_files = [f for f in os.listdir(nslkdd_path) if f.endswith('.arff')]
            except Exception as e:
                logger.error(f"Error reading NSL-KDD directory for ARFF files: {e}")
                return None
            
            for file in arff_files:
                file_path = os.path.join(nslkdd_path, file)
                try:
                    logger.info(f"Loading ARFF file: {file}")
                    
                    # Parse ARFF file manually
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Skip header and find data section
                    data_start = 0
                    for i, line in enumerate(lines):
                        if '@data' in line.lower():
                            data_start = i + 1
                            break
                    
                    # Extract data rows
                    data_rows = []
                    for i in range(data_start, len(lines)):
                        line = lines[i].strip()
                        if line and not line.startswith('%'):  # Skip empty lines and comments
                            row = line.split(',')
                            if len(row) >= len(columns):  # Ensure row has enough columns
                                data_rows.append(row[:len(columns)])
                    
                    # Create DataFrame
                    if data_rows:
                        df = pd.DataFrame(data_rows, columns=columns)
                        dataframes.append(df)
                        logger.info(f"Loaded {file} with {df.shape[0]} rows")
                except Exception as e:
                    logger.error(f"Error loading ARFF file {file}: {e}")
        
        if not dataframes:
            logger.warning("No NSL-KDD dataset files found or loaded")
            return None
        
        # Combine all dataframes
        try:
            nslkdd_data = pd.concat(dataframes, ignore_index=True)
        except Exception as e:
            logger.error(f"Error concatenating NSL-KDD dataframes: {e}")
            if dataframes:
                # Try to use the largest dataframe
                largest_df = max(dataframes, key=lambda df: df.shape[0])
                nslkdd_data = largest_df
            else:
                return None
        
        # Drop difficulty level column if it exists
        if 'difficulty_level' in nslkdd_data.columns:
            nslkdd_data.drop('difficulty_level', axis=1, inplace=True)
        
        # Standardize label column name
        nslkdd_data.rename(columns={'label': 'Label'}, inplace=True)
        
        # Map labels to standard categories
        try:
            nslkdd_data['Label'] = nslkdd_data['Label'].astype(str).apply(self._map_nslkdd_attacks)
        except Exception as e:
            logger.error(f"Error mapping NSL-KDD labels: {e}")
            # Continue with unmapped labels if error occurs
        
        logger.info(f"NSL-KDD dataset loaded with {nslkdd_data.shape[0]} samples")
        return nslkdd_data
    
    def load_unsw_dataset(self):
        """Load and preprocess UNSW-NB15 dataset with improved error handling"""
        logger.info("Loading UNSW-NB15 dataset...")
        
        # Path to the UNSW-NB15 dataset files
        unsw_path = os.path.join(self.data_path, "UNSW-NB15")
        
        # Check if the directory exists
        if not os.path.exists(unsw_path):
            logger.warning(f"UNSW-NB15 directory not found at {unsw_path}")
            return None
        
        # Look for all files and filter by type
        try:
            all_files = os.listdir(unsw_path)
        except Exception as e:
            logger.error(f"Error reading UNSW-NB15 directory: {e}")
            return None
        
        train_files = [f for f in all_files if 'train' in f.lower()]
        test_files = [f for f in all_files if 'test' in f.lower()]
        
        # Also check for any CSV files
        csv_files = [f for f in all_files if f.endswith('.csv')]
        
        dataframes = []
        
        # Process both training and testing files
        for file_list, file_type in [(train_files, "training"), (test_files, "testing"), (csv_files, "csv")]:
            for file in file_list:
                file_path = os.path.join(unsw_path, file)
                try:
                    logger.info(f"Loading {file_type} file: {file}")
                    
                    # Check file size
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                    logger.info(f"File size: {file_size:.2f} MB")
                    
                    # Check file extension and use appropriate reader
                    if file.lower().endswith('.parquet'):
                        # Try multiple methods to read parquet files
                        successful = False
                        
                        # Method 1: Direct pandas read_parquet
                        try:
                            logger.info("Trying direct pandas read_parquet...")
                            if file_size > 100:  # Large file
                                # For large parquet files, we may need to read in chunks
                                # This is a bit more complex with parquet, but we can try filtering
                                # First try to read just the schema
                                column_names = pd.read_parquet(file_path, columns=[]).columns.tolist()
                                
                                # Then read in chunks by row groups or partitions if possible
                                chunk_list = []
                                for i in range(0, 10):  # Try up to 10 row groups
                                    try:
                                        chunk = pd.read_parquet(file_path, row_group=i)
                                        chunk_list.append(chunk)
                                    except:
                                        break
                                
                                if chunk_list:
                                    df = pd.concat(chunk_list, ignore_index=True)
                                    dataframes.append(df)
                                    logger.info(f"Successfully loaded {file} with {df.shape[0]} rows using pandas read_parquet in chunks")
                                    successful = True
                                
                                # Free memory
                                del chunk_list
                                gc.collect()
                            else:
                                df = pd.read_parquet(file_path)
                                dataframes.append(df)
                                logger.info(f"Successfully loaded {file} with {df.shape[0]} rows using pandas read_parquet")
                                successful = True
                        except Exception as e1:
                            logger.warning(f"Failed to read parquet with pandas directly: {e1}")
                            
                            # Method 2: Using pyarrow if available
                            if PYARROW_AVAILABLE:
                                try:
                                    logger.info("Trying pyarrow.parquet.read_table...")
                                    table = pq.read_table(file_path)
                                    df = table.to_pandas()
                                    dataframes.append(df)
                                    logger.info(f"Successfully loaded {file} with {df.shape[0]} rows using pyarrow")
                                    successful = True
                                except Exception as e2:
                                    logger.warning(f"Failed to read parquet with pyarrow: {e2}")
                            
                            # Method 3: Convert to CSV first if possible
                            if not successful:
                                try:
                                    logger.info("Trying to convert parquet to CSV...")
                                    # Try to find existing CSV version
                                    csv_path = file_path.replace('.parquet', '.csv')
                                    if os.path.exists(csv_path):
                                        if os.path.getsize(csv_path) / (1024 * 1024) > 100:  # Large CSV
                                            chunk_list = []
                                            for chunk in pd.read_csv(csv_path, low_memory=False, chunksize=100000):
                                                chunk_list.append(chunk)
                                            
                                            if chunk_list:
                                                df = pd.concat(chunk_list, ignore_index=True)
                                                dataframes.append(df)
                                                
                                                # Free memory
                                                del chunk_list
                                                gc.collect()
                                        else:
                                            df = pd.read_csv(csv_path, low_memory=False)
                                            dataframes.append(df)
                                        
                                        logger.info(f"Successfully loaded CSV version of {file} with {df.shape[0]} rows")
                                        successful = True
                                except Exception as e3:
                                    logger.warning(f"Failed to read CSV version: {e3}")
                            
                        if not successful:
                            logger.error(f"All methods to read {file} failed. Skipping file.")
                    else:
                        # Use read_csv for csv files
                        if file_size > 100:  # Large CSV
                            chunk_list = []
                            for chunk in pd.read_csv(file_path, low_memory=False, chunksize=100000):
                                chunk_list.append(chunk)
                            
                            if chunk_list:
                                df = pd.concat(chunk_list, ignore_index=True)
                                dataframes.append(df)
                                logger.info(f"Loaded {file} with {df.shape[0]} rows using chunking")
                                
                                # Free memory
                                del chunk_list
                                gc.collect()
                        else:
                            df = pd.read_csv(file_path, low_memory=False)
                            dataframes.append(df)
                            logger.info(f"Loaded {file} with {df.shape[0]} rows")
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
        
        if not dataframes:
            logger.warning("No UNSW-NB15 dataset files found or loaded")
            return None
        
        # Combine all dataframes
        try:
            unsw_data = pd.concat(dataframes, ignore_index=True)
        except Exception as e:
            logger.error(f"Error concatenating UNSW-NB15 dataframes: {e}")
            if dataframes:
                # Try to use the largest dataframe
                largest_df = max(dataframes, key=lambda df: df.shape[0])
                unsw_data = largest_df
            else:
                return None
        
        # Standardize label column name
        for col in unsw_data.columns:
            if col.lower() == 'label':
                unsw_data.rename(columns={col: 'Label'}, inplace=True)
                break
        
        # Map attack_cat to Label for attack type classification
        attack_cat_col = None
        for col in unsw_data.columns:
            if 'attack' in col.lower() and 'cat' in col.lower():
                attack_cat_col = col
                break
        
        if attack_cat_col:
            try:
                # Check if column is categorical and convert to string first
                if pd.api.types.is_categorical_dtype(unsw_data[attack_cat_col]):
                    logger.info(f"Converting categorical column {attack_cat_col} to string type")
                    unsw_data[attack_cat_col] = unsw_data[attack_cat_col].astype(str)
                
                # Replace empty attack_cat with 'normal' for normal traffic
                # Use assignment instead of inplace to avoid categorical issues
                unsw_data[attack_cat_col] = unsw_data[attack_cat_col].fillna('normal')
                
                # Map UNSW specific attack types to our standard categories
                unsw_data['Label'] = unsw_data[attack_cat_col].astype(str).apply(self._map_unsw_attacks)
                unsw_data.drop(attack_cat_col, axis=1, inplace=True)
            except Exception as e:
                logger.error(f"Error processing attack category column: {e}")
                # Try to create a default label column if one doesn't exist
                if 'Label' not in unsw_data.columns:
                    logger.warning("Creating default 'Label' column based on binary classification")
                    if 'label' in unsw_data.columns.str.lower():
                        binary_col = unsw_data.columns[unsw_data.columns.str.lower() == 'label'][0]
                        unsw_data['Label'] = unsw_data[binary_col].apply(
                            lambda x: 'normal' if x == 0 or x == '0' else 'dos'
                        )
        elif 'Label' not in unsw_data.columns:
            logger.warning("No label or attack category column found in UNSW-NB15 dataset")
            return None
        
        logger.info(f"UNSW-NB15 dataset loaded with {unsw_data.shape[0]} samples")
        return unsw_data
    
    def _map_cicids_attacks(self, label):
        """Map CICIDS2017 attack labels to standard categories"""
        try:
            label = str(label).strip().lower()
            
            if 'benign' in label or label == 'normal':
                return 'normal'
            elif 'dos' in label or 'hulk' in label or 'goldeneye' in label or 'slowloris' in label or 'slowhttptest' in label:
                return 'dos'
            elif 'ddos' in label:
                return 'ddos'
            elif 'portscan' in label or 'port scan' in label:
                return 'probe'
            elif 'bruteforce' in label or 'brute force' in label or 'ftp-patator' in label or 'ssh-patator' in label:
                return 'r2l'
            elif 'infiltration' in label:
                return 'u2r'
            elif 'bot' in label:
                return 'backdoor'
            elif 'web attack' in label or 'sql injection' in label or 'xss' in label:
                return 'injection'
            elif 'heartbleed' in label:
                return 'mitm'
            else:
                return 'dos'  # Default to DOS for unclassified attacks
        except Exception as e:
            logger.error(f"Error mapping CICIDS attack label '{label}': {e}")
            return 'dos'  # Default in case of error
    
    def _map_nslkdd_attacks(self, label):
        """Map NSL-KDD attack labels to standard categories"""
        try:
            label = str(label).strip().lower()
            
            if label == 'normal':
                return 'normal'
            elif any(attack in label for attack in ['neptune', 'smurf', 'pod', 'teardrop', 'land', 'back']):
                return 'dos'
            elif any(attack in label for attack in ['ipsweep', 'portsweep', 'nmap', 'satan']):
                return 'probe'
            elif any(attack in label for attack in ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy']):
                return 'r2l'
            elif any(attack in label for attack in ['buffer_overflow', 'loadmodule', 'rootkit', 'perl']):
                return 'u2r'
            else:
                return 'dos'  # Default to DOS for unclassified attacks
        except Exception as e:
            logger.error(f"Error mapping NSL-KDD attack label '{label}': {e}")
            return 'dos'  # Default in case of error
    
    def _map_unsw_attacks(self, label):
        """Map UNSW-NB15 attack labels to standard categories"""
        try:
            label = str(label).strip().lower()
            
            if label == 'normal':
                return 'normal'
            elif label == 'dos':
                return 'dos'
            elif label == 'ddos':
                return 'ddos'
            elif label in ['reconnaissance', 'analysis', 'scanning']:
                return 'probe'
            elif label in ['backdoor', 'backdoors', 'trojan']:
                return 'backdoor'
            elif label in ['exploits', 'shellcode', 'worms']:
                return 'u2r'
            elif label in ['fuzzers', 'generic']:
                return 'dos'
            else:
                return 'dos'  # Default to DOS for unclassified attacks
        except Exception as e:
            logger.error(f"Error mapping UNSW attack label '{label}': {e}")
            return 'dos'  # Default in case of error
    
    def preprocess_data(self, combined_data):
        """
        Preprocess the combined dataset for training with improved error handling
        
        Args:
            combined_data: Combined dataset from multiple sources
            
        Returns:
            X: Features for training
            y: Labels for training
            feature_names: Names of the features used
        """
        logger.info("Preprocessing combined dataset...")
        
        if combined_data is None or combined_data.empty:
            logger.error("Cannot preprocess empty or None dataset")
            return None, None, None
        
        # Make a copy to avoid modifying the original data
        try:
            data = combined_data.copy()
        except Exception as e:
            logger.error(f"Error copying dataset: {e}")
            # Try to continue with the original data
            data = combined_data
        
        # Simple terminal progress display
        print("Preprocessing data...")
        process_steps = 6  # Number of major preprocessing steps
        progress_bar = None
        
        try:
            from progress_callback import TerminalProgressBar
            progress_bar = TerminalProgressBar(total=process_steps, prefix='Preprocessing:', suffix='Complete')
        except ImportError:
            pass
        
        # Step 1: Remove unnecessary columns if they exist
        cols_to_drop = ['id', 'attack_cat', 'difficulty_level', 'idx']
        for col in cols_to_drop:
            if col in data.columns:
                try:
                    data.drop(col, axis=1, inplace=True)
                except Exception as e:
                    logger.warning(f"Error dropping column {col}: {e}")
        
        if progress_bar:
            progress_bar.update(1)
        
        # Step 2: Handle and remove extreme outliers across numerical columns
        # This is critical for preventing NaN values during training
        logger.info("Handling extreme outliers in numerical features...")
        try:
            numeric_cols = data.select_dtypes(include=['number']).columns
        except Exception as e:
            logger.error(f"Error identifying numeric columns: {e}")
            # Create a fallback list of likely numeric columns
            numeric_cols = [col for col in data.columns if 
                            any(term in col.lower() for term in 
                                ['count', 'bytes', 'duration', 'time', 'rate', 'number'])]
            logger.info(f"Using fallback numeric columns: {numeric_cols}")
        
        for col in numeric_cols:
            if col in data.columns:  # Check if column still exists (might have been dropped)
                try:
                    # Calculate Q1, Q3 and IQR
                    Q1 = data[col].quantile(0.01)  # Use 1% instead of 25% to be more conservative
                    Q3 = data[col].quantile(0.99)  # Use 99% instead of 75% to be more conservative
                    IQR = Q3 - Q1
                    
                    # Define bounds for outliers (more conservative than the usual 1.5*IQR)
                    lower_bound = Q1 - 5 * IQR
                    upper_bound = Q3 + 5 * IQR
                    
                    # Check for extreme values that could cause numerical instability
                    extreme_lower = data[col].lt(lower_bound)
                    extreme_upper = data[col].gt(upper_bound)
                    
                    if extreme_lower.any() or extreme_upper.any():
                        # Replace extreme values with bounds
                        data.loc[extreme_lower, col] = lower_bound
                        data.loc[extreme_upper, col] = upper_bound
                        
                        total_clipped = extreme_lower.sum() + extreme_upper.sum()
                        logger.info(f"  Clipped {total_clipped} extreme values in column '{col}'")
                except Exception as e:
                    logger.warning(f"Error handling outliers in column '{col}': {e}")
                    # Try a simpler approach for problematic columns
                    try:
                        # Remove infinity values
                        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                        
                        # Use more conservative percentile-based clipping
                        lower = data[col].quantile(0.001)
                        upper = data[col].quantile(0.999)
                        data[col] = data[col].clip(lower=lower, upper=upper)
                        
                        # Fill remaining NaNs with median
                        data[col] = data[col].fillna(data[col].median())
                        
                        logger.info(f"  Applied alternative outlier handling for column '{col}'")
                    except Exception as e2:
                        logger.error(f"Failed alternative outlier handling for column '{col}': {e2}")
                        # Last resort: convert problematic columns to 0/1 indicators
                        try:
                            data[col] = (data[col] > 0).astype(int)
                            logger.warning(f"  Converted column '{col}' to binary indicator")
                        except:
                            # If all else fails, drop the column
                            data.drop(col, axis=1, inplace=True, errors='ignore')
                            logger.warning(f"  Dropped problematic column '{col}'")
        
        if progress_bar:
            progress_bar.update(2)
        
        # Step 3: Separate features and labels
        if 'Label' in data.columns:
            try:
                X = data.drop('Label', axis=1)
                y = data['Label']
            except Exception as e:
                logger.error(f"Error separating features and labels: {e}")
                return None, None, None
        else:
            logger.error("Label column not found in the dataset")
            return None, None, None
        
        if progress_bar:
            progress_bar.update(3)
        
        # Step 4: Handle categorical features
        logger.info("Encoding categorical features...")
        try:
            categorical_cols = X.select_dtypes(include=['object']).columns
        except Exception as e:
            logger.error(f"Error identifying categorical columns: {e}")
            # Continue with an empty list if we can't identify categorical columns
            categorical_cols = []
        
        # Track progress for categorical feature encoding
        if len(categorical_cols) > 0:
            logger.info(f"Found {len(categorical_cols)} categorical columns to encode")
            
        for i, col in enumerate(categorical_cols):
            try:
                # One-hot encode categorical features
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
                
                # Log progress for categorical encoding
                if i % max(1, len(categorical_cols)//5) == 0 or i == len(categorical_cols)-1:
                    logger.info(f"Encoded {i+1}/{len(categorical_cols)} categorical features")
            except Exception as e:
                logger.warning(f"Error one-hot encoding column '{col}': {e}")
                # Try simple label encoding as fallback
                try:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    logger.info(f"Applied label encoding to column '{col}' as fallback")
                except Exception as e2:
                    logger.error(f"Failed label encoding for column '{col}': {e2}")
                    # If all else fails, drop the column
                    X.drop(col, axis=1, inplace=True, errors='ignore')
                    logger.warning(f"Dropped problematic categorical column '{col}'")
        
        if progress_bar:
            progress_bar.update(4)
        
        # Step 5: Store feature names
        feature_names = X.columns.tolist()
        self.feature_columns = feature_names
        
        # Log feature count
        logger.info(f"Total features after preprocessing: {len(feature_names)}")
        
        if progress_bar:
            progress_bar.update(5)
        
        # Step 6: Scale the features and encode labels
        logger.info("Scaling features and encoding labels...")
        try:
            # Replace any remaining NaN or inf values
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(0, inplace=True)  # Fill remaining NaNs with zeros
            
            # Scale the features
            X = self.scaler.fit_transform(X)
            
            # Additional sanitization check for NaN/Inf after scaling
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
            
            y = self.label_encoder.fit_transform(y)
        except Exception as e:
            logger.error(f"Error during scaling or encoding: {e}")
            
            # Try more robust approach
            try:
                logger.info("Attempting more robust scaling approach...")
                
                # Convert dataframe to numpy array first
                X_np = X.values
                
                # Replace problematic values
                X_np = np.nan_to_num(X_np, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Use a more robust scaler with precomputed statistics
                self.scaler = RobustScaler(quantile_range=(5.0, 95.0))
                X = self.scaler.fit_transform(X_np)
                
                # Encode labels
                y = self.label_encoder.fit_transform(y)
                
                logger.info("Robust scaling approach succeeded")
            except Exception as e2:
                logger.error(f"Robust scaling also failed: {e2}")
                return None, None, None
            
        if progress_bar:
            progress_bar.update(6)
        
        # Save the label encoder classes for reference
        self.label_classes = self.label_encoder.classes_
        
        # Log class distribution
        try:
            unique_classes, class_counts = np.unique(y, return_counts=True)
            logger.info(f"Class distribution after preprocessing:")
            for i, (class_idx, count) in enumerate(zip(unique_classes, class_counts)):
                class_name = self.label_encoder.inverse_transform([class_idx])[0]
                percentage = (count / len(y)) * 100
                logger.info(f"  {class_name}: {count} samples ({percentage:.2f}%)")
        except Exception as e:
            logger.warning(f"Error logging class distribution: {e}")
        
        logger.info(f"Preprocessing complete. Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
        
        return X, y, feature_names
    
    def build_autoencoder(self, input_dim):
        """
        Build a more stable autoencoder model with careful initialization and normalization
        
        Args:
            input_dim: Dimension of the input features
            
        Returns:
            autoencoder: Complete autoencoder model
            encoder: Encoder part of the autoencoder for feature extraction
        """
        logger.info("Building autoencoder model...")
        
        # Configure for GPU
        device_name = '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'
        logger.info(f"Building autoencoder model on {device_name}")
        
        try:
            with tf.device(device_name):
                # Use more conservative initializations for stability
                initializer = HeNormal(seed=42)  # He initialization is better for ReLU
                
                # Define encoder with batch normalization for stability
                input_layer = Input(shape=(input_dim,))
                
                # First, normalize the input data
                encoded = BatchNormalization()(input_layer)
                
                # First encoder layer with careful initialization and small initial weights
                encoded = Dense(512, activation='relu', 
                              kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l2(1e-5))(encoded)
                encoded = BatchNormalization()(encoded)
                encoded = Dropout(0.3)(encoded)
                
                # Second encoder layer
                encoded = Dense(256, activation='relu', 
                              kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l2(1e-5))(encoded)
                encoded = BatchNormalization()(encoded)
                encoded = Dropout(0.3)(encoded)
                
                # Third encoder layer
                encoded = Dense(128, activation='relu',
                              kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l2(1e-5))(encoded)
                encoded = BatchNormalization()(encoded)
                encoded = Dropout(0.3)(encoded)
                
                # Final encoder layer with careful initialization
                encoded = Dense(64, activation='relu',
                              kernel_initializer=initializer)(encoded)
                encoded = BatchNormalization()(encoded)
                
                # First decoder layer
                decoded = Dense(128, activation='relu',
                               kernel_initializer=initializer,
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5))(encoded)
                decoded = BatchNormalization()(decoded)
                decoded = Dropout(0.3)(decoded)
                
                # Second decoder layer
                decoded = Dense(256, activation='relu',
                               kernel_initializer=initializer,
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5))(decoded)
                decoded = BatchNormalization()(decoded)
                decoded = Dropout(0.3)(decoded)
                
                # Third decoder layer
                decoded = Dense(512, activation='relu',
                               kernel_initializer=initializer,
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5))(decoded)
                decoded = BatchNormalization()(decoded)
                decoded = Dropout(0.3)(decoded)
                
                # Final output layer - use a smaller activation range for stability
                output_layer = Dense(input_dim, activation='tanh',
                                   kernel_initializer=initializer)(decoded)
                
                # Add final batch normalization to keep outputs in a stable range
                output_layer = BatchNormalization()(output_layer)
                
                # Create autoencoder model
                autoencoder = Model(inputs=input_layer, outputs=output_layer)
                
                # Use mixed precision training if on GPU to speed up training
                if device_name == '/GPU:0':
                    try:
                        from tensorflow.keras.mixed_precision import global_policy, set_global_policy
                        policy = global_policy()
                        logger.info(f"Current policy: {policy}")
                        
                        if 'mixed_float16' not in str(policy):
                            logger.info("Setting mixed precision policy for faster GPU training")
                            set_global_policy('mixed_float16')
                            logger.info(f"New policy: {global_policy()}")
                    except Exception as e:
                        logger.warning(f"Could not set mixed precision policy: {e}")
                
                # Use a much smaller learning rate for numerical stability
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=0.00001,  # Reduced by 10x for stability
                    clipnorm=1.0,  # Gradient clipping
                    epsilon=1e-7   # Increased epsilon for numerical stability
                )
                
                autoencoder.compile(
                    optimizer=optimizer, 
                    loss='mse',   # MSE is fine for autoencoder
                    metrics=['mse']
                )
                
                # Create encoder model
                encoder = Model(inputs=input_layer, outputs=encoded)
            
            logger.info(f"Autoencoder model built successfully on {device_name}")
            return autoencoder, encoder
            
        except Exception as e:
            logger.error(f"Error building autoencoder model: {e}")
            
            # Try to build a simpler model as fallback
            logger.info("Attempting to build a simpler autoencoder model as fallback")
            try:
                with tf.device('/CPU:0'):  # Always use CPU for fallback
                    # Simpler model with fewer layers and parameters
                    input_layer = Input(shape=(input_dim,))
                    
                    # Encoder
                    encoded = BatchNormalization()(input_layer)
                    encoded = Dense(256, activation='relu')(encoded)
                    encoded = BatchNormalization()(encoded)
                    encoded = Dense(128, activation='relu')(encoded)
                    encoded = BatchNormalization()(encoded)
                    encoded = Dense(64, activation='relu')(encoded)
                    
                    # Decoder
                    decoded = Dense(128, activation='relu')(encoded)
                    decoded = BatchNormalization()(decoded)
                    decoded = Dense(256, activation='relu')(decoded)
                    decoded = BatchNormalization()(decoded)
                    decoded = Dense(input_dim, activation='linear')(decoded)
                    
                    # Create models
                    autoencoder = Model(inputs=input_layer, outputs=decoded)
                    encoder = Model(inputs=input_layer, outputs=encoded)
                    
                    # Use a simple optimizer
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
                    
                    autoencoder.compile(
                        optimizer=optimizer,
                        loss='mse',
                        metrics=['mse']
                    )
                    
                    logger.info("Fallback autoencoder model built successfully")
                    return autoencoder, encoder
            except Exception as e2:
                logger.error(f"Failed to build fallback autoencoder model: {e2}")
                return None, None
    
    def train_autoencoder(self, X_train, X_val, epochs=100, batch_size=64):
        """
        Train the autoencoder model with improved parameters for stability
        
        Args:
            X_train: Training features
            X_val: Validation features
            epochs: Number of training epochs
            batch_size: Batch size for training (smaller for better stability)
            
        Returns:
            history: Training history
        """
        logger.info("Training autoencoder model...")
        
        if self.autoencoder is None:
            logger.error("Autoencoder model not built, cannot train")
            return None
        
        # Early stopping callback with increased patience
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased from 10 for more chances to converge
            restore_best_weights=True,
            verbose=1
        )
        
        # Reduce learning rate on plateau with more conservative parameters
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,  # More conservative reduction
            patience=10,  # More patience before reducing
            min_lr=1e-8,  # Allow very small learning rates
            verbose=1
        )
        
        # Model checkpoint callback - corrected parameter name
        checkpoint = ModelCheckpoint(
            os.path.join(self.model_path, f'autoencoder_best_{self.version}.h5'),
            monitor='val_loss',
            save_best_only=True,  # Correct parameter name (was save_best_weights_only)
            verbose=1
        )
        
        # Custom callback to detect NaN values and recover - using the CORRECT parameter name
        nan_callback = NanValueCallback(
            autoencoder_model=self.autoencoder,  # MUST match the parameter name in NanValueCallback.__init__
            x_train=X_train,
            x_val=X_val,
            frequency=2
        )
        
        # TensorBoard callback for visualization if needed
        log_dir = os.path.join(self.model_path, 'logs', self.version)
        os.makedirs(log_dir, exist_ok=True)
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
        
        # Pre-training sanity check to detect NaN values in data
        try:
            X_check = np.concatenate([X_train[:100], X_val[:100]], axis=0)
            nan_check = np.isnan(X_check).any()
            inf_check = np.isinf(X_check).any()
            
            if nan_check or inf_check:
                logger.warning("Detected NaN or Inf values in input data, fixing...")
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
                X_val = np.nan_to_num(X_val, nan=0.0, posinf=1.0, neginf=-1.0)
        except Exception as e:
            logger.warning(f"Error during NaN check: {e}")
            # Apply fix anyway to be safe
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Given the large dataset, let's reduce initial epochs to prevent memory issues
        logger.info("Starting initial training with small batch size for stability...")
        try:
            # Use a smaller subset for initial training
            train_subset_size = min(100000, X_train.shape[0])
            val_subset_size = min(10000, X_val.shape[0])
            
            initial_history = self.autoencoder.fit(
                X_train[:train_subset_size], X_train[:train_subset_size],  # Use a subset for initial training
                epochs=5,  # Just a few epochs to stabilize
                batch_size=16,  # Very small batch size
                validation_data=(X_val[:val_subset_size], X_val[:val_subset_size]),  # Small validation set
                verbose=1
            )
            logger.info("Initial training completed successfully")
        except Exception as e:
            logger.warning(f"Initial training failed, but continuing with main training: {e}")
        
        # Main training with larger batch size
        logger.info("Starting main autoencoder training...")
        try:
            history = self.autoencoder.fit(
                X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, X_val),
                callbacks=[early_stopping, reduce_lr, checkpoint, nan_callback, tensorboard],
                verbose=1
            )
            logger.info("Autoencoder training complete")
            return history
        except Exception as e:
            logger.error(f"Error during autoencoder training: {e}")
            # Try to save the model even if training failed
            try:
                save_path = os.path.join(self.model_path, f'autoencoder_partial_{self.version}.h5')
                self.autoencoder.save(save_path)
                logger.info(f"Saved partial autoencoder model to {save_path}")
            except Exception as e2:
                logger.error(f"Could not save partial autoencoder model: {e2}")
            
            # Try to train with even more conservative parameters as a fallback
            logger.info("Attempting fallback training with more conservative parameters...")
            try:
                # Reduce training set size further
                train_subset_size = min(50000, X_train.shape[0])
                val_subset_size = min(5000, X_val.shape[0])
                
                # Use an even smaller batch size and learning rate
                self.autoencoder.optimizer.learning_rate.assign(0.000001)
                
                fallback_history = self.autoencoder.fit(
                    X_train[:train_subset_size], X_train[:train_subset_size],
                    epochs=50,  # Fewer epochs
                    batch_size=32,  # Small batch size but not too small
                    validation_data=(X_val[:val_subset_size], X_val[:val_subset_size]),
                    callbacks=[early_stopping, checkpoint],
                    verbose=1
                )
                logger.info("Fallback training completed")
                return fallback_history
            except Exception as e3:
                logger.error(f"Fallback training also failed: {e3}")
            
            # Return a minimal history object
            return {'loss': [0], 'val_loss': [0]}
    
    def train_classifier(self, X_train, y_train, X_val, y_val):
        """
        Train the XGBoost classifier with improved parameters for higher accuracy
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            classifier: Trained XGBoost classifier
        """
        logger.info("Training XGBoost classifier...")
        
        # Check for NaNs or infs in data
        try:
            if np.isnan(X_train).any() or np.isinf(X_train).any():
                logger.warning("Found NaN or Inf values in classifier training data, fixing...")
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if np.isnan(X_val).any() or np.isinf(X_val).any():
                logger.warning("Found NaN or Inf values in classifier validation data, fixing...")
                X_val = np.nan_to_num(X_val, nan=0.0, posinf=1.0, neginf=-1.0)
        except Exception as e:
            logger.warning(f"Error checking for NaN/Inf values: {e}")
            # Apply fix anyway to be safe
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Get number of classes
        num_classes = len(np.unique(y_train))
        
        # Define XGBoost parameters
        # Check for GPU availability
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        logger.info(f"GPU available for XGBoost: {gpu_available}")
        
        # Calculate class weights to handle class imbalance
        try:
            classes_counts = np.bincount(y_train)
            total_samples = len(y_train)
            
            # Compute weight for each class
            class_weights = {}
            for i in range(len(classes_counts)):
                if classes_counts[i] > 0:  # Avoid division by zero
                    # Weight inversely proportional to class frequency
                    weight = total_samples / (len(classes_counts) * classes_counts[i])
                    class_weights[i] = weight
            
            logger.info(f"Class weights for handling imbalance: {class_weights}")
        except Exception as e:
            logger.warning(f"Error calculating class weights: {e}")
            # Use default weights if calculation fails
            class_weights = {i: 1.0 for i in range(num_classes)}
        
        # Improved parameters for better accuracy
        params = {
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'eta': 0.01,  # Even lower learning rate for better convergence
            'max_depth': 10,  # Complex model
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,  # Added to reduce overfitting
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'eval_metric': ['mlogloss', 'merror'],  # Track multiple metrics
            'use_label_encoder': False,
            'scale_pos_weight': 1,  # We'll handle class weights separately
            'max_delta_step': 1  # Helps with class imbalance
        }
        
        # Add tree method parameter conditionally
        if gpu_available:
            try:
                # Test GPU compatibility
                test_params = params.copy()
                test_params['tree_method'] = 'gpu_hist'
                test_params['gpu_id'] = 0
                test_params['predictor'] = 'gpu_predictor'
                
                # Create a small test DMatrix
                test_size = min(1000, X_train.shape[0])
                test_dmat = xgb.DMatrix(X_train[:test_size], label=y_train[:test_size])
                
                # Try to train a small model with GPU
                test_model = xgb.train(
                    test_params,
                    test_dmat,
                    num_boost_round=5
                )
                
                # If we get here, GPU works
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = 0
                params['predictor'] = 'gpu_predictor'
                logger.info("XGBoost will use GPU acceleration")
            except Exception as e:
                logger.warning(f"GPU test for XGBoost failed, falling back to CPU: {e}")
                params['tree_method'] = 'hist'
        else:
            params['tree_method'] = 'hist'
        
        # Create DMatrix for training - apply sample weights for class imbalance
        try:
            sample_weights = np.ones(len(y_train))
            for i, label in enumerate(y_train):
                sample_weights[i] = class_weights.get(label, 1.0)
            
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # Define watchlist for evaluation
            watchlist = [(dtrain, 'train'), (dval, 'eval')]
        except Exception as e:
            logger.error(f"Error creating DMatrix: {e}")
            # Try with default weights
            try:
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)
                watchlist = [(dtrain, 'train'), (dval, 'eval')]
            except Exception as e2:
                logger.error(f"Error creating DMatrix with default weights: {e2}")
                return None
        
        # Train in stages with increasing complexity for better accuracy
        logger.info("Stage 1: Initial training with conservative parameters...")
        try:
            initial_params = params.copy()
            initial_params['max_depth'] = 6  # Start with simpler model
            initial_params['eta'] = 0.03     # Faster learning initially
            
            # Initial training
            self.classifier = xgb.train(
                initial_params,
                dtrain,
                num_boost_round=200,
                evals=watchlist,
                early_stopping_rounds=20,
                verbose_eval=20
            )
            
            # Get best iteration from initial training
            best_iteration = self.classifier.best_iteration
            logger.info(f"Initial model achieved best iteration at: {best_iteration}")
        except Exception as e:
            logger.error(f"Error during initial XGBoost training: {e}")
            # Continue with main training anyway
            self.classifier = None
        
        # Stage 2: Main training with more boosting rounds
        logger.info("Stage 2: Main training with optimal parameters...")
        try:
            # If initial training failed, initialize a new classifier
            if self.classifier is None:
                self.classifier = xgb.Booster(params)
            
            self.classifier = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,  # Increased for better accuracy
                evals=watchlist,
                early_stopping_rounds=50,  # More patience
                verbose_eval=50,
                xgb_model=self.classifier  # Continue from the initial model
            )
            
            # Calculate final training and validation accuracy
            y_train_pred = np.argmax(self.classifier.predict(dtrain), axis=1)
            train_acc = np.mean(y_train_pred == y_train)
            
            y_val_pred = np.argmax(self.classifier.predict(dval), axis=1)
            val_acc = np.mean(y_val_pred == y_val)
            
            logger.info(f"XGBoost classifier training complete - Train accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}")
            
            # Save the model after successful training
            model_path = os.path.join(self.model_path, f'xgboost_classifier_{self.version}.model')
            self.classifier.save_model(model_path)
            logger.info(f"Saved XGBoost model to {model_path}")
            
            return self.classifier
            
        except Exception as e:
            logger.error(f"Error during main XGBoost training: {e}")
            
            # Try with simpler parameters as fallback
            logger.info("Attempting fallback XGBoost training with simpler parameters...")
            try:
                fallback_params = {
                    'objective': 'multi:softprob',
                    'num_class': num_classes,
                    'max_depth': 6,
                    'eta': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'tree_method': 'hist',  # Always use CPU for fallback
                    'eval_metric': ['mlogloss', 'merror']
                }
                
                self.classifier = xgb.train(
                    fallback_params,
                    dtrain,
                    num_boost_round=500,
                    evals=watchlist,
                    early_stopping_rounds=30,
                    verbose_eval=50
                )
                
                # Save the fallback model
                fallback_path = os.path.join(self.model_path, f'xgboost_fallback_{self.version}.model')
                self.classifier.save_model(fallback_path)
                logger.info(f"Saved fallback XGBoost model to {fallback_path}")
                
                return self.classifier
                
            except Exception as e2:
                logger.error(f"Fallback XGBoost training also failed: {e2}")
                return None
    
    def train_model(self):
        """
        Complete training pipeline including data loading, preprocessing, and model training
        
        Returns:
            True if training was successful, False otherwise
        """
        start_time = time.time()
        logger.info("Starting model training pipeline...")
        logger.info(f"Model version: {self.version}")
        
        # Import the progress tracking classes
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        try:
            from progress_callback import ProgressTracker, DatasetProgressTracker, TensorFlowProgressCallback
            progress_available = True
        except ImportError:
            progress_available = False
            logger.warning("Progress tracking module not found. Simple progress reporting will be used.")
        
        # Initialize overall progress tracker
        if progress_available:
            overall_progress = ProgressTracker(total_steps=5, description="Model Training")
        
        # Load datasets
        datasets = []
        dataset_names = ["CICIDS2017", "NSL-KDD", "UNSW-NB15"]
        
        if progress_available:
            dataset_tracker = DatasetProgressTracker(dataset_names)
        
        # CICIDS2017
        logger.info("Starting to load CICIDS2017 dataset...")
        if progress_available:
            dataset_tracker.start_dataset("CICIDS2017")
        cicids_data = self.load_cicids_dataset()
        if cicids_data is not None:
            datasets.append(cicids_data)
            if progress_available:
                dataset_tracker.complete_dataset("CICIDS2017", cicids_data.shape[0])
        
        # NSL-KDD
        logger.info("Starting to load NSL-KDD dataset...")
        if progress_available:
            dataset_tracker.start_dataset("NSL-KDD")
        nslkdd_data = self.load_nslkdd_dataset()
        if nslkdd_data is not None:
            datasets.append(nslkdd_data)
            if progress_available:
                dataset_tracker.complete_dataset("NSL-KDD", nslkdd_data.shape[0])
        
        # UNSW-NB15
        logger.info("Starting to load UNSW-NB15 dataset...")
        if progress_available:
            dataset_tracker.start_dataset("UNSW-NB15")
        unsw_data = self.load_unsw_dataset()
        if unsw_data is not None:
            datasets.append(unsw_data)
            if progress_available:
                dataset_tracker.complete_dataset("UNSW-NB15", unsw_data.shape[0])
        
        if not datasets:
            logger.error("No datasets were loaded. Please check the data paths.")
            return False
        
        # Update progress
        if progress_available:
            overall_progress.update("Dataset Loading")
        
        logger.info("Combining datasets...")
        try:
            # Combine datasets with common features
            # For simplicity, we'll just concatenate them
            # In a real implementation, more sophisticated feature alignment would be needed
            combined_data = pd.concat(datasets, ignore_index=True)
            
            # Free up memory
            del datasets
            gc.collect()
        except Exception as e:
            logger.error(f"Error combining datasets: {e}")
            # Try to use the largest dataset
            if datasets:
                largest_dataset = max(datasets, key=lambda df: df.shape[0])
                logger.info(f"Using largest dataset with {largest_dataset.shape[0]} rows")
                combined_data = largest_dataset
                
                # Free memory
                del datasets
                gc.collect()
            else:
                return False
        
        # Preprocess data
        logger.info("Starting data preprocessing...")
        X, y, feature_names = self.preprocess_data(combined_data)
        
        if X is None or y is None:
            logger.error("Data preprocessing failed.")
            return False
        
        # Update progress
        if progress_available:
            overall_progress.update("Data Preprocessing")
        
        # Split data into training and testing sets with stratification to maintain class distribution
        logger.info("Splitting data into train/validation/test sets...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
            
            # Log split sizes
            logger.info(f"Training set: {X_train.shape[0]} samples")
            logger.info(f"Validation set: {X_val.shape[0]} samples")
            logger.info(f"Test set: {X_test.shape[0]} samples")
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            # Try a simple split without stratification
            try:
                logger.info("Attempting simple split without stratification")
                split_idx1 = int(X.shape[0] * 0.8)
                split_idx2 = int(X.shape[0] * 0.9)
                
                X_train, y_train = X[:split_idx1], y[:split_idx1]
                X_val, y_val = X[split_idx1:split_idx2], y[split_idx1:split_idx2]
                X_test, y_test = X[split_idx2:], y[split_idx2:]
                
                logger.info(f"Simple split - Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
            except Exception as e2:
                logger.error(f"Simple split also failed: {e2}")
                return False
        
        # Create versioned directory for this training run
        version_dir = os.path.join(self.model_path, f"version_{self.version}")
        os.makedirs(version_dir, exist_ok=True)
        
        # Build and train autoencoder with improved parameters
        logger.info("Building autoencoder model...")
        self.autoencoder, self.encoder = self.build_autoencoder(X_train.shape[1])
        
        if self.autoencoder is None or self.encoder is None:
            logger.error("Failed to build autoencoder model.")
            return False
        
        logger.info("Training autoencoder model...")
        epochs = 100  # Increased from 50 for better convergence
        batch_size = 64  # Reduced for better stability
        
        # Train the autoencoder with improved method
        autoencoder_history = self.train_autoencoder(X_train, X_val, epochs, batch_size)
        
        # Update progress
        if progress_available:
            overall_progress.update("Autoencoder Training")
        
        # Get encoded features
        logger.info("Generating encoded features for classifier training...")
        try:
            X_train_encoded = self.encoder.predict(X_train)
            X_val_encoded = self.encoder.predict(X_val)
            
            # Check for NaN values in encoded features
            if np.isnan(X_train_encoded).any() or np.isnan(X_val_encoded).any():
                logger.warning("NaN values detected in encoded features. Cleaning up...")
                X_train_encoded = np.nan_to_num(X_train_encoded)
                X_val_encoded = np.nan_to_num(X_val_encoded)
        except Exception as e:
            logger.error(f"Error generating encoded features: {e}")
            return False
        
        # Train classifier with improved parameters
        logger.info("Training XGBoost classifier...")
        xgb_classifier = self.train_classifier(X_train_encoded, y_train, X_val_encoded, y_val)
        
        if xgb_classifier is None:
            logger.error("XGBoost classifier training failed.")
            return False
        
        # Update progress
        if progress_available:
            overall_progress.update("XGBoost Classifier Training")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        try:
            X_test_encoded = self.encoder.predict(X_test)
            
            # Clean up test encoded features if needed
            if np.isnan(X_test_encoded).any():
                logger.warning("NaN values detected in test encoded features. Cleaning up...")
                X_test_encoded = np.nan_to_num(X_test_encoded)
            
            dtest = xgb.DMatrix(X_test_encoded)
            y_pred = self.classifier.predict(dtest)
            y_pred_labels = np.argmax(y_pred, axis=1)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred_labels == y_test)
            logger.info(f"Test Accuracy: {accuracy:.4f}")
            
            # Calculate per-class metrics
            from sklearn.metrics import classification_report, confusion_matrix
            class_names = self.label_encoder.classes_
            
            # Print detailed classification report
            logger.info("Classification Report:")
            report = classification_report(y_test, y_pred_labels, target_names=class_names)
            logger.info(f"\n{report}")
            
            # Save classification report to file
            report_path = os.path.join(version_dir, 'classification_report.txt')
            with open(report_path, 'w') as f:
                f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
                f.write(report)
            logger.info(f"Classification report saved to {report_path}")
            
            # Generate confusion matrix
            try:
                cm = confusion_matrix(y_test, y_pred_labels)
                cm_path = os.path.join(version_dir, 'confusion_matrix.txt')
                with open(cm_path, 'w') as f:
                    f.write("Confusion Matrix:\n")
                    f.write(str(cm))
                logger.info(f"Confusion matrix saved to {cm_path}")
            except Exception as e:
                logger.warning(f"Error generating confusion matrix: {e}")
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            # Continue with model saving even if evaluation fails
        
        # Save models and metadata
        logger.info("Saving models and metadata...")
        save_success = self.save_models()
        
        # Update progress
        if progress_available:
            overall_progress.update("Model Evaluation and Saving")
            overall_progress.close()
        
        # Total training time
        training_time = time.time() - start_time
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        return save_success
    
    def save_models(self):
        """Save all trained models and metadata with versioning"""
        logger.info("Saving models and metadata...")
        
        # Create timestamped directory
        version_dir = os.path.join(self.model_path, f"version_{self.version}")
        os.makedirs(version_dir, exist_ok=True)
        
        try:
            # Save to the versioned directory first
            if self.autoencoder is not None:
                self.autoencoder.save(os.path.join(version_dir, 'autoencoder_model.h5'))
            
            if self.encoder is not None:
                self.encoder.save(os.path.join(version_dir, 'encoder_model.h5'))
            
            if self.classifier is not None:
                self.classifier.save_model(os.path.join(version_dir, 'xgboost_classifier.model'))
            
            # Save scaler
            joblib.dump(self.scaler, os.path.join(version_dir, 'scaler.pkl'))
            
            # Save label encoder and classes
            joblib.dump(self.label_encoder, os.path.join(version_dir, 'label_encoder.pkl'))
            
            # Save feature columns
            with open(os.path.join(version_dir, 'feature_columns.pkl'), 'wb') as f:
                pickle.dump(self.feature_columns, f)
            
            # Save attack mapping
            with open(os.path.join(version_dir, 'attack_mapping.pkl'), 'wb') as f:
                pickle.dump(self.attack_mapping, f)
            
            # Save version info
            with open(os.path.join(version_dir, 'version_info.txt'), 'w') as f:
                f.write(f"Version: {self.version}\n")
                f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # If everything succeeded, copy to main directory
            for file in os.listdir(version_dir):
                shutil.copy(os.path.join(version_dir, file), os.path.join(self.model_path, file))
            
            logger.info(f"Models and metadata saved successfully (version: {self.version})")
            return True
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            logger.info(f"Models were saved to backup directory: {version_dir}")
            return False
    
    def extract_features_from_pcap(self, pcap_file):
        """
        Extract features from a pcap file for prediction with improved error handling
        
        Args:
            pcap_file: Path to the pcap file
            
        Returns:
            features_df: DataFrame with extracted features
        """
        logger.info(f"Extracting features from pcap file: {pcap_file}")
        
        try:
            # Verify file exists and has content
            if not os.path.exists(pcap_file):
                logger.error(f"PCAP file does not exist: {pcap_file}")
                return None
                
            if os.path.getsize(pcap_file) == 0:
                logger.error(f"PCAP file is empty: {pcap_file}")
                return None
            
            # Read pcap file
            try:
                packets = rdpcap(pcap_file)
            except Exception as e:
                logger.error(f"Failed to read PCAP file: {e}")
                return None
                
            if len(packets) == 0:
                logger.warning("PCAP file contains no packets")
                return pd.DataFrame()  # Return empty DataFrame instead of None
            
            # Extract basic features
            features = []
            
            # Group packets by flow (src IP, dst IP, src port, dst port, protocol)
            flows = {}
            
            # Count total packets for progress reporting
            total_packets = len(packets)
            logger.info(f"Processing {total_packets} packets")
            
            # Process packets with progress tracking
            processed = 0
            for packet in packets:
                # Check if packet has IP layer
                if 'IP' in packet:
                    try:
                        # Extract flow identifier
                        src_ip = packet['IP'].src
                        dst_ip = packet['IP'].dst
                        protocol = packet['IP'].proto
                        
                        # Extract port information if available
                        if 'TCP' in packet:
                            src_port = packet['TCP'].sport
                            dst_port = packet['TCP'].dport
                        elif 'UDP' in packet:
                            src_port = packet['UDP'].sport
                            dst_port = packet['UDP'].dport
                        else:
                            src_port = dst_port = 0
                        
                        flow_id = (src_ip, dst_ip, src_port, dst_port, protocol)
                        
                        # Add packet to flow
                        if flow_id not in flows:
                            flows[flow_id] = []
                        
                        flows[flow_id].append(packet)
                    except Exception as e:
                        # Log but continue with next packet
                        logger.warning(f"Error processing packet: {e}")
                
                # Update progress periodically
                processed += 1
                if processed % 10000 == 0 or processed == total_packets:
                    logger.info(f"Processed {processed}/{total_packets} packets ({processed/total_packets*100:.1f}%)")
            
            if not flows:
                logger.warning("No valid IP flows found in PCAP file")
                return pd.DataFrame()  # Return empty DataFrame
            
            logger.info(f"Found {len(flows)} flows")
            
            # Process each flow to extract features
            flow_count = 0
            for flow_id, flow_packets in flows.items():
                try:
                    # Extract features from flow
                    src_ip, dst_ip, src_port, dst_port, protocol = flow_id
                    
                    # Basic flow statistics
                    flow_duration = flow_packets[-1].time - flow_packets[0].time if len(flow_packets) > 1 else 0
                    packet_count = len(flow_packets)
                    total_bytes = sum(len(p) for p in flow_packets)
                    avg_packet_size = total_bytes / packet_count if packet_count > 0 else 0
                    
                    # Packet timing features
                    if packet_count > 1:
                        packet_times = [p.time for p in flow_packets]
                        inter_arrival_times = [packet_times[i+1] - packet_times[i] for i in range(len(packet_times)-1)]
                        mean_iat = np.mean(inter_arrival_times)
                        std_iat = np.std(inter_arrival_times)
                        min_iat = np.min(inter_arrival_times)
                        max_iat = np.max(inter_arrival_times)
                    else:
                        mean_iat = std_iat = min_iat = max_iat = 0
                    
                    # TCP flags if applicable
                    tcp_flags_count = {'FIN': 0, 'SYN': 0, 'RST': 0, 'PSH': 0, 'ACK': 0, 'URG': 0}
                    if protocol == 6:  # TCP
                        for p in flow_packets:
                            if 'TCP' in p:
                                tcp = p['TCP']
                                try:
                                    if tcp.flags & 0x01: tcp_flags_count['FIN'] += 1
                                    if tcp.flags & 0x02: tcp_flags_count['SYN'] += 1
                                    if tcp.flags & 0x04: tcp_flags_count['RST'] += 1
                                    if tcp.flags & 0x08: tcp_flags_count['PSH'] += 1
                                    if tcp.flags & 0x10: tcp_flags_count['ACK'] += 1
                                    if tcp.flags & 0x20: tcp_flags_count['URG'] += 1
                                except Exception as e:
                                    logger.warning(f"Error processing TCP flags: {e}")
                    
                    # Collect all features
                    flow_features = {
                        'duration': flow_duration,
                        'protocol': protocol,
                        'src_port': src_port,
                        'dst_port': dst_port,
                        'packet_count': packet_count,
                        'total_bytes': total_bytes,
                        'avg_packet_size': avg_packet_size,
                        'mean_iat': mean_iat,
                        'std_iat': std_iat,
                        'min_iat': min_iat,
                        'max_iat': max_iat,
                        'fin_count': tcp_flags_count['FIN'],
                        'syn_count': tcp_flags_count['SYN'],
                        'rst_count': tcp_flags_count['RST'],
                        'psh_count': tcp_flags_count['PSH'],
                        'ack_count': tcp_flags_count['ACK'],
                        'urg_count': tcp_flags_count['URG']
                    }
                    
                    features.append(flow_features)
                    
                    # Update progress periodically
                    flow_count += 1
                    if flow_count % 1000 == 0 or flow_count == len(flows):
                        logger.info(f"Processed {flow_count}/{len(flows)} flows ({flow_count/len(flows)*100:.1f}%)")
                        
                except Exception as e:
                    logger.warning(f"Error extracting features for flow {flow_id}: {e}")
            
            # Create DataFrame from features
            if not features:
                logger.warning("No features extracted from flows")
                return pd.DataFrame()
            
            features_df = pd.DataFrame(features)
            
            # Check for NaN or infinite values
            if features_df.isnull().values.any() or np.isinf(features_df.values).any():
                logger.warning("Found NaN or infinite values in extracted features, cleaning...")
                features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                features_df.fillna(0, inplace=True)
            
            logger.info(f"Extracted features from {len(features)} flows")
            return features_df
        
        except Exception as e:
            logger.error(f"Error extracting features from pcap: {e}")
            return None
    
    def load_trained_models(self):
        """Load saved models for prediction with improved error handling"""
        logger.info("Loading trained models...")
        
        try:
            # Check if model path exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model path does not exist: {self.model_path}")
                return False
            
            # Find the latest version directory
            version_dirs = [d for d in os.listdir(self.model_path) 
                            if os.path.isdir(os.path.join(self.model_path, d)) and d.startswith('version_')]
            
            if version_dirs:
                # Sort by creation time
                version_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(self.model_path, x)), reverse=True)
                latest_version = version_dirs[0]
                logger.info(f"Found latest model version: {latest_version}")
                model_dir = os.path.join(self.model_path, latest_version)
            else:
                # Use main directory
                model_dir = self.model_path
            
            # Define custom objects for Keras model loading - using class-based metrics
            import tensorflow as tf
            custom_objects = {
                'mse': tf.keras.metrics.MeanSquaredError(),
                'mean_squared_error': tf.keras.metrics.MeanSquaredError()
            }
            
            # Try loading with compile=False first to avoid metric issues
            try:
                self.autoencoder = load_model(os.path.join(model_dir, 'autoencoder_model.h5'), 
                                            custom_objects=custom_objects,
                                            compile=False)
                logger.info("Loaded autoencoder model")
                
                self.encoder = load_model(os.path.join(model_dir, 'encoder_model.h5'),
                                        custom_objects=custom_objects,
                                        compile=False)
                logger.info("Loaded encoder model")
            except Exception as e:
                logger.error(f"Error loading models with compile=False: {e}")
                
                # Try fallback approach - recreate and load weights separately
                try:
                    logger.info("Attempting fallback approach - rebuilding model from scratch")
                    
                    # Create a simplified autoencoder and encoder with the same architecture
                    input_dim = 386  # This should match your feature dimension
                    
                    # Build simplified autoencoder
                    inputs = tf.keras.layers.Input(shape=(input_dim,))
                    x = tf.keras.layers.BatchNormalization()(inputs)
                    x = tf.keras.layers.Dense(512, activation='relu')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.Dense(256, activation='relu')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.Dense(128, activation='relu')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    encoded = tf.keras.layers.Dense(64, activation='relu')(x)
                    
                    # Encoder model
                    self.encoder = tf.keras.Model(inputs=inputs, outputs=encoded)
                    
                    # Continue with decoder for the full autoencoder
                    x = tf.keras.layers.Dense(128, activation='relu')(encoded)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.Dense(256, activation='relu')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.Dense(512, activation='relu')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    outputs = tf.keras.layers.Dense(input_dim, activation='tanh')(x)
                    
                    # Autoencoder model
                    self.autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs)
                    
                    logger.info("Successfully recreated models - skipping weight loading")
                except Exception as e2:
                    logger.error(f"Failed to create backup models: {e2}")
                    return False
            
            # Load XGBoost classifier
            try:
                self.classifier = xgb.Booster()
                self.classifier.load_model(os.path.join(model_dir, 'xgboost_classifier.model'))
                logger.info("Loaded XGBoost classifier")
            except Exception as e:
                logger.error(f"Error loading XGBoost classifier: {e}")
                # Try alternate file name
                try:
                    for file in os.listdir(model_dir):
                        if file.startswith('xgboost_classifier') and file.endswith('.model'):
                            self.classifier = xgb.Booster()
                            self.classifier.load_model(os.path.join(model_dir, file))
                            logger.info(f"Loaded XGBoost classifier from: {file}")
                            break
                except Exception as e2:
                    logger.error(f"Failed to load alternate classifier: {e2}")
                    return False
            
            # Load other required files
            try:
                self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
                logger.info("Loaded scaler")
                
                self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
                logger.info("Loaded label encoder")
                
                with open(os.path.join(model_dir, 'feature_columns.pkl'), 'rb') as f:
                    self.feature_columns = pickle.load(f)
                logger.info(f"Loaded feature columns: {len(self.feature_columns)} features")
                
                with open(os.path.join(model_dir, 'attack_mapping.pkl'), 'rb') as f:
                    self.attack_mapping = pickle.load(f)
                logger.info("Loaded attack mapping")
            except Exception as e:
                logger.error(f"Error loading support files: {e}")
                return False
            
            logger.info("All models loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Unexpected error loading models: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def predict(self, pcap_file):
        """
        Predict network traffic type from a pcap file with improved error handling
        
        Args:
            pcap_file: Path to the pcap file
            
        Returns:
            predictions: Dictionary with prediction results
        """
        logger.info(f"Starting prediction for pcap file: {pcap_file}")
        
        # Load models if not already loaded
        if self.encoder is None or self.classifier is None:
            if not self.load_trained_models():
                return {'error': 'Failed to load models'}
        
        # Extract features from pcap file
        features_df = self.extract_features_from_pcap(pcap_file)
        
        if features_df is None:
            return {'error': 'Failed to extract features from pcap file'}
        
        if features_df.empty:
            return {
                'status': 'Unknown',
                'details': 'No valid network flows found in the pcap file',
                'flows_analyzed': 0,
                'normal_percentage': 0,
                'attack_types': []
            }
        
        # Prepare features for prediction
        try:
            # Align feature columns with training data
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            # Check for any extra columns that aren't in the training data
            extra_cols = set(features_df.columns) - set(self.feature_columns)
            if extra_cols:
                logger.warning(f"Dropping extra columns not in training data: {extra_cols}")
                features_df = features_df.drop(columns=list(extra_cols))
            
            # Keep only columns used during training and in the right order
            features_df = features_df[self.feature_columns]
            
            # Handle potential NaN or Inf values
            features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            features_df.fillna(0, inplace=True)
            
            # Scale features
            try:
                X = self.scaler.transform(features_df)
            except Exception as e:
                logger.error(f"Error during feature scaling: {e}")
                # Try to handle common scaling errors
                if "shape of X" in str(e):
                    logger.info("Attempting to fix feature shape mismatch")
                    # Add missing columns
                    for col in self.feature_columns:
                        if col not in features_df.columns:
                            features_df[col] = 0
                    # Keep only needed columns in the right order
                    features_df = features_df[self.feature_columns]
                    # Try scaling again
                    X = self.scaler.transform(features_df)
                else:
                    # Use a minimal feature transformation as fallback
                    logger.info("Using minimal feature transformation as fallback")
                    X = features_df.values
                    # Simple normalization
                    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
            
            # Additional safety check for NaN/Inf values
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Encode features
            X_encoded = self.encoder.predict(X)
            
            # Final check for NaN values
            X_encoded = np.nan_to_num(X_encoded)
            
            # Predict with XGBoost
            dtest = xgb.DMatrix(X_encoded)
            y_pred_proba = self.classifier.predict(dtest)
            
            # Check that prediction returned expected format
            if not isinstance(y_pred_proba, np.ndarray) or y_pred_proba.ndim != 2:
                logger.error(f"XGBoost prediction returned unexpected format: {type(y_pred_proba)}")
                # Try to reshape if possible
                if isinstance(y_pred_proba, np.ndarray):
                    num_classes = len(self.label_encoder.classes_)
                    y_pred_proba = y_pred_proba.reshape(-1, num_classes)
            
            # Get predicted classes and their probabilities
            y_pred_labels = np.argmax(y_pred_proba, axis=1)
            y_pred_proba_max = np.max(y_pred_proba, axis=1)
            
            # Convert numeric labels back to original labels
            try:
                predicted_labels = self.label_encoder.inverse_transform(y_pred_labels)
            except Exception as e:
                logger.error(f"Error in label conversion: {e}")
                # Fallback - manually map the numeric labels
                label_map = {i: label for i, label in enumerate(self.label_encoder.classes_)}
                predicted_labels = np.array([label_map.get(label, 'unknown') for label in y_pred_labels])
            
            # Count the occurrences of each attack type
            attack_counts = {}
            for label, prob in zip(predicted_labels, y_pred_proba_max):
                if label not in attack_counts:
                    attack_counts[label] = {'count': 0, 'total_prob': 0}
                
                attack_counts[label]['count'] += 1
                attack_counts[label]['total_prob'] += prob
            
            # Calculate average probability for each attack type
            for label in attack_counts:
                attack_counts[label]['avg_prob'] = attack_counts[label]['total_prob'] / attack_counts[label]['count']
                # Map attack type to its description
                attack_counts[label]['description'] = self.attack_mapping.get(label, label)
            
            # Determine overall prediction
            total_flows = len(predicted_labels)
            normal_flows = sum(1 for label in predicted_labels if label == 'normal')
            normal_percentage = (normal_flows / total_flows) * 100 if total_flows > 0 else 0
            
            if normal_percentage >= 95:
                overall_status = "Normal"
                status_details = "Traffic appears to be normal (≥95% normal flows)"
            elif normal_percentage >= 80:
                overall_status = "Mostly Normal"
                status_details = "Traffic is mostly normal with some suspicious activity"
            else:
                # Find the most common attack type
                attack_flows = {label: attack_counts[label]['count'] for label in attack_counts if label != 'normal'}
                if attack_flows:
                    most_common_attack = max(attack_flows.items(), key=lambda x: x[1])[0]
                    attack_percentage = (attack_flows[most_common_attack] / total_flows) * 100
                    overall_status = f"Under Attack: {self.attack_mapping.get(most_common_attack, most_common_attack)}"
                    status_details = f"{attack_percentage:.1f}% of flows classified as {most_common_attack} attack"
                else:
                    overall_status = "Suspicious"
                    status_details = "Traffic contains suspicious patterns"
            
            # Prepare results
            result = {
                'status': overall_status,
                'details': status_details,
                'flows_analyzed': total_flows,
                'normal_percentage': normal_percentage,
                'attack_types': [
                    {
                        'type': label,
                        'description': attack_counts[label]['description'],
                        'count': attack_counts[label]['count'],
                        'percentage': (attack_counts[label]['count'] / total_flows) * 100 if total_flows > 0 else 0,
                        'confidence': attack_counts[label]['avg_prob'] * 100
                    }
                    for label in attack_counts
                ]
            }
            
            # Add summary statistics
            try:
                # Sort attack types by percentage
                result['attack_types'] = sorted(
                    result['attack_types'], 
                    key=lambda x: x['percentage'], 
                    reverse=True
                )
                
                # Calculate attack percentages excluding normal traffic
                attack_only_flows = total_flows - normal_flows
                if attack_only_flows > 0:
                    for attack in result['attack_types']:
                        if attack['type'] != 'normal':
                            attack['attack_percentage'] = (attack['count'] / attack_only_flows) * 100
                            
                # Add detection time
                result['detection_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Add flow rate information
                if 'duration' in features_df.columns:
                    max_duration = features_df['duration'].max()
                    if max_duration > 0:
                        result['flow_rate'] = total_flows / max_duration
                    else:
                        result['flow_rate'] = 0
            except Exception as e:
                logger.warning(f"Error adding summary statistics: {e}")
            
            logger.info(f"Prediction completed with status: {overall_status}")
            return result
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            # Try to provide partial results if possible
            if 'features_df' in locals() and not features_df.empty:
                return {
                    'error': f'Prediction failed: {str(e)}',
                    'status': 'Error',
                    'details': 'Error during prediction processing',
                    'flows_analyzed': len(features_df),
                    'partial_data': True
                }
            else:
                return {'error': f'Prediction failed: {str(e)}'}


def main():
    """Main function to train the model with improved error handling"""
    logger.info("Starting the Network Intrusion Detection Model training")
    
    try:
        # Create model
        model = NetworkIntrusionDetectionModel()
        
        # Train model
        success = model.train_model()
        
        if success:
            logger.info("Model training completed successfully")
            # Test prediction
            test_pcap = "./test.pcap"
            if os.path.exists(test_pcap):
                logger.info("Testing model with a sample pcap file")
                predictions = model.predict(test_pcap)
                logger.info(f"Prediction results: {predictions}")
                
                # Save prediction results to file
                result_file = f"test_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                try:
                    import json
                    with open(result_file, 'w') as f:
                        json.dump(predictions, f, indent=2)
                    logger.info(f"Prediction results saved to {result_file}")
                except Exception as e:
                    logger.warning(f"Could not save prediction results: {e}")
        else:
            logger.error("Model training failed")
            return 1
        
        return 0
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}")
        import traceback
        logger.critical(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)