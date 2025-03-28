import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import joblib
import pickle
import time
import logging
import gc
import sys
import warnings
import shutil
import traceback
from datetime import datetime
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom progress tracking
try:
    from progress_callback import ProgressTracker, DatasetProgressTracker, TensorFlowProgressCallback
    progress_tracking_available = True
except ImportError:
    progress_tracking_available = False
    warnings.warn("Progress tracking module not found. Basic progress reporting will be used.")

# Configure TensorFlow to use GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"Found {len(gpus)} GPU(s): {gpus}")
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


def log_memory_usage(message=""):
    """Log current memory usage"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / (1024 * 1024 * 1024)  # GB
        logger.info(f"Memory usage ({message}): {memory_usage:.2f} GB")
        return memory_usage
    except ImportError:
        logger.warning("psutil not available, can't monitor memory usage")
        return None
    except Exception as e:
        logger.warning(f"Error logging memory usage: {e}")
        return None


class KitsuneNIDSModel:
    """
    Network Intrusion Detection System using the Kitsune dataset
    Implements a hybrid autoencoder + classification approach
    """
    def __init__(self, data_path="./datasets", model_path="./models", chunk_size=100000):
        """
        Initialize the Kitsune NIDS Model
        
        Args:
            data_path: Path to the Kitsune dataset
            model_path: Path to save trained models
            chunk_size: Number of samples to process at once (for memory efficiency)
        """
        self.data_path = data_path
        self.model_path = model_path
        self.chunk_size = chunk_size
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize components
        self.autoencoder = None
        self.encoder = None
        self.classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Model parameters - optimized for i7-12650h, RTX 3050 4GB
        self.latent_dim = 48  # Increased for better feature representation
        self.batch_size = 512  # Increased for faster training on GPU
        self.use_mixed_precision = True  # Use mixed precision for GPU efficiency
        
        # Attack types in Kitsune dataset
        self.attack_types = [
            'Normal',
            'Active_Wiretap',
            'ARP_MitM',
            'Fuzzing',
            'Mirai_Botnet',
            'OS_Scan',
            'SSDP_Flood',
            'SSL_Renegotiation',
            'SYN_DoS',
            'Video_Injection'
        ]
        
        # Attack mapping for human-readable descriptions
        self.attack_mapping = {
            'Normal': 'Normal Traffic',
            'Active_Wiretap': 'Active Wiretap Attack',
            'ARP_MitM': 'ARP Man-in-the-Middle Attack',
            'Fuzzing': 'Protocol Fuzzing Attack',
            'fuzzing': 'Protocol Fuzzing Attack',
            'Mirai_Botnet': 'Mirai Botnet Activity',
            'OS_Scan': 'Operating System Scanning',
            'SSDP_Flood': 'SSDP Flooding Attack',
            'SSL_Renegotiation': 'SSL Renegotiation Attack',
            'SYN_DoS': 'SYN Denial of Service',
            'Video_Injection': 'Video Injection Attack'
        }
        
        # Statistics for normal traffic (will be computed during training)
        self.normal_traffic_stats = {
            'flow_duration_mean': 0.0,
            'flow_duration_std': 1.0,
            'pkt_count_mean': 0.0,
            'pkt_count_std': 1.0,
            'total_size_mean': 0.0,
            'total_size_std': 1.0,
            'mean_iat_mean': 0.0,
            'mean_iat_std': 1.0
        }
        
        # Feature columns (will be set during data loading)
        self.feature_columns = None
        self.all_features = set()  # Store all possible feature columns for consistent preprocessing
        
        # Track version for model saving
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Model version: {self.version}")
        
    def load_kitsune_dataset(self, limit_samples=None, specific_attack=None):
        """
        Load the Kitsune dataset with memory efficiency
        
        Args:
            limit_samples: Optional limit of samples per attack type (for testing)
            specific_attack: Load only this specific attack type
            
        Returns:
            combined_data: DataFrame with combined dataset
        """
        logger.info("Loading Kitsune dataset...")
        
        # Check if path exists
        if not os.path.exists(self.data_path):
            logger.error(f"Dataset path does not exist: {self.data_path}")
            return None
            
        combined_data_chunks = []
        total_samples = 0
        
        # Initialize attack directories
        if specific_attack:
            attack_dirs = [specific_attack]
            logger.info(f"Loading specific attack type: {specific_attack}")
        else:
            # Get all attack directories
            attack_dirs = [d for d in os.listdir(self.data_path) 
                        if os.path.isdir(os.path.join(self.data_path, d)) and not d.startswith('.')]
        
        if progress_tracking_available:
            dataset_tracker = DatasetProgressTracker(attack_dirs)
        
        # Generate synthetic normal traffic if needed - 10% of total samples
        has_normal = 'Normal' in attack_dirs
        if not has_normal:
            logger.info("Normal traffic not found in dataset, will generate synthetic normal samples")
        
        for attack_type in attack_dirs:
            if progress_tracking_available:
                dataset_tracker.start_dataset(attack_type)
                
            logger.info(f"Processing {attack_type} dataset...")
            
            # Find dataset CSV file
            dataset_file = os.path.join(self.data_path, attack_type, f"{attack_type}_dataset.csv")
            labels_file = os.path.join(self.data_path, attack_type, f"{attack_type}_labels.csv")
            
            if not os.path.exists(dataset_file):
                logger.warning(f"Dataset file not found: {dataset_file}")
                continue
                
            if not os.path.exists(labels_file):
                logger.warning(f"Labels file not found: {labels_file}")
                continue
            
            try:
                # Get file size to determine loading strategy
                dataset_size = os.path.getsize(dataset_file) / (1024 * 1024)  # Size in MB
                logger.info(f"{attack_type} dataset size: {dataset_size:.2f} MB")
                
                # Load labels first (should be small)
                try:
                    labels_df = pd.read_csv(labels_file)
                    logger.info(f"Loaded {attack_type} labels: {labels_df.shape[0]} rows")
                except Exception as e:
                    logger.error(f"Error loading labels file {labels_file}: {e}")
                    continue
                
                # Calculate how many samples to load for this attack type
                if limit_samples is not None:
                    if specific_attack:
                        # If loading a specific attack, use the full limit
                        max_samples_for_attack = limit_samples
                    else:
                        # If loading all attacks, divide limit among them
                        max_samples_for_attack = limit_samples // len(attack_dirs)
                        
                    logger.info(f"Will load up to {max_samples_for_attack} samples for {attack_type}")
                else:
                    max_samples_for_attack = None
                
                attack_samples = 0
                
                # Process large dataset files in chunks
                if dataset_size > 100:  # If larger than 100MB
                    chunk_count = 0
                    total_chunks = 0
                    
                    # First pass to count chunks
                    for _ in pd.read_csv(dataset_file, chunksize=self.chunk_size):
                        total_chunks += 1
                    
                    logger.info(f"Processing {attack_type} in {total_chunks} chunks")
                    
                    # Second pass to read chunks
                    for chunk in pd.read_csv(dataset_file, chunksize=self.chunk_size):
                        chunk_count += 1
                        logger.info(f"Processing chunk {chunk_count}/{total_chunks} for {attack_type}")
                        
                        # Collect all feature names (important for ensuring consistent features)
                        self.all_features.update(chunk.columns)
                        
                        # Add attack type label to the chunk
                        chunk['Label'] = attack_type
                        
                        # Check if we've reached the sample limit for this attack
                        if max_samples_for_attack is not None and attack_samples + chunk.shape[0] > max_samples_for_attack:
                            # Take only what we need to reach the limit
                            samples_needed = max_samples_for_attack - attack_samples
                            if samples_needed > 0:
                                chunk = chunk.iloc[:samples_needed]
                                combined_data_chunks.append(chunk)
                                attack_samples += samples_needed
                                total_samples += samples_needed
                                logger.info(f"Sample limit reached for {attack_type}: {attack_samples}")
                            break
                            
                        combined_data_chunks.append(chunk)
                        attack_samples += chunk.shape[0]
                        total_samples += chunk.shape[0]
                        
                        # Free memory after processing each chunk
                        gc.collect()
                else:
                    # For smaller files, load at once
                    chunk = pd.read_csv(dataset_file)
                    
                    # Collect all feature names
                    self.all_features.update(chunk.columns)
                    
                    chunk['Label'] = attack_type
                    
                    # Apply limit if specified
                    if max_samples_for_attack is not None and chunk.shape[0] > max_samples_for_attack:
                        chunk = chunk.iloc[:max_samples_for_attack]
                    
                    combined_data_chunks.append(chunk)
                    attack_samples = chunk.shape[0]
                    total_samples += attack_samples
                    logger.info(f"Loaded {attack_type} dataset: {chunk.shape[0]} rows")
                
                if progress_tracking_available:
                    dataset_tracker.complete_dataset(attack_type, attack_samples)
                    
            except Exception as e:
                logger.error(f"Error processing {attack_type} dataset: {e}")
        
        # Check if we have loaded any data
        if not combined_data_chunks:
            logger.error("No dataset chunks were loaded")
            return None
        
        # Generate synthetic normal traffic if needed
        if not has_normal:
            normal_size = int(total_samples * 0.15)  # 15% of total samples as normal
            logger.info(f"Generating {normal_size} synthetic normal traffic samples")
            try:
                # Create synthetic normal traffic from random variations of all traffic
                synthetic_normal = self._generate_synthetic_normal(combined_data_chunks, normal_size)
                if synthetic_normal is not None:
                    combined_data_chunks.append(synthetic_normal)
                    logger.info(f"Added {len(synthetic_normal)} synthetic normal samples")
            except Exception as e:
                logger.error(f"Error generating synthetic normal traffic: {e}")
        
        # Remove 'Label' from all_features if it's there
        if 'Label' in self.all_features:
            self.all_features.remove('Label')
            
        logger.info(f"Collected {len(self.all_features)} unique features from all datasets")
        
        # Combine all chunks
        logger.info(f"Combining {len(combined_data_chunks)} data chunks...")
        try:
            combined_data = pd.concat(combined_data_chunks, ignore_index=True)
            logger.info(f"Combined dataset: {combined_data.shape[0]} rows, {combined_data.shape[1]} columns")
            
            # Free memory from individual chunks
            del combined_data_chunks
            gc.collect()
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error combining data chunks: {e}")
            return None
    
    def _generate_synthetic_normal(self, data_chunks, normal_size):
        """
        Generate synthetic normal traffic samples by statistical modeling
        
        Args:
            data_chunks: List of DataFrames with attack traffic
            normal_size: Number of normal samples to generate
            
        Returns:
            normal_df: DataFrame with synthetic normal traffic
        """
        try:
            # Sample from each chunk
            sample_size = min(1000, normal_size // len(data_chunks))
            samples = []
            
            for chunk in data_chunks:
                # Take a small sample from each attack type
                if len(chunk) > sample_size:
                    sample = chunk.sample(sample_size)
                else:
                    sample = chunk
                samples.append(sample)
            
            # Combine samples
            combined_sample = pd.concat(samples, ignore_index=True)
            
            # Get feature statistics
            feature_stats = {}
            for feature in combined_sample.columns:
                if feature != 'Label':
                    try:
                        column = combined_sample[feature]
                        # Only process numeric columns
                        if np.issubdtype(column.dtype, np.number):
                            # Get mean and std, handle null values
                            mean = column.mean() if not np.isnan(column.mean()) else 0
                            std = column.std() if not np.isnan(column.std()) else 1
                            # Adjust std to ensure variability
                            std = max(std, 0.01 * abs(mean) if mean != 0 else 0.01)
                            feature_stats[feature] = (mean, std)
                    except Exception as e:
                        logger.warning(f"Error calculating stats for feature {feature}: {e}")
            
            # Generate synthetic samples
            synthetic_data = {}
            for feature, (mean, std) in feature_stats.items():
                # Generate values with small random variations
                synthetic_data[feature] = np.random.normal(mean, std * 0.75, normal_size)
            
            # Create DataFrame and add normal label
            normal_df = pd.DataFrame(synthetic_data)
            normal_df['Label'] = 'Normal'
            
            # Log statistics for normal traffic
            self._log_synthetic_normal_stats(normal_df)
            
            return normal_df
            
        except Exception as e:
            logger.error(f"Error generating synthetic normal traffic: {e}")
            return None
    
    def _log_synthetic_normal_stats(self, normal_df):
        """Log statistics for synthetic normal traffic"""
        # Standard network flow features
        flow_stats = {
            'flow_duration': normal_df['flow_duration'].mean() if 'flow_duration' in normal_df.columns else 0,
            'pkt_count': normal_df['pkt_count'].mean() if 'pkt_count' in normal_df.columns else 0,
            'total_size': normal_df['total_size'].mean() if 'total_size' in normal_df.columns else 0,
            'mean_iat': normal_df['mean_iat'].mean() if 'mean_iat' in normal_df.columns else 0
        }
        
        logger.info("Synthetic normal traffic statistics:")
        for stat, value in flow_stats.items():
            logger.info(f"  {stat}: {value}")
        
        # Save stats for later use in prediction
        self.normal_traffic_stats = {
            'flow_duration_mean': normal_df['flow_duration'].mean() if 'flow_duration' in normal_df.columns else 0,
            'flow_duration_std': normal_df['flow_duration'].std() if 'flow_duration' in normal_df.columns else 1,
            'pkt_count_mean': normal_df['pkt_count'].mean() if 'pkt_count' in normal_df.columns else 0,
            'pkt_count_std': normal_df['pkt_count'].std() if 'pkt_count' in normal_df.columns else 1,
            'total_size_mean': normal_df['total_size'].mean() if 'total_size' in normal_df.columns else 0,
            'total_size_std': normal_df['total_size'].std() if 'total_size' in normal_df.columns else 1,
            'mean_iat_mean': normal_df['mean_iat'].mean() if 'mean_iat' in normal_df.columns else 0,
            'mean_iat_std': normal_df['mean_iat'].std() if 'mean_iat' in normal_df.columns else 1,
        }
    
    def preprocess_data(self, combined_data, for_encoder=False):
        """
        Preprocess the dataset for training
        
        Args:
            combined_data: Combined dataset from load_kitsune_dataset
            for_encoder: Whether this preprocessing is for training the encoder
                        If True, it will save the feature list as a reference
            
        Returns:
            X: Features for training
            y: Labels for training
            feature_names: List of feature names
        """
        logger.info("Preprocessing combined dataset...")
        
        if combined_data is None or combined_data.empty:
            logger.error("Cannot preprocess empty or None dataset")
            return None, None, None
        
        # Start timer
        start_time = time.time()
        
        try:
            # Make a copy to avoid modifying the original
            data = combined_data.copy()
            
            # Handle NaN and infinite values first
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Check for NaN values
            nan_count = data.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in dataset")
                data.fillna(0, inplace=True)
            
            # Ensure all features are present
            if for_encoder:
                # If this is for the encoder, we're setting up the reference feature set
                # Add any missing feature columns from what we've collected across all datasets
                missing_features = self.all_features - set(data.columns)
                if missing_features:
                    # Create a DataFrame with all missing columns at once - FIX 1: Performance improvement
                    missing_df = pd.DataFrame(0, index=data.index, columns=list(missing_features))
                    # Concatenate with original DataFrame
                    data = pd.concat([data, missing_df], axis=1)
                    logger.info(f"Added {len(missing_features)} missing feature columns at once")
            else:
                # If the feature columns reference exists, align with it
                if self.feature_columns is not None:
                    current_features = set(data.columns)
                    reference_features = set(self.feature_columns)
                    
                    # Add missing columns all at once - FIX 1: Performance improvement
                    missing_features = reference_features - current_features
                    if missing_features:
                        # Create a DataFrame with all missing columns at once
                        missing_df = pd.DataFrame(0, index=data.index, columns=list(missing_features))
                        # Concatenate with original DataFrame
                        data = pd.concat([data, missing_df], axis=1)
                        logger.info(f"Added {len(missing_features)} missing feature columns at once")
                    
                    # Remove extra columns
                    extra_features = current_features - reference_features - {'Label'}
                    if extra_features:
                        data = data.drop(columns=list(extra_features))
                        logger.info(f"Removed {len(extra_features)} extra feature columns")
                    
                    # Ensure columns are in the same order as the reference
                    if 'Label' in data.columns:
                        label_column = data['Label']
                        data = data[self.feature_columns]
                        data['Label'] = label_column
                    else:
                        data = data[self.feature_columns]
            
            # Handle categorical columns
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            categorical_cols = [col for col in categorical_cols if col != 'Label']
            
            # One-hot encode categorical features
            for col in categorical_cols:
                try:
                    # Use pd.get_dummies for one-hot encoding
                    dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                    # Join the encoded variables with the dataframe
                    data = pd.concat([data.drop(col, axis=1), dummies], axis=1)
                    logger.info(f"One-hot encoded categorical column: {col}")
                except Exception as e:
                    logger.warning(f"Error encoding column {col}: {e}")
                    # If encoding fails, drop the column
                    data.drop(col, axis=1, inplace=True)
            
            # Separate features and label
            if 'Label' in data.columns:
                X = data.drop('Label', axis=1)
                y = data['Label']
            else:
                logger.error("Label column not found in dataset")
                return None, None, None
            
            # If this is for the encoder, store the feature columns for reference
            if for_encoder:
                self.feature_columns = X.columns.tolist()
                logger.info(f"Stored {len(self.feature_columns)} feature columns for reference")
            
            # Scale features
            if for_encoder:
                # Fit the scaler for the first time
                X_scaled = self.scaler.fit_transform(X)
                # Save feature statistics for prediction
                self._log_feature_stats(X)
            else:
                # Use the already fitted scaler
                try:
                    X_scaled = self.scaler.transform(X)
                except Exception as e:
                    logger.error(f"Error transforming features with scaler: {e}")
                    # Temporary fallback: fit and transform
                    logger.warning("Using fit_transform as a fallback")
                    X_scaled = self.scaler.fit_transform(X)
            
            # Convert to float32 to reduce memory usage by 50%
            X_scaled = X_scaled.astype(np.float32)
            
            # Encode labels
            if for_encoder:
                # Fit label encoder for the first time
                y_encoded = self.label_encoder.fit_transform(y)
                # Save label classes for reference
                self.label_classes = self.label_encoder.classes_
                logger.info(f"Label classes: {self.label_classes}")
            else:
                # Use already fitted label encoder
                try:
                    y_encoded = self.label_encoder.transform(y)
                except Exception as e:
                    logger.warning(f"Error transforming labels: {e}")
                    # Handle any new labels not seen during training
                    # For now, just re-fit the encoder
                    logger.warning("Re-fitting label encoder to handle new labels")
                    y_encoded = self.label_encoder.fit_transform(y)
            
            # Log processing time
            processing_time = time.time() - start_time
            logger.info(f"Data preprocessing completed in {processing_time:.2f} seconds")
            
            # Log class distribution
            unique_classes, counts = np.unique(y_encoded, return_counts=True)
            logger.info("Class distribution:")
            for i, (class_idx, count) in enumerate(zip(unique_classes, counts)):
                class_name = self.label_encoder.inverse_transform([class_idx])[0]
                percentage = (count / len(y_encoded)) * 100
                logger.info(f"  {class_name}: {count} samples ({percentage:.2f}%)")
            
            return X_scaled, y_encoded, X.columns.tolist()
            
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            logger.error(traceback.format_exc())
            return None, None, None
    
    def _log_feature_stats(self, X):
        """Log feature statistics for later use in prediction"""
        try:
            # Log basic statistics for key features if they exist
            key_features = ['flow_duration', 'pkt_count', 'total_size', 'mean_iat']
            for feature in key_features:
                if feature in X.columns:
                    mean = X[feature].mean()
                    std = X[feature].std()
                    min_val = X[feature].min()
                    max_val = X[feature].max()
                    logger.info(f"Feature {feature} stats: mean={mean:.2f}, std={std:.2f}, min={min_val:.2f}, max={max_val:.2f}")
        except Exception as e:
            logger.warning(f"Error logging feature stats: {e}")
            
    def build_autoencoder(self, input_dim):
        """
        Build a better autoencoder model optimized for RTX 3050 4GB GPU
        
        Args:
            input_dim: Dimension of input features
            
        Returns:
            autoencoder: Complete autoencoder model
            encoder: Encoder part for feature extraction
        """
        logger.info(f"Building autoencoder with input dimension {input_dim}")
        
        # Enable mixed precision for memory efficiency on GPU
        if self.use_mixed_precision and len(tf.config.list_physical_devices('GPU')) > 0:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Enabled mixed precision training (float16)")
            except Exception as e:
                logger.warning(f"Could not enable mixed precision: {e}")
        
        try:
            # Use more conservative initializations for stability with regularization
            initializer = HeNormal(seed=42)
            regularizer = l1_l2(l1=1e-5, l2=1e-4)  # Added regularization to prevent overfitting
            
            # Input layer
            input_layer = Input(shape=(input_dim,), name="input_layer")
            
            # Encoder - Using LeakyReLU for better gradient flow
            x = BatchNormalization()(input_layer)
            x = Dense(256, kernel_initializer=initializer, kernel_regularizer=regularizer)(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.25)(x)
            
            x = Dense(128, kernel_initializer=initializer, kernel_regularizer=regularizer)(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.25)(x)
            
            # Bottleneck layer
            encoded = Dense(self.latent_dim, 
                           kernel_initializer=initializer, 
                           kernel_regularizer=regularizer,
                           name="bottleneck")(x)
            encoded = LeakyReLU(alpha=0.1)(encoded)
            
            # Decoder
            x = Dense(128, kernel_initializer=initializer, kernel_regularizer=regularizer)(encoded)
            x = LeakyReLU(alpha=0.1)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.25)(x)
            
            x = Dense(256, kernel_initializer=initializer, kernel_regularizer=regularizer)(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.25)(x)
            
            # Output layer
            decoded = Dense(input_dim, activation='linear', kernel_initializer=initializer)(x)
            
            # Create models
            autoencoder = Model(inputs=input_layer, outputs=decoded)
            encoder = Model(inputs=input_layer, outputs=encoded)
            
            # Use adaptive learning rate
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            
            # Compile model with additional metrics for monitoring
            autoencoder.compile(
                optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mean_squared_error', 'mean_absolute_error']
            )
            
            # Model summary
            logger.info("Autoencoder model summary:")
            autoencoder.summary(print_fn=logger.info)
            
            return autoencoder, encoder
            
        except Exception as e:
            logger.error(f"Error building autoencoder: {e}")
            logger.error(traceback.format_exc())
            return None, None
    
    def train_autoencoder(self, X_train, X_val, epochs=150):  # Increased epochs for better training
        """
        Train the autoencoder model with proper callbacks and GPU optimization
        
        Args:
            X_train: Training data
            X_val: Validation data
            epochs: Maximum number of training epochs
            
        Returns:
            history: Training history
        """
        logger.info("Training autoencoder model...")
        
        if self.autoencoder is None:
            logger.error("Autoencoder not built, cannot train")
            return None
        
        try:
            # Configure callbacks
            callbacks = [
                # Early stopping to prevent overfitting, longer patience for better convergence
                EarlyStopping(
                    monitor='val_loss',
                    patience=20,  # Increased patience
                    restore_best_weights=True,
                    verbose=1
                ),
                # Model checkpoint to save best model
                ModelCheckpoint(
                    filepath=os.path.join(self.model_path, 'best_autoencoder.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                ),
                # Reduce learning rate when plateau is reached
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,  # Increased patience
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # Add TensorFlow progress callback if available
            if progress_tracking_available:
                tf_callback = TensorFlowProgressCallback(
                    total_epochs=epochs,
                    model_name="Autoencoder"
                )
                callbacks.append(tf_callback)
            
            # Train model with validation
            history = self.autoencoder.fit(
                X_train, X_train,  # Autoencoder trains on input=output
                epochs=epochs,
                batch_size=self.batch_size,
                shuffle=True,
                validation_data=(X_val, X_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Verify training quality
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            
            logger.info(f"Autoencoder training completed - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Check if model is learning properly
            if val_loss > 0.9 * X_train.shape[1]:
                logger.warning("Autoencoder validation loss is very high. Model may not be learning properly.")
            
            # Evaluate reconstruction quality on samples
            sample_size = min(1000, X_val.shape[0])
            X_sample = X_val[:sample_size]
            X_reconstructed = self.autoencoder.predict(X_sample)
            mse = np.mean(np.square(X_sample - X_reconstructed))
            logger.info(f"Sample reconstruction MSE: {mse:.6f}")
            
            # Check encoder output distribution
            encoded_sample = self.encoder.predict(X_sample)
            mean_encoded = np.mean(encoded_sample)
            std_encoded = np.std(encoded_sample)
            min_encoded = np.min(encoded_sample)
            max_encoded = np.max(encoded_sample)
            
            logger.info(f"Encoded features distribution: mean={mean_encoded:.4f}, std={std_encoded:.4f}, min={min_encoded:.4f}, max={max_encoded:.4f}")
            
            return history
            
        except Exception as e:
            logger.error(f"Error during autoencoder training: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def train_classifier(self, X_train, y_train, X_val, y_val, use_xgboost=True):
        """
        Train either XGBoost or LightGBM classifier on encoded features
        
        Args:
            X_train: Encoded training features
            y_train: Training labels
            X_val: Encoded validation features
            y_val: Validation labels
            use_xgboost: Whether to use XGBoost (True) or LightGBM (False)
            
        Returns:
            classifier: Trained classifier model
        """
        logger.info(f"Training {'XGBoost' if use_xgboost else 'LightGBM'} classifier...")
        
        try:
            # Get number of classes
            num_classes = len(np.unique(y_train))
            logger.info(f"Training classifier for {num_classes} classes")
            
            # Calculate class weights for imbalanced dataset
            class_counts = np.bincount(y_train)
            total_samples = len(y_train)
            class_weights = {
                i: total_samples / (len(class_counts) * count) if count > 0 else 1.0
                for i, count in enumerate(class_counts)
            }
            
            if use_xgboost:
                # XGBoost parameters optimized for better learning
                params = {
                    'objective': 'multi:softprob',
                    'num_class': num_classes,
                    'eta': 0.03,  # Slower learning rate for better generalization
                    'max_depth': 6,  # Reduced depth to prevent overfitting
                    'min_child_weight': 2,  # Increased to reduce overfitting
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'colsample_bylevel': 0.8,  # Added feature sampling at each level
                    'gamma': 0.2,  # Increased regularization
                    'eval_metric': ['mlogloss', 'merror'],
                    'scale_pos_weight': 1,
                    'verbosity': 1,
                    'seed': 42  # Fixed seed for reproducibility
                }
                
                # Set tree method based on GPU availability
                if len(tf.config.list_physical_devices('GPU')) > 0:
                    try:
                        # Test GPU compatibility
                        test_params = params.copy()
                        test_params['tree_method'] = 'gpu_hist'
                        test_params['gpu_id'] = 0
                        
                        # Try with small subset first
                        test_size = min(1000, X_train.shape[0])
                        test_dmat = xgb.DMatrix(X_train[:test_size], label=y_train[:test_size])
                        test_model = xgb.train(test_params, test_dmat, num_boost_round=5)
                        
                        # If we get here, GPU works
                        params['tree_method'] = 'gpu_hist'
                        params['gpu_id'] = 0
                        logger.info("XGBoost will use GPU acceleration")
                    except Exception as e:
                        logger.warning(f"XGBoost GPU test failed, falling back to CPU: {e}")
                        params['tree_method'] = 'hist'
                else:
                    params['tree_method'] = 'hist'
                
                # Create DMatrix for training with sample weights
                sample_weights = np.array([class_weights.get(label, 1.0) for label in y_train])
                dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
                dval = xgb.DMatrix(X_val, label=y_val)
                
                # Create watchlist
                watchlist = [(dtrain, 'train'), (dval, 'eval')]
                
                # Train XGBoost model with more rounds
                self.classifier = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=1000,  # Increased number of rounds
                    early_stopping_rounds=50,  # More rounds before stopping
                    evals=watchlist,
                    verbose_eval=20
                )
                
                # Save model
                model_path = os.path.join(self.model_path, 'xgboost_classifier.model')
                self.classifier.save_model(model_path)
                logger.info(f"XGBoost model saved to {model_path}")
                
            else:
                # LightGBM parameters (more memory efficient than XGBoost)
                params = {
                    'objective': 'multiclass',
                    'num_class': num_classes,
                    'learning_rate': 0.03,  # Slower learning rate
                    'max_depth': 6,
                    'num_leaves': 40,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'min_child_samples': 20,  # Prevent overfitting
                    'reg_alpha': 0.1,  # L1 regularization
                    'reg_lambda': 0.1,  # L2 regularization
                    'verbose': -1,
                    'random_state': 42  # Fixed seed for reproducibility
                }
                
                # Create dataset with weights
                weight = np.array([class_weights.get(label, 1.0) for label in y_train])
                lgb_train = lgb.Dataset(X_train, y_train, weight=weight)
                lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
                
                # Train LightGBM model with more rounds
                self.classifier = lgb.train(
                    params,
                    lgb_train,
                    num_boost_round=1000,  # Increased number of rounds
                    valid_sets=[lgb_train, lgb_eval],
                    callbacks=[
                        lgb.early_stopping(50, verbose=True),  # More rounds before stopping
                        lgb.log_evaluation(20)
                    ]
                )
                
                # Save model
                model_path = os.path.join(self.model_path, 'lightgbm_classifier.txt')
                self.classifier.save_model(model_path)
                logger.info(f"LightGBM model saved to {model_path}")
            
            # Calculate final training and validation accuracy
            if use_xgboost:
                y_train_pred = np.argmax(self.classifier.predict(dtrain), axis=1)
                train_acc = np.mean(y_train_pred == y_train)
                
                y_val_pred = np.argmax(self.classifier.predict(dval), axis=1)
                val_acc = np.mean(y_val_pred == y_val)
                
                # Check predictions diversity - IMPORTANT FIX FOR CRITICAL ISSUE
                y_val_proba = self.classifier.predict(dval)
                self._verify_prediction_diversity(y_val_proba)
                
            else:
                y_train_pred = np.argmax(self.classifier.predict(X_train), axis=1)
                train_acc = np.mean(y_train_pred == y_train)
                
                y_val_pred = np.argmax(self.classifier.predict(X_val), axis=1)
                val_acc = np.mean(y_val_pred == y_val)
                
                # Check predictions diversity - IMPORTANT FIX FOR CRITICAL ISSUE
                y_val_proba = self.classifier.predict(X_val)
                self._verify_prediction_diversity(y_val_proba)
            
            logger.info(f"Training accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}")
            
            # Make sure we're achieving high accuracy as requested (90%+)
            if val_acc < 0.9:
                logger.warning(f"Validation accuracy {val_acc:.4f} is below 90%. Model might not meet accuracy requirements.")
                
                # Suggestions for improvement
                logger.info("Consider these improvements for higher accuracy:")
                logger.info("1. Increase training data size")
                logger.info("2. Further tune hyperparameters")
                logger.info("3. Try a more complex model architecture")
                logger.info("4. Improve feature engineering")
            else:
                logger.info(f"Achieved validation accuracy of {val_acc:.4f}, which meets the 90%+ requirement")
            
            return self.classifier
            
        except Exception as e:
            logger.error(f"Error training classifier: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _verify_prediction_diversity(self, predictions):
        """
        Verify that the model is making diverse predictions
        Critical fix for ensuring the model doesn't predict the same class for all inputs
        """
        try:
            # Check if predictions are uniform (same prediction for all samples)
            if len(predictions.shape) > 1:
                # For multi-class probabilities
                max_probs = np.max(predictions, axis=1)
                pred_classes = np.argmax(predictions, axis=1)
                
                # Check if all predictions are the same class
                unique_classes = np.unique(pred_classes)
                if len(unique_classes) == 1:
                    logger.warning(f"Model is predicting only one class: {unique_classes[0]}")
                    logger.warning("This indicates a potential training issue!")
                else:
                    logger.info(f"Model is predicting {len(unique_classes)} different classes")
                
                # Check for diversity in confidence
                mean_conf = np.mean(max_probs)
                std_conf = np.std(max_probs)
                
                logger.info(f"Prediction confidence: mean={mean_conf:.4f}, std={std_conf:.4f}")
                
                if std_conf < 0.01:
                    logger.warning("Very low diversity in prediction confidence!")
                    logger.warning("Model may be giving the same confidence for all inputs!")
                    
                # Check range of probabilities for each class
                for i in range(predictions.shape[1]):
                    class_probs = predictions[:, i]
                    class_mean = np.mean(class_probs)
                    class_min = np.min(class_probs)
                    class_max = np.max(class_probs)
                    class_std = np.std(class_probs)
                    
                    logger.info(f"Class {i} probability stats: mean={class_mean:.4f}, min={class_min:.4f}, max={class_max:.4f}, std={class_std:.4f}")
                    
                    if class_std < 0.001:
                        logger.warning(f"Class {i} has very low probability variance!")
                        
        except Exception as e:
            logger.warning(f"Error verifying prediction diversity: {e}")
    
    def train_model_batch(self, batch_size=13500000, use_xgboost=True):
        """
        Train the model using batch processing to handle large datasets
        
        Args:
            batch_size: Size of batches to process at once (default is large enough for 1.5M per attack type)
            use_xgboost: Whether to use XGBoost (True) or LightGBM (False)
            
        Returns:
            True if training successful, False otherwise
        """
        start_time = time.time()
        logger.info(f"Starting NIDS batch training with batch size: {batch_size}")
        
        # Set up overall progress tracking
        if progress_tracking_available:
            overall_progress = ProgressTracker(total_steps=6, description="NIDS Training")
        
        # Step 1: Get all available attack types
        logger.info("Step 1: Identifying available attack types...")
        attack_dirs = []
        for d in os.listdir(self.data_path):
            path = os.path.join(self.data_path, d)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, f"{d}_dataset.csv")):
                attack_dirs.append(d)
        
        if not attack_dirs:
            logger.error("No valid attack datasets found")
            return False
            
        logger.info(f"Found {len(attack_dirs)} attack types: {attack_dirs}")
        
        if progress_tracking_available:
            overall_progress.update("Attack Type Identification")
        
        # Step 2: Load a subset of data for initial model training
        logger.info("Step 2: Loading initial subset for autoencoder training...")
        
        # Calculate samples per attack for initial subset
        # Increased initial subset size for better model training
        samples_per_attack = min(150000, batch_size // len(attack_dirs))
        logger.info(f"Using {samples_per_attack} samples per attack type for initial autoencoder training")
        
        initial_data = []
        for attack_type in attack_dirs:
            # Load subset of this attack
            attack_data = self.load_kitsune_dataset(
                limit_samples=samples_per_attack,
                specific_attack=attack_type
            )
            
            if attack_data is not None and not attack_data.empty:
                initial_data.append(attack_data)
                log_memory_usage(f"After loading {attack_type}")
        
        if not initial_data:
            logger.error("Failed to load any data for initial training")
            return False
        
        # Combine initial data
        try:
            subset_data = pd.concat(initial_data, ignore_index=True)
            logger.info(f"Initial subset: {subset_data.shape[0]} rows, {subset_data.shape[1]} columns")
            
            # Free memory
            del initial_data
            gc.collect()
            log_memory_usage("After combining initial data")
            
        except Exception as e:
            logger.error(f"Error combining initial data: {e}")
            return False
        
        # Step 3: Preprocess initial subset and build autoencoder
        logger.info("Step 3: Preprocessing initial subset...")
        # Note the for_encoder=True flag to establish the reference feature set
        X_initial, y_initial, feature_names = self.preprocess_data(subset_data, for_encoder=True)
        
        if X_initial is None or y_initial is None:
            logger.error("Initial data preprocessing failed")
            return False
        
        # Free memory
        del subset_data
        gc.collect()
        log_memory_usage("After preprocessing")
        
        if progress_tracking_available:
            overall_progress.update("Initial Data Loading")
        
        # Split data for autoencoder training
        logger.info("Step 4: Training autoencoder on initial subset...")
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_initial, y_initial, test_size=0.2, random_state=42, stratify=y_initial
            )
            
            logger.info(f"Autoencoder training set: {X_train.shape[0]} samples")
            logger.info(f"Autoencoder validation set: {X_val.shape[0]} samples")
            
            # Build and train autoencoder
            self.autoencoder, self.encoder = self.build_autoencoder(X_train.shape[1])
            
            if self.autoencoder is None or self.encoder is None:
                logger.error("Failed to build autoencoder")
                return False
            
            autoencoder_history = self.train_autoencoder(X_train, X_val, epochs=150)  # Increased epochs
            
            if autoencoder_history is None:
                logger.error("Autoencoder training failed")
                return False
            
            # Validate encoder output patterns
            encoded_train = self.encoder.predict(X_train[:1000])
            encoded_val = self.encoder.predict(X_val[:1000])
            
            # Check for consistent patterns
            train_mean = np.mean(encoded_train, axis=0)
            val_mean = np.mean(encoded_val, axis=0)
            train_std = np.std(encoded_train, axis=0)
            val_std = np.std(encoded_val, axis=0)
            
            logger.info(f"Encoded train mean std: {np.std(train_mean):.4f}")
            logger.info(f"Encoded val mean std: {np.std(val_mean):.4f}")
            logger.info(f"Mean train std: {np.mean(train_std):.4f}")
            logger.info(f"Mean val std: {np.mean(val_std):.4f}")
            
            # Check if encoder is working properly
            if np.mean(train_std) < 0.01 or np.mean(val_std) < 0.01:
                logger.warning("Encoder may not be learning useful representations!")
                
            # Free memory
            del X_initial, y_initial, X_train, X_val
            gc.collect()
            log_memory_usage("After autoencoder training")
            
        except Exception as e:
            logger.error(f"Error during autoencoder training phase: {e}")
            return False
        
        if progress_tracking_available:
            overall_progress.update("Autoencoder Training")
        
        # Step 5: Process each attack type and collect encoded features
        logger.info("Step 5: Processing all attack types and collecting encoded features...")
        encoded_features = []
        encoded_labels = []
        
        # Calculate samples per attack type for classifier training
        # Increased samples for better training
        samples_per_attack_for_classifier = 2000000  # 2 million samples per attack type
        logger.info(f"Using up to {samples_per_attack_for_classifier} samples per attack type for classifier training")
        
        for attack_type in attack_dirs:
            logger.info(f"Processing attack type: {attack_type} for classifier training")
            
            # Load a batch of this attack type
            attack_data = self.load_kitsune_dataset(
                limit_samples=samples_per_attack_for_classifier,
                specific_attack=attack_type
            )
            
            if attack_data is None or attack_data.empty:
                logger.warning(f"No data loaded for {attack_type}, skipping...")
                continue
            
            # Log actual sample count obtained
            logger.info(f"Loaded {len(attack_data)} samples for {attack_type}")
                
            # Preprocess this batch - Note the for_encoder=False to use existing feature list
            logger.info(f"Preprocessing {attack_type} batch...")
            X_attack, y_attack, _ = self.preprocess_data(attack_data, for_encoder=False)
            
            if X_attack is None or y_attack is None:
                logger.warning(f"Preprocessing failed for {attack_type}, skipping...")
                continue
            
            # Free memory
            del attack_data
            gc.collect()
            
            # Encode features
            logger.info(f"Encoding {attack_type} features...")
            try:
                # Log the shape to debug any issues
                logger.info(f"Input shape before encoding: {X_attack.shape}")
                # Make sure it matches what the encoder expects
                if X_attack.shape[1] != len(self.feature_columns):
                    logger.error(f"Shape mismatch: encoder expects {len(self.feature_columns)} features, got {X_attack.shape[1]}")
                    continue
                    
                X_encoded = self.encoder.predict(X_attack)
                logger.info(f"Encoded features shape: {X_encoded.shape}")
                
                # Verify encoded features have variance
                enc_mean = np.mean(X_encoded)
                enc_std = np.std(X_encoded)
                logger.info(f"Encoded features stats: mean={enc_mean:.4f}, std={enc_std:.4f}")
                
                # Add to collection
                encoded_features.append(X_encoded)
                encoded_labels.append(y_attack)
                logger.info(f"Added {len(y_attack)} samples of {attack_type} to classifier dataset")
            except Exception as e:
                logger.error(f"Error encoding features for {attack_type}: {e}")
                continue
            
            # Free memory
            del X_attack, y_attack
            gc.collect()
            log_memory_usage(f"After processing {attack_type}")
        
        if not encoded_features:
            logger.error("Failed to collect any encoded features for classifier training")
            return False
        
        # Step 6: Train classifier on encoded features
        logger.info("Step 6: Training classifier on encoded features...")
        try:
            # Combine all encoded features and labels
            X_all = np.vstack(encoded_features)
            y_all = np.concatenate(encoded_labels)
            
            logger.info(f"Combined encoded features: {X_all.shape[0]} samples, {X_all.shape[1]} features")
            
            # Free original arrays
            del encoded_features, encoded_labels
            gc.collect()
            log_memory_usage("After combining encoded features")
            
            # Split data for classifier training
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            logger.info(f"Classifier training set: {X_train.shape[0]} samples")
            logger.info(f"Classifier validation set: {X_val.shape[0]} samples")
            logger.info(f"Test set: {X_test.shape[0]} samples")
            
            # Free combined data to save memory
            del X_all, y_all
            gc.collect()
            
            # Train classifier
            classifier = self.train_classifier(
                X_train, y_train,
                X_val, y_val,
                use_xgboost=use_xgboost
            )
            
            if classifier is None:
                logger.error("Classifier training failed")
                return False
            
        except Exception as e:
            logger.error(f"Error during classifier training phase: {e}")
            return False
        
        if progress_tracking_available:
            overall_progress.update("Classifier Training")
        
        # Step 7: Evaluate on test set
        logger.info("Step 7: Evaluating model on test set...")
        try:
            if use_xgboost:
                dtest = xgb.DMatrix(X_test)
                y_pred_proba = self.classifier.predict(dtest)
            else:
                y_pred_proba = self.classifier.predict(X_test)
                
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Overall accuracy
            accuracy = np.mean(y_pred == y_test)
            logger.info(f"Test accuracy: {accuracy:.4f}")
            
            # Log confidence scores
            conf_scores = np.max(y_pred_proba, axis=1)
            mean_conf = np.mean(conf_scores)
            std_conf = np.std(conf_scores)
            min_conf = np.min(conf_scores)
            max_conf = np.max(conf_scores)
            
            logger.info(f"Confidence scores: mean={mean_conf:.4f}, std={std_conf:.4f}, min={min_conf:.4f}, max={max_conf:.4f}")
            
            # Detailed metrics
            from sklearn.metrics import classification_report, confusion_matrix
            
            # Get class names
            class_names = self.label_encoder.classes_
            
            # Classification report
            report = classification_report(y_test, y_pred, target_names=class_names)
            logger.info(f"Classification report:\n{report}")
            
            # Save report to file
            with open(os.path.join(self.model_path, 'classification_report.txt'), 'w') as f:
                f.write(f"Test accuracy: {accuracy:.4f}\n\n")
                f.write(report)
            
            # Save confusion matrix visualization
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_path, 'confusion_matrix.png'))
            
            # Feature importance for XGBoost
            if use_xgboost:
                try:
                    importance_scores = self.classifier.get_score(importance_type='gain')
                    importance_df = pd.DataFrame({
                        'Feature': list(importance_scores.keys()),
                        'Importance': list(importance_scores.values())
                    })
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    
                    # Plot feature importance
                    plt.figure(figsize=(12, 6))
                    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
                    plt.title('Top 20 Feature Importance')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.model_path, 'feature_importance.png'))
                except Exception as e:
                    logger.warning(f"Error generating feature importance plot: {e}")
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
        
        if progress_tracking_available:
            overall_progress.update("Model Evaluation")
        
        # Step 8: Save all models and metadata
        logger.info("Step 8: Saving models and metadata...")
        self.save_models()
        
        # Log total training time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
        return True
    
    def train_model(self, limit_samples=None, use_xgboost=True):
        """
        Complete training pipeline for the NIDS model
        
        Args:
            limit_samples: Optional limit on samples per attack (for testing)
            use_xgboost: Whether to use XGBoost (True) or LightGBM (False)
            
        Returns:
            True if training successful, False otherwise
        """
        # If limit_samples is set, use the original training approach
        if limit_samples is not None:
            logger.info(f"Training with sample limit: {limit_samples}")
            return self._train_model_original(limit_samples, use_xgboost)
        else:
            # Otherwise, use the memory-optimized batch approach
            logger.info("Training with batch processing (memory-optimized)")
            return self.train_model_batch(batch_size=13500000, use_xgboost=use_xgboost)
    
    def _train_model_original(self, limit_samples=None, use_xgboost=True):
        """
        Original training pipeline that loads all data at once
        
        Args:
            limit_samples: Optional limit on samples per attack (for testing)
            use_xgboost: Whether to use XGBoost (True) or LightGBM (False)
            
        Returns:
            True if training successful, False otherwise
        """
        start_time = time.time()
        logger.info("Starting Kitsune NIDS model training (original method)...")
        
        # Set up overall progress tracking
        if progress_tracking_available:
            overall_progress = ProgressTracker(total_steps=5, description="NIDS Training")
        
        # 1. Load and preprocess dataset
        logger.info("Step 1: Loading Kitsune dataset...")
        combined_data = self.load_kitsune_dataset(limit_samples)
        
        if combined_data is None:
            logger.error("Failed to load dataset")
            return False
        
        if progress_tracking_available:
            overall_progress.update("Dataset Loading")
        
        # 2. Preprocess data
        logger.info("Step 2: Preprocessing data...")
        # Note the for_encoder=True flag to establish the reference feature set
        X, y, feature_names = self.preprocess_data(combined_data, for_encoder=True)
        
        if X is None or y is None:
            logger.error("Data preprocessing failed")
            return False
        
        if progress_tracking_available:
            overall_progress.update("Data Preprocessing")
        
        # 3. Split data
        logger.info("Step 3: Splitting data...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            logger.info(f"Training set: {X_train.shape[0]} samples")
            logger.info(f"Validation set: {X_val.shape[0]} samples")
            logger.info(f"Test set: {X_test.shape[0]} samples")
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return False
        
        if progress_tracking_available:
            overall_progress.update("Data Splitting")
        
        # 4. Build and train autoencoder
        logger.info("Step 4: Building and training autoencoder...")
        self.autoencoder, self.encoder = self.build_autoencoder(X_train.shape[1])
        
        if self.autoencoder is None or self.encoder is None:
            logger.error("Failed to build autoencoder")
            return False
        
        autoencoder_history = self.train_autoencoder(X_train, X_val, epochs=150)  # Increased epochs
        
        if autoencoder_history is None:
            logger.error("Autoencoder training failed")
            return False
        
        if progress_tracking_available:
            overall_progress.update("Autoencoder Training")
        
        # 5. Generate encoded features and train classifier
        logger.info("Step 5: Training classifier on encoded features...")
        try:
            # Generate encoded features
            X_train_encoded = self.encoder.predict(X_train)
            X_val_encoded = self.encoder.predict(X_val)
            X_test_encoded = self.encoder.predict(X_test)
            
            logger.info(f"Encoded feature dimension: {X_train_encoded.shape[1]}")
            
            # Train classifier
            classifier = self.train_classifier(
                X_train_encoded, y_train, 
                X_val_encoded, y_val,
                use_xgboost=use_xgboost
            )
            
            if classifier is None:
                logger.error("Classifier training failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in classifier training phase: {e}")
            return False
        
        if progress_tracking_available:
            overall_progress.update("Classifier Training")
        
        # 6. Evaluate model on test set
        logger.info("Step 6: Evaluating model...")
        try:
            if use_xgboost:
                dtest = xgb.DMatrix(X_test_encoded)
                y_pred_proba = self.classifier.predict(dtest)
            else:
                y_pred_proba = self.classifier.predict(X_test_encoded)
                
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Overall accuracy
            accuracy = np.mean(y_pred == y_test)
            logger.info(f"Test accuracy: {accuracy:.4f}")
            
            # Check for prediction diversity - IMPORTANT FIX FOR CRITICAL ISSUE
            self._verify_prediction_diversity(y_pred_proba)
            
            # Detailed metrics
            from sklearn.metrics import classification_report, confusion_matrix
            
            # Classification report
            class_names = self.label_encoder.classes_
            report = classification_report(y_test, y_pred, target_names=class_names)
            logger.info(f"Classification report:\n{report}")
            
            # Save report to file
            with open(os.path.join(self.model_path, 'classification_report.txt'), 'w') as f:
                f.write(f"Test accuracy: {accuracy:.4f}\n\n")
                f.write(report)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            # Save confusion matrix visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_path, 'confusion_matrix.png'))
            
            # Feature importance for XGBoost
            if use_xgboost:
                importance_scores = self.classifier.get_score(importance_type='gain')
                importance_df = pd.DataFrame({
                    'Feature': list(importance_scores.keys()),
                    'Importance': list(importance_scores.values())
                })
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                # Plot feature importance
                plt.figure(figsize=(12, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
                plt.title('Top 20 Feature Importance')
                plt.tight_layout()
                plt.savefig(os.path.join(self.model_path, 'feature_importance.png'))
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
        
        # 7. Save all models and metadata
        logger.info("Step 7: Saving models and metadata...")
        self.save_models()
        
        # Log total training time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
        return True
    
    def save_models(self):
        """Save all trained models and metadata"""
        logger.info("Saving models and metadata...")
        
        try:
            # Create version directory
            version_dir = os.path.join(self.model_path, f"version_{self.version}")
            os.makedirs(version_dir, exist_ok=True)
            
            # Save autoencoder and encoder
            if self.autoencoder is not None:
                self.autoencoder.save(os.path.join(version_dir, 'autoencoder_model.h5'))
                
            if self.encoder is not None:
                self.encoder.save(os.path.join(version_dir, 'encoder_model.h5'))
            
            # Save classifier
            if self.classifier is not None:
                if isinstance(self.classifier, xgb.Booster):
                    self.classifier.save_model(os.path.join(version_dir, 'xgboost_classifier.model'))
                else:
                    self.classifier.save_model(os.path.join(version_dir, 'lightgbm_classifier.txt'))
            
            # Save scaler and label encoder
            joblib.dump(self.scaler, os.path.join(version_dir, 'scaler.pkl'))
            joblib.dump(self.label_encoder, os.path.join(version_dir, 'label_encoder.pkl'))
            
            # Save feature columns
            with open(os.path.join(version_dir, 'feature_columns.pkl'), 'wb') as f:
                pickle.dump(self.feature_columns, f)
            
            # Save attack mapping
            with open(os.path.join(version_dir, 'attack_mapping.pkl'), 'wb') as f:
                pickle.dump(self.attack_mapping, f)
                
            # Save normal traffic stats
            with open(os.path.join(version_dir, 'normal_traffic_stats.pkl'), 'wb') as f:
                pickle.dump(self.normal_traffic_stats, f)
            
            # Save version info
            with open(os.path.join(version_dir, 'version_info.txt'), 'w') as f:
                f.write(f"Version: {self.version}\n")
                f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Feature dimension: {len(self.feature_columns)}\n")
                f.write(f"Latent dimension: {self.latent_dim}\n")
                f.write(f"Classes: {', '.join(self.label_encoder.classes_)}\n")
            
            # Copy to main model directory for easy access
            for filename in os.listdir(version_dir):
                source = os.path.join(version_dir, filename)
                destination = os.path.join(self.model_path, filename)
                shutil.copy2(source, destination)
                
            logger.info(f"Models and metadata saved to {version_dir} and {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self):
        """Load trained models for inference"""
        logger.info("Loading trained models...")
        
        try:
            # Find latest version directory
            version_dirs = sorted(
                [d for d in os.listdir(self.model_path) if d.startswith('version_')],
                reverse=True
            )
            
            if version_dirs:
                model_dir = os.path.join(self.model_path, version_dirs[0])
                logger.info(f"Loading from version directory: {model_dir}")
            else:
                model_dir = self.model_path
                logger.info(f"Loading from main model directory: {model_dir}")
            
            # Load encoder
            encoder_path = os.path.join(model_dir, 'encoder_model.h5')
            if os.path.exists(encoder_path):
                self.encoder = load_model(encoder_path)
                logger.info(f"Loaded encoder from {encoder_path}")
            else:
                logger.error(f"Encoder model not found at {encoder_path}")
                return False
            
            # Load classifier
            xgb_path = os.path.join(model_dir, 'xgboost_classifier.model')
            lgb_path = os.path.join(model_dir, 'lightgbm_classifier.txt')
            
            if os.path.exists(xgb_path):
                self.classifier = xgb.Booster()
                self.classifier.load_model(xgb_path)
                logger.info(f"Loaded XGBoost classifier from {xgb_path}")
            elif os.path.exists(lgb_path):
                self.classifier = lgb.Booster(model_file=lgb_path)
                logger.info(f"Loaded LightGBM classifier from {lgb_path}")
            else:
                logger.error("No classifier model found")
                return False
            
            # Load scaler and label encoder
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded scaler from {scaler_path}")
            else:
                logger.error(f"Scaler not found at {scaler_path}")
                return False
            
            label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
            if os.path.exists(label_encoder_path):
                self.label_encoder = joblib.load(label_encoder_path)
                logger.info(f"Loaded label encoder from {label_encoder_path}")
            else:
                logger.error(f"Label encoder not found at {label_encoder_path}")
                return False
            
            # Load feature columns
            feature_columns_path = os.path.join(model_dir, 'feature_columns.pkl')
            if os.path.exists(feature_columns_path):
                with open(feature_columns_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)
                logger.info(f"Loaded feature columns ({len(self.feature_columns)} features)")
            else:
                logger.error(f"Feature columns not found at {feature_columns_path}")
                return False
            
            # Load attack mapping
            attack_mapping_path = os.path.join(model_dir, 'attack_mapping.pkl')
            if os.path.exists(attack_mapping_path):
                with open(attack_mapping_path, 'rb') as f:
                    self.attack_mapping = pickle.load(f)
                logger.info(f"Loaded attack mapping")
            else:
                logger.warning(f"Attack mapping not found at {attack_mapping_path}")
                # Use default attack mapping
                
            # Load normal traffic stats if available
            normal_stats_path = os.path.join(model_dir, 'normal_traffic_stats.pkl')
            if os.path.exists(normal_stats_path):
                with open(normal_stats_path, 'rb') as f:
                    self.normal_traffic_stats = pickle.load(f)
                logger.info(f"Loaded normal traffic statistics")
            else:
                logger.warning(f"Normal traffic stats not found at {normal_stats_path}")
                # Use default stats
            
            logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.error(traceback.format_exc())
            return False
            
    def extract_features_from_pcap(self, pcap_file):
        """
        Extract features from a pcap file for prediction
        Enhanced to ensure all required fields are available
        
        Args:
            pcap_file: Path to pcap file
            
        Returns:
            features: DataFrame with extracted features
        """
        logger.info(f"Extracting features from pcap: {pcap_file}")
        
        try:
            from scapy.all import rdpcap
            import numpy as np
            import pandas as pd
            import traceback
            
            # Verify file exists
            if not os.path.exists(pcap_file):
                logger.error(f"PCAP file not found: {pcap_file}")
                return None
            
            # Read pcap file
            packets = rdpcap(pcap_file)
            logger.info(f"Loaded {len(packets)} packets from {pcap_file}")
            
            if len(packets) == 0:
                logger.warning("No packets found in pcap file")
                return pd.DataFrame()
            
            # Extract features
            features = []
            
            # Group packets by flow - ensuring all components are native Python types
            flows = {}
            for i, pkt in enumerate(packets):
                try:
                    if 'IP' in pkt:
                        # Extract flow key (ensuring all values are native Python types, not numpy types)
                        src_ip = str(pkt['IP'].src)
                        dst_ip = str(pkt['IP'].dst)
                        protocol = int(pkt['IP'].proto) if isinstance(pkt['IP'].proto, np.integer) else pkt['IP'].proto
                        
                        # Extract ports if available (ensuring they're native Python ints)
                        src_port = 0
                        dst_port = 0
                        
                        if 'TCP' in pkt:
                            src_port = int(pkt['TCP'].sport) if isinstance(pkt['TCP'].sport, np.integer) else pkt['TCP'].sport
                            dst_port = int(pkt['TCP'].dport) if isinstance(pkt['TCP'].dport, np.integer) else pkt['TCP'].dport
                        elif 'UDP' in pkt:
                            src_port = int(pkt['UDP'].sport) if isinstance(pkt['UDP'].sport, np.integer) else pkt['UDP'].sport
                            dst_port = int(pkt['UDP'].dport) if isinstance(pkt['UDP'].dport, np.integer) else pkt['UDP'].dport
                        
                        # Create flow key as a string to avoid any type issues with tuples
                        flow_key = f"{src_ip}:{dst_ip}:{src_port}:{dst_port}:{protocol}"
                        
                        # Add packet to flow
                        if flow_key not in flows:
                            flows[flow_key] = []
                        
                        flows[flow_key].append(pkt)
                except Exception as e:
                    logger.warning(f"Error processing packet {i}: {e}")
            
            logger.info(f"Identified {len(flows)} flows in the pcap file")
            
            # Process each flow to extract features
            for flow_key, flow_packets in flows.items():
                try:
                    # Parse the flow key back into components
                    src_ip, dst_ip, src_port, dst_port, protocol = flow_key.split(':')
                    src_port = int(src_port)
                    dst_port = int(dst_port)
                    protocol = int(protocol)
                    
                    # Basic stats
                    pkt_count = len(flow_packets)
                    
                    # Safely convert time values to native Python floats
                    packet_times = []
                    for p in flow_packets:
                        if hasattr(p, 'time'):
                            # Convert numpy.float64 to native Python float
                            if isinstance(p.time, np.floating):
                                packet_times.append(float(p.time))
                            else:
                                packet_times.append(p.time)
                    
                    if len(packet_times) > 1:
                        flow_duration = packet_times[-1] - packet_times[0]
                    else:
                        flow_duration = 0.0
                    
                    # Safe conversion for total size
                    total_size = 0
                    for p in flow_packets:
                        if hasattr(p, 'len'):
                            # Convert numpy.int to native Python int if necessary
                            if isinstance(p.len, np.integer):
                                total_size += int(p.len)
                            else:
                                total_size += p.len
                        else:
                            total_size += len(p)
                    
                    # More stats with safe conversions
                    avg_pkt_size = float(total_size) / pkt_count if pkt_count > 0 else 0.0
                    
                    # Inter-arrival time stats
                    if len(packet_times) > 1:
                        iats = [packet_times[i+1] - packet_times[i] for i in range(len(packet_times)-1)]
                        # Use Python's built-in functions instead of numpy for safety
                        mean_iat = sum(iats) / len(iats) if iats else 0.0
                        
                        # Manual variance and std_dev calculation to avoid numpy
                        if len(iats) > 1:
                            variance = sum((x - mean_iat) ** 2 for x in iats) / len(iats)
                            std_iat = variance ** 0.5  # sqrt
                        else:
                            std_iat = 0.0
                        
                        min_iat = min(iats) if iats else 0.0
                        max_iat = max(iats) if iats else 0.0
                    else:
                        mean_iat = std_iat = min_iat = max_iat = 0.0
                    
                    # TCP flags if applicable
                    syn_count = fin_count = rst_count = psh_count = ack_count = urg_count = 0
                    
                    if protocol == 6:  # TCP
                        for p in flow_packets:
                            if 'TCP' in p:
                                # Convert numpy.int to native Python int if necessary
                                flags = int(p['TCP'].flags) if isinstance(p['TCP'].flags, np.integer) else p['TCP'].flags
                                if flags & 0x02: syn_count += 1
                                if flags & 0x01: fin_count += 1
                                if flags & 0x04: rst_count += 1
                                if flags & 0x08: psh_count += 1
                                if flags & 0x10: ack_count += 1
                                if flags & 0x20: urg_count += 1
                
                    # Calculate additional derived features
                    pkt_rate = pkt_count / flow_duration if flow_duration > 0 else 0
                    byte_rate = total_size / flow_duration if flow_duration > 0 else 0
                    
                    # Calculate rates
                    syn_rate = syn_count / pkt_count if pkt_count > 0 else 0
                    fin_rate = fin_count / pkt_count if pkt_count > 0 else 0
                    rst_rate = rst_count / pkt_count if pkt_count > 0 else 0
                    
                    # Create feature dictionary with native Python types
                    # Make sure to include the 'proto' field specifically for attack signature detection
                    flow_features = {
                        'protocol': int(protocol),  # Standard name
                        'proto': int(protocol),     # Include both names to be safe
                        'pkt_count': int(pkt_count),
                        'flow_duration': float(flow_duration),
                        'total_size': int(total_size),
                        'avg_pkt_size': float(avg_pkt_size),
                        'mean_iat': float(mean_iat),
                        'std_iat': float(std_iat),
                        'min_iat': float(min_iat),
                        'max_iat': float(max_iat),
                        'src_port': int(src_port),
                        'dst_port': int(dst_port),
                        'syn_count': int(syn_count),
                        'fin_count': int(fin_count),
                        'rst_count': int(rst_count),
                        'psh_count': int(psh_count),
                        'ack_count': int(ack_count),
                        'urg_count': int(urg_count),
                        'pkt_rate': float(pkt_rate),
                        'byte_rate': float(byte_rate),
                        'syn_rate': float(syn_rate),
                        'fin_rate': float(fin_rate),
                        'rst_rate': float(rst_rate)
                    }
                    
                    features.append(flow_features)
                    
                except Exception as e:
                    logger.warning(f"Error extracting features for flow {flow_key}: {e}")
                    logger.warning(traceback.format_exc())
            
            # Create DataFrame
            if not features:
                logger.warning("No features extracted from flows")
                # Return an empty DataFrame with the expected columns to avoid errors
                return pd.DataFrame(columns=['protocol', 'proto', 'pkt_count', 'flow_duration', 
                                         'total_size', 'avg_pkt_size', 'mean_iat', 'std_iat',
                                         'min_iat', 'max_iat', 'src_port', 'dst_port',
                                         'syn_count', 'fin_count', 'rst_count', 'psh_count',
                                         'ack_count', 'urg_count', 'pkt_rate', 'byte_rate',
                                         'syn_rate', 'fin_rate', 'rst_rate'])
            
            features_df = pd.DataFrame(features)
            logger.info(f"Extracted features from {len(features)} flows")
            
            # Double check to make sure critical fields exist
            if 'proto' not in features_df.columns and 'protocol' in features_df.columns:
                features_df['proto'] = features_df['protocol']
                logger.info("Added 'proto' field based on 'protocol' field")
                
            # Log all available column names for debugging
            logger.info(f"Available feature columns: {features_df.columns.tolist()}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error extracting features from pcap: {e}")
            logger.error(traceback.format_exc())
            return None
            
    def predict(self, pcap_file):
        """
        Predict attack types from a pcap file with improved confidence handling and error recovery
        
        Args:
            pcap_file: Path to pcap file
            
        Returns:
            result: Dictionary with prediction results
        """
        logger.info(f"Starting prediction for pcap file: {pcap_file}")
        
        # Load models if not already loaded
        if self.encoder is None or self.classifier is None:
            if not self.load_models():
                return {'error': 'Failed to load models'}
        
        # Extract features
        features_df = self.extract_features_from_pcap(pcap_file)
        
        if features_df is None:
            return {'error': 'Failed to extract features from pcap file'}
        
        if features_df.empty:
            return {
                'status': 'Unknown',
                'details': 'No valid network flows found in the pcap file',
                'flows_analyzed': 0,
                'attack_types': []
            }
        
        try:
            # Log column names for debugging
            logger.info(f"Extracted features columns: {features_df.columns.tolist()}")
            
            # Align features with trained model
            logger.info(f"Aligning features with model (expected {len(self.feature_columns)} features)")
            
            # Add missing features using DataFrame concatenation for better performance
            missing_features = set(self.feature_columns) - set(features_df.columns)
            if missing_features:
                # Create DataFrame with missing columns
                missing_df = pd.DataFrame(0, index=features_df.index, columns=list(missing_features))
                # Concatenate with original DataFrame
                features_df = pd.concat([features_df, missing_df], axis=1)
                logger.info(f"Added {len(missing_features)} missing features")
                
            # Remove extra features
            extra_features = set(features_df.columns) - set(self.feature_columns)
            if extra_features:
                features_df = features_df.drop(columns=list(extra_features))
                logger.info(f"Removed {len(extra_features)} extra features")
            
            # Ensure features are in same order as training
            features_df = features_df[self.feature_columns]
            
            logger.info(f"Feature alignment complete: {features_df.shape}")
            
            # Handle NaN values
            features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            features_df.fillna(0, inplace=True)
            
            # Save a copy of the raw features for attack signatures before scaling
            raw_features_df = features_df.copy()
            
            # Scale features
            X_scaled = self.scaler.transform(features_df)
            
            # Convert to float32 to match training data
            X_scaled = X_scaled.astype(np.float32)
            
            # Try to run attack signature detection with error handling
            try:
                raw_predictions = self._check_attack_signatures(raw_features_df)
                logger.info(f"Attack signature detection completed, found {len(raw_predictions)} matches")
            except Exception as e:
                logger.warning(f"Error in attack signature detection, using ML prediction only: {e}")
                logger.warning(traceback.format_exc())
                raw_predictions = {}
            
            # Encode with autoencoder
            X_encoded = self.encoder.predict(X_scaled)
            
            # Verify encoded features have variance
            encoded_mean = np.mean(X_encoded)
            encoded_std = np.std(X_encoded)
            encoded_min = np.min(X_encoded)
            encoded_max = np.max(X_encoded)
            
            logger.info(f"Encoded features stats: mean={encoded_mean:.4f}, std={encoded_std:.4f}, min={encoded_min:.4f}, max={encoded_max:.4f}")
            
            # Check if encoded features are suspicious (all identical would be a red flag)
            if encoded_std < 0.001:
                logger.warning("WARNING: Encoded features have very low variance! This may indicate model issues.")
            
            # Predict with classifier
            if isinstance(self.classifier, xgb.Booster):
                dmatrix = xgb.DMatrix(X_encoded)
                # Log model info for debugging
                logger.info(f"XGBoost model type: {type(self.classifier).__name__}")
                logger.info(f"Number of trees: {self.classifier.num_boosted_rounds()}")
                # Get prediction probabilities
                y_pred_proba = self.classifier.predict(dmatrix)
                # Log the shape and a sample of predictions
                logger.info(f"Prediction shape: {y_pred_proba.shape}")
                logger.info(f"Sample prediction (first 5 rows, first 5 classes):\n{y_pred_proba[:5, :5]}")
                
            else:  # LightGBM
                # Get prediction probabilities
                y_pred_proba = self.classifier.predict(X_encoded)
                
            # Check for prediction diversity
            self._verify_prediction_diversity(y_pred_proba)
            
            # Get predicted classes
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Debug info for first 10 predictions
            for i in range(min(10, len(y_pred))):
                class_idx = y_pred[i]
                class_name = self.label_encoder.inverse_transform([class_idx])[0]
                class_prob = y_pred_proba[i, class_idx]
                logger.info(f"Sample {i}: Class {class_idx} ({class_name}) with prob {class_prob:.4f} ({class_prob*100:.2f}%)")
            
            # Calculate confidences properly for each prediction
            confidences = []
            for i, pred_idx in enumerate(y_pred):
                # Get the probability for the predicted class
                if isinstance(y_pred_proba[i], np.ndarray):
                    conf = float(y_pred_proba[i][pred_idx]) * 100  # Convert to percentage
                else:
                    # If y_pred_proba is not a 2D array, adjust accordingly
                    conf = float(y_pred_proba[i]) * 100 if isinstance(y_pred_proba[i], (np.number, float)) else 50.0
                
                # Add any additional confidence from attack signatures
                signature_conf = raw_predictions.get(i, {}).get('confidence', 0)
                if signature_conf > 0:
                    # Use the higher of the two confidence values
                    conf = max(conf, signature_conf)
                    
                confidences.append(conf)
                
            # Debug confidence statistics
            if confidences:
                logger.info(f"Confidence stats - Avg: {np.mean(confidences):.2f}%, Min: {np.min(confidences):.2f}%, Max: {np.max(confidences):.2f}%, Std: {np.std(confidences):.2f}%")
                
            # Convert to original labels
            predicted_labels = self.label_encoder.inverse_transform(y_pred)
            
            # Override predictions with high-confidence signature matches
            for i, raw_pred in raw_predictions.items():
                if i < len(predicted_labels) and raw_pred['confidence'] >= 90:
                    logger.info(f"Overriding prediction {i} with signature match: {raw_pred['type']}")
                    predicted_labels[i] = raw_pred['type']
                    confidences[i] = raw_pred['confidence']
            
            # Count by attack type with actual confidence values
            attack_counts = {}
            for label, confidence in zip(predicted_labels, confidences):
                if label not in attack_counts:
                    attack_counts[label] = {
                        'count': 0,
                        'confidence_sum': 0,
                        'all_confidences': []  # Store all confidences for debugging
                    }
                
                attack_counts[label]['count'] += 1
                attack_counts[label]['confidence_sum'] += confidence
                attack_counts[label]['all_confidences'].append(confidence)
            
            # Debug attack type stats
            for attack_type, stats in attack_counts.items():
                confidences = stats['all_confidences']
                logger.info(f"Attack type {attack_type}: Count {stats['count']}, Avg conf: {np.mean(confidences):.2f}%, Min: {np.min(confidences):.2f}%, Max: {np.max(confidences):.2f}%")
            
            # Calculate statistics
            total_flows = len(predicted_labels)
            normal_flows = attack_counts.get('Normal', {}).get('count', 0)
            normal_percentage = (normal_flows / total_flows) * 100 if total_flows > 0 else 0
            
            # Prepare attack types for result with proper confidence calculation
            attack_types = []
            for label, stats in attack_counts.items():
                avg_confidence = stats['confidence_sum'] / stats['count']
                percentage = (stats['count'] / total_flows) * 100
                    
                attack_types.append({
                    'type': label,
                    'description': self.attack_mapping.get(label, label),
                    'count': stats['count'],
                    'percentage': percentage,
                    'confidence': avg_confidence
                })
            
            # Sort by percentage
            attack_types = sorted(attack_types, key=lambda x: x['percentage'], reverse=True)
            
            # Determine overall status with 70% confidence threshold
            # This is the critical fix for showing "Uncertain Traffic Pattern" when confidence is below 70%
            if normal_percentage >= 95:
                status = "Normal Traffic"
                details = "Traffic appears normal (≥95% normal flows)"
            elif normal_percentage >= 80:
                status = "Mostly Normal Traffic"
                details = "Traffic is mostly normal with some suspicious activities"
            else:
                # Find dominant attack type
                attack_type_counts = {k: v['count'] for k, v in attack_counts.items() if k != 'Normal'}
                if attack_type_counts:
                    dominant_attack = max(attack_type_counts.items(), key=lambda x: x[1])[0]
                    attack_percentage = (attack_type_counts[dominant_attack] / total_flows) * 100
                    
                    # Get the average confidence for this attack type
                    dominant_attack_confidence = attack_counts[dominant_attack]['confidence_sum'] / attack_counts[dominant_attack]['count']
                    
                    # Only classify as an attack if confidence > 70%
                    if dominant_attack_confidence >= 70:
                        status = f"Attack Detected: {dominant_attack}"
                        details = f"{attack_percentage:.1f}% of flows classified as {dominant_attack} attack (confidence: {dominant_attack_confidence:.1f}%)"
                    else:
                        status = "Uncertain Traffic Pattern"
                        details = f"Traffic shows patterns similar to {dominant_attack} but with low confidence ({dominant_attack_confidence:.1f}%)"
                else:
                    status = "Uncertain Traffic Pattern"
                    details = "Traffic contains unusual patterns but could not be classified with confidence"
            
            # Prepare result
            result = {
                'status': status,
                'details': details,
                'flows_analyzed': total_flows,
                'normal_percentage': normal_percentage,
                'attack_types': attack_types,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Prediction completed: {status}")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            logger.error(traceback.format_exc())
            
            # Try to provide a minimal result even if there's an error
            try:
                return {
                    'status': "Error during analysis",
                    'details': f"An error occurred during analysis: {str(e)}",
                    'flows_analyzed': len(features_df) if features_df is not None else 0,
                    'error': str(e),
                    'attack_types': [],
                    'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            except:
                return {'error': f'Prediction failed: {str(e)}'}
    
    def _check_attack_signatures(self, features_df):
        """
        Check for known attack signatures in raw features
        This helps identify attacks based on network characteristics without relying on the ML model
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            dict: Dictionary mapping row indices to attack types and confidence
        """
        signatures = {}
        
        # Check for required columns first
        required_columns = ['proto', 'syn_count', 'pkt_count', 'pkt_rate', 'dst_port', 'byte_rate', 
                            'fin_rate', 'rst_rate']
        
        # Check if any of the required columns are missing
        missing_columns = [col for col in required_columns if col not in features_df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns for signature detection: {missing_columns}")
            # If missing columns are critical, modify column names if similar ones exist
            column_map = {}
            # Map potential alternative column names
            if 'proto' in missing_columns and 'protocol' in features_df.columns:
                column_map['protocol'] = 'proto'
            if 'syn_count' in missing_columns and 'SYN_count' in features_df.columns:
                column_map['SYN_count'] = 'syn_count'
            
            # Rename columns if alternatives found
            if column_map:
                features_df = features_df.rename(columns=column_map)
                logger.info(f"Renamed columns: {column_map}")
                # Update the missing columns list
                missing_columns = [col for col in required_columns if col not in features_df.columns]
        
        # If critical columns are still missing after attempting renames, use heuristic approach
        if 'proto' in missing_columns or 'syn_count' in missing_columns:
            logger.info("Using heuristic signature detection due to missing columns")
            return self._heuristic_attack_detection(features_df)
        
        # Check each flow for attack signatures
        for i, flow in features_df.iterrows():
            # Initialize with no match
            match = {'type': None, 'confidence': 0}
            
            # Check for SYN flood attack
            if flow['proto'] == 6 and flow['syn_count'] > 0:
                syn_ratio = flow['syn_count'] / flow['pkt_count'] if flow['pkt_count'] > 0 else 0
                if syn_ratio > 0.8 and flow['pkt_rate'] > 100:
                    match = {'type': 'SYN_DoS', 'confidence': min(95, 70 + syn_ratio * 20)}
            
            # Check for SSDP Flood (UDP to port 1900 with high packet rate)
            if flow['proto'] == 17 and flow['dst_port'] == 1900 and flow['pkt_rate'] > 50:
                match = {'type': 'SSDP_Flood', 'confidence': min(95, 70 + flow['pkt_rate'] / 10)}
            
            # Check for OS scan (many SYN packets to different ports)
            if flow['proto'] == 6 and flow['syn_count'] > 10 and flow['syn_rate'] > 0.7:
                match = {'type': 'OS_Scan', 'confidence': min(90, 70 + flow['syn_count'] / 5)}
            
            # Check for SSL renegotiation attack
            if (flow['proto'] == 6 and (flow['dst_port'] == 443 or flow['src_port'] == 443) and 
                flow['pkt_count'] > 50 and flow['byte_rate'] > 1000):
                match = {'type': 'SSL_Renegotiation', 'confidence': 85}
            
            # Check for normal traffic patterns
            if match['type'] is None:
                # Check if flow matches normal traffic characteristics
                if (flow['syn_rate'] < 0.1 and flow['fin_rate'] < 0.1 and 
                    flow['rst_rate'] < 0.1 and flow['pkt_rate'] < 50):
                    match = {'type': 'Normal', 'confidence': 75}
            
            # Store the match if a signature was found
            if match['type'] is not None and match['confidence'] > 0:
                signatures[i] = match
        
        # Log signature detection results
        sig_count = len(signatures)
        if sig_count > 0:
            logger.info(f"Detected {sig_count} attack signatures in raw features")
            sig_types = {}
            for match in signatures.values():
                sig_types[match['type']] = sig_types.get(match['type'], 0) + 1
            for sig_type, count in sig_types.items():
                logger.info(f"  {sig_type}: {count} flows")
        
        return signatures
    
    def _heuristic_attack_detection(self, features_df):
        """
        Fallback method for attack detection when protocol info is missing
        Uses statistical approaches instead of protocol-specific signatures
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            dict: Dictionary mapping row indices to attack types and confidence
        """
        signatures = {}
        
        # Log the available columns for debugging
        logger.info(f"Available columns for heuristic detection: {features_df.columns.tolist()}")
        
        # Key metrics to look for
        flow_metrics = {
            'avg_pkt_size': features_df['avg_pkt_size'].mean() if 'avg_pkt_size' in features_df.columns else 0,
            'pkt_count': features_df['pkt_count'].mean() if 'pkt_count' in features_df.columns else 0,
            'flow_duration': features_df['flow_duration'].mean() if 'flow_duration' in features_df.columns else 0
        }
        
        logger.info(f"Flow metrics for heuristic detection: {flow_metrics}")
        
        # Check each flow for statistical anomalies
        for i, flow in features_df.iterrows():
            # Initialize with no match
            match = {'type': None, 'confidence': 0}
            
            # Use available metrics
            if 'pkt_count' in flow and 'flow_duration' in flow and flow['flow_duration'] > 0:
                # Calculate packets per second
                pps = flow['pkt_count'] / flow['flow_duration']
                
                # High packet rates may indicate DoS
                if pps > 1000:
                    match = {'type': 'SYN_DoS', 'confidence': min(90, 60 + pps/100)}
                
                # Extremely high packet rates indicate SSDP Flood
                if pps > 5000:
                    match = {'type': 'SSDP_Flood', 'confidence': min(95, 60 + pps/500)}
            
            # Check for small packet size (often seen in scanning)
            if 'avg_pkt_size' in flow and flow['avg_pkt_size'] < 60:
                match = {'type': 'OS_Scan', 'confidence': 75}
            
            # Check source/destination ports if available
            if 'src_port' in flow and 'dst_port' in flow:
                # If destination port is 80/443 with high traffic, may be HTTP flood
                if (flow['dst_port'] == 80 or flow['dst_port'] == 443) and 'pkt_count' in flow and flow['pkt_count'] > 100:
                    match = {'type': 'SYN_DoS', 'confidence': 80}
                
                # Common ports for SSL
                if flow['dst_port'] == 443 and 'total_size' in flow and flow['total_size'] > 10000:
                    match = {'type': 'SSL_Renegotiation', 'confidence': 75}
                    
                # Check for suspicious source ports (common in some attacks)
                if flow['src_port'] < 1024 and 'pkt_count' in flow and flow['pkt_count'] > 50:
                    match = {'type': 'ARP_MitM', 'confidence': 70}
            
            # Check packet timing if available
            if 'mean_iat' in flow and flow['mean_iat'] < 0.001:  # Very small inter-arrival time
                match = {'type': 'SYN_DoS', 'confidence': 85}
                
            # Statistical approach for normal traffic
            if match['type'] is None:
                # Check if flow matches normal traffic characteristics based on available metrics
                normal_score = 0
                checks = 0
                
                if 'pkt_count' in flow:
                    normal_score += 1 if 5 <= flow['pkt_count'] <= 500 else 0
                    checks += 1
                    
                if 'flow_duration' in flow:
                    normal_score += 1 if 0.1 <= flow['flow_duration'] <= 60 else 0
                    checks += 1
                    
                if 'avg_pkt_size' in flow:
                    normal_score += 1 if 80 <= flow['avg_pkt_size'] <= 1500 else 0
                    checks += 1
                    
                if 'mean_iat' in flow:
                    normal_score += 1 if 0.01 <= flow['mean_iat'] <= 1 else 0
                    checks += 1
                    
                # If most checks pass, consider it normal traffic
                if checks > 0 and (normal_score / checks) > 0.6:
                    match = {'type': 'Normal', 'confidence': 70 + (normal_score * 5)}
            
            # Store the match if a signature was found
            if match['type'] is not None and match['confidence'] > 0:
                signatures[i] = match
        
        # Log signature detection results
        sig_count = len(signatures)
        if sig_count > 0:
            logger.info(f"Detected {sig_count} attack signatures using heuristic detection")
            sig_types = {}
            for match in signatures.values():
                sig_types[match['type']] = sig_types.get(match['type'], 0) + 1
            for sig_type, count in sig_types.items():
                logger.info(f"  {sig_type}: {count} flows")
        
        return signatures

def main():
    """Main function to train the model"""
    parser = argparse.ArgumentParser(description='Train NIDS model on Kitsune dataset')
    parser.add_argument('--data-path', type=str, default='./datasets',
                       help='Path to Kitsune dataset')
    parser.add_argument('--model-path', type=str, default='./models',
                       help='Path to save trained models')
    parser.add_argument('--limit-samples', type=int, default=None,
                       help='Limit samples per attack type (for testing)')
    parser.add_argument('--use-lightgbm', action='store_true',
                       help='Use LightGBM instead of XGBoost (more memory efficient)')
    parser.add_argument('--chunk-size', type=int, default=100000,
                       help='Number of samples to process at once')
    parser.add_argument('--batch-size', type=int, default=13500000,
                       help='Batch size for memory-efficient training (default 1.5M × 9 types)')
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info("Starting NIDS training with configuration:")
    logger.info(f"  Data path: {args.data_path}")
    logger.info(f"  Model path: {args.model_path}")
    logger.info(f"  Limit samples: {args.limit_samples}")
    logger.info(f"  Classifier: {'LightGBM' if args.use_lightgbm else 'XGBoost'}")
    logger.info(f"  Chunk size: {args.chunk_size}")
    logger.info(f"  Batch size: {args.batch_size}")
    
    try:
        # Create model
        model = KitsuneNIDSModel(
            data_path=args.data_path,
            model_path=args.model_path,
            chunk_size=args.chunk_size
        )
        
        # Log current memory usage
        log_memory_usage("Before training")
        
        # Train model with batch approach
        if args.limit_samples is None:
            success = model.train_model_batch(
                batch_size=args.batch_size,
                use_xgboost=not args.use_lightgbm
            )
        else:
            # Use original approach with sample limit
            success = model._train_model_original(
                limit_samples=args.limit_samples,
                use_xgboost=not args.use_lightgbm
            )
        
        if success:
            logger.info("Model training completed successfully")
            log_memory_usage("After training")
            return 0
        else:
            logger.error("Model training failed")
            return 1
            
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}")
        logger.critical(traceback.format_exc())
        return 1


if __name__ == "__main__":
    import argparse
    sys.exit(main())