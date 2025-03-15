import time
import sys
import tensorflow as tf
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class ProgressTracker:
    """
    A class to track and display overall training progress
    """
    def __init__(self, total_steps=4, description="Overall Progress", console_output=True):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.console_output = console_output
        self.start_time = time.time()
        self.step_times = {}
        
        # Initialize progress bar
        if self.console_output:
            self.progress_bar = tqdm(total=total_steps, desc=description, 
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        # Print initial message
        logger.info(f"Starting {description}...")
        
    def update(self, step_name=None):
        """Update progress by one step"""
        self.current_step += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if step_name:
            self.step_times[step_name] = current_time
            logger.info(f"Completed: {step_name} ({self.current_step}/{self.total_steps})")
        
        if self.console_output:
            self.progress_bar.update(1)
            
        # Calculate and log percentage
        percentage = (self.current_step / self.total_steps) * 100
        logger.info(f"Overall Progress: {percentage:.1f}% ({self.current_step}/{self.total_steps})")
        
        return self.current_step, percentage
        
    def close(self):
        """Close the progress tracker"""
        if self.console_output:
            self.progress_bar.close()
        
        # Log total time
        total_time = time.time() - self.start_time
        logger.info(f"Total time for {self.description}: {total_time:.2f} seconds")
        
        # Log time for each step if available
        if self.step_times:
            logger.info("Time breakdown by step:")
            prev_time = self.start_time
            for step_name, step_time in self.step_times.items():
                step_duration = step_time - prev_time
                logger.info(f"  {step_name}: {step_duration:.2f} seconds")
                prev_time = step_time


class DatasetProgressTracker:
    """
    Track progress for dataset loading and processing
    """
    def __init__(self, dataset_names):
        self.dataset_names = dataset_names
        self.total_datasets = len(dataset_names)
        self.current_dataset = 0
        
        logger.info(f"Preparing to load {self.total_datasets} datasets: {', '.join(dataset_names)}")
        
    def start_dataset(self, dataset_name):
        """Mark the start of processing a dataset"""
        self.current_dataset += 1
        logger.info(f"[{self.current_dataset}/{self.total_datasets}] Loading dataset: {dataset_name}")
        return dataset_name
        
    def complete_dataset(self, dataset_name, sample_count):
        """Mark the completion of processing a dataset"""
        logger.info(f"Completed loading {dataset_name} with {sample_count} samples")
        percentage = (self.current_dataset / self.total_datasets) * 100
        logger.info(f"Dataset loading progress: {percentage:.1f}% ({self.current_dataset}/{self.total_datasets})")


class TensorFlowProgressCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to track and display training progress for TensorFlow models
    """
    def __init__(self, total_epochs, model_name="Model"):
        super(TensorFlowProgressCallback, self).__init__()
        self.total_epochs = total_epochs
        self.model_name = model_name
        self.epoch_progress_bar = None
        self.batch_progress_bar = None
        
    def on_train_begin(self, logs=None):
        print(f"\n{'-'*30}")
        print(f"Training {self.model_name} for {self.total_epochs} epochs")
        print(f"{'-'*30}")
        self.train_start_time = time.time()
        
    def on_epoch_begin(self, epoch, logs=None):
        if self.epoch_progress_bar is None:
            self.epoch_progress_bar = tqdm(total=self.total_epochs, 
                                          desc="Epochs", 
                                          position=0,
                                          leave=True)
        
        self.epoch_start_time = time.time()
        total_epochs_completed = epoch
        self.epoch_progress_bar.update(1 if epoch > 0 else 0)
        self.epoch_progress_bar.set_description(f"Epoch {epoch+1}/{self.total_epochs}")
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        
        # Get metrics
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        
        # Update epoch progress bar with metrics
        metrics_str = f"loss: {loss:.4f}"
        if val_loss:
            metrics_str += f", val_loss: {val_loss:.4f}"
            
        self.epoch_progress_bar.set_postfix_str(metrics_str)
        
        # Log to file as well
        logger.info(f"Epoch {epoch+1}/{self.total_epochs} - {metrics_str} - {epoch_time:.2f}s")
        
        # Calculate remaining time
        elapsed_time = time.time() - self.train_start_time
        estimated_total_time = elapsed_time / (epoch + 1) * self.total_epochs
        remaining_time = estimated_total_time - elapsed_time
        
        logger.info(f"Estimated remaining time: {remaining_time:.2f}s")
        
    def on_train_end(self, logs=None):
        # Close all progress bars
        if self.epoch_progress_bar:
            self.epoch_progress_bar.close()
            
        if self.batch_progress_bar:
            self.batch_progress_bar.close()
            
        # Log total training time
        total_time = time.time() - self.train_start_time
        logger.info(f"{self.model_name} training completed in {total_time:.2f} seconds")
        print(f"\n{'-'*30}")
        print(f"{self.model_name} training completed in {total_time:.2f} seconds")
        print(f"{'-'*30}")


class TerminalProgressBar:
    """Simple ASCII progress bar for the terminal"""
    def __init__(self, total, prefix='', suffix='', length=50):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.iteration = 0
        self.start_time = time.time()
        
    def update(self, iteration=None):
        if iteration is not None:
            self.iteration = iteration
        else:
            self.iteration += 1
            
        percent = (self.iteration / self.total) * 100
        filled_length = int(self.length * self.iteration // self.total)
        bar = '█' * filled_length + '-' * (self.length - filled_length)
        
        elapsed_time = time.time() - self.start_time
        if self.iteration > 0:
            estimated_total = elapsed_time * (self.total / self.iteration)
            remaining = estimated_total - elapsed_time
            time_info = f" | {elapsed_time:.1f}s elapsed, {remaining:.1f}s remaining"
        else:
            time_info = ""
            
        sys.stdout.write(f'\r{self.prefix} |{bar}| {percent:.1f}% {self.suffix}{time_info}')
        sys.stdout.flush()
        
        if self.iteration == self.total:
            print()