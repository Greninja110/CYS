import time
import sys
import tensorflow as tf
from tqdm import tqdm
import logging
import math
import matplotlib.pyplot as plt
from IPython.display import clear_output

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
        self.step_descriptions = {}
        
        # Initialize progress bar
        if self.console_output:
            self.progress_bar = tqdm(
                total=total_steps, 
                desc=description, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        
        # Print initial message
        logger.info(f"Starting {description}...")
        
    def update(self, step_name=None):
        """Update progress by one step"""
        self.current_step += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if step_name:
            self.step_times[step_name] = current_time
            self.step_descriptions[self.current_step] = step_name
            logger.info(f"Completed: {step_name} ({self.current_step}/{self.total_steps})")
        
        if self.console_output:
            # Update progress bar description to include the step name
            if step_name:
                self.progress_bar.set_description(f"{self.description}: {step_name}")
            self.progress_bar.update(1)
            
        # Calculate and log percentage
        percentage = (self.current_step / self.total_steps) * 100
        logger.info(f"Overall Progress: {percentage:.1f}% ({self.current_step}/{self.total_steps})")
        
        # Estimate remaining time
        if self.current_step > 0:
            time_per_step = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            estimated_remaining = time_per_step * remaining_steps
            
            # Format time for display
            hours, remainder = divmod(estimated_remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                time_str = f"{int(hours)}h {int(minutes)}m"
            else:
                time_str = f"{int(minutes)}m {int(seconds)}s"
                
            logger.info(f"Estimated time remaining: {time_str}")
        
        return self.current_step, percentage
        
    def close(self):
        """Close the progress tracker"""
        if self.console_output:
            self.progress_bar.close()
        
        # Log total time
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            time_str = f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
        else:
            time_str = f"{int(minutes)}m {seconds:.1f}s"
            
        logger.info(f"Total time for {self.description}: {time_str}")
        
        # Log time for each step if available
        if self.step_times:
            logger.info("Time breakdown by step:")
            prev_time = self.start_time
            
            for step in range(1, self.total_steps + 1):
                if step in self.step_descriptions:
                    step_name = self.step_descriptions[step]
                    if step_name in self.step_times:
                        step_time = self.step_times[step_name]
                        step_duration = step_time - prev_time
                        
                        hours, remainder = divmod(step_duration, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        
                        if hours > 0:
                            duration_str = f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
                        else:
                            duration_str = f"{int(minutes)}m {seconds:.1f}s"
                            
                        logger.info(f"  Step {step}: {step_name} - {duration_str}")
                        prev_time = step_time
        
        return total_time


class DatasetProgressTracker:
    """
    Track progress for dataset loading and processing
    """
    def __init__(self, dataset_names):
        self.dataset_names = dataset_names
        self.total_datasets = len(dataset_names)
        self.current_dataset = 0
        self.start_time = time.time()
        self.dataset_times = {}
        
        logger.info(f"Preparing to load {self.total_datasets} datasets: {', '.join(dataset_names)}")
        
        # Initialize progress bar
        try:
            self.progress_bar = tqdm(
                total=self.total_datasets,
                desc="Loading Datasets",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        except:
            self.progress_bar = None
        
    def start_dataset(self, dataset_name):
        """Mark the start of processing a dataset"""
        self.current_dataset += 1
        logger.info(f"[{self.current_dataset}/{self.total_datasets}] Loading dataset: {dataset_name}")
        
        # Update progress bar
        if self.progress_bar:
            self.progress_bar.set_description(f"Loading {dataset_name}")
            
        # Record start time for this dataset
        self.dataset_times[dataset_name] = {'start': time.time()}
        
        return dataset_name
        
    def complete_dataset(self, dataset_name, sample_count):
        """Mark the completion of processing a dataset"""
        if dataset_name in self.dataset_times:
            self.dataset_times[dataset_name]['end'] = time.time()
            self.dataset_times[dataset_name]['samples'] = sample_count
            
            # Calculate time taken for this dataset
            duration = self.dataset_times[dataset_name]['end'] - self.dataset_times[dataset_name]['start']
            samples_per_second = sample_count / duration if duration > 0 else 0
            
            logger.info(f"Completed loading {dataset_name} with {sample_count} samples " +
                       f"in {duration:.2f}s ({samples_per_second:.1f} samples/sec)")
        else:
            logger.info(f"Completed loading {dataset_name} with {sample_count} samples")
        
        # Update progress bar
        if self.progress_bar:
            self.progress_bar.update(1)
            
        # Log overall progress
        percentage = (self.current_dataset / self.total_datasets) * 100
        logger.info(f"Dataset loading progress: {percentage:.1f}% ({self.current_dataset}/{self.total_datasets})")
        
        # Estimate remaining time
        if self.current_dataset > 0:
            elapsed = time.time() - self.start_time
            time_per_dataset = elapsed / self.current_dataset
            remaining_datasets = self.total_datasets - self.current_dataset
            estimated_remaining = time_per_dataset * remaining_datasets
            
            # Format time for display
            hours, remainder = divmod(estimated_remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                time_str = f"{int(hours)}h {int(minutes)}m"
            else:
                time_str = f"{int(minutes)}m {int(seconds)}s"
                
            logger.info(f"Estimated time remaining for dataset loading: {time_str}")
            
        return percentage
        
    def close(self):
        """Close the dataset progress tracker"""
        if self.progress_bar:
            self.progress_bar.close()
            
        # Log performance statistics
        total_time = time.time() - self.start_time
        total_samples = sum(data.get('samples', 0) for data in self.dataset_times.values())
        
        logger.info("Dataset loading performance:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Total samples: {total_samples}")
        
        if total_time > 0:
            logger.info(f"  Overall throughput: {total_samples / total_time:.1f} samples/sec")
            
        # Log per-dataset statistics
        logger.info("Per-dataset statistics:")
        for dataset, data in self.dataset_times.items():
            if 'start' in data and 'end' in data and 'samples' in data:
                duration = data['end'] - data['start']
                samples = data['samples']
                throughput = samples / duration if duration > 0 else 0
                
                logger.info(f"  {dataset}: {samples} samples, {duration:.2f}s, {throughput:.1f} samples/sec")


class TensorFlowProgressCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to track and display training progress for TensorFlow models
    """
    def __init__(self, total_epochs, model_name="Model", plot_metrics=True):
        super(TensorFlowProgressCallback, self).__init__()
        self.total_epochs = total_epochs
        self.model_name = model_name
        self.plot_metrics = plot_metrics
        self.epoch_progress_bar = None
        self.batch_progress_bar = None
        self.metrics_history = {'loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def on_train_begin(self, logs=None):
        print(f"\n{'-'*50}")
        print(f"Training {self.model_name} for {self.total_epochs} epochs")
        print(f"{'-'*50}")
        self.train_start_time = time.time()
        
    def on_epoch_begin(self, epoch, logs=None):
        if self.epoch_progress_bar is None:
            self.epoch_progress_bar = tqdm(
                total=self.total_epochs, 
                desc=f"Training {self.model_name}", 
                position=0,
                leave=True
            )
        
        self.epoch_start_time = time.time()
        self.epoch_progress_bar.update(1 if epoch > 0 else 0)
        self.epoch_progress_bar.set_description(f"Epoch {epoch+1}/{self.total_epochs}")
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        epoch_time = time.time() - self.epoch_start_time
        
        # Extract metrics
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        
        # Update metrics history
        self.metrics_history['loss'].append(loss)
        if 'val_loss' in logs:
            self.metrics_history['val_loss'].append(val_loss)
            
        # Check if this is the best model so far
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            is_best = " (best so far)"
        else:
            is_best = ""
        
        # Build metrics string
        metrics_str = f"loss: {loss:.4f}"
        if 'val_loss' in logs:
            metrics_str += f", val_loss: {val_loss:.4f}{is_best}"
            
        # Update progress bar with metrics
        self.epoch_progress_bar.set_postfix_str(metrics_str)
        
        # Log to file as well
        logger.info(f"Epoch {epoch+1}/{self.total_epochs} - {metrics_str} - {epoch_time:.2f}s")
        
        # Plot metrics if requested
        if self.plot_metrics and epoch % 5 == 0:
            self._plot_metrics()
        
        # Calculate remaining time
        elapsed_time = time.time() - self.train_start_time
        estimated_total_time = elapsed_time / (epoch + 1) * self.total_epochs
        remaining_time = estimated_total_time - elapsed_time
        
        hours, remainder = divmod(remaining_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            time_str = f"{int(hours)}h {int(minutes)}m"
        else:
            time_str = f"{int(minutes)}m {int(seconds)}s"
            
        logger.info(f"Estimated remaining time: {time_str}")
        
    def on_train_end(self, logs=None):
        # Close all progress bars
        if self.epoch_progress_bar:
            self.epoch_progress_bar.close()
            
        if self.batch_progress_bar:
            self.batch_progress_bar.close()
            
        # Log total training time
        total_time = time.time() - self.train_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            time_str = f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
        else:
            time_str = f"{int(minutes)}m {seconds:.1f}s"
            
        logger.info(f"{self.model_name} training completed in {time_str}")
        
        # Log best epoch
        if self.best_epoch > 0:
            logger.info(f"Best model at epoch {self.best_epoch+1} with val_loss: {self.best_val_loss:.6f}")
        
        # Final metrics plot
        if self.plot_metrics:
            self._plot_metrics()
            
        print(f"\n{'-'*50}")
        print(f"{self.model_name} training completed in {time_str}")
        if self.best_epoch > 0:
            print(f"Best model at epoch {self.best_epoch+1} with val_loss: {self.best_val_loss:.6f}")
        print(f"{'-'*50}")
        
    def _plot_metrics(self):
        """Plot training metrics"""
        try:
            # Skip if no validation data
            if len(self.metrics_history['val_loss']) == 0:
                return
                
            plt.figure(figsize=(12, 4))
            
            # Plot training & validation loss
            plt.subplot(1, 2, 1)
            plt.plot(self.metrics_history['loss'], label='Training Loss')
            plt.plot(self.metrics_history['val_loss'], label='Validation Loss')
            if self.best_epoch > 0:
                plt.axvline(x=self.best_epoch, color='r', linestyle='--', alpha=0.3)
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot zoomed validation loss
            plt.subplot(1, 2, 2)
            val_losses = self.metrics_history['val_loss']
            if len(val_losses) > 5:  # Only plot if we have enough data
                # Plot from 5th epoch onwards to avoid initial high losses
                start_idx = min(5, len(val_losses) // 3)
                plt.plot(range(start_idx, len(val_losses)), val_losses[start_idx:], 'g-')
                if self.best_epoch >= start_idx:
                    plt.axvline(x=self.best_epoch, color='r', linestyle='--', alpha=0.3)
                plt.title('Validation Loss (Zoomed)')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Try to display in notebook if possible
            try:
                clear_output(wait=True)
                plt.show()
            except:
                pass
            
            # Save the plot
            plt.savefig(f'training_progress_{self.model_name}.png')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error plotting metrics: {e}")


class MemoryMonitorCallback(tf.keras.callbacks.Callback):
    """
    Monitor and log memory usage during training
    """
    def __init__(self, check_interval=5):
        super(MemoryMonitorCallback, self).__init__()
        self.check_interval = check_interval
        self.memory_usage = []
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.check_interval == 0:
            self._log_memory_usage(f"Epoch {epoch+1} begin")
            
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.check_interval == 0:
            self._log_memory_usage(f"Epoch {epoch+1} end")
            
    def _log_memory_usage(self, point):
        """Log memory usage at a specific point"""
        try:
            import psutil
            import tensorflow as tf
            
            # Get CPU memory usage
            process = psutil.Process()
            cpu_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Get GPU memory usage if available
            gpu_memory = None
            if hasattr(tf.config, 'experimental') and hasattr(tf.config.experimental, 'get_memory_info'):
                for device in tf.config.list_logical_devices('GPU'):
                    try:
                        memory_info = tf.config.experimental.get_memory_info(device.name)
                        gpu_memory = memory_info['current'] / (1024 * 1024)  # MB
                        break  # Just get the first GPU
                    except:
                        pass
            
            # Log memory usage
            if gpu_memory:
                logger.info(f"Memory usage at {point} - CPU: {cpu_memory:.0f} MB, GPU: {gpu_memory:.0f} MB")
                self.memory_usage.append((point, cpu_memory, gpu_memory))
            else:
                logger.info(f"Memory usage at {point} - CPU: {cpu_memory:.0f} MB")
                self.memory_usage.append((point, cpu_memory, None))
                
        except Exception as e:
            logger.warning(f"Error logging memory usage: {e}")
            

class TerminalProgressBar:
    """
    Simple ASCII progress bar for the terminal with time estimation
    """
    def __init__(self, total, prefix='', suffix='', length=50):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.iteration = 0
        self.start_time = time.time()
        
    def update(self, iteration=None):
        """Update the progress bar"""
        if iteration is not None:
            self.iteration = iteration
        else:
            self.iteration += 1
            
        percent = (self.iteration / self.total) * 100
        filled_length = int(self.length * self.iteration // self.total)
        bar = '█' * filled_length + '-' * (self.length - filled_length)
        
        # Calculate time information
        elapsed_time = time.time() - self.start_time
        if self.iteration > 0:
            estimated_total = elapsed_time * (self.total / self.iteration)
            remaining = estimated_total - elapsed_time
            
            # Format time strings
            elapsed_hours, elapsed_remainder = divmod(elapsed_time, 3600)
            elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
            
            remaining_hours, remaining_remainder = divmod(remaining, 3600)
            remaining_minutes, remaining_seconds = divmod(remaining_remainder, 60)
            
            if elapsed_hours > 0:
                elapsed_str = f"{int(elapsed_hours)}h {int(elapsed_minutes)}m"
            else:
                elapsed_str = f"{int(elapsed_minutes)}m {int(elapsed_seconds)}s"
                
            if remaining_hours > 0:
                remaining_str = f"{int(remaining_hours)}h {int(remaining_minutes)}m"
            else:
                remaining_str = f"{int(remaining_minutes)}m {int(remaining_seconds)}s"
                
            time_info = f" | {elapsed_str} elapsed, {remaining_str} remaining"
        else:
            time_info = ""
            
        # Print the progress bar
        print(f'\r{self.prefix} |{bar}| {percent:.1f}% {self.suffix}{time_info}', end='')
        
        # Print newline when complete
        if self.iteration == self.total:
            print()
            
    def close(self):
        """Close the progress bar and print final statistics"""
        if self.iteration < self.total:
            self.update(self.total)
            
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            time_str = f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
        else:
            time_str = f"{int(minutes)}m {seconds:.1f}s"
            
        print(f"\nCompleted in {time_str}")


class ChunkProcessor:
    """
    Helper class to process large datasets in chunks with progress tracking
    """
    def __init__(self, total_chunks, description="Processing Chunks"):
        self.total_chunks = total_chunks
        self.description = description
        self.current_chunk = 0
        self.start_time = time.time()
        
        # Initialize progress bar
        self.progress_bar = tqdm(
            total=total_chunks,
            desc=description,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
    def update(self, chunk_description=None):
        """Update progress after processing a chunk"""
        self.current_chunk += 1
        
        # Update progress bar
        if chunk_description:
            self.progress_bar.set_description(f"{self.description}: {chunk_description}")
        self.progress_bar.update(1)
        
        # Calculate progress percentage
        percentage = (self.current_chunk / self.total_chunks) * 100
        
        # Estimate remaining time
        elapsed = time.time() - self.start_time
        if self.current_chunk > 0:
            time_per_chunk = elapsed / self.current_chunk
            remaining_chunks = self.total_chunks - self.current_chunk
            estimated_remaining = time_per_chunk * remaining_chunks
            
            # Format time for display
            hours, remainder = divmod(estimated_remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                time_str = f"{int(hours)}h {int(minutes)}m"
            else:
                time_str = f"{int(minutes)}m {int(seconds)}s"
                
            logger.info(f"Chunk {self.current_chunk}/{self.total_chunks} completed " +
                       f"({percentage:.1f}%). Estimated time remaining: {time_str}")
        else:
            logger.info(f"Chunk {self.current_chunk}/{self.total_chunks} completed " +
                       f"({percentage:.1f}%)")
            
        return percentage
        
    def close(self):
        """Close the chunk processor"""
        self.progress_bar.close()
        
        # Log total processing time
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            time_str = f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
        else:
            time_str = f"{int(minutes)}m {seconds:.1f}s"
            
        logger.info(f"{self.description} completed in {time_str}")
        logger.info(f"Average time per chunk: {total_time / self.total_chunks:.2f}s")
        
        return total_time