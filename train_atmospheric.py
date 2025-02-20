import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import random
import psutil
from collections import OrderedDict
import multiprocessing

from atmospheric_vae.models.vae.ConvVAE import CNNVAE
from atmospheric_vae.training.trainer import train_epoch, test_epoch
from atmospheric_vae.utils.utils import Utils


class AtmosphericDataset(Dataset):
    # Class-level cache that can be shared between instances
    _shared_cache = OrderedDict()
    _shared_cache_size = 0
    
    def __init__(self, data_dir, file_list, shape=(512, 768, 3, 94), dtype=np.float32, transform=None, cache_size=10, shared_cache=False):
        """
        Initialize dataset from a list of .dat files with lazy loading
        
        Args:
            data_dir (str): Directory path containing .dat files
            file_list (list): List of .dat files to use
            shape (tuple): Expected shape of each .dat file
            dtype: Data type for reading files
            transform: Optional transform to be applied on a sample
            cache_size (int): Number of files to keep in memory cache
            shared_cache (bool): Whether to use shared cache between dataset instances
        """
        self.data_dir = data_dir
        self.shape = shape
        self.dtype = dtype
        self.transform = transform
        self.use_shared_cache = shared_cache
        
        # Calculate memory usage per file
        single_file_size = np.prod(shape) * np.dtype(dtype).itemsize / (1024 ** 3)  # Size in GB
        print(f"Estimated memory per file: {single_file_size:.2f} GB")
        
        # Suggest optimal cache size based on available system memory
        total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Total memory in GB
        available_memory = psutil.virtual_memory().available / (1024 ** 3)  # Available memory in GB
        suggested_cache = min(
            int(available_memory * 0.5 / single_file_size),  # Use up to 50% of available memory
            len(file_list)
        )
        
        print(f"System memory: {total_memory:.1f} GB total, {available_memory:.1f} GB available")
        print(f"Suggested cache size: {suggested_cache} files")
        print(f"Current cache size: {cache_size} files")
        
        # Store file information instead of actual data
        self.sample_info = []  # List of (file_path, time_index) tuples
        
        print(f"Processing {len(file_list)} .dat files")
        
        # Validate files and store their information
        for dat_file in file_list:
            file_path = os.path.join(data_dir, dat_file)
            try:
                # Just check file size without loading the data
                file_size = os.path.getsize(file_path)
                expected_size = np.prod(shape) * np.dtype(dtype).itemsize
                if file_size != expected_size:
                    print(f"Warning: Skipping {dat_file} due to size mismatch")
                    continue
                
                # Store information for each time point
                for t in range(shape[3]):  # shape[3] is the time dimension
                    self.sample_info.append((file_path, t))
                print(f"Validated {dat_file}")
                
            except Exception as e:
                print(f"Error validating {dat_file}: {str(e)}")
                continue
        
        if not self.sample_info:
            raise RuntimeError("No valid samples were found")
        
        print(f"Total number of samples: {len(self.sample_info)}")
        
        # Use the suggested cache size if none specified
        if cache_size <= 0:
            self.cache_size = suggested_cache
        else:
            self.cache_size = min(cache_size, suggested_cache)
        
        print(f"Using cache size: {self.cache_size} files")
        
        # Initialize cache
        if self.use_shared_cache:
            if AtmosphericDataset._shared_cache_size == 0:
                AtmosphericDataset._shared_cache_size = self.cache_size
                print(f"Initializing shared cache with size: {self.cache_size} files")
            self.cache = AtmosphericDataset._shared_cache
            print(f"Using shared cache (size: {AtmosphericDataset._shared_cache_size} files)")
        else:
            self.cache = OrderedDict()
            print(f"Using private cache (size: {self.cache_size} files)")
        
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def __len__(self):
        return len(self.sample_info)
    
    def _load_file(self, file_path):
        """Helper method to load and cache file data with statistics"""
        self.total_requests += 1
        cache_size = (AtmosphericDataset._shared_cache_size 
                     if self.use_shared_cache 
                     else self.cache_size)
        
        if file_path in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(file_path)
            data = self.cache[file_path]
        else:
            self.cache_misses += 1
            data = np.fromfile(file_path, dtype=self.dtype)
            data = data.reshape(self.shape, order='F')
            data = Utils.process_terrain_data(data, verbose=False)
            
            if len(self.cache) >= cache_size:
                self.cache.popitem(last=False)
            
            self.cache[file_path] = data
        
        # Print cache statistics every 1000 requests
        if self.total_requests % 1000 == 0:
            hit_rate = self.cache_hits / self.total_requests * 100
            cache_usage = len(self.cache) / cache_size * 100
            print(f"Cache statistics ({('shared' if self.use_shared_cache else 'private')}):\n"
                  f"- Hit rate: {hit_rate:.1f}%\n"
                  f"- Hits: {self.cache_hits}, Misses: {self.cache_misses}\n"
                  f"- Cache usage: {cache_usage:.1f}% ({len(self.cache)}/{cache_size} files)")
        
        return self.cache[file_path]
    
    def __getitem__(self, idx):
        file_path, time_idx = self.sample_info[idx]
        
        # Load the file if not in cache
        data = self._load_file(file_path)
        
        # Get the specific time point
        sample = data[:, :, :, time_idx].copy()
        sample = sample.astype(np.float32)
        
        # Safely normalize data to [0, 1]
        sample_min = np.min(sample)
        sample_max = np.max(sample)
        if sample_max > sample_min:
            sample = (sample - sample_min) / (sample_max - sample_min)
        else:
            sample = np.zeros_like(sample)
        
        # Transpose to (C, H, W) format
        sample = np.transpose(sample, (2, 0, 1))
        if self.transform:
            sample = self.transform(sample)
        else:
            sample = torch.from_numpy(sample)
        
        sample = torch.clamp(sample, 0, 1)
        return sample, 0

def create_train_test_datasets(data_dir, shape=(512, 768, 3, 94), dtype=np.float32, train_ratio=0.8, seed=42, cache_config=None):
    """
    Create train and test datasets by splitting at file level with shared cache
    
    Args:
        cache_config (dict): Cache configuration containing cache_size, use_shared_cache, etc.
    """
    # Get all .dat files
    dat_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    if not dat_files:
        raise ValueError(f"No .dat files found in {data_dir}")
    
    print(f"Found {len(dat_files)} total .dat files")
    
    # Use cache settings from config
    cache_size = cache_config.get('cache_size', 256) if cache_config else 256
    use_shared_cache = cache_config.get('use_shared_cache', True) if cache_config else True
    
    # Split files into train and test sets
    train_size = int(len(dat_files) * train_ratio)
    train_files = dat_files[:train_size]
    test_files = dat_files[train_size:]
    
    print(f"Training files: {len(train_files)}")
    print(f"Testing files: {len(test_files)}")
    
    # Create datasets with shared cache
    train_dataset = AtmosphericDataset(
        data_dir, 
        train_files, 
        shape, 
        dtype, 
        shared_cache=use_shared_cache,
        cache_size=cache_size
    )
    
    test_dataset = AtmosphericDataset(
        data_dir, 
        test_files, 
        shape, 
        dtype, 
        shared_cache=use_shared_cache,
        cache_size=cache_size
    )
    
    return train_dataset, test_dataset

def main():
    # Load configuration
    with open('configs/train_config.json', 'r') as f:
        config = json.load(f)
    
    # Create train and test datasets with cache configuration
    train_dataset, test_dataset = create_train_test_datasets(
        data_dir=config['data']['data_dir'],
        shape=tuple(config['data']['shape']),
        dtype=np.dtype(config['data']['dtype']),
        train_ratio=config['training']['train_ratio'],
        seed=config['training']['seed'],
        cache_config=config['data']['cache_settings']
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    # Set training parameters from config
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    learning_rate = config['training']['learning_rate']
    seed = config['training']['seed']
    train_ratio = config['training']['train_ratio']

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set multiprocessing start method
    if multiprocessing.get_start_method() != 'spawn':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    # Get dataloader settings from config
    dataloader_settings = config['data'].get('dataloader_settings', {})
    
    # Create DataLoaders with settings from config
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        **dataloader_settings
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    # Initialize model with configuration
    model = CNNVAE(
        config=config['model'],
        latent_dim=config['model']['latent_dim']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_epoch(model, device, train_loader, optimizer, epoch, log_interval=1)
        test_epoch(model, device, test_loader)

    # Save trained model
    model_path = "results/vae_atmospheric.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    # Ensure proper multiprocessing behavior
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()