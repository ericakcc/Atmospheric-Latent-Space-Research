import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import random
import torch.multiprocessing
from torchvision import transforms

from atmospheric_vae.models.vae.ConvVAE import CNNVAE
from atmospheric_vae.training.trainer import train_epoch, test_epoch
from atmospheric_vae.utils.utils import Utils
from atmospheric_vae.config.experiment_config import ExperimentConfig
from atmospheric_vae.utils.experiment_logger import ExperimentLogger


class AtmosphericDataset(Dataset):
    def __init__(self, data_dir, file_list, shape=(512, 768, 3, 94), dtype=np.float32, transform=None):
        """
        Initialize dataset by loading all data into memory
        
        Args:
            data_dir (str): Directory path containing .dat files
            file_list (list): List of .dat files to use
            shape (tuple): Expected shape of each .dat file
            dtype: Data type for reading files
            transform: Optional transform to be applied on a sample
        """
        self.transform = transform
        
        # Load all data into memory
        self.all_samples = []
        print(f"Loading {len(file_list)} .dat files into memory...")
        
        for dat_file in file_list:
            file_path = os.path.join(data_dir, dat_file)
            try:
                # Load and process data
                data = np.fromfile(file_path, dtype=dtype)
                data = data.reshape(shape, order='F')
                data = Utils.process_terrain_data(data, verbose=False)
                
                # Preprocess each time point
                for t in range(shape[3]):
                    sample = data[:, :, :, t].copy()
                    sample = sample.astype(np.float32)
                    
                    # Normalize to [0, 1]
                    sample_min = np.min(sample)
                    sample_max = np.max(sample)
                    if sample_max > sample_min:
                        sample = (sample - sample_min) / (sample_max - sample_min)
                    else:
                        sample = np.zeros_like(sample)
                    
                    # Convert to (C, H, W) format and to tensor
                    sample = np.transpose(sample, (2, 0, 1))
                    sample = torch.from_numpy(sample)
                    sample = torch.clamp(sample, 0, 1)
                    
                    self.all_samples.append(sample)
                
                print(f"Loaded {dat_file}")
                
            except Exception as e:
                print(f"Error loading {dat_file}: {str(e)}")
                continue
        
        if not self.all_samples:
            raise RuntimeError("No valid samples were loaded")
        
        print(f"Total samples loaded: {len(self.all_samples)}")
    
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

def create_train_test_datasets(data_dir, shape=(512, 768, 3, 94), dtype=np.float32, train_ratio=0.8, seed=42):
    """
    Create train and test datasets by splitting at file level
    
    Args:
        data_dir (str): Directory containing .dat files
        shape (tuple): Expected shape of each .dat file
        dtype: Data type for reading files
        train_ratio (float): Ratio of files to use for training
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    # Get all .dat files
    dat_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    if not dat_files:
        raise ValueError(f"No .dat files found in {data_dir}")
    
    print(f"Found {len(dat_files)} total .dat files")
    
    # Randomly shuffle files
    random.seed(seed)
    random.shuffle(dat_files)
    
    # Split files into train and test sets
    train_size = int(len(dat_files) * train_ratio)
    train_files = dat_files[:train_size]
    test_files = dat_files[train_size:]
    
    print(f"Training files: {len(train_files)}")
    print(f"Testing files: {len(test_files)}")
    
    transform = transforms.Compose([
        transforms.Resize((256, 384), antialias=True),
        transforms.Lambda(lambda x: x.clone())  
    ])
    
    train_dataset = AtmosphericDataset(data_dir, train_files, shape, dtype, transform=transform)
    test_dataset = AtmosphericDataset(data_dir, test_files, shape, dtype, transform=transform)
    
    return train_dataset, test_dataset

def main():
    # Create experiment configuration
    config = ExperimentConfig()
    config.experiment_name = "exp_002_adjusted_vae"
    config.description = "VAE with adjusted architecture and training parameters"
    
    config.model_config.update({
        "latent_dim": 64,
        "in_channels": 3,
        "input_height": 256,
        "input_width": 384,
        "conv1_out": 32,
        "conv2_out": 64,
        "conv3_out": 128,
        "decoder_input_shape": [128, 64, 96],
    })
    
    config.training_config.update({
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 100,
        "beta": 0.001,
        "loss_weights": {
            "bce": 1.0,
            "mse": 1.0,        
            "l1": 0.1          
        }
    })
    
    # Initialize logger
    logger = ExperimentLogger(config)
    
    # Set random seed
    torch.manual_seed(config.training_config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_dataset, test_dataset = create_train_test_datasets(
        data_dir="data/dcape/",
        shape=(512, 768, 3, 94),
        dtype=np.float32,
        train_ratio=0.8,
        seed=42
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training_config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training_config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model and optimizer
    model = CNNVAE(config=config.model_config).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training_config["learning_rate"],
        weight_decay=0.01
    )
    
    for epoch in range(1, config.training_config["epochs"] + 1):
        train_loss = train_epoch(model, device, train_loader, optimizer, epoch, config, logger)
        test_loss = test_epoch(model, device, test_loader, config)
        logger.log_epoch(epoch, train_loss, test_loss, model)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()