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
        
        # 直接將所有數據載入記憶體
        self.all_samples = []
        print(f"Loading {len(file_list)} .dat files into memory...")
        
        for dat_file in file_list:
            file_path = os.path.join(data_dir, dat_file)
            try:
                # 載入並處理數據
                data = np.fromfile(file_path, dtype=dtype)
                data = data.reshape(shape, order='F')
                data = Utils.process_terrain_data(data, verbose=False)
                
                # 對每個時間點進行預處理
                for t in range(shape[3]):
                    sample = data[:, :, :, t].copy()
                    sample = sample.astype(np.float32)
                    
                    # 正規化到 [0, 1]
                    sample_min = np.min(sample)
                    sample_max = np.max(sample)
                    if sample_max > sample_min:
                        sample = (sample - sample_min) / (sample_max - sample_min)
                    else:
                        sample = np.zeros_like(sample)
                    
                    # 轉換為 (C, H, W) 格式並轉換為 tensor
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
    
    # 添加 resize 轉換
    transform = transforms.Compose([
        transforms.Resize((256, 384), antialias=True),  # 添加 antialias
        transforms.Lambda(lambda x: x.clone())  # 確保尺寸一致
    ])
    
    # 創建數據集時添加 transform
    train_dataset = AtmosphericDataset(data_dir, train_files, shape, dtype, transform=transform)
    test_dataset = AtmosphericDataset(data_dir, test_files, shape, dtype, transform=transform)
    
    return train_dataset, test_dataset

def main():
    # 超參數設定
    batch_size = 16
    epochs = 50
    learning_rate = 1e-4
    seed = 42

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 數據文件夾路徑
    data_dir = "data/dcape/"  # 替換為實際的資料夾路徑
    
    # 創建訓練集和測試集
    train_dataset, test_dataset = create_train_test_datasets(
        data_dir=data_dir,
        shape=(512, 768, 3, 94),
        dtype=np.float32,
        train_ratio=0.8,
        seed=42
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    # 創建 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # 改為 0 表示不使用多進程
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 定義一個與數據尺寸匹配的 CNNVAE 配置（此配置需根據你的需求進行調整）
    custom_config = {
        "in_channels": 3,
        "input_height": 256,
        "input_width": 384,
        
        # Encoder
        "conv1_out": 16,
        "conv1_kernel": 3,
        "conv1_stride": 1,
        "conv1_padding": 1,
        "pool1_kernel": 2,    
        "pool1_stride": 2,    # 128x192
        
        "conv2_out": 32,
        "conv2_kernel": 3,
        "conv2_stride": 1,
        "conv2_padding": 1,
        "pool2_kernel": 2,
        "pool2_stride": 2,    # 64x96
        
        "conv3_out": 64,
        "conv3_kernel": 3,
        "conv3_stride": 1,    
        "conv3_padding": 1,   # 64x96

        # 確保這個尺寸正確
        "decoder_input_shape": [64, 64, 96],
        
        # Decoder - 確保每一步的上採樣都精確對應
        "deconv1_out": 32,
        "deconv1_kernel": 4,  # 改用 4x4 kernel
        "deconv1_stride": 2,  # 128x192
        "deconv1_padding": 1,
        "upsample1_scale": 1,

        "deconv2_out": 16,
        "deconv2_kernel": 4,  # 改用 4x4 kernel
        "deconv2_stride": 2,  # 256x384
        "deconv2_padding": 1,
        "upsample2_scale": 1,

        "deconv3_out": 3,
        "deconv3_kernel": 3,
        "deconv3_stride": 1,
        "deconv3_padding": 1
    }
    latent_dim = 128  # 從 256 改為 128
    model = CNNVAE(config=custom_config, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs("results", exist_ok=True)

    # 訓練循環
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_epoch(model, device, train_loader, optimizer, epoch, log_interval=1)
        test_epoch(model, device, test_loader)

    # 保存訓練後模型
    model_path = "results/vae_atmospheric.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    # 設置多進程啟動方法
    torch.multiprocessing.set_start_method('spawn')
    main()