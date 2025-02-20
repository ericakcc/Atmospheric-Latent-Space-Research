import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

class SimpleVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(SimpleVAE, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8,8))
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)
        
        # Decoder
        self.fc_decoder = nn.Linear(latent_dim, 64 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # 添加額外的上採樣層以達到原始尺寸
        self.upsample = nn.Upsample(size=(512, 768), mode='bilinear', align_corners=True)
        
    def encode(self, x):
        print(f"\nEncoder steps:")
        print(f"Input: {x.shape}")
        
        x = F.relu(self.conv1(x))
        print(f"After conv1: {x.shape}")
        
        x = F.relu(self.conv2(x))
        print(f"After conv2: {x.shape}")
        
        x = F.relu(self.conv3(x))
        print(f"After conv3: {x.shape}")
        
        x = self.adaptive_pool(x)
        print(f"After adaptive_pool: {x.shape}")
        
        x = x.view(x.size(0), -1)
        print(f"After flatten: {x.shape}")
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        print(f"mu: {mu.shape}, logvar: {logvar.shape}")
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        print(f"\nDecoder steps:")
        x = self.fc_decoder(z)
        print(f"After fc_decoder: {x.shape}")
        
        x = x.view(x.size(0), 64, 8, 8)
        print(f"After reshape: {x.shape}")
        
        x = F.relu(self.deconv1(x))
        print(f"After deconv1: {x.shape}")
        
        x = F.relu(self.deconv2(x))
        print(f"After deconv2: {x.shape}")
        
        x = torch.sigmoid(self.deconv3(x))
        print(f"After deconv3: {x.shape}")
        
        x = self.upsample(x)
        print(f"After upsample: {x.shape}")
        
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class AtmosphericDataset(Dataset):
    def __init__(self, file_path, shape=(512, 768, 3, 94), dtype=np.float32, transform=None):
        self.file_path = file_path
        self.shape = shape
        self.dtype = dtype
        self.transform = transform
        try:
            data = np.fromfile(file_path, dtype=dtype)
            expected_size = np.prod(shape)
            if data.size != expected_size:
                raise ValueError(f"Data size mismatch: expected {expected_size} elements, got {data.size}")
            data = data.reshape(shape, order='F')
            
            # 處理 NaN 值
            print("Original data statistics:")
            print(f"Number of NaN values: {np.isnan(data).sum()}")
            print(f"Data range: [{np.nanmin(data)}, {np.nanmax(data)}]")
            
            # 將 NaN 替換為 0 或其他合適的值
            data = np.nan_to_num(data, nan=0.0)
            
            print("\nAfter NaN handling:")
            print(f"Number of NaN values: {np.isnan(data).sum()}")
            print(f"Data range: [{np.min(data)}, {np.max(data)}]")
            
            print(f"\nSuccessfully loaded data from {file_path} with shape {data.shape}")
        except Exception as e:
            raise RuntimeError(f"Error reading file {file_path}: {str(e)}")
        
        self.samples = [data[:, :, :, t] for t in range(data.shape[3])]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample = sample.astype(np.float32)
        
        # 確保數據是有效的
        if np.isnan(sample).any():
            sample = np.nan_to_num(sample, nan=0.0)
        
        # 安全的歸一化
        sample_min = np.min(sample)
        sample_max = np.max(sample)
        if sample_max > sample_min:  # 避免除以零
            sample = (sample - sample_min) / (sample_max - sample_min)
        else:
            sample = np.zeros_like(sample)
        
        sample = np.transpose(sample, (2, 0, 1))
        if self.transform:
            sample = self.transform(sample)
        else:
            sample = torch.from_numpy(sample)
        
        # 最後確認一次數據是否在正確範圍內
        sample = torch.clamp(sample, 0, 1)
        return sample

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        
        # 檢查數據是否包含 NaN 或 Inf
        if torch.isnan(data).any() or torch.isinf(data).any():
            print(f"Warning: Batch {batch_idx} contains NaN or Inf values, skipping...")
            continue
        
        optimizer.zero_grad()
        try:
            recon_batch, mu, logvar = model(data)
            
            # 檢查模型輸出
            if torch.isnan(recon_batch).any() or torch.isinf(recon_batch).any():
                print(f"Warning: Model output contains NaN or Inf values in batch {batch_idx}")
                continue
                
            loss = loss_function(recon_batch, data, mu, logvar)
            
            # 檢查損失值
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Loss is NaN or Inf in batch {batch_idx}")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
        
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            torch.cuda.empty_cache()
            continue
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    return test_loss

def main():
    # 超參數設定
    batch_size = 1  # 使用小batch size來debug
    epochs = 5
    learning_rate = 1e-4  # 從 1e-3 改為 1e-4
    latent_dim = 128
    seed = 1

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 數據文件路徑
    data_file = "data/dcape/tpe20050702nor.dat"

    # 建立數據集
    dataset = AtmosphericDataset(data_file, shape=(512, 768, 3, 94), dtype=np.float32)
    
    # 分割數據集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 創建模型
    model = SimpleVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 創建結果目錄
    os.makedirs("results", exist_ok=True)

    # 訓練循環
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, device, train_loader, optimizer, epoch)
        test_loss = test_epoch(model, device, test_loader)

    # 保存模型
    model_path = "results/simple_vae_atmospheric.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # 添加 GPU 記憶體清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 