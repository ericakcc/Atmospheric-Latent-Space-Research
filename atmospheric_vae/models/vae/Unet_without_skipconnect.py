import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseVAE

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class CNNVAE_UNet(BaseVAE):
    def __init__(self, config, latent_dim=20):
        super().__init__()
        self.config = config
        in_channels = config.get("in_channels", 1)
        self.in_channels = in_channels
        self.input_height = config.get("input_height", 128)
        self.input_width = config.get("input_width", 128)
        self.debug_mode = False  # 添加調試模式標誌供訓練使用

        # Encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = UNetBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        dummy = torch.zeros(1, in_channels, self.input_height, self.input_width)
        x = self.pool1(self.enc1(dummy))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        self.flatten_dim = x.view(1, -1).size(1)

        # Latent space
        self.latent_dim = config.get("latent_dim", latent_dim)
        self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_decode = nn.Linear(self.latent_dim, self.flatten_dim)

        # Decoder (without skip-connections)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3 = UNetBlock(256, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = UNetBlock(128, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        if self.debug_mode:
            print(f"Encode input shape: {x.shape}")
            
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.pool2(x)
        x = self.enc3(x)
        x = self.pool3(x)

        flat = x.view(x.size(0), -1)
        
        if self.debug_mode:
            print(f"壓平後的特徵維度: {flat.shape}")
            
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar

    def decode(self, z):
        if self.debug_mode:
            print(f"Decode z shape: {z.shape}")
            
        x = self.fc_decode(z)
        B = z.size(0)
        H, W = self.input_height // 8, self.input_width // 8
        
        try:
            x = x.view(B, 256, H, W)
        except RuntimeError as e:
            # 處理維度不匹配的情況
            if self.debug_mode:
                print(f"重塑失敗: {str(e)}")
                print(f"嘗試重新計算尺寸...")
            
            actual_size = x.numel()
            C = 256
            # 計算新的H和W，確保C*H*W等於actual_size/B
            total_spatial = actual_size // (B * C)
            # 嘗試保持寬高比
            ratio = self.input_width / self.input_height
            new_H = int((total_spatial / ratio) ** 0.5)
            new_W = int(total_spatial / new_H)
            
            if new_H * new_W == total_spatial:
                H, W = new_H, new_W
                if self.debug_mode:
                    print(f"使用調整後的尺寸: {[B, C, H, W]}")
                x = x.view(B, C, H, W)
            else:
                # 如果無法保持比例，直接使用平方形狀
                spatial_dim = int((actual_size / (B * C)) ** 0.5)
                if self.debug_mode:
                    print(f"使用平方形狀: {[B, C, spatial_dim, spatial_dim]}")
                x = x.view(B, C, spatial_dim, spatial_dim)
        
        if self.debug_mode:
            print(f"重塑後 - x: {x.shape}")

        # 檢查並替換NaN和Inf值
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        x = self.up3(x)
        x = self.dec3(x)
        # 確保中間特徵圖不包含NaN或Inf
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        
        if self.debug_mode:
            print(f"dec3後 - x: {x.shape}")

        x = self.up2(x)
        x = self.dec2(x)
        # 確保中間特徵圖不包含NaN或Inf
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        
        if self.debug_mode:
            print(f"dec2後 - x: {x.shape}")

        x = self.up1(x)
        x = self.dec1(x)
        
        if self.debug_mode:
            print(f"dec1後 - x: {x.shape}")

        # Resize to match input size
        if x.shape[2:] != (self.input_height, self.input_width):
            x = F.interpolate(x, size=(self.input_height, self.input_width), mode="bilinear", align_corners=False)
        
        # 最終確保輸出值在[0,1]範圍內
        x = torch.clamp(x, min=0.0, max=1.0)
        
        # 檢查並處理任何殘留的NaN或Inf值
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            if self.debug_mode:
                print("警告：處理了NaN或Inf值")
            
        if self.debug_mode:
            print(f"最終輸出形狀: {x.shape}")
            print(f"輸出值範圍: [{x.min().item()}, {x.max().item()}]")

        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        if self.debug_mode:
            print(f"輸入形狀: {x.shape}, 輸出形狀: {recon.shape}")
            
        return recon, mu, logvar 