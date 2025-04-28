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
        self.debug_mode = False  # 添加調試模式標誌
        
        # Encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = UNetBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # 計算編碼器的輸出尺寸
        dummy = torch.zeros(1, in_channels, self.input_height, self.input_width)
        e1 = self.enc1(dummy)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # 記錄特徵圖的形狀以便在解碼時使用
        self.encoded_h = p3.size(2)
        self.encoded_w = p3.size(3)
        
        self.flatten_dim = p3.view(1, -1).size(1)
        print(f"初始化時 - 壓平的特徵維度: {self.flatten_dim}, 編碼形狀: [{p3.size(2)}, {p3.size(3)}]")

        # Latent space
        latent_dim = config.get("latent_dim", latent_dim)
        self.latent_dim = latent_dim
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        # Decoder - 修正通道數以避免不匹配
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        # 確保上採樣後合併的通道數正確
        self.dec3 = UNetBlock(256 + 256, 128)  # 256(解碼) + 256(編碼e3)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = UNetBlock(128 + 128, 64)   # 128(dec3) + 128(編碼e2)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1),  # 64(dec2) + 64(編碼e1)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        # 輸出調試信息
        if self.debug_mode:
            print(f"Encode input shape: {x.shape}")
            
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # 輸出調試信息
        if self.debug_mode:
            print(f"特徵尺寸 - e1: {e1.shape}, e2: {e2.shape}, e3: {e3.shape}")
            print(f"池化尺寸 - p1: {p1.shape}, p2: {p2.shape}, p3: {p3.shape}")

        flat = p3.view(p3.size(0), -1)
        
        # 輸出調試信息
        if self.debug_mode:
            print(f"壓平後的特徵維度: {flat.shape}")
            
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        
        # 將特徵圖形狀保存為屬性以便在decode中使用
        self.last_batch_size = x.size(0)
        self.last_encoded_h = p3.size(2)
        self.last_encoded_w = p3.size(3)
        self.last_encoded_c = p3.size(1)
        
        return mu, logvar, [e1, e2, e3]

    def decode(self, z, enc_feats):
        # 輸出調試信息
        if self.debug_mode:
            print(f"Decode z shape: {z.shape}")
            print(f"enc_feats shapes: {[f.shape for f in enc_feats]}")
        
        x = self.fc_decode(z)
        
        # 使用編碼過程中保存的特徵圖形狀，確保重塑後的尺寸正確
        B = z.size(0)
        C = self.last_encoded_c if hasattr(self, 'last_encoded_c') else 256
        H = self.last_encoded_h if hasattr(self, 'last_encoded_h') else (self.input_height // 8)
        W = self.last_encoded_w if hasattr(self, 'last_encoded_w') else (self.input_width // 8)
        
        # 檢查重塑的尺寸是否合理
        expected_size = B * C * H * W
        actual_size = x.numel()
        
        if self.debug_mode:
            print(f"重塑前 - x: {x.shape}, 預期重塑尺寸: {[B, C, H, W]}")
            print(f"預期元素數量: {expected_size}, 實際元素數量: {actual_size}")
        
        # 如果尺寸不匹配，嘗試適應
        if expected_size != actual_size:
            if self.debug_mode:
                print(f"尺寸不匹配，嘗試適應大小")
            # 計算新的H和W，確保C*H*W等於actual_size/B
            total_spatial = actual_size // (B * C)
            # 嘗試保持寬高比
            ratio = W / H
            new_H = int((total_spatial / ratio) ** 0.5)
            new_W = int(total_spatial / new_H)
            
            if new_H * new_W == total_spatial:
                H, W = new_H, new_W
                if self.debug_mode:
                    print(f"調整後的尺寸: {[B, C, H, W]}")
            else:
                # 如果無法保持比例，直接使用全平方形狀
                spatial_dim = int((actual_size / (B * C)) ** 0.5)
                H = W = spatial_dim
                if self.debug_mode:
                    print(f"使用平方形狀: {[B, C, H, W]}")
        
        try:
            x = x.view(B, C, H, W)
        except RuntimeError as e:
            # 如果仍然出錯，使用最安全的方法 - 直接計算
            if self.debug_mode:
                print(f"重塑失敗，使用最安全的方法: {str(e)}")
            spatial_size = actual_size // (B * C)
            spatial_dim = int(spatial_size ** 0.5)
            x = x.view(B, C, spatial_dim, spatial_dim)
            H, W = spatial_dim, spatial_dim
            
        if self.debug_mode:
            print(f"重塑後 - x: {x.shape}")

        x = self.up3(x)
        if self.debug_mode:
            print(f"第一次上採樣後 - x: {x.shape}, 將與 e3: {enc_feats[2].shape} 連接")
            
        # 檢查尺寸是否匹配，如果不匹配則調整大小
        if x.shape[2:] != enc_feats[2].shape[2:]:
            x = F.interpolate(x, size=enc_feats[2].shape[2:], mode='bilinear', align_corners=False)
            if self.debug_mode:
                print(f"調整 x 大小以匹配 e3: {x.shape}")
                
        x = torch.cat([x, enc_feats[2]], dim=1)
        if self.debug_mode:
            print(f"連接後 - x: {x.shape}")
            
        x = self.dec3(x)
        if self.debug_mode:
            print(f"dec3後 - x: {x.shape}")

        x = self.up2(x)
        if self.debug_mode:
            print(f"第二次上採樣後 - x: {x.shape}, 將與 e2: {enc_feats[1].shape} 連接")
            
        # 檢查尺寸是否匹配，如果不匹配則調整大小
        if x.shape[2:] != enc_feats[1].shape[2:]:
            x = F.interpolate(x, size=enc_feats[1].shape[2:], mode='bilinear', align_corners=False)
            if self.debug_mode:
                print(f"調整 x 大小以匹配 e2: {x.shape}")
                
        x = torch.cat([x, enc_feats[1]], dim=1)
        if self.debug_mode:
            print(f"連接後 - x: {x.shape}")
            
        x = self.dec2(x)
        if self.debug_mode:
            print(f"dec2後 - x: {x.shape}")

        x = self.up1(x)
        if self.debug_mode:
            print(f"第三次上採樣後 - x: {x.shape}, 將與 e1: {enc_feats[0].shape} 連接")
            
        # 檢查尺寸是否匹配，如果不匹配則調整大小
        if x.shape[2:] != enc_feats[0].shape[2:]:
            x = F.interpolate(x, size=enc_feats[0].shape[2:], mode='bilinear', align_corners=False)
            if self.debug_mode:
                print(f"調整 x 大小以匹配 e1: {x.shape}")
                
        x = torch.cat([x, enc_feats[0]], dim=1)
        if self.debug_mode:
            print(f"連接後 - x: {x.shape}")
            
        x = self.dec1(x)
        if self.debug_mode:
            print(f"dec1後 - x: {x.shape}")

        # 確保輸出大小與輸入大小匹配
        if x.shape[2:] != (self.input_height, self.input_width):
            x = F.interpolate(x, size=(self.input_height, self.input_width), mode="bilinear", align_corners=False)
            
        # 添加調試輸出
        if self.debug_mode:
            print(f"UNetVAE decode output shape: {x.shape}, expected channels: {self.in_channels}")

        return x

    def forward(self, x):
        """
        覆蓋BaseVAE的forward方法，以適應特殊的encode輸出格式
        此方法處理UNetVAE的特殊encode輸出，並返回與BaseVAE相容的結果
        """
        input_shape = x.shape
        mu, logvar, enc_feats = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, enc_feats)
        
        # 添加調試輸出
        if self.debug_mode:
            print(f"UNetVAE input shape: {input_shape}, output shape: {recon.shape}")
            
        return recon, mu, logvar
