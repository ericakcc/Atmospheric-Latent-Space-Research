import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import sys
from datetime import datetime

# 添加項目根目錄到路徑
sys.path.append('/home/erica/Atmospheric-Latent-Space-Research')

# 導入項目模塊
from atmospheric_vae.utils.utils import Utils
from notebooks.train_atmospheric import AtmosphericDataset, create_train_test_datasets
from atmospheric_vae.training.trainer import vae_loss_function
from atmospheric_vae.config.experiment_config import ExperimentConfig

def check_terrain_masking():
    """
    Check if terrain masking is correctly applied to gradient calculation
    """
    # 創建輸出目錄
    output_dir = "terrain_mask_check"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("Loading dataset...")
    # 載入小部分數據用於測試
    try:
        # 獲取所有.dat文件並只選擇第一個用於測試
        data_dir = "data/dcape/"
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
        if not all_files:
            print(f"No .dat files found in {data_dir}")
            return
        
        # 只使用第一個文件創建數據集
        sample_dataset = AtmosphericDataset(
            data_dir=data_dir,
            file_list=[all_files[0]],
            shape=(512, 768, 3, 94),
            dtype=np.float32,
            transform=None
        )
        
        # 創建數據加載器
        sample_loader = DataLoader(
            sample_dataset,
            batch_size=4,
            shuffle=False
        )
        
        # 獲取一個批次的數據
        data_batch, _ = next(iter(sample_loader))
        
        print(f"Loaded data shape: {data_batch.shape}")
        print(f"Data range: [{data_batch.min().item():.6f}, {data_batch.max().item():.6f}]")
        
        # 1. 檢查地形掩碼是否正確創建
        print("\nChecking terrain mask creation...")
        # 創建地形掩碼（與trainer.py中相同的方法）
        mask = (data_batch.sum(dim=1, keepdim=True) > 0).float()
        mask = mask.repeat(1, data_batch.size(1), 1, 1)
        
        # 計算有效像素的比例
        valid_ratio = mask.sum() / mask.numel()
        print(f"Valid terrain pixel ratio: {valid_ratio.item():.2%}")
        
        # 檢查是否所有像素都是非零的
        if valid_ratio.item() > 0.99:
            print("WARNING: Almost all pixels are non-zero, which means all areas are treated as valid terrain.")
            print("This can cause issues with the masking test, as there's no distinction between terrain and non-terrain.")
            
            # 嘗試創建一個人工掩碼以進行測試
            print("\nCreating artificial mask for testing...")
            # 創建一個只在中間區域為1的掩碼
            h, w = data_batch.shape[2], data_batch.shape[3]
            artificial_mask = torch.zeros_like(mask)
            artificial_mask[:, :, h//4:3*h//4, w//4:3*w//4] = 1.0
            
            # 使用這個人工掩碼代替原始掩碼
            mask = artificial_mask
            valid_ratio = mask.sum() / mask.numel()
            print(f"New artificial mask created with valid ratio: {valid_ratio.item():.2%}")
        
        # 2. 模擬損失計算
        print("\nSimulating loss calculation...")
        # 創建虛擬重構數據（實際應用中這將是模型輸出）
        recon_batch = data_batch.clone()
        
        # 創建實驗配置
        config = ExperimentConfig()
        
        # 分別計算帶掩碼和不帶掩碼的損失
        # 創建一個全1掩碼用於不掩碼的情況
        all_ones_mask = torch.ones_like(mask)
        
        # 計算對比
        mu = torch.zeros(data_batch.size(0), 64)  # 假設latent_dim為64
        logvar = torch.zeros(data_batch.size(0), 64)
        
        masked_loss = vae_loss_function(recon_batch, data_batch, mu, logvar, mask, config)
        unmasked_loss = vae_loss_function(recon_batch, data_batch, mu, logvar, all_ones_mask, config)
        
        print(f"Loss with mask: {masked_loss.item():.6f}")
        print(f"Loss without mask: {unmasked_loss.item():.6f}")
        print(f"Loss difference: {unmasked_loss.item() - masked_loss.item():.6f}")
        
        # 3. 視覺化地形掩碼
        print("\nGenerating visualizations...")
        # 從批次中選擇一個樣本
        sample_idx = 0
        sample_data = data_batch[sample_idx].detach().cpu().numpy()
        sample_mask = mask[sample_idx, 0].detach().cpu().numpy()  # 取第一個通道的掩碼
        
        # 創建對比圖
        plt.figure(figsize=(15, 5))
        
        # 原始數據
        plt.subplot(131)
        # 顯示每個通道的平均值
        plt.imshow(np.mean(sample_data, axis=0))
        plt.title("Original Data (Channel Average)")
        plt.colorbar()
        
        # 地形掩碼
        plt.subplot(132)
        plt.imshow(sample_mask, cmap='gray')
        plt.title("Terrain Mask")
        plt.colorbar()
        
        # 應用掩碼後的數據
        plt.subplot(133)
        masked_sample = sample_data * sample_mask[np.newaxis, :, :]
        plt.imshow(np.mean(masked_sample, axis=0))
        plt.title("Masked Data (Channel Average)")
        plt.colorbar()
        
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"terrain_mask_visualization_{timestamp}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Visualization saved to: {fig_path}")
        
        # 4. 分析每個通道的掩碼效果
        plt.figure(figsize=(15, 10))
        for channel in range(sample_data.shape[0]):
            plt.subplot(3, 3, channel+1)
            channel_data = sample_data[channel]
            masked_channel = channel_data * sample_mask
            plt.imshow(masked_channel)
            plt.title(f"Channel {channel+1} (Masked)")
            plt.colorbar()
        
        plt.tight_layout()
        channel_fig_path = os.path.join(output_dir, f"channel_masks_{timestamp}.png")
        plt.savefig(channel_fig_path, dpi=300)
        plt.close()
        print(f"Channel masks visualization saved to: {channel_fig_path}")
        
        # 5. 分析梯度中是否只有地形區域有貢獻
        print("\nAnalyzing gradient contribution...")
        # 創建一個有需要梯度的數據拷貝
        grad_data = data_batch.clone().detach().requires_grad_(True)
        
        # 計算帶掩碼的損失並反向傳播
        dummy_mu = torch.zeros(grad_data.size(0), 64)
        dummy_logvar = torch.zeros(grad_data.size(0), 64)
        
        # 啟用異常檢測
        prev_anomaly_detection = torch.is_anomaly_enabled()
        torch.set_anomaly_enabled(True)
        
        masked_loss = vae_loss_function(grad_data, data_batch, dummy_mu, dummy_logvar, mask, config)
        print(f"Loss for gradient computation: {masked_loss.item():.6f}")
        masked_loss.backward()
        
        # 還原異常檢測設置
        torch.set_anomaly_enabled(prev_anomaly_detection)
        
        # 檢查梯度中非零元素的位置是否與掩碼一致
        if grad_data.grad is None:
            print("ERROR: No gradients were computed!")
        else:
            # 打印梯度的基本統計信息
            grad_stats = {
                "min": grad_data.grad.min().item(),
                "max": grad_data.grad.max().item(),
                "mean": grad_data.grad.mean().item(),
                "non_zero": (grad_data.grad != 0).float().sum().item(),
                "total": grad_data.grad.numel()
            }
            print(f"Gradient statistics:")
            print(f"  Min: {grad_stats['min']:.6f}")
            print(f"  Max: {grad_stats['max']:.6f}")
            print(f"  Mean: {grad_stats['mean']:.6f}")
            print(f"  Non-zero elements: {grad_stats['non_zero']} / {grad_stats['total']} ({grad_stats['non_zero']/grad_stats['total']*100:.2f}%)")
            
            # 檢查梯度掩碼
            grad_mask = (grad_data.grad != 0).float()
            
            # 打印掩碼的基本統計信息
            mask_stats = {
                "terrain": mask.sum().item(),
                "gradient": grad_mask.sum().item(),
                "total": mask.numel()
            }
            print(f"Mask statistics:")
            print(f"  Terrain mask non-zero elements: {mask_stats['terrain']} / {mask_stats['total']} ({mask_stats['terrain']/mask_stats['total']*100:.2f}%)")
            print(f"  Gradient mask non-zero elements: {mask_stats['gradient']} / {mask_stats['total']} ({mask_stats['gradient']/mask_stats['total']*100:.2f}%)")
            
            # 計算梯度掩碼與地形掩碼的重疊情況
            overlap = (grad_mask * mask).sum().item()
            if mask_stats['terrain'] > 0:
                terrain_coverage = overlap / mask_stats['terrain'] * 100
            else:
                terrain_coverage = 0
                
            if mask_stats['gradient'] > 0:
                gradient_coverage = overlap / mask_stats['gradient'] * 100
            else:
                gradient_coverage = 0
                
            print(f"Overlap statistics:")
            print(f"  Overlap elements: {overlap}")
            print(f"  Percentage of terrain covered by gradient: {terrain_coverage:.2f}%")
            print(f"  Percentage of gradient in terrain area: {gradient_coverage:.2f}%")
            
            # 計算梯度掩碼與地形掩碼的一致性
            consistency = ((grad_mask > 0) == (mask > 0)).float().mean().item()
            print(f"Consistency between gradient mask and terrain mask: {consistency*100:.2f}%")
        
            # 可視化梯度掩碼
            sample_grad_mask = grad_mask[sample_idx, 0].detach().cpu().numpy()
            
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(sample_mask, cmap='gray')
            plt.title("Terrain Mask")
            plt.colorbar()
            
            plt.subplot(132)
            plt.imshow(sample_grad_mask, cmap='gray')
            plt.title("Gradient Mask")
            plt.colorbar()
            
            plt.subplot(133)
            # 顯示兩者的差異
            difference = sample_grad_mask - sample_mask
            plt.imshow(difference, cmap='bwr')
            plt.title("Difference (Gradient Mask - Terrain Mask)")
            plt.colorbar()
            
            plt.tight_layout()
            grad_fig_path = os.path.join(output_dir, f"gradient_mask_comparison_{timestamp}.png")
            plt.savefig(grad_fig_path, dpi=300)
            plt.close()
            print(f"Gradient mask comparison saved to: {grad_fig_path}")
        
        print("\nCheck completed!")
        
    except Exception as e:
        print(f"Error during check: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_terrain_masking()