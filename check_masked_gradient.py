import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import sys
from datetime import datetime

# 添加項目根目錄到路徑
sys.path.append('/home/erica/Atmospheric-Latent-Space-Research')

# 導入項目模塊
from atmospheric_vae.config.experiment_config import ExperimentConfig

class RawAtmosphericDataset(Dataset):
    """
    A dataset that preserves NaN and Inf values as they are, and creates masks from them
    """
    def __init__(self, data_dir, file_list, shape=(512, 768, 3, 94), dtype=np.float32, transform=None, max_samples=5):
        """
        Initialize dataset by loading data without converting NaN/Inf to zeros
        
        Args:
            data_dir (str): Directory path containing .dat files
            file_list (list): List of .dat files to use
            shape (tuple): Expected shape of each .dat file
            dtype: Data type for reading files
            transform: Optional transform to be applied on a sample
            max_samples: Maximum number of time points to load (to limit memory usage)
        """
        self.transform = transform
        
        # Load all data into memory
        self.all_samples = []
        self.all_masks = []
        print(f"Loading {len(file_list)} .dat files into memory (limited to {max_samples} time points)...")
        
        for dat_file in file_list:
            file_path = os.path.join(data_dir, dat_file)
            try:
                # Load data without processing NaN/Inf
                data = np.fromfile(file_path, dtype=dtype)
                data = data.reshape(shape, order='F')
                
                # Print original stats before any processing
                nan_count = np.isnan(data).sum()
                inf_count = np.isinf(data).sum()
                print(f"Original data stats for {dat_file}:")
                print(f"  NaN count: {nan_count}")
                print(f"  Inf count: {inf_count}")
                print(f"  Total invalid pixels: {nan_count + inf_count} out of {data.size} ({(nan_count + inf_count)/data.size*100:.2f}%)")
                
                # Preprocess only a limited number of time points
                sample_count = 0
                for t in range(shape[3]):
                    if sample_count >= max_samples:
                        break
                        
                    sample = data[:, :, :, t].copy()
                    
                    # Create mask: 0 for NaN/Inf, 1 for valid values
                    invalid_mask = np.isnan(sample) | np.isinf(sample)
                    valid_mask = ~invalid_mask
                    
                    # Convert mask to float32 and save it (1 for valid data, 0 for invalid)
                    mask = valid_mask.astype(np.float32)
                    
                    # For visualization purposes, we need to replace NaN/Inf with some value
                    # but we'll keep the mask to know which pixels are originally NaN/Inf
                    vis_sample = sample.copy()
                    vis_sample[invalid_mask] = 0.0  # Replace only for visualization
                    
                    # Normalize valid parts to [0, 1]
                    if np.any(valid_mask):  # Check if there are any valid values
                        valid_min = np.min(vis_sample[valid_mask])
                        valid_max = np.max(vis_sample[valid_mask])
                        if valid_max > valid_min:
                            # Only normalize the valid parts
                            normalized = np.zeros_like(vis_sample)
                            normalized[valid_mask] = (vis_sample[valid_mask] - valid_min) / (valid_max - valid_min)
                            vis_sample = normalized
                    
                    # Convert to (C, H, W) format
                    vis_sample = np.transpose(vis_sample, (2, 0, 1))
                    mask = np.transpose(mask, (2, 0, 1))
                    
                    # Convert to tensors
                    vis_sample = torch.from_numpy(vis_sample.astype(np.float32))
                    mask_tensor = torch.from_numpy(mask)
                    
                    # Store both the sample and its mask
                    self.all_samples.append(vis_sample)
                    self.all_masks.append(mask_tensor)
                    sample_count += 1
                
                print(f"Loaded {dat_file} with {sample_count} time points")
                
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
        mask = self.all_masks[idx]
        
        if self.transform:
            sample = self.transform(sample)
            # Apply same transform to mask if needed
        
        return sample, mask  # Return both the sample and its mask

def masked_loss_function(pred, target, mask, loss_type='mse'):
    """
    Calculate loss only on valid pixels (where mask == 1)
    
    Args:
        pred: Prediction tensor
        target: Target tensor
        mask: Mask tensor (1 for valid pixels, 0 for invalid)
        loss_type: Type of loss to use ('mse', 'bce', 'l1')
    
    Returns:
        Masked loss
    """
    # Apply mask to both prediction and target
    masked_pred = pred * mask
    masked_target = target * mask
    
    # Count valid pixels for normalization
    valid_pixels = mask.sum() + 1e-8  # Add small constant to avoid division by zero
    
    if loss_type == 'mse':
        # Mean squared error on masked data
        loss = F.mse_loss(masked_pred, masked_target, reduction='sum') / valid_pixels
    elif loss_type == 'bce':
        # Binary cross entropy on masked data
        loss = F.binary_cross_entropy(masked_pred, masked_target, reduction='sum') / valid_pixels
    elif loss_type == 'l1':
        # L1 loss on masked data
        loss = F.l1_loss(masked_pred, masked_target, reduction='sum') / valid_pixels
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return loss

def test_gradient_masking(sample_batch, mask_batch, output_dir="terrain_mask_analysis"):
    """
    Test if gradients are correctly affected by masking
    """
    print("\nTesting gradient masking...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use only the first sample to reduce memory usage
    sample = sample_batch[0:1].clone()
    mask = mask_batch[0:1].clone()
    
    # Create a copy of the sample that requires gradients
    sample_grad = sample.clone().detach().requires_grad_(True)
    target = sample.clone()  # Use the same data as target for testing
    
    # Try both with and without masking
    print("Comparing gradients with and without masking...")
    
    # 1. Without masking - all pixels contribute
    loss_no_mask = F.mse_loss(sample_grad, target)
    loss_no_mask.backward(retain_graph=True)
    
    # Get gradients
    grad_no_mask = sample_grad.grad.clone()
    
    # Reset gradients
    sample_grad.grad.zero_()
    
    # 2. With masking - only valid pixels contribute
    loss_with_mask = masked_loss_function(sample_grad, target, mask)
    loss_with_mask.backward()
    
    # Get gradients
    grad_with_mask = sample_grad.grad.clone()
    
    # Analyze gradient differences
    print(f"Loss without mask: {loss_no_mask.item():.6f}")
    print(f"Loss with mask: {loss_with_mask.item():.6f}")
    
    # Create binary masks of non-zero gradients
    grad_no_mask_binary = (grad_no_mask != 0).float()
    grad_with_mask_binary = (grad_with_mask != 0).float()
    
    # Check if masked gradients are zero in invalid regions
    invalid_regions = (mask == 0).float()
    grad_in_invalid = (grad_with_mask_binary * invalid_regions).sum()
    
    print(f"Gradient statistics:")
    print(f"  Non-zero gradient pixels without mask: {grad_no_mask_binary.sum().item()}")
    print(f"  Non-zero gradient pixels with mask: {grad_with_mask_binary.sum().item()}")
    print(f"  Non-zero gradient pixels in invalid regions: {grad_in_invalid.item()}")
    
    if grad_in_invalid.item() == 0:
        print("SUCCESS: No gradients in invalid regions - masking is working correctly!")
    else:
        print("WARNING: There are gradients in invalid regions - masking may not be working correctly.")
    
    # Visualize the gradients
    plt.figure(figsize=(15, 5))
    
    # First channel for visualization
    channel_idx = 0
    
    # Gradient without mask
    plt.subplot(131)
    plt.imshow(grad_no_mask_binary[0, channel_idx].detach().cpu().numpy())
    plt.title("Gradient Mask\n(Without Masking Loss)")
    plt.colorbar()
    
    # Gradient with mask
    plt.subplot(132)
    plt.imshow(grad_with_mask_binary[0, channel_idx].detach().cpu().numpy())
    plt.title("Gradient Mask\n(With Masking Loss)")
    plt.colorbar()
    
    # Terrain mask
    plt.subplot(133)
    plt.imshow(mask[0, channel_idx].detach().cpu().numpy(), cmap='gray')
    plt.title("Terrain Mask\n(1=valid, 0=invalid)")
    plt.colorbar()
    
    plt.tight_layout()
    grad_fig_path = os.path.join(output_dir, f"gradient_mask_comparison_{timestamp}.png")
    plt.savefig(grad_fig_path, dpi=300)
    plt.close()
    print(f"Gradient comparison saved to: {grad_fig_path}")
    
    # Detailed comparison of mask and gradient
    plt.figure(figsize=(15, 5))
    
    # Terrain mask as reference
    plt.subplot(131)
    plt.imshow(mask[0, channel_idx].detach().cpu().numpy(), cmap='gray')
    plt.title("Terrain Mask\n(1=valid, 0=invalid)")
    plt.colorbar()
    
    # Gradient only in valid regions
    valid_grad = grad_with_mask_binary[0, channel_idx] * mask[0, channel_idx]
    plt.subplot(132)
    plt.imshow(valid_grad.detach().cpu().numpy())
    plt.title("Gradient in Valid Regions\n(Should match terrain mask)")
    plt.colorbar()
    
    # Difference (should be zero)
    terrain_mask = mask[0, channel_idx]
    diff = grad_with_mask_binary[0, channel_idx] - terrain_mask
    plt.subplot(133)
    plt.imshow(diff.detach().cpu().numpy(), cmap='bwr')
    plt.title("Difference\n(Gradient - Terrain Mask)")
    plt.colorbar()
    
    plt.tight_layout()
    detail_fig_path = os.path.join(output_dir, f"gradient_detail_comparison_{timestamp}.png")
    plt.savefig(detail_fig_path, dpi=300)
    plt.close()
    print(f"Detailed gradient-mask comparison saved to: {detail_fig_path}")
    
    return grad_with_mask, mask

def analyze_terrain_masking(data_dir="data/dcape/"):
    """
    Analyze terrain masks created from NaN/Inf values
    """
    # 創建輸出目錄
    output_dir = "terrain_mask_analysis"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("Loading dataset with preserved NaN/Inf values...")
    try:
        # 獲取所有.dat文件並只選擇第一個用於測試
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
        if not all_files:
            print(f"No .dat files found in {data_dir}")
            return
        
        # 只使用第一個文件創建數據集，並限制時間點數量
        sample_dataset = RawAtmosphericDataset(
            data_dir=data_dir,
            file_list=[all_files[0]],
            shape=(512, 768, 3, 94),
            dtype=np.float32,
            max_samples=5  # 只使用5個時間點
        )
        
        # 創建數據加載器，使用較小的batch size
        sample_loader = DataLoader(
            sample_dataset,
            batch_size=2,  # 使用較小的batch size
            shuffle=False
        )
        
        # 獲取一個批次的數據和掩碼
        sample_batch, mask_batch = next(iter(sample_loader))
        
        print(f"Loaded data shape: {sample_batch.shape}")
        print(f"Mask shape: {mask_batch.shape}")
        
        # 計算有效地形的比例
        valid_ratio = mask_batch.sum() / mask_batch.numel()
        print(f"Valid terrain pixel ratio: {valid_ratio.item():.2%}")
        
        # 分析幾個時間點的地形掩碼
        plt.figure(figsize=(15, 5))
        num_samples = min(3, len(sample_dataset))
        
        for i in range(num_samples):
            sample, mask = sample_dataset[i]
            
            # 選取第一個通道的掩碼（所有通道的掩碼應該相同）
            mask_channel = mask[0].numpy()
            
            # 分析掩碼的分佈
            valid_count = np.count_nonzero(mask_channel)
            total_count = mask_channel.size
            valid_percentage = (valid_count / total_count) * 100
            
            # 繪製掩碼
            plt.subplot(1, num_samples, i+1)
            plt.imshow(mask_channel, cmap='gray')
            plt.title(f"Terrain Mask (Time {i})\nValid: {valid_percentage:.2f}%")
            plt.colorbar()
        
        plt.tight_layout()
        mask_fig_path = os.path.join(output_dir, f"terrain_masks_{timestamp}.png")
        plt.savefig(mask_fig_path, dpi=300)
        plt.close()
        print(f"Terrain masks visualization saved to: {mask_fig_path}")
        
        # 可視化原始數據與掩碼後的數據
        plt.figure(figsize=(15, 10))
        sample_idx = 0  # 使用第一個樣本
        
        sample = sample_batch[sample_idx].numpy()
        mask = mask_batch[sample_idx].numpy()
        
        for channel in range(sample.shape[0]):
            # 原始數據（已替換NaN/Inf為0）
            plt.subplot(3, 3, channel*3+1)
            plt.imshow(sample[channel])
            plt.title(f"Channel {channel+1}\nOriginal (NaN/Inf set to 0)")
            plt.colorbar()
            
            # 掩碼
            plt.subplot(3, 3, channel*3+2)
            plt.imshow(mask[channel], cmap='gray')
            plt.title(f"Channel {channel+1}\nMask (1=valid, 0=NaN/Inf)")
            plt.colorbar()
            
            # 應用掩碼後的數據
            plt.subplot(3, 3, channel*3+3)
            masked_data = sample[channel] * mask[channel]
            plt.imshow(masked_data)
            plt.title(f"Channel {channel+1}\nMasked Data")
            plt.colorbar()
        
        plt.tight_layout()
        data_fig_path = os.path.join(output_dir, f"data_with_masks_{timestamp}.png")
        plt.savefig(data_fig_path, dpi=300)
        plt.close()
        print(f"Data with masks visualization saved to: {data_fig_path}")
        
        # 測試梯度掩碼效果
        test_gradient_masking(sample_batch, mask_batch, output_dir)
        
        # 返回樣本和掩碼供後續處理
        return sample_batch, mask_batch
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    analyze_terrain_masking() 