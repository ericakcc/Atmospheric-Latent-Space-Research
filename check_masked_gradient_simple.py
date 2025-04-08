import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 設置輸出目錄
output_dir = "terrain_mask_analysis"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def create_synthetic_data(height=32, width=32):
    """
    創建合成數據和掩碼，模擬地形數據（使用更小的尺寸）
    """
    # 創建一個簡單的示例數據
    channels = 1  # 只使用一個通道以減少數據量
    batch_size = 1
    
    # 創建一個隨機數據
    sample = torch.rand(batch_size, channels, height, width)
    
    # 創建一個有差異的目標
    target = sample.clone() + 0.1 * torch.rand(batch_size, channels, height, width)
    target = torch.clamp(target, 0, 1)  # 確保值在[0,1]範圍內
    
    # 創建一個掩碼，模擬地形（中間區域有效，邊緣無效）
    mask = torch.zeros(batch_size, channels, height, width)
    
    # 圓形有效區域
    center_h, center_w = height // 2, width // 2
    radius = min(height, width) // 3
    
    for h in range(height):
        for w in range(width):
            if ((h - center_h)**2 + (w - center_w)**2) < radius**2:
                mask[:, :, h, w] = 1.0
    
    print(f"Created synthetic data with shape {sample.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # 掩碼覆蓋比例
    valid_ratio = mask.sum() / mask.numel()
    print(f"Valid terrain pixel ratio: {valid_ratio.item():.2%}")
    
    return sample, target, mask

def masked_loss_function(pred, target, mask, loss_type='mse'):
    """
    Calculate loss only on valid pixels (where mask == 1)
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

def test_gradient_masking():
    """
    Test if gradients are correctly affected by masking
    """
    print("\nTesting gradient masking with synthetic data...")
    
    # Create synthetic data with different input and target
    sample_batch, target_batch, mask_batch = create_synthetic_data()
    
    # Visualize the data and mask
    plt.figure(figsize=(15, 5))
    
    # First channel for visualization (we only have one channel)
    channel_idx = 0
    
    # Sample data
    plt.subplot(131)
    plt.imshow(sample_batch[0, channel_idx].detach().cpu().numpy())
    plt.title("Input Data")
    plt.colorbar()
    
    # Target data
    plt.subplot(132)
    plt.imshow(target_batch[0, channel_idx].detach().cpu().numpy())
    plt.title("Target Data")
    plt.colorbar()
    
    # Mask
    plt.subplot(133)
    plt.imshow(mask_batch[0, channel_idx].detach().cpu().numpy(), cmap='gray')
    plt.title("Terrain Mask\n(1=valid, 0=invalid)")
    plt.colorbar()
    
    plt.tight_layout()
    data_fig_path = os.path.join(output_dir, f"synthetic_data_{timestamp}.png")
    plt.savefig(data_fig_path, dpi=300)
    plt.close()
    print(f"Synthetic data visualization saved to: {data_fig_path}")
    
    # Visualize masked data
    plt.figure(figsize=(15, 5))
    
    # Input with mask
    plt.subplot(131)
    masked_input = sample_batch[0, channel_idx] * mask_batch[0, channel_idx]
    plt.imshow(masked_input.detach().cpu().numpy())
    plt.title("Masked Input")
    plt.colorbar()
    
    # Target with mask
    plt.subplot(132)
    masked_target = target_batch[0, channel_idx] * mask_batch[0, channel_idx]
    plt.imshow(masked_target.detach().cpu().numpy())
    plt.title("Masked Target")
    plt.colorbar()
    
    # Difference (masked)
    plt.subplot(133)
    diff = masked_input - masked_target
    plt.imshow(diff.detach().cpu().numpy(), cmap='bwr')
    plt.title("Masked Difference\n(Input - Target)")
    plt.colorbar()
    
    plt.tight_layout()
    masked_fig_path = os.path.join(output_dir, f"masked_data_{timestamp}.png")
    plt.savefig(masked_fig_path, dpi=300)
    plt.close()
    print(f"Masked data visualization saved to: {masked_fig_path}")
    
    # Test gradients
    sample_grad = sample_batch.clone().detach().requires_grad_(True)
    
    # Try both with and without masking
    print("Comparing gradients with and without masking...")
    
    # 1. Without masking - all pixels contribute
    loss_no_mask = F.mse_loss(sample_grad, target_batch)
    loss_no_mask.backward(retain_graph=True)
    
    # Get gradients
    grad_no_mask = sample_grad.grad.clone()
    
    # Reset gradients
    sample_grad.grad.zero_()
    
    # 2. With masking - only valid pixels contribute
    loss_with_mask = masked_loss_function(sample_grad, target_batch, mask_batch)
    loss_with_mask.backward()
    
    # Get gradients
    grad_with_mask = sample_grad.grad.clone()
    
    # Analyze gradient differences
    print(f"Loss without mask: {loss_no_mask.item():.6f}")
    print(f"Loss with mask: {loss_with_mask.item():.6f}")
    
    # Create binary masks of non-zero gradients
    grad_no_mask_binary = (grad_no_mask.abs() > 1e-6).float()  # Use threshold to handle very small values
    grad_with_mask_binary = (grad_with_mask.abs() > 1e-6).float()
    
    # Check if masked gradients are zero in invalid regions
    invalid_regions = (mask_batch == 0).float()
    grad_in_invalid = (grad_with_mask_binary * invalid_regions).sum()
    
    print(f"Gradient statistics:")
    print(f"  Non-zero gradient pixels without mask: {grad_no_mask_binary.sum().item()}")
    print(f"  Non-zero gradient pixels with mask: {grad_with_mask_binary.sum().item()}")
    print(f"  Non-zero gradient pixels in invalid regions: {grad_in_invalid.item()}")
    
    if grad_in_invalid.item() == 0:
        print("SUCCESS: No gradients in invalid regions - masking is working correctly!")
    else:
        print(f"WARNING: Found {grad_in_invalid.item()} gradient pixels in invalid regions!")
    
    # Visualize the gradients
    plt.figure(figsize=(15, 5))
    
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
    plt.imshow(mask_batch[0, channel_idx].detach().cpu().numpy(), cmap='gray')
    plt.title("Terrain Mask\n(1=valid, 0=invalid)")
    plt.colorbar()
    
    plt.tight_layout()
    grad_fig_path = os.path.join(output_dir, f"gradient_mask_comparison_{timestamp}.png")
    plt.savefig(grad_fig_path, dpi=300)
    plt.close()
    print(f"Gradient comparison saved to: {grad_fig_path}")
    
    # Visualize the gradient values themselves (not just binary)
    plt.figure(figsize=(15, 5))
    
    # Gradient without mask
    plt.subplot(131)
    plt.imshow(grad_no_mask[0, channel_idx].detach().cpu().numpy())
    plt.title("Gradient Values\n(Without Masking)")
    plt.colorbar()
    
    # Gradient with mask
    plt.subplot(132)
    plt.imshow(grad_with_mask[0, channel_idx].detach().cpu().numpy())
    plt.title("Gradient Values\n(With Masking)")
    plt.colorbar()
    
    # Difference between gradients
    plt.subplot(133)
    diff = grad_with_mask[0, channel_idx] - grad_no_mask[0, channel_idx]
    plt.imshow(diff.detach().cpu().numpy(), cmap='bwr')
    plt.title("Gradient Difference\n(Masked - Unmasked)")
    plt.colorbar()
    
    plt.tight_layout()
    grad_val_path = os.path.join(output_dir, f"gradient_values_comparison_{timestamp}.png")
    plt.savefig(grad_val_path, dpi=300)
    plt.close()
    print(f"Gradient values comparison saved to: {grad_val_path}")
    
    # Test with higher precision
    # Create a flattened view of the data for easier analysis
    flat_grad = grad_with_mask.view(-1).detach().cpu().numpy()
    flat_mask = mask_batch.view(-1).detach().cpu().numpy()
    
    # Find the non-zero gradients in invalid regions
    invalid_mask = flat_mask == 0
    invalid_grad = flat_grad[invalid_mask]
    
    if len(invalid_grad) > 0:
        non_zero_invalid = invalid_grad[np.abs(invalid_grad) > 1e-6]
        print(f"Analysis of gradients in invalid regions:")
        print(f"  Total invalid pixels: {len(invalid_grad)}")
        print(f"  Non-zero gradient invalid pixels: {len(non_zero_invalid)}")
        if len(non_zero_invalid) > 0:
            print(f"  Min gradient in invalid regions: {non_zero_invalid.min()}")
            print(f"  Max gradient in invalid regions: {non_zero_invalid.max()}")
            print(f"  Mean abs gradient in invalid regions: {np.abs(non_zero_invalid).mean()}")
    
    return grad_with_mask, mask_batch

if __name__ == "__main__":
    test_gradient_masking() 