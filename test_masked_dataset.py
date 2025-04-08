import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from datetime import datetime

# 導入我們的掩碼數據集
from notebooks.atmospheric_masked_dataset import MaskedAtmosphericDataset, create_masked_train_test_datasets

def test_masked_dataset(data_dir="data/dcape/", max_samples=3):
    """測試掩碼數據集的加載和掩碼創建"""
    
    print("測試掩碼數據集...")
    
    # 創建輸出目錄
    output_dir = "mask_test_output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 獲取所有.dat文件並只選擇第一個用於測試
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    if not all_files:
        print(f"在 {data_dir} 中沒有找到.dat文件")
        return
    
    # 只使用第一個文件創建數據集
    sample_dataset = MaskedAtmosphericDataset(
        data_dir=data_dir,
        file_list=[all_files[0]],
        shape=(512, 768, 3, 94),
        dtype=np.float32,
        max_samples=max_samples  # 限制時間點數量
    )
    
    # 創建數據加載器
    sample_loader = DataLoader(
        sample_dataset,
        batch_size=1,  # 使用較小的批次大小
        shuffle=False
    )
    
    # 獲取一個批次的數據和掩碼
    sample_batch, mask_batch = next(iter(sample_loader))
    
    print(f"加載的數據形狀: {sample_batch.shape}")
    print(f"掩碼形狀: {mask_batch.shape}")
    
    # 計算有效地形的比例
    valid_ratio = mask_batch.sum() / mask_batch.numel()
    print(f"有效地形像素比例: {valid_ratio.item():.2%}")
    
    # 可視化原始數據與掩碼
    plt.figure(figsize=(15, 10))
    
    sample = sample_batch[0].numpy()  # [C, H, W]
    mask = mask_batch[0].numpy()      # [C, H, W]
    
    for channel in range(sample.shape[0]):
        # 原始數據（NaN/Inf已替換為0）
        plt.subplot(3, 3, channel*3+1)
        plt.imshow(sample[channel])
        plt.title(f"通道 {channel+1}\n原始數據 (NaN/Inf已設為0)")
        plt.colorbar()
        
        # 掩碼
        plt.subplot(3, 3, channel*3+2)
        plt.imshow(mask[channel], cmap='gray')
        plt.title(f"通道 {channel+1}\n掩碼 (1=有效, 0=NaN/Inf)")
        plt.colorbar()
        
        # 應用掩碼後的數據
        plt.subplot(3, 3, channel*3+3)
        masked_data = sample[channel] * mask[channel]
        plt.imshow(masked_data)
        plt.title(f"通道 {channel+1}\n掩碼後數據")
        plt.colorbar()
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"masked_data_visualization_{timestamp}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"已保存可視化結果到: {fig_path}")
    
    # 測試所有樣本的掩碼統計
    all_valid_ratios = []
    
    for i in range(len(sample_dataset)):
        sample, mask = sample_dataset[i]
        valid_ratio = mask.sum().item() / mask.numel()
        all_valid_ratios.append(valid_ratio)
        print(f"樣本 {i+1}: 有效地形比例 = {valid_ratio:.2%}")
    
    # 輸出掩碼統計摘要
    avg_valid = np.mean(all_valid_ratios)
    min_valid = np.min(all_valid_ratios)
    max_valid = np.max(all_valid_ratios)
    
    print("\n掩碼統計摘要:")
    print(f"平均有效地形比例: {avg_valid:.2%}")
    print(f"最小有效地形比例: {min_valid:.2%}")
    print(f"最大有效地形比例: {max_valid:.2%}")
    
    return sample_batch, mask_batch

def test_create_datasets(data_dir="data/dcape/", max_samples=3):
    """測試創建訓練和測試數據集函數"""
    
    print("\n測試創建訓練和測試數據集...")
    
    # 創建訓練和測試數據集
    train_dataset, test_dataset = create_masked_train_test_datasets(
        data_dir=data_dir,
        shape=(512, 768, 3, 94),
        dtype=np.float32,
        train_ratio=0.8,
        seed=42,
        max_samples=max_samples
    )
    
    print(f"訓練集大小: {len(train_dataset)} 樣本")
    print(f"測試集大小: {len(test_dataset)} 樣本")
    
    # 創建數據加載器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    # 獲取一個批次檢查形狀
    sample_batch, mask_batch = next(iter(train_loader))
    
    print(f"批次數據形狀: {sample_batch.shape}")
    print(f"批次掩碼形狀: {mask_batch.shape}")
    
    # 檢查批次中的有效地形比例
    valid_ratio = mask_batch.sum() / mask_batch.numel()
    print(f"批次中有效地形像素比例: {valid_ratio.item():.2%}")
    
    return train_dataset, test_dataset

if __name__ == "__main__":
    # 測試單個數據集
    sample_batch, mask_batch = test_masked_dataset(max_samples=5)
    
    # 測試創建完整的數據集
    train_dataset, test_dataset = test_create_datasets(max_samples=5) 