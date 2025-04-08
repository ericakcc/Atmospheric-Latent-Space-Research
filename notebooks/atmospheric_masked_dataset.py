import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class MaskedAtmosphericDataset(Dataset):
    def __init__(self, data_dir, file_list, shape=(512, 768, 3, 94), dtype=np.float32, transform=None, max_samples=None):
        """
        初始化數據集，從.dat文件加載數據，保留NaN和Inf值作為掩碼
        
        參數:
            data_dir (str): 包含.dat文件的目錄路徑
            file_list (list): 要使用的.dat文件列表
            shape (tuple): 每個.dat文件的預期形狀
            dtype: 讀取文件的數據類型
            transform: 可選的數據轉換
            max_samples: 每個文件要加載的最大時間點數量（None表示加載所有）
        """
        self.transform = transform
        
        # 加載所有數據到內存
        self.all_samples = []  # 原始數據（NaN/Inf被替換為0）
        self.all_masks = []    # 掩碼（1表示有效值，0表示NaN/Inf）
        print(f"正在加載 {len(file_list)} 個.dat文件到內存...")
        
        for dat_file in file_list:
            file_path = os.path.join(data_dir, dat_file)
            try:
                # 加載數據
                data = np.fromfile(file_path, dtype=dtype)
                data = data.reshape(shape, order='F')
                
                # 統計原始數據中的無效值
                nan_count = np.isnan(data).sum()
                inf_count = np.isinf(data).sum()
                total_invalid = nan_count + inf_count
                print(f"{dat_file} 原始數據統計:")
                print(f"  NaN數量: {nan_count}")
                print(f"  Inf數量: {inf_count}")
                print(f"  總無效像素: {total_invalid} / {data.size} ({total_invalid/data.size*100:.2f}%)")
                
                # 處理每個時間點
                sample_count = 0
                for t in range(shape[3]):
                    if max_samples is not None and sample_count >= max_samples:
                        break
                        
                    # 獲取當前時間點的數據
                    sample = data[:, :, :, t].copy()
                    
                    # 創建掩碼：0表示NaN/Inf，1表示有效值
                    invalid_mask = np.isnan(sample) | np.isinf(sample)
                    valid_mask = ~invalid_mask
                    
                    # 將掩碼轉換為float32並保存（1表示有效數據，0表示無效）
                    mask = valid_mask.astype(np.float32)
                    
                    # 為了可視化目的，我們需要替換NaN/Inf為某個值
                    # 但我們會保留掩碼以知道哪些像素原本是NaN/Inf
                    vis_sample = sample.copy()
                    vis_sample[invalid_mask] = 0.0  # 僅為了可視化而替換
                    
                    # 只對有效部分標準化到[0, 1]
                    if np.any(valid_mask):  # 檢查是否有任何有效值
                        valid_min = np.min(vis_sample[valid_mask])
                        valid_max = np.max(vis_sample[valid_mask])
                        if valid_max > valid_min:
                            # 只標準化有效部分
                            normalized = np.zeros_like(vis_sample)
                            normalized[valid_mask] = (vis_sample[valid_mask] - valid_min) / (valid_max - valid_min)
                            vis_sample = normalized
                    
                    # 轉換為(C, H, W)格式
                    vis_sample = np.transpose(vis_sample, (2, 0, 1))
                    mask = np.transpose(mask, (2, 0, 1))
                    
                    # 轉換為張量
                    vis_sample = torch.from_numpy(vis_sample.astype(np.float32))
                    mask_tensor = torch.from_numpy(mask)
                    
                    # 存儲樣本和掩碼
                    self.all_samples.append(vis_sample)
                    self.all_masks.append(mask_tensor)
                    sample_count += 1
                
                print(f"從 {dat_file} 加載了 {sample_count} 個時間點")
                
            except Exception as e:
                print(f"加載 {dat_file} 時出錯: {str(e)}")
                continue
        
        if not self.all_samples:
            raise RuntimeError("沒有加載有效樣本")
        
        # 計算有效地形的比例
        total_valid = sum(mask.sum().item() for mask in self.all_masks)
        total_pixels = sum(mask.numel() for mask in self.all_masks)
        valid_ratio = total_valid / total_pixels
        print(f"總加載樣本: {len(self.all_samples)}")
        print(f"有效地形像素比例: {valid_ratio:.2%}")
    
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        mask = self.all_masks[idx]
        
        if self.transform:
            # 注意：需要確保轉換同時適用於樣本和掩碼
            sample = self.transform(sample)
            # 對於某些轉換（如調整大小），需要對掩碼進行相同的處理
            if isinstance(self.transform, transforms.Compose):
                for t in self.transform.transforms:
                    if isinstance(t, transforms.Resize):
                        mask = t(mask)
            
        return sample, mask  # 返回樣本和對應的掩碼

def create_masked_train_test_datasets(data_dir, shape=(512, 768, 3, 94), dtype=np.float32, 
                               train_ratio=0.8, seed=42, max_samples=None):
    """
    創建包含掩碼的訓練和測試數據集
    
    參數:
        data_dir (str): 包含.dat文件的目錄
        shape (tuple): 每個.dat文件的預期形狀
        dtype: 讀取文件的數據類型
        train_ratio (float): 用於訓練的文件比例
        seed (int): 隨機種子以確保可重現性
        max_samples: 每個文件要加載的最大時間點數量
    
    返回:
        tuple: (train_dataset, test_dataset)
    """
    # 獲取所有.dat文件
    dat_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    if not dat_files:
        raise ValueError(f"在 {data_dir} 中沒有找到.dat文件")
    
    print(f"總共找到 {len(dat_files)} 個.dat文件")
    
    # 隨機打亂文件
    random.seed(seed)
    random.shuffle(dat_files)
    
    # 將文件分割為訓練集和測試集
    train_size = int(len(dat_files) * train_ratio)
    train_files = dat_files[:train_size]
    test_files = dat_files[train_size:]
    
    print(f"訓練文件: {len(train_files)}")
    print(f"測試文件: {len(test_files)}")
    
    transform = transforms.Compose([
        transforms.Resize((256, 384), antialias=True),
        transforms.Lambda(lambda x: x.clone())  
    ])
    
    train_dataset = MaskedAtmosphericDataset(data_dir, train_files, shape, dtype, 
                                     transform=transform, max_samples=max_samples)
    test_dataset = MaskedAtmosphericDataset(data_dir, test_files, shape, dtype, 
                                    transform=transform, max_samples=max_samples)
    
    return train_dataset, test_dataset

def masked_loss_function(pred, target, mask, loss_type='mse'):
    """
    計算僅在有效像素（掩碼==1）上的損失
    
    參數:
        pred: 預測張量
        target: 目標張量
        mask: 掩碼張量（1表示有效像素，0表示無效）
        loss_type: 要使用的損失類型（'mse', 'bce', 'l1'）
    
    返回:
        掩碼損失
    """
    import torch.nn.functional as F
    
    # 對預測和目標應用掩碼
    masked_pred = pred * mask
    masked_target = target * mask
    
    # 計算有效像素數量用於標準化
    valid_pixels = mask.sum() + 1e-8  # 添加小常數以避免除以零
    
    if loss_type == 'mse':
        # 掩碼數據上的均方誤差
        loss = F.mse_loss(masked_pred, masked_target, reduction='sum') / valid_pixels
    elif loss_type == 'bce':
        # 掩碼數據上的二元交叉熵
        loss = F.binary_cross_entropy(masked_pred, masked_target, reduction='sum') / valid_pixels
    elif loss_type == 'l1':
        # 掩碼數據上的L1損失
        loss = F.l1_loss(masked_pred, masked_target, reduction='sum') / valid_pixels
    else:
        raise ValueError(f"不支持的損失類型: {loss_type}")
    
    return loss

# 用法示例:
if __name__ == "__main__":
    data_dir = "data/dcape/"
    train_dataset, test_dataset = create_masked_train_test_datasets(
        data_dir=data_dir,
        shape=(512, 768, 3, 94),
        dtype=np.float32,
        train_ratio=0.8,
        max_samples=5  # 限制每個文件的時間點數量以減少內存使用
    )
    
    # 創建資料加載器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,
        shuffle=True,
        num_workers=0
    )
    
    # 獲取一個批次的數據
    sample_batch, mask_batch = next(iter(train_loader))
    print(f"樣本批次形狀: {sample_batch.shape}")
    print(f"掩碼批次形狀: {mask_batch.shape}")
    print(f"有效地形像素比例: {mask_batch.sum() / mask_batch.numel():.2%}") 