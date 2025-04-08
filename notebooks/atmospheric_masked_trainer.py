import os
import torch
import torch.nn.functional as F
from torchvision import utils

def masked_vae_loss_function(recon_x, x, mu, logvar, mask, config):
    """
    使用掩碼計算VAE損失，僅考慮有效地形區域
    
    參數:
        recon_x: 重建的輸入
        x: 原始輸入
        mu: 均值向量
        logvar: 對數方差向量
        mask: 掩碼（1表示有效地形區域，0表示NaN/Inf區域）
        config: 配置對象，包含損失權重和beta值
    
    返回:
        計算得到的總損失
    """
    # 對重建和原始數據應用掩碼
    masked_recon = recon_x * mask
    masked_x = x * mask
    
    # 計算有效像素數量用於標準化
    valid_pixels = mask.sum() + 1e-8  # 添加小常數以避免除以零
    
    loss = 0
    
    # BCE損失（僅在有效區域）
    if config.training_config["loss_weights"]["bce"] > 0:
        BCE = F.binary_cross_entropy(masked_recon, masked_x, reduction='sum') / valid_pixels
        loss += config.training_config["loss_weights"]["bce"] * BCE
    
    # MSE損失（僅在有效區域）
    if config.training_config["loss_weights"]["mse"] > 0:
        MSE = F.mse_loss(masked_recon, masked_x, reduction='sum') / valid_pixels
        loss += config.training_config["loss_weights"]["mse"] * MSE
    
    # L1損失（僅在有效區域）
    if config.training_config["loss_weights"]["l1"] > 0:
        L1 = F.l1_loss(masked_recon, masked_x, reduction='sum') / valid_pixels
        loss += config.training_config["loss_weights"]["l1"] * L1
    
    # KL散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # 總損失 = 重建損失 + beta * KL散度
    return loss + config.training_config["beta"] * KLD

def train_masked_epoch(model, device, train_loader, optimizer, epoch, config, logger, log_interval=1):
    """
    使用掩碼數據集訓練VAE一個epoch
    
    參數:
        model: 要訓練的VAE模型
        device: 計算設備（CPU/GPU）
        train_loader: 提供掩碼數據的DataLoader
        optimizer: 優化器
        epoch: 當前epoch
        config: 配置對象
        logger: 實驗記錄器
        log_interval: 多少批次記錄一次
    
    返回:
        平均訓練損失
    """
    model.train()
    train_loss = 0
    
    # 創建調整大小的轉換
    input_height = config.model_config.get("input_height", 256)
    input_width = config.model_config.get("input_width", 384)
    resize_transform = torch.nn.Sequential(
        torch.nn.Upsample(size=(input_height, input_width), mode='bilinear', align_corners=False)
    ).to(device)
    
    for batch_idx, (data, mask) in enumerate(train_loader):
        data = data.to(device)
        mask = mask.to(device)
        
        # 調整數據和掩碼大小以匹配模型配置
        if data.shape[2] != input_height or data.shape[3] != input_width:
            data = resize_transform(data)
            mask = resize_transform(mask)
            print(f"已將數據調整為形狀: {data.shape}")
        
        optimizer.zero_grad()
        
        # 前向傳播
        recon_batch, mu, logvar = model(data)
        
        # 使用掩碼計算損失
        loss = masked_vae_loss_function(recon_batch, data, mu, logvar, mask, config)
        
        # 反向傳播和優化
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        # 記錄和可視化
        if batch_idx % log_interval == 0:
            print(f'訓練Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t損失: {loss.item():.6f}')
            
            # 每個epoch的第一個批次保存重建圖像
            if batch_idx == 0:
                n = min(data.size(0), 8)
                
                # 為可視化目的，對原始和重建的圖像應用掩碼
                masked_data = data[:n] * mask[:n]
                masked_recon = recon_batch[:n] * mask[:n]
                
                # 創建比較圖
                comparison = torch.cat([masked_data, masked_recon])
                save_path = os.path.join(logger.exp_dir, f"reconstruction_epoch_{epoch}.png")
                utils.save_image(comparison.cpu(), save_path, nrow=n)
                
                # 確保張量維度匹配
                # 注意：這裡是錯誤發生的地方，需要確保數據通道維度匹配
                unmasked_recon = recon_batch[:n]
                
                # 檢查並確保數據維度匹配
                if data[:n].shape[1] == unmasked_recon.shape[1]:
                    unmasked_comparison = torch.cat([data[:n], unmasked_recon])
                    unmasked_save_path = os.path.join(logger.exp_dir, f"unmasked_reconstruction_epoch_{epoch}.png")
                    utils.save_image(unmasked_comparison.cpu(), unmasked_save_path, nrow=n)
                else:
                    print(f"警告：數據通道不匹配 - 原始數據: {data[:n].shape}, 重建數據: {unmasked_recon.shape}")
                    # 如果通道不匹配，就只保存重建結果
                    unmasked_save_path = os.path.join(logger.exp_dir, f"unmasked_recon_epoch_{epoch}.png")
                    utils.save_image(unmasked_recon.cpu(), unmasked_save_path, nrow=n)
                
                # 保存掩碼本身作為參考
                mask_save_path = os.path.join(logger.exp_dir, f"masks_epoch_{epoch}.png")
                utils.save_image(mask[:n].cpu(), mask_save_path, nrow=n)
    
    # 計算平均損失
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} 平均損失: {avg_loss:.4f}')
    
    return avg_loss

def test_masked_epoch(model, device, test_loader, config):
    """
    使用掩碼數據集評估VAE模型
    
    參數:
        model: 要評估的VAE模型
        device: 計算設備（CPU/GPU）
        test_loader: 提供掩碼數據的測試DataLoader
        config: 配置對象
    
    返回:
        平均測試損失
    """
    model.eval()
    test_loss = 0
    
    # 創建調整大小的轉換
    input_height = config.model_config.get("input_height", 256)
    input_width = config.model_config.get("input_width", 384)
    resize_transform = torch.nn.Sequential(
        torch.nn.Upsample(size=(input_height, input_width), mode='bilinear', align_corners=False)
    ).to(device)
    
    with torch.no_grad():
        for data, mask in test_loader:
            data = data.to(device)
            mask = mask.to(device)
            
            # 調整數據和掩碼大小以匹配模型配置
            if data.shape[2] != input_height or data.shape[3] != input_width:
                data = resize_transform(data)
                mask = resize_transform(mask)
            
            # 前向傳播
            recon, mu, logvar = model(data)
            
            # 使用掩碼計算損失
            test_loss += masked_vae_loss_function(recon, data, mu, logvar, mask, config).item()
    
    # 計算平均損失
    test_loss /= len(test_loader.dataset)
    print(f'====> 測試集損失: {test_loss:.4f}')
    
    return test_loss

def visualize_masked_reconstructions(model, data_loader, device, output_dir, max_samples=8, prefix="", input_height=None, input_width=None):
    """
    在測試集上可視化掩碼重建結果
    
    參數:
        model: 要評估的VAE模型
        data_loader: 提供掩碼數據的DataLoader
        device: 計算設備（CPU/GPU）
        output_dir: 保存可視化結果的目錄
        max_samples: 要可視化的最大樣本數
        prefix: 輸出文件名前綴
        input_height: 可選的輸入高度，如果提供則優先使用
        input_width: 可選的輸入寬度，如果提供則優先使用
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # 創建調整大小的轉換 - 優先使用直接傳入的參數
    if input_height is not None and input_width is not None:
        # 使用直接提供的參數
        pass
    # 如果沒有直接提供，嘗試從模型獲取
    elif hasattr(model, 'config'):
        input_height = model.config.get("input_height", 256)
        input_width = model.config.get("input_width", 384)
    else:
        # 如果模型沒有直接的 config 屬性，嘗試從內部獲取
        input_height = getattr(model, 'input_height', 256)
        input_width = getattr(model, 'input_width', 384)
        
    # 打印參數進行調試
    print(f"可視化使用的圖像尺寸: {input_height}x{input_width}")
    
    resize_transform = torch.nn.Sequential(
        torch.nn.Upsample(size=(input_height, input_width), mode='bilinear', align_corners=False)
    ).to(device)
    
    with torch.no_grad():
        # 獲取一個批次的數據
        data, mask = next(iter(data_loader))
        data = data.to(device)
        mask = mask.to(device)
        
        # 調整數據和掩碼大小以匹配模型配置
        if data.shape[2] != input_height or data.shape[3] != input_width:
            data = resize_transform(data)
            mask = resize_transform(mask)
        
        # 對數據進行編碼和解碼
        recon, mu, logvar = model(data)
        
        # 限制樣本數量
        n = min(data.size(0), max_samples)
        
        # 掩碼原始和重建的數據
        masked_data = data[:n] * mask[:n]
        masked_recon = recon[:n] * mask[:n]
        
        # 計算未掩碼重建
        unmasked_recon = recon[:n]
        
        # 根據潛在空間變量生成樣本
        z = torch.randn_like(mu[:n])
        samples = model.decode(z)
        
        # 確保掩碼可以應用到樣本上
        if samples.shape[1] == mask[:n].shape[1]:
            masked_samples = samples * mask[:n]
        else:
            print(f"警告：樣本通道不匹配 - 樣本: {samples.shape}, 掩碼: {mask[:n].shape}")
            masked_samples = samples  # 使用未掩碼樣本
        
        # 保存原始/重建比較
        comparison = torch.cat([masked_data, masked_recon])
        save_path = os.path.join(output_dir, f"{prefix}masked_reconstruction.png")
        utils.save_image(comparison.cpu(), save_path, nrow=n)
        print(f"已保存掩碼重建到: {save_path}")
        
        # 保存未掩碼比較，確保通道匹配
        if data[:n].shape[1] == unmasked_recon.shape[1]:
            unmasked_comparison = torch.cat([data[:n], unmasked_recon])
            unmasked_save_path = os.path.join(output_dir, f"{prefix}unmasked_reconstruction.png")
            utils.save_image(unmasked_comparison.cpu(), unmasked_save_path, nrow=n)
            print(f"已保存未掩碼重建到: {unmasked_save_path}")
        else:
            # 如果通道不匹配，就只保存重建結果
            unmasked_save_path = os.path.join(output_dir, f"{prefix}unmasked_recon.png")
            utils.save_image(unmasked_recon.cpu(), unmasked_save_path, nrow=n)
            print(f"已保存未掩碼重建到: {unmasked_save_path}（數據通道不匹配）")
        
        # 保存掩碼和隨機生成，確保通道匹配
        if mask[:n].shape[1] == masked_samples.shape[1]:
            mask_and_samples = torch.cat([mask[:n], masked_samples])
            samples_save_path = os.path.join(output_dir, f"{prefix}masks_and_samples.png")
            utils.save_image(mask_and_samples.cpu(), samples_save_path, nrow=n)
            print(f"已保存掩碼和生成樣本到: {samples_save_path}")
        else:
            # 分開保存
            mask_save_path = os.path.join(output_dir, f"{prefix}masks.png")
            samples_save_path = os.path.join(output_dir, f"{prefix}samples.png")
            utils.save_image(mask[:n].cpu(), mask_save_path, nrow=n)
            utils.save_image(samples.cpu(), samples_save_path, nrow=n)
            print(f"已保存掩碼到: {mask_save_path}")
            print(f"已保存生成樣本到: {samples_save_path}（通道不匹配，無法合併）") 