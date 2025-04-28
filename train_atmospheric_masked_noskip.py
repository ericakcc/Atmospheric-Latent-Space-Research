import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 設置環境變量，限制只使用0號GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 導入模型和工具
from atmospheric_vae.models.vae.Unet_without_skipconnect import CNNVAE_UNet
from atmospheric_vae.config.experiment_config import ExperimentConfig
from atmospheric_vae.utils.experiment_logger import ExperimentLogger

# 導入我們新實現的掩碼數據集和訓練方法
from notebooks.atmospheric_masked_dataset import MaskedAtmosphericDataset, create_masked_train_test_datasets
from notebooks.atmospheric_masked_trainer import train_masked_epoch, test_masked_epoch, visualize_masked_reconstructions

def main():
    # 創建實驗配置
    config = ExperimentConfig()
    config.experiment_name = "masked_unet_noskip_test"
    config.description = "使用地形掩碼的UNet-VAE模型（無跳躍連接版本），測試版本"
    
    # 獲取數據集的基本信息以確保配置與數據集一致
    print("加載數據集樣本以檢查形狀...")
    # 先獲取一個樣本來檢查形狀
    try:
        # 只加載一個文件的一個時間點用於檢查
        data_dir = "data/dcape/"
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
        if not all_files:
            raise ValueError(f"在 {data_dir} 中沒有找到.dat文件")
        
        sample_dataset = MaskedAtmosphericDataset(
            data_dir=data_dir,
            file_list=[all_files[0]],
            shape=(512, 768, 3, 94),
            dtype=np.float32,
            max_samples=1
        )
        
        # 檢查樣本的形狀
        if len(sample_dataset) > 0:
            sample, mask = sample_dataset[0]
            print(f"數據樣本形狀: {sample.shape}")
            print(f"掩碼形狀: {mask.shape}")
            
            # 更新配置以匹配數據集
            input_channels = sample.shape[0]
            input_height = sample.shape[1]
            input_width = sample.shape[2]
            
            # 調整大小以平衡處理速度和解析度
            input_height = 256  # 使用較高解析度
            input_width = 384   # 保持比例
        else:
            print("警告：樣本數據集為空，使用默認配置")
            input_channels = 3
            input_height = 128
            input_width = 192
            
    except Exception as e:
        print(f"檢查樣本時出錯: {str(e)}")
        print("使用默認配置")
        input_channels = 3
        input_height = 128
        input_width = 192
    
    # 更新配置
    config.model_config.update({
        "latent_dim": 64,  # 潛在空間維度
        "in_channels": input_channels,
        "input_height": input_height,
        "input_width": input_width,
    })
    
    config.training_config.update({
        "batch_size": 8,  # 使用較小的批次大小以提高穩定性
        "learning_rate": 5e-5,  # 使用較小的學習率
        "epochs": 200,     
        "beta": 0.0001,  # 降低KL散度的權重，以增加重建準確性
        "loss_weights": {
            "bce": 0.5,  # 降低BCE權重，以減少可能的數值不穩定
            "mse": 1.0,  # 主要使用MSE損失  
            "l1": 0.3    # 增加L1損失權重，改善細節      
        }
    })
    
    # 初始化記錄器
    logger = ExperimentLogger(config)
    
    # 設置隨機種子
    torch.manual_seed(config.training_config.get("seed", 42))
    
    # 指定只使用0號GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 設置為僅使用0號GPU
        device = torch.device("cuda:0")
        print(f"限制使用GPU 0: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
    
    print(f"使用設備: {device}")
    
    # 創建掩碼數據集，使用少量資料用於測試
    print("創建掩碼數據集...")
    train_dataset, test_dataset = create_masked_train_test_datasets(
        data_dir="data/dcape/",
        shape=(512, 768, 3, 94),
        dtype=np.float32,
        train_ratio=0.8,
        seed=42,
        # max_samples=10  # 限制每個文件只使用前10個時間點
    )
    
    # 創建調整大小的轉換
    transform = torch.nn.Sequential(
        torch.nn.Upsample(size=(input_height, input_width), mode='bilinear', align_corners=False)
    )
    
    # 創建數據加載器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training_config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training_config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 創建無跳躍連接的UNet VAE模型
    model = CNNVAE_UNet(config=config.model_config).to(device)
    
    # 設置調試模式，在第一個epoch中檢查形狀
    model.debug_mode = True
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training_config["learning_rate"],
        weight_decay=0.01
    )
    
    print(f"開始訓練，配置: {config.experiment_name}")
    print(f"模型配置: {config.model_config}")
    print(f"訓練配置: {config.training_config}")
    print(f"模型類型: UNet VAE (無跳躍連接)")
    
    # 檢查一個批次數據的形狀
    print("檢查數據批次的形狀...")
    for data, mask in train_loader:
        print(f"批次數據形狀: {data.shape}")
        print(f"批次掩碼形狀: {mask.shape}")
        # 手動調整大小以檢查是否工作
        resized_data = transform(data)
        resized_mask = transform(mask)
        print(f"調整大小後的數據形狀: {resized_data.shape}")
        print(f"調整大小後的掩碼形狀: {resized_mask.shape}")
        break
    
    # 訓練循環
    for epoch in range(1, config.training_config["epochs"] + 1):
        # 在第一個epoch後關閉調試模式
        if epoch > 1:
            model.debug_mode = False
            
        print(f"\n開始Epoch {epoch}/{config.training_config['epochs']}...")
        
        # 使用掩碼訓練方法
        train_loss = train_masked_epoch(
            model, device, train_loader, optimizer, epoch, config, logger
        )
        
        # 使用掩碼測試方法
        test_loss = test_masked_epoch(
            model, device, test_loader, config
        )
        
        # 記錄指標
        logger.log_epoch(epoch, train_loss, test_loss, model)
        
        # 保存檢查點
        if epoch % 5 == 0 or epoch == config.training_config["epochs"]:
            checkpoint_path = os.path.join(logger.exp_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'config': config
            }, checkpoint_path)
            print(f"已保存檢查點到: {checkpoint_path}")
    
    # 訓練完成後，在測試集上可視化結果
    visualization_dir = os.path.join(logger.exp_dir, "visualizations")
    
    # 傳遞尺寸參數以確保可視化正確
    print("開始生成可視化結果...")
    try:
        # 創建參數字典
        viz_params = {
            "model": model,
            "data_loader": test_loader,
            "device": device,
            "output_dir": visualization_dir,
            "max_samples": 4,
            "prefix": "final_"
        }
        
        # 添加尺寸參數
        viz_params["input_height"] = config.model_config.get("input_height", 128)
        viz_params["input_width"] = config.model_config.get("input_width", 192)
        
        visualize_masked_reconstructions(**viz_params)
    except Exception as e:
        print(f"可視化過程中出錯: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"訓練完成! 結果保存在: {logger.exp_dir}")

if __name__ == "__main__":
    main() 