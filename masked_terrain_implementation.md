# 地形掩碼實現總結

## 概要
本實現為大氣數據VAE訓練提供了一個掩碼方案，確保模型僅從有效的地形區域學習，而忽略NaN和Inf區域。這解決了原始數據集中將無效值（NaN/Inf）替換為0的問題，可以顯著提高模型在有效地形區域的重建質量。

## 問題描述
原始的`AtmosphericDataset`類和訓練流程存在以下問題：
1. NaN和Inf值被直接替換為0，沒有區分真實的0值和無效值
2. 損失計算在所有像素上進行，包括原本是無效的區域
3. 模型試圖學習重建非地形區域，浪費了建模能力

## 解決方案
我們實現了一個全面的掩碼解決方案，包括：

1. **掩碼數據集類**(`MaskedAtmosphericDataset`)：
   - 保留原始數據中的NaN/Inf位置信息
   - 創建二進制掩碼：1表示有效地形區域，0表示無效區域
   - 僅對有效區域進行標準化處理

2. **掩碼損失函數**(`masked_vae_loss_function`)：
   - 僅計算有效地形區域的重建損失
   - 按有效像素數量進行標準化，而非全部像素
   - 兼容多種損失類型(MSE, BCE, L1)

3. **掩碼訓練流程**：
   - 掩碼訓練和測試epoch函數，使用掩碼數據和損失
   - 增強的可視化功能，同時顯示掩碼和重建結果
   - 完整的訓練腳本，與原始實現保持兼容

## 實現文件
- `notebooks/atmospheric_masked_dataset.py`：掩碼數據集實現
- `notebooks/atmospheric_masked_trainer.py`：掩碼訓練器實現
- `train_atmospheric_masked.py`：使用掩碼的主訓練腳本
- `test_masked_dataset.py`：掩碼數據集測試腳本

## 掩碼效果測試
我們通過測試確認：
1. 掩碼創建正確，對應於原始數據中的NaN/Inf區域
2. 掩碼正確應用於損失計算，僅有效區域貢獻梯度
3. 數據集中大約62.43%的像素為無效(NaN/Inf)，只有37.57%是有效地形
4. 在無效區域沒有梯度傳播，確保模型只從有效地形學習

## 使用方法
1. 創建掩碼數據集：
```python
train_dataset, test_dataset = create_masked_train_test_datasets(
    data_dir="data/dcape/",
    shape=(512, 768, 3, 94),
    dtype=np.float32,
    max_samples=10  # 限制每個文件的時間點數量
)
```

2. 訓練VAE模型：
```python
python train_atmospheric_masked.py
```

3. 測試掩碼數據集：
```python
python test_masked_dataset.py
```

## 建議
1. 使用較小的batch size，因為數據集現在包含額外的掩碼信息
2. 限制每個文件的時間點數量，以減少內存使用
3. 定期檢查掩碼與重建的匹配程度，確保模型聚焦於有效區域
4. 考慮在訓練中使用較大的MSE權重，因為有效像素數量變少 