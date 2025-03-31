import os
import torch
import torch.nn.functional as F
from torchvision import utils

def vae_loss_function(recon_x, x, mu, logvar, mask, config):
    """
    Calculate the VAE loss with configurable weights
    """
    # Apply mask to both reconstruction and original data
    masked_recon = recon_x * mask
    masked_x = x * mask
    
    valid_pixels = mask.sum()
    loss = 0
    
    # BCE loss
    if config.training_config["loss_weights"]["bce"] > 0:
        BCE = F.binary_cross_entropy(masked_recon, masked_x, reduction='sum') / (valid_pixels + 1e-8)
        loss += config.training_config["loss_weights"]["bce"] * BCE
    
    # MSE loss
    if config.training_config["loss_weights"]["mse"] > 0:
        MSE = F.mse_loss(masked_recon, masked_x, reduction='sum') / (valid_pixels + 1e-8)
        loss += config.training_config["loss_weights"]["mse"] * MSE
    
    # L1 loss
    if config.training_config["loss_weights"]["l1"] > 0:
        L1 = F.l1_loss(masked_recon, masked_x, reduction='sum') / (valid_pixels + 1e-8)
        loss += config.training_config["loss_weights"]["l1"] * L1
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    return loss + config.training_config["beta"] * KLD

def train_epoch(model, device, train_loader, optimizer, epoch, config, logger, log_interval=1):
    """
    Train the VAE for one epoch.
    """
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        
        # Create mask for terrain (assuming non-zero values indicate terrain)
        # Sum across channels to detect any non-zero values
        mask = (data.sum(dim=1, keepdim=True) > 0).float()
        mask = mask.repeat(1, data.size(1), 1, 1)  # Repeat mask for each channel
        
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss = vae_loss_function(recon_batch, data, mu, logvar, mask, config)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            if batch_idx == 0:
                n = min(data.size(0), 8)
                # For visualization, apply mask to both original and reconstructed images
                masked_data = data[:n] * mask[:n]
                masked_recon = recon_batch[:n] * mask[:n]
                comparison = torch.cat([masked_data, masked_recon])
                save_path = os.path.join(logger.exp_dir, f"reconstruction_epoch_{epoch}.png")
                utils.save_image(comparison.cpu(), save_path, nrow=n)
                
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

def test_epoch(model, device, test_loader, config):
    """
    Test the VAE over the test dataset.
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            # Create mask for terrain (same as in train_epoch)
            mask = (data.sum(dim=1, keepdim=True) > 0).float()
            mask = mask.repeat(1, data.size(1), 1, 1)
            
            recon, mu, logvar = model(data)
            test_loss += vae_loss_function(recon, data, mu, logvar, mask, config).item()
            
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    return test_loss