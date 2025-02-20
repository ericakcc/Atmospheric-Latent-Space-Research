import os
import torch
import torch.nn.functional as F
from torchvision import utils

def vae_loss_function(recon_x, x, mu, logvar):
    """
    Calculate the VAE loss as the sum of reconstruction loss and KL divergence.
    """
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_epoch(model, device, train_loader, optimizer, epoch, log_interval=1):
    """
    Train the VAE for one epoch.
    """
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        # Only flatten images if using SimpleVAE; CNNVAE expects 4D tensor.
        if model.__class__.__name__ == "SimpleVAE":
            data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss = vae_loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
            if batch_idx == 0:
                n = min(data.size(0), 8)
                if model.__class__.__name__ == "SimpleVAE":
                    # For SimpleVAE, reshape to 28x28 images.
                    comparison = torch.cat([data[:n].view(-1, 1, 28, 28),
                                            recon_batch[:n].view(-1, 1, 28, 28)])
                else:
                    # For CNNVAE, assume that data is already [batch, channels, H, W]
                    comparison = torch.cat([data[:n], recon_batch[:n]])
                os.makedirs("results", exist_ok=True)
                utils.save_image(comparison.cpu(), f"results/reconstruction_epoch_{epoch}.png", nrow=n)
                
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

def test_epoch(model, device, test_loader):
    """
    Test the VAE over the test dataset.
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            # Only flatten images if using SimpleVAE.
            if model.__class__.__name__ == "SimpleVAE":
                data = data.view(data.size(0), -1)
            recon, mu, logvar = model(data)
            test_loss += vae_loss_function(recon, data, mu, logvar).item()
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    return test_loss