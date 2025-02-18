import torch
import six
# Define six.string_classes to satisfy torchvision's checks, if not already defined.
if not hasattr(six, "string_classes"):
    six.string_classes = (str, bytes)
torch._six = six  # Monkey-patch torch._six for compatibility with torchvision

import os
import argparse
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# Import our VAE implementation. For MNIST, we use the simple, fully connected VAE.
from atmospheric_vae.models.vae.basic import SimpleVAE

def loss_function(recon_x, x, mu, logvar):
    """
    Calculate the VAE loss as the sum of reconstruction loss and KL divergence.
    """
    # Reconstruction loss using binary cross entropy
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    """
    Train the VAE for one epoch.
    """
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        # Flatten each 28x28 image into a vector of 784 elements
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}")
            
            # Save a sample reconstruction (first batch only per epoch)
            if batch_idx == 0:
                n = min(data.size(0), 8)
                # Concatenate original and reconstructed images for comparison
                comparison = torch.cat([data[:n].view(-1, 1, 28, 28),
                                        recon_batch[:n].view(-1, 1, 28, 28)])
                os.makedirs("results", exist_ok=True)
                utils.save_image(comparison.cpu(),
                                 f"results/reconstruction_epoch_{epoch}.png",
                                 nrow=n)
    avg_loss = train_loss / len(train_loader.dataset)
    print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")

def test(model, device, test_loader):
    """
    Evaluate the VAE on the test dataset.
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            data = data.view(data.size(0), -1)
            recon, mu, logvar = model(data)
            test_loss += loss_function(recon, data, mu, logvar).item()
    test_loss /= len(test_loader.dataset)
    print(f"====> Test set loss: {test_loss:.4f}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="VAE MNIST Training")
    parser.add_argument("--batch-size", type=int, default=128, help="input batch size for training (default: 128)")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, help="seed for random number generator")
    parser.add_argument("--log-interval", type=int, default=100, help="how many batches to wait before logging training status")
    parser.add_argument("--save-model", action="store_true", default=False, help="For saving the current model")
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # MNIST dataset transform: convert to tensor (values in [0,1])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the VAE model for MNIST. Note: input_dim = 28 x 28 = 784.
    model = SimpleVAE(input_dim=784, hidden_dim=400, latent_dim=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("results", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval=args.log_interval)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "results/vae_mnist.pt")

if __name__ == "__main__":
    main()
