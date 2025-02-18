import torch
import six
# Define six.string_classes to satisfy torchvision's checks, if not already defined.
if not hasattr(six, "string_classes"):
    six.string_classes = (str, bytes)
torch._six = six  # Monkey-patch torch._six for compatibility with torchvision

import os
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import our VAE implementation and the common training functions.
from atmospheric_vae.models.vae.basic import SimpleVAE
from atmospheric_vae.training.trainer import train_epoch, test_epoch

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
        train_epoch(model, device, train_loader, optimizer, epoch, log_interval=args.log_interval)
        test_epoch(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "results/vae_mnist.pt")

if __name__ == "__main__":
    main()
