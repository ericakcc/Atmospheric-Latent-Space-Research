import torch
import six
# Define six.string_classes to satisfy torchvision's checks, if not already defined.
if not hasattr(six, "string_classes"):
    six.string_classes = (str, bytes)
torch._six = six  # Monkey-patch torch._six for compatibility with torchvision

import os
import argparse
import json
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import our VAE implementation and the common training functions.
from atmospheric_vae.models.vae.basic import SimpleVAE
from atmospheric_vae.models.vae.ConvVAE import CNNVAE
from atmospheric_vae.training.trainer import train_epoch, test_epoch
from atmospheric_vae.utils import utils

def main():
    # Parse command-line arguments, including a new argument for model type and config path.
    parser = argparse.ArgumentParser(description="VAE MNIST Training")
    parser.add_argument("--batch-size", type=int, default=128, help="input batch size for training (default: 128)")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, help="seed for random number generator")
    parser.add_argument("--log-interval", type=int, default=100, help="how many batches to wait before logging training status")
    parser.add_argument("--save-model", action="store_true", default=False, help="For saving the current model")
    parser.add_argument("--model-type", type=str, default="simple", choices=["simple", "cnn"],
                        help="Choose VAE model: 'simple' for SimpleVAE, 'cnn' for CNNVAE")
    parser.add_argument("--config-path", type=str, default="atmospheric_vae/configs/cnn_config_mnist.json",
                        help="Path to the CNNVAE configuration JSON file (used when model-type is 'cnn')")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if not args.no_cuda and torch.cuda.is_available():
        device = utils.Utils.get_device()
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        current_device = torch.cuda.current_device()
        print("Current device index:", current_device)
        print("Current device name:", torch.cuda.get_device_name(current_device))

    # MNIST dataset transform: for both models, we use ToTensor().
    # (Note: For CNNVAE, ensure that your trainer bypasses flattening so that the image shape remains
    # (batch, channels, height, width), as expected by the model.)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model based on the chosen model type.
    if args.model_type == "cnn":
        # Load the configurable CNNVAE architecture from a JSON config file.
        with open(args.config_path, "r") as f:
            cnn_config = json.load(f)
        model = CNNVAE(config=cnn_config, latent_dim=20).to(device)
    else:
        # For SimpleVAE: note that input_dim = 28x28 = 784.
        model = SimpleVAE(input_dim=784, hidden_dim=400, latent_dim=20).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("results", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, device, train_loader, optimizer, epoch, log_interval=args.log_interval)
        test_epoch(model, device, test_loader)

    if args.save_model:
        model_filename = f"results/vae_{args.model_type}_mnist.pt"
        torch.save(model.state_dict(), model_filename)
        print(f"Saved model to {model_filename}")

if __name__ == "__main__":
    main()
