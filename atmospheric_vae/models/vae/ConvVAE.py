import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseVAE

class CNNVAE(BaseVAE):
    def __init__(self, latent_dim=2):
        """
        Build a VAE using a CNN architecture, assuming the input dimensions are (batch, 2, 61, 61)
        Args:
            latent_dim (int): Dimension of the latent space, default is 2
        """
        super(CNNVAE, self).__init__()
        # Encoder: Convolutional layers + pooling layers
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=0)    # (batch, 8, 59, 59)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)                  # (batch, 8, 58, 58)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=0)     # (batch, 32, 56, 56)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)     # (batch, 64, 27, 27)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)    # (batch, 128, 25, 25)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                    # (batch, 128, 12, 12)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0)    # (batch, 256, 5, 5)

        # Compute the flattened dimension (256 * 5 * 5)
        self.flatten_dim = 256 * 5 * 5
        # Fully connected layers to generate the latent space's mean and log variance
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder: Reconstruct the original image size from the latent vector
        # First, use a fully connected layer to reshape into the initial feature map for the convolutional layers
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        # Transposed convolution layers to sequentially upsample the feature map to the original size
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0)  # (batch, 128, ~11, ~11)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')                      # Upsample to ~22×22
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0)    # (batch, 64, ~24, ~24)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=0)     # (batch, 16, ~49, ~49)
        self.deconv4 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=1, padding=0)      # (batch, 8, ~53, ~53)
        self.deconv5 = nn.ConvTranspose2d(8, 2, kernel_size=5, stride=1, padding=0)       # (batch, 2, ~57, ~57)
        self.deconv6 = nn.ConvTranspose2d(2, 2, kernel_size=5, stride=1, padding=0)       # (batch, 2, 61, 61)

    def encode(self, x):
        """
        Encode the input image into the latent space
        Args:
            x (torch.Tensor): Input data with shape (batch, 2, 61, 61)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The mean (mu) and log variance (logvar) of the latent space
        """
        x = F.relu(self.conv1(x))   # (batch, 8, 59, 59)
        x = self.pool1(x)           # (batch, 8, 58, 58)
        x = F.relu(self.conv2(x))   # (batch, 32, 56, 56)
        x = F.relu(self.conv3(x))   # (batch, 64, 27, 27)
        x = F.relu(self.conv4(x))   # (batch, 128, 25, 25)
        x = self.pool2(x)           # (batch, 128, 12, 12)
        x = F.relu(self.conv5(x))   # (batch, 256, 5, 5)
        x = x.view(-1, self.flatten_dim)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        """
        Reconstruct the input image from the latent vector
        Args:
            z (torch.Tensor): Latent vector
        Returns:
            torch.Tensor: Reconstructed image with shape (batch, 2, 61, 61)
        """
        x = self.fc_decode(z)
        x = x.view(-1, 256, 5, 5)
        x = F.relu(self.deconv1(x))  # (batch, 128, ~11, ~11)
        x = self.upsample(x)         # (batch, 128, ~22, ~22)
        x = F.relu(self.deconv2(x))  # (batch, 64, ~24, ~24)
        x = F.relu(self.deconv3(x))  # (batch, 16, ~49, ~49)
        x = F.relu(self.deconv4(x))  # (batch, 8, ~53, ~53)
        x = F.relu(self.deconv5(x))  # (batch, 2, ~57, ~57)
        # The final layer applies tanh to restrict the output between -1 and 1, consistent with the original data preprocessing
        x = torch.tanh(self.deconv6(x))  # (batch, 2, 61, 61)
        return x

# For testing: perform a single forward pass
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNVAE(latent_dim=2).to(device)
    dummy_input = torch.randn(4, 2, 61, 61).to(device)
    reconstructed, mu, logvar = model(dummy_input)
    print("Reconstructed shape:", reconstructed.shape)
