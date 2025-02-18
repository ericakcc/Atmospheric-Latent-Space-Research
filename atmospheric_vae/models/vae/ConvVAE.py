import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseVAE

class CNNVAE(BaseVAE):
    def __init__(self, config, latent_dim=20):
        """
        Build a CNN-based VAE with a configurable architecture.
        The architecture is driven by a config dictionary so that users
        can adjust the CNNVAE architecture without resorting to image resizing.
        
        Args:
            config (dict): Dictionary containing configuration parameters for each layer.
            latent_dim (int): Dimension of the latent space.
        """
        super(CNNVAE, self).__init__()
        self.config = config

        # Encoder: create convolutional and pooling layers based on the config.
        # For MNIST, we expect the images to have 1 channel (28x28).
        in_channels = self.config.get("in_channels", 1)
        self.conv1 = nn.Conv2d(
            in_channels,
            self.config.get("conv1_out", 8),
            kernel_size=self.config.get("conv1_kernel", 3),
            stride=self.config.get("conv1_stride", 1),
            padding=self.config.get("conv1_padding", 1)
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=self.config.get("pool1_kernel", 2),
            stride=self.config.get("pool1_stride", 2)
        )
        self.conv2 = nn.Conv2d(
            self.config.get("conv1_out", 8),
            self.config.get("conv2_out", 16),
            kernel_size=self.config.get("conv2_kernel", 3),
            stride=self.config.get("conv2_stride", 1),
            padding=self.config.get("conv2_padding", 1)
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=self.config.get("pool2_kernel", 2),
            stride=self.config.get("pool2_stride", 2)
        )
        self.conv3 = nn.Conv2d(
            self.config.get("conv2_out", 16),
            self.config.get("conv3_out", 32),
            kernel_size=self.config.get("conv3_kernel", 3),
            stride=self.config.get("conv3_stride", 1),
            padding=self.config.get("conv3_padding", 1)
        )
        
        # Use a dummy input to dynamically compute the flatten dimension.
        input_height = self.config.get("input_height", 28)
        input_width = self.config.get("input_width", 28)
        dummy = torch.zeros(1, in_channels, input_height, input_width)
        x = F.relu(self.conv1(dummy))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        self.flatten_dim = x.view(1, -1).size(1)  # For MNIST, expect 32*7*7 = 1568
        
        # Fully connected layers for the latent space.
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder: fully connected layer to convert latent vector back to feature map.
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        # The decoder_input_shape should be provided in the config.
        # For MNIST with the above encoder, [32, 7, 7] is expected.
        self.decoder_input_shape = self.config.get("decoder_input_shape", [32, 7, 7])
        
        # Create decoder layers based on the config.
        # First transposed convolution block.
        self.deconv1 = nn.ConvTranspose2d(
            self.decoder_input_shape[0],
            self.config.get("deconv1_out", 16),
            kernel_size=self.config.get("deconv1_kernel", 3),
            stride=self.config.get("deconv1_stride", 1),
            padding=self.config.get("deconv1_padding", 1)
        )
        self.upsample1 = nn.Upsample(scale_factor=self.config.get("upsample1_scale", 2), mode='nearest')
        # Second transposed convolution block.
        self.deconv2 = nn.ConvTranspose2d(
            self.config.get("deconv1_out", 16),
            self.config.get("deconv2_out", 8),
            kernel_size=self.config.get("deconv2_kernel", 3),
            stride=self.config.get("deconv2_stride", 1),
            padding=self.config.get("deconv2_padding", 1)
        )
        self.upsample2 = nn.Upsample(scale_factor=self.config.get("upsample2_scale", 2), mode='nearest')
        # Final transposed convolution to reconstruct the image.
        self.deconv3 = nn.ConvTranspose2d(
            self.config.get("deconv2_out", 8),
            self.config.get("deconv3_out", 1),
            kernel_size=self.config.get("deconv3_kernel", 3),
            stride=self.config.get("deconv3_stride", 1),
            padding=self.config.get("deconv3_padding", 1)
        )
        
    def encode(self, x):
        """
        Encode the input image into a latent vector.
        
        Args:
            x (torch.Tensor): Input image tensor with shape (batch, in_channels, H, W).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log variance of the latent space.
        """
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z):
        """
        Decode the latent vector back to the image space.
        
        Args:
            z (torch.Tensor): Latent vector.
            
        Returns:
            torch.Tensor: Reconstructed image with shape (batch, out_channels, H, W).
        """
        x = self.fc_decode(z)
        x = x.view(-1, self.decoder_input_shape[0], self.decoder_input_shape[1], self.decoder_input_shape[2])
        x = F.relu(self.deconv1(x))
        x = self.upsample1(x)
        x = F.relu(self.deconv2(x))
        x = self.upsample2(x)
        # Using sigmoid to constrain the output between 0 and 1 (suitable for image data)
        x = torch.sigmoid(self.deconv3(x))
        return x
    
    def forward(self, x):
        """
        Perform a full forward pass through the VAE: encode and then decode.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed image, mean, and log variance.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# For testing: perform a single forward pass
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNVAE(latent_dim=2).to(device)
    dummy_input = torch.randn(4, 2, 61, 61).to(device)
    reconstructed, mu, logvar = model(dummy_input)
    print("Reconstructed shape:", reconstructed.shape)
