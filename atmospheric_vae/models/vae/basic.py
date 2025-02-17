import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseVAE

class SimpleVAE(BaseVAE):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        """
        Initializes the architecture of SimpleVAE.

        Args:
            input_dim (int): The dimension of the input data (default is 784, e.g., a flattened MNIST image).
            hidden_dim (int): The number of neurons in the hidden layer.
            latent_dim (int): The dimension of the latent vector.
        """
        super(SimpleVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder layers: from the input to the hidden representation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers: from the latent vector to the reconstructed output
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        Encodes the flattened input and computes the parameters (mean and log variance) of the latent distribution.

        Args:
            x (torch.Tensor): The flattened input tensor with shape [batch_size, input_dim].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The mean and log variance of the latent distribution.
        """
        h1 = F.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar

    def decode(self, z):
        """
        Decodes the latent vector to reconstruct the input.

        Args:
            z (torch.Tensor): The latent vector.

        Returns:
            torch.Tensor: The reconstructed input.
        """
        h3 = F.relu(self.fc3(z))
        # Use Sigmoid to constrain the output values to the range [0, 1] (suitable for image data)
        return torch.sigmoid(self.fc4(h3)) 