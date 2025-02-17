import torch
import torch.nn as nn
import abc

class BaseVAE(nn.Module, abc.ABC):
    @abc.abstractmethod
    def encode(self, x):
        """
        Encodes the input into latent space parameters.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mu, logvar)
        """
        pass

    @abc.abstractmethod
    def decode(self, z):
        """
        Decodes the latent variable into a reconstruction of the input.

        Args:
            z (torch.Tensor): The latent variable.

        Returns:
            torch.Tensor: The reconstructed input.
        """
        pass

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick to sample from a Gaussian distribution.

        Args:
            mu (torch.Tensor): Mean of the latent Gaussian.
            logvar (torch.Tensor): Log variance of the latent Gaussian.

        Returns:
            torch.Tensor: A sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Performs the forward pass through the VAE.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            (reconstructed input, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar