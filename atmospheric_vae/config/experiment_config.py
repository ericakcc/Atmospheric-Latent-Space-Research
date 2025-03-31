class ExperimentConfig:
    def __init__(self):
        # Setting
        self.experiment_name = "exp_001"
        self.description = "Base VAE with masked loss"
        
        # Model parameters
        self.model_config = {
            "latent_dim": 128,
            "in_channels": 3,
            "architecture": "CNNVAE",
        }
        
        # Training parameters
        self.training_config = {
            "batch_size": 16,
            "learning_rate": 1e-4,
            "epochs": 50,
            "beta": 0.1,  # KL loss weight
            "loss_weights": {
                "bce": 1.0,
                "mse": 0.0,
                "l1": 0.0
            }
        }
        
        # Data processing parameters
        self.data_config = {
            "input_size": (256, 384),
            "normalization": "minmax",  # or "standard"
            "augmentation": False
        } 