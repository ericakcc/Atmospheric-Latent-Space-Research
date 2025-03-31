import json
import os
from datetime import datetime
import torch
import matplotlib.pyplot as plt

class ExperimentLogger:
    def __init__(self, config, base_dir="experiments"):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, f"{config.experiment_name}_{self.timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.train_losses = []
        self.test_losses = []
        self.best_loss = float('inf')
        
        # Save experiment configuration
        self.save_config()
    
    def save_config(self):
        config_path = os.path.join(self.exp_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)
    
    def log_epoch(self, epoch, train_loss, test_loss, model):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        
        # Save best model
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.save_model(model, "best_model.pt")
        
        # Periodically save checkpoints
        if epoch % 10 == 0:
            self.save_model(model, f"checkpoint_epoch_{epoch}.pt")
        
        # Update loss curve plot
        self.plot_losses()
    
    def save_model(self, model, filename):
        path = os.path.join(self.exp_dir, filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config.__dict__,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses
        }, path)
    
    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.exp_dir, 'loss_curve.png'))
        plt.close() 