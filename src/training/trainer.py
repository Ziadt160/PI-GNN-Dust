import torch
import torch.nn as nn
from tqdm import tqdm
from src.physics.loss import physics_loss

class Trainer:
    """
    Orchestrates the training process for the PI-GNN model.
    Implements homoscedastic uncertainty weighting for multi-task loss (supervised + physics).
    """
    def __init__(self, model, optimizer, device, log_vars=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.mse_loss = nn.MSELoss()
        
        # Homoscedastic uncertainty log-variances
        if log_vars is None:
            self.log_vars = nn.Parameter(torch.zeros(2, device=device))
            self.optimizer.add_param_group({'params': [self.log_vars]})
        else:
            self.log_vars = log_vars

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        
        for data in tqdm(loader, desc="Training", leave=False):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            pm10_next, _ = self.model(data)
            
            # 1. Supervised Loss
            l_sup = self.mse_loss(pm10_next, data.y)
            
            # 2. Physics Loss
            l_phys = physics_loss(
                data.x[:, 0:1], # Previous state
                pm10_next,      # Predicted next state
                data.x[:, 1],   # Wind U
                data.x[:, 2]    # Wind V
            )
            
            # Combined Loss with learned weighting
            loss = (0.5 * torch.exp(-self.log_vars[0]) * l_sup + 0.5 * self.log_vars[0]) + \
                   (0.5 * torch.exp(-self.log_vars[1]) * l_phys + 0.5 * self.log_vars[1])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                pm10_next, _ = self.model(data)
                
                l_sup = self.mse_loss(pm10_next, data.y)
                l_phys = physics_loss(data.x[:, 0:1], pm10_next, data.x[:, 1], data.x[:, 2])
                
                loss = (0.5 * torch.exp(-self.log_vars[0]) * l_sup + 0.5 * self.log_vars[0]) + \
                       (0.5 * torch.exp(-self.log_vars[1]) * l_phys + 0.5 * self.log_vars[1])
                
                total_loss += loss.item()
                
        return total_loss / len(loader)
