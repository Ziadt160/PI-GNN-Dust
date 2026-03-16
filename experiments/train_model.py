import torch
import os
import yaml
from torch_geometric.loader import DataLoader
from src.models.pi_gnn import PIDustModel
from src.training.dataset import GraphDustDataset
from src.training.trainer import Trainer

def main(config_path="configs/default_config.yaml"):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare Datasets
    data_dir = config['data']['dir']
    sfc_file = os.path.join(data_dir, config['data']['sfc_file'])
    train_years = config['training']['years']
    val_years = config['validation']['years']
    
    train_era = [os.path.join(data_dir, f'era5_{year}.nc') for year in train_years]
    val_era = [os.path.join(data_dir, f'era5_{year}.nc') for year in val_years]
    
    print("Creating Datasets...")
    train_dataset = GraphDustDataset(sfc_file, train_era)
    val_dataset = GraphDustDataset(sfc_file, val_era)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # 2. Model & Optimization
    model = PIDustModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    trainer = Trainer(model, optimizer, device)
    
    best_val_loss = float('inf')
    epochs = config['training']['epochs']
    
    print("Starting Training...")
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('results', exist_ok=True)
            torch.save(model.state_dict(), 'results/best_model.pth')
            print("  -> Saved Best Model to results/best_model.pth")

if __name__ == "__main__":
    main()
