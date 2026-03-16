
import os
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"PyTorch Version: {torch.__version__}")
print(f"PennyLane Version: {qml.__version__}")
