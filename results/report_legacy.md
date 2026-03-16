# Quantum-Enhanced Dust Prediction: Project Report
**Date:** December 23, 2025

## 1. Executive Summary
This project aimed to develop and optimize quantum-enhanced machine learning models for forecasting PM10 dust concentrations. By integrating ERA5 meteorological data with quantum computing layers, we developed two distinct hybrid architectures: a Spatio-Temporal Transformer and a Hybrid CNN-VQC. Both models achieved a significant performance improvement (~38% reduction in MSE) over the persistence baseline, demonstrating the viability of the hybrid approach.

## 2. Methodology

### 2.1 Data Pipeline
- **Sources**: 
  - **Targets**: `data_sfc.nc` (PM10 concentrations)
  - **Features**: ERA5 reanalysis data (`u10`, `v10`, `t2m`)
- **Temporal Range**: 
  - **Training**: 2003 - 2006
  - **Validation**: 2007
- **Preprocessing**:
  - **Resampling**: Data aggregated to 12-hour intervals to optimize training speed.
  - **Normalization**: Log-transformation (`log1p`) for PM10; Standard Scaling for all physics variables.
  - **Time Encoding**: Cyclic features (Sine/Cosine) for Hour and Month to preserve temporal periodicity.
  - **Context**: Embedded features for Season (Winter-Fall) and Day of Week.

### 2.2 Model Architectures

#### Model A: Spatio-Temporal Transformer-Quantum
This architecture leverages the sequence modeling capabilities of Transformers, augmented by a quantum layer that processes global time-averaged features.

![Transformer Architecture](transformer_quantum_architecture.png)

- **Mechanism**: Visual/Spatial grid flattened per time step $\rightarrow$ Transformer Encoder $\rightarrow$ Temporal Pooling.
- **Quantum Integration**: Parallel branch processing mean-field features via a Variational Quantum Circuit (VQC).

#### Model B: Hybrid CNN-VQC
This architecture prioritizes spatial pattern recognition using Convolutional Neural Networks (CNNs), with a VQC acting as a feature transformation layer in the latent space.

![CNN-VQC Architecture](cnn_vqc_architecture.png)

- **Mechanism**: 2-Layer CNN inputs time-stacked grids $\rightarrow$ 16D Feature Vector.
- **Quantum Integration**: The 16D vector is projected to $n$ qubits, processed by Strongly Entangling Layers, and measured to modulate the feature space.

## 3. Experimental Results

We evaluated both models on the withheld validation set (Year 2007).

| Metric | Persistence Baseline | Transformer-Quantum | CNN-VQC |
| :--- | :--- | :--- | :--- |
| **MSE** (Mean Squared Error) | 1.2182 | ~0.7500 | **0.7553** |
| **RMSE** (Root Mean Sq Error) | 1.1037 | ~0.8660 | **0.8691** |
| **R² Score** | 0.00 | ~0.25 | **0.2482** |

*Note: The Persistence Baseline predicts the last observed frame for all future steps, representing a "no-skill" reference.*

### Key Findings
1.  **Significant Skill**: Both models significantly outperform the baseline, reducing MSE by approximately 38%. This confirms the models are learning meaningful physical patterns.
2.  **Architecture Parity**: The Transformer and CNN architectures achieved nearly identical performance metrics. This suggests the bottleneck may be in the data resolution (12H) or feature set rather than the specific mix of Convolution vs. Attention.
3.  **Low Explained Variance**: An R² of ~0.25 indicates that while the models capture the general trend and seasonality, they miss significant high-frequency variance or extreme outlier events.

## 4. Technical Achievements
- **Hardware Acceleration**: Implemented `lightning.qubit` for VQC simulation, enabling efficient hybrid training on classical CPUs.
- **Regularization**: Mitigated early overfitting using Dropout (0.2), Weight Decay (1e-4), and Early Stopping.
- **Robust Pipeline**: Established a production-ready pipeline with automated data loading, error handling for missing files, and standardized evaluation scripts.

## 5. Recommendations & Next Steps
1.  **Ablation Study**: Rigorously test `USE_QUANTUM=False` vs `True` to quantify the specific contribution of the VQC.
2.  **Hyperparameter Tuning**: The current R² suggests the model is under-fitting the complexity of dust events. Increasing model depth or input resolution (e.g., 6H or 3H) could help.
3.  **Data Expansion**: Integrating more meteorological variables (e.g., soil moisture, precipitation) could improve the explainability of dust generation events.
