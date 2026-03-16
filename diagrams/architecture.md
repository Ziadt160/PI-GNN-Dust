# PI-GNN Architecture Documentation

This document describes the scientific architecture of the Physics-Informed Graph Neural Network (PI-GNN) for dust forecasting.

## 1. Graph Construction
We represent the atmospheric grid as a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$.
- **Nodes $\mathcal{V}$**: Each node represents a grid cell $(lat, lon)$.
- **Edges $\mathcal{E}$**: Edges connect immediate neighbors (North, South, East, West).
- **Node Features $X$**: $X = [C, u, v, T]$, where $C$ is PM10 concentration, $u, v$ are wind components, and $T$ is temperature.

## 2. Physics Flux Message Passing
The model uses a `PhysicsFluxLayer` derived from the `MessagePassing` class in PyG. 
The message $m_{j \to i}$ from node $j$ to node $i$ is calculated as:

$$m_{j \to i} = \text{MLP}(X_i, X_j, e_{ij}, \text{adv}_{ij})$$

Where:
- $e_{ij}$ is the projection of the wind vector $\vec{w}_i$ onto the edge vector $\vec{r}_{ij} = \vec{pos}_i - \vec{pos}_j$.
- $\text{adv}_{ij} = e_{ij} \cdot (C_j - C_i)$ represents the explicit advection term between the two cells.

The aggregation step sums these fluxes to update the node's latent state:
$$\hat{X}_i = \sum_{j \in \mathcal{N}(i)} m_{j \to i}$$

## 3. Temporal Prediction Mechanism
The model predicts the concentration change $\Delta C$ over a time step $\Delta t$:
$$\Delta C_i = \text{FC}(\hat{X}_i)$$
$$C_i(t+\Delta t) = \text{Softplus}(C_i(t) + \Delta C_i)$$

The use of `Softplus` ensures that concentrations remain strictly non-negative, a critical physical constraint.

## 4. Loss Functions
The model is trained using a composite loss function:
$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{physics}}$$

### Supervised Loss (Data Loss)
$\mathcal{L}_{\text{data}} = \text{MSE}(\hat{C}(t+\Delta t), C_{\text{GT}}(t+\Delta t))$

### Physics Loss (Residual Loss)
The physics loss enforces the advection-diffusion equation:
$$\frac{\partial C}{\partial t} + \vec{u} \cdot \nabla C = D \nabla^2 C + S$$
Ignoring diffusion $D$ and sources $S$ at the transport scale, we minimize the residual:
$$\mathcal{L}_{\text{physics}} = \left\| (C(t+\Delta t) - C(t)) + (\vec{u} \cdot \nabla C) \Delta t \right\|^2$$

Spatial gradients $\nabla C$ are calculated using central differences on the reconstructed grid from the graph nodes.

## 5. Homoscedastic Uncertainty Weighting
To automatically balance $\mathcal{L}_{\text{data}}$ and $\mathcal{L}_{\text{physics}}$, we learn the weights using homoscedastic uncertainty:
$$\mathcal{L}_{total} = \sum_j \frac{1}{2\sigma_j^2} \mathcal{L}_j + \log \sigma_j$$
where $\sigma_j$ are learnable parameters.
