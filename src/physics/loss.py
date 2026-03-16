import torch

def physics_loss(pm10_old, pm10_new, u10, v10, delta_t=3600*12, dx=83000.0, dy=83000.0, grid_size=(7, 12)):
    """
    Computes the physics residual based on the Advection-Diffusion equation.
    Residual R = dC/dt + u·∇C
    
    Args:
        pm10_old (Tensor): Concentration at time t.
        pm10_new (Tensor): Concentration at time t+1 (predicted).
        u10 (Tensor): Eastward wind component.
        v10 (Tensor): Northward wind component.
        delta_t (float): Time step in seconds.
        dx, dy (float): Spatial grid resolution in meters.
        grid_size (tuple): (Latitude dimensions, Longitude dimensions).
    """
    # 1. Reshape to grid for spatial gradient calculation
    pm10_flat = pm10_old.view(-1)
    batch_size = pm10_flat.size(0) // (grid_size[0] * grid_size[1])
    pm10_grid = pm10_flat.view(batch_size, grid_size[0], grid_size[1])
    
    # 2. Compute spatial gradients using Central Differences
    # grad[0] is dy (lat), grad[1] is dx (lon)
    grad = torch.gradient(pm10_grid, spacing=(dy, dx), dim=(1, 2)) 
    
    grad_y = grad[0].reshape(-1, 1) # Gradient along Latitude
    grad_x = grad[1].reshape(-1, 1) # Gradient along Longitude
    
    # 3. Advection term: u * dC/dx + v * dC/dy
    u10 = u10.view(-1, 1)
    v10 = v10.view(-1, 1)
    advection = u10 * grad_x + v10 * grad_y
    
    # 4. Residual of the conservation law
    # dC + (u·∇C) * dt = 0
    change = pm10_new - pm10_old
    residual = change + (advection * delta_t) 
    
    # 5. Mass Conservation Penalty
    # Ensures that the global mass doesn't vanish or explode without external sources.
    mass_conservation = (torch.mean(pm10_new) - torch.mean(pm10_old)) ** 2
    
    return torch.mean(residual**2) + mass_conservation
