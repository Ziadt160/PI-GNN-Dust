import torch

# Grid spacing for a 0.75 deg lat/lon mesh centred near ~26 N (Arabian Gulf region).
#   dy (latitude)  = 0.75 deg * 111_320 m/deg                 ~= 83_490 m
#   dx (longitude) = 0.75 deg * 111_320 m/deg * cos(26.25 deg) ~= 74_880 m
# Longitude spacing is shorter than latitude spacing at this latitude; using a
# single 83 km value for both (as the original code did) over-weights zonal advection.
DEFAULT_DT = 43200.0      # 12 h in seconds
DEFAULT_DX = 74880.0      # zonal (east-west) spacing in metres
DEFAULT_DY = 83490.0      # meridional (north-south) spacing in metres


def physics_loss(pm10_old, pm10_new, u10, v10,
                 delta_t=DEFAULT_DT, dx=DEFAULT_DX, dy=DEFAULT_DY, grid_size=(7, 12)):
    """
    Advection residual of the transport equation ``dC/dt + u . grad(C) = 0``.

    Discretised over one time step (explicit Euler):
        R = (C_{t+1} - C_t) + (u . grad C_t) * dt
    minimised in the mean-squared sense.

    Notes on orientation: node order is row-major with row 0 = northern-most
    latitude (latitude *decreases* with row index). ``torch.gradient`` along the
    row axis therefore returns the derivative in the *southward* direction, so it
    is negated to obtain the true northward gradient before pairing it with the
    (northward-positive) ``v10`` component.

    Args:
        pm10_old (Tensor): Concentration at time t (log space) [N, 1] or [N].
        pm10_new (Tensor): Predicted concentration at time t+1 (log space).
        u10 (Tensor): Eastward (zonal) wind component [N] (m/s).
        v10 (Tensor): Northward (meridional) wind component [N] (m/s).
        delta_t (float): Time step in seconds.
        dx, dy (float): Zonal / meridional grid spacing in metres.
        grid_size (tuple): (lat_dim, lon_dim).
    """
    lat_dim, lon_dim = grid_size
    cells = lat_dim * lon_dim

    pm10_flat = pm10_old.reshape(-1)
    batch_size = pm10_flat.size(0) // cells
    pm10_grid = pm10_flat.view(batch_size, lat_dim, lon_dim)

    # Spatial gradients via central differences.
    #   grad[0] -> d/d(row)  with spacing dy  (row increases southward)
    #   grad[1] -> d/d(col)  with spacing dx  (col increases eastward)
    grad = torch.gradient(pm10_grid, spacing=(dy, dx), dim=(1, 2))
    grad_north = (-grad[0]).reshape(-1, 1)  # negate: row+ is south, so flip sign
    grad_east = grad[1].reshape(-1, 1)

    u10 = u10.reshape(-1, 1)
    v10 = v10.reshape(-1, 1)
    advection = u10 * grad_east + v10 * grad_north

    change = pm10_new.reshape(-1, 1) - pm10_old.reshape(-1, 1)
    residual = change + advection * delta_t

    return torch.mean(residual ** 2)


def mass_conservation_error(pm10_pred, pm10_true):
    """
    Diagnostic (not a training penalty): relative error in the domain-mean
    concentration, ``|mean(pred) - mean(true)| / |mean(true)|``.

    Reported as a sanity check only -- an open domain with advective boundary
    flux and sources/sinks is *not* expected to conserve domain-mean mass, so
    this is deliberately excluded from the optimisation objective.
    """
    pred_mean = pm10_pred.mean()
    true_mean = pm10_true.mean()
    return (torch.abs(pred_mean - true_mean) / torch.clamp(torch.abs(true_mean), min=1e-8)).item()
