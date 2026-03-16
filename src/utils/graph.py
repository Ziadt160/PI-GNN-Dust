import torch

def build_dust_graph(lat_dim=7, lon_dim=12):
    """
    Converts a rectangular grid into a graph suitable for Graph Neural Networks.
    Nodes are grid cells, and edges connect immediate neighbors (N, S, E, W).
    
    Args:
        lat_dim (int): Number of latitude grid cells.
        lon_dim (int): Number of longitude grid cells.
        
    Returns:
        edge_index (torch.Tensor): Graph connectivity in COO format [2, E].
        pos (torch.Tensor): Node positions (grid coordinates) [N, 2].
    """
    edge_index = []
    pos = []
    
    for r in range(lat_dim):
        for c in range(lon_dim):
            node_idx = r * lon_dim + c
            # Connect to 4 neighbors (N, S, E, W)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < lat_dim and 0 <= nc < lon_dim:
                    edge_index.append([node_idx, nr * lon_dim + nc])
            pos.append([r, c])
            
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous(), torch.tensor(pos, dtype=torch.float)
