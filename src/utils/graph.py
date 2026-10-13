import numpy as np
import torch


def build_masked_graph(mask, neighbors=8):
    """
    Build a graph over only the VALID cells of a (possibly irregular) masked grid.

    Used for the full Saudi domain, where ~2/3 of the rectangular grid is masked
    ocean / out-of-region. Nodes are created only for ``mask == True`` cells, in
    row-major order, and edges connect a valid cell to its valid 4- or 8-neighbours.

    Args:
        mask (np.ndarray): 2-D boolean array [lat_dim, lon_dim]; True = valid cell.
        neighbors (int): 4 (N/S/E/W) or 8 (adds diagonals).

    Returns:
        edge_index (torch.LongTensor): connectivity [2, E] in node-index space.
        pos (torch.FloatTensor): geographically-oriented positions [N, 2] = (east, north).
        node_flat_idx (np.ndarray): flat grid index (row*lon_dim + col) for each node,
            so a dataset can pull that node's features from a flattened field.
    """
    mask = np.asarray(mask, dtype=bool)
    lat_dim, lon_dim = mask.shape

    # Map each valid (row, col) to a contiguous node id.
    node_id = -np.ones((lat_dim, lon_dim), dtype=np.int64)
    node_flat_idx, pos, coords = [], [], []
    nid = 0
    for r in range(lat_dim):
        for c in range(lon_dim):
            if mask[r, c]:
                node_id[r, c] = nid
                node_flat_idx.append(r * lon_dim + c)
                pos.append([c, (lat_dim - 1 - r)])  # (east, north)
                coords.append((r, c))
                nid += 1

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if neighbors == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    edge_index = []
    for (r, c) in coords:
        i = node_id[r, c]
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < lat_dim and 0 <= nc < lon_dim and mask[nr, nc]:
                edge_index.append([i, node_id[nr, nc]])

    return (
        torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        torch.tensor(pos, dtype=torch.float),
        np.asarray(node_flat_idx, dtype=np.int64),
    )


def build_dust_graph(lat_dim=7, lon_dim=12, neighbors=4):
    """
    Converts a rectangular lat/lon grid into a graph for Graph Neural Networks.
    Nodes are grid cells; edges connect immediate neighbours: N/S/E/W for
    ``neighbors=4`` or additionally the four diagonals for ``neighbors=8``
    (so winds at ~45 deg can be represented in a single message-passing hop).

    Node ordering is row-major with row 0 = northern-most latitude (matching the
    descending-latitude layout of the ERA5/CAMS files, e.g. 28.5 deg -> 24.0 deg).

    `pos` is returned in a geographically-oriented frame so that wind projections
    in :class:`~src.layers.flux.PhysicsFluxLayer` carry the correct sign:
        pos[:, 0] = eastward index  (increases with longitude / column)
        pos[:, 1] = northward index (increases with latitude, i.e. decreasing row)

    Args:
        lat_dim (int): Number of latitude grid cells.
        lon_dim (int): Number of longitude grid cells.

    Returns:
        edge_index (torch.Tensor): Graph connectivity in COO format [2, E].
        pos (torch.Tensor): Geographically-oriented node positions [N, 2] = (east, north).
    """
    edge_index = []
    pos = []

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if neighbors == 8:
        offsets = offsets + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for r in range(lat_dim):
        for c in range(lon_dim):
            node_idx = r * lon_dim + c
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < lat_dim and 0 <= nc < lon_dim:
                    edge_index.append([node_idx, nr * lon_dim + nc])
            # east = column, north = flipped row (row 0 is the northern-most band)
            pos.append([c, (lat_dim - 1 - r)])

    return (
        torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        torch.tensor(pos, dtype=torch.float),
    )
