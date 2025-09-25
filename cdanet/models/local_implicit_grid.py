"""Local implicit grid query function based on reference sourcecodeCDAnet."""

import torch
import torch.nn as nn
import numpy as np


def regular_nd_grid_interpolation_coefficients(grid, query_pts, xmin, xmax):
    """Compute interpolation coefficients for regular ND grid.

    This is a simplified version that handles the basic interpolation needed.
    """
    batch_size, *grid_dims, num_channels = grid.shape
    batch_size_q, num_points, spatial_dim = query_pts.shape

    assert batch_size == batch_size_q, "Batch sizes must match"
    assert len(grid_dims) == spatial_dim, "Grid dimensions must match query point dimensions"

    # Normalize query points to grid coordinates
    grid_coords = []
    for i in range(spatial_dim):
        # Scale from [xmin, xmax] to [0, grid_dims[i]-1]
        normalized = (query_pts[:, :, i] - xmin[i]) / (xmax[i] - xmin[i])
        grid_coord = normalized * (grid_dims[i] - 1)
        grid_coords.append(grid_coord)

    grid_coords = torch.stack(grid_coords, dim=-1)  # [batch, num_points, spatial_dim]

    # Get integer and fractional parts
    grid_coords_floor = torch.floor(grid_coords).long()
    grid_coords_frac = grid_coords - grid_coords_floor.float()

    # Clamp to valid range
    for i in range(spatial_dim):
        grid_coords_floor[:, :, i] = torch.clamp(grid_coords_floor[:, :, i], 0, grid_dims[i] - 2)

    # Generate corner indices (2^spatial_dim corners)
    num_corners = 2 ** spatial_dim
    corner_values = []
    corner_weights = []

    for corner_idx in range(num_corners):
        # Determine which corner this is (binary representation)
        corner_offset = []
        corner_weight = torch.ones(batch_size, num_points, device=query_pts.device)

        for dim_idx in range(spatial_dim):
            bit = (corner_idx >> dim_idx) & 1
            corner_offset.append(bit)

            if bit == 0:
                corner_weight *= (1 - grid_coords_frac[:, :, dim_idx])
            else:
                corner_weight *= grid_coords_frac[:, :, dim_idx]

        # Get corner indices
        corner_indices = grid_coords_floor.clone()
        for dim_idx in range(spatial_dim):
            corner_indices[:, :, dim_idx] += corner_offset[dim_idx]

        # Extract values at corners
        if spatial_dim == 3:
            corner_vals = grid[torch.arange(batch_size)[:, None],
                              corner_indices[:, :, 0],
                              corner_indices[:, :, 1],
                              corner_indices[:, :, 2]]
        else:
            raise NotImplementedError("Only 3D grid supported for now")

        corner_values.append(corner_vals)
        corner_weights.append(corner_weight)

    corner_values = torch.stack(corner_values, dim=2)  # [batch, num_points, num_corners, num_channels]
    corner_weights = torch.stack(corner_weights, dim=2)  # [batch, num_points, num_corners]

    # Relative coordinates for each query point
    x_relative = grid_coords_frac  # [batch, num_points, spatial_dim]

    return corner_values, corner_weights, x_relative


def query_local_implicit_grid(model, latent_grid, query_pts, xmin, xmax):
    """Function for querying local implicit grid.

    The latent feature grid can be of arbitrary physical dimensions. query_pts are query points
    representing the coordinates at which the grid is queried. xmin and xmax are the bounds of
    the grid.

    Args:
        model: nn.Module instance, model for decoding local latents. Must accept input of length
        d+c.
        latent_grid: tensor of shape [b, n1, n2, ..., nd, c] where b is the batch size, n1, ..., nd
        are the spatial resolution in each dimension, c is the number of latent channels.
        query_pts: tensor of shape [b, num_pts, d] where num_pts is the number of query points, d is
        the dimension of the query points. The query points must fall within xmin and xmax, or else
        will be clipped to xmin and xmax.
        xmin: float or tuple of floats or tensor. If float, automatically broadcast to the
        corresponding dimensions. Reference spatial coordinate of the lower left corner of the grid.
        xmax:float or tuple of floats or tensor. If float, automatically broadcast to the
        corresponding dimensions. Reference spatial coordinate of the upper right corner of the
        grid.
    Returns:
        query_vals: tensor of shape [b, num_pts, o], queried values at query points, where o is the
        number output channels from the model.
    """
    # Handle xmin/xmax format
    if isinstance(xmin, float):
        spatial_dim = query_pts.shape[-1]
        xmin = torch.tensor([xmin] * spatial_dim, device=query_pts.device)
    elif isinstance(xmin, (list, tuple)):
        xmin = torch.tensor(xmin, device=query_pts.device)

    if isinstance(xmax, float):
        spatial_dim = query_pts.shape[-1]
        xmax = torch.tensor([xmax] * spatial_dim, device=query_pts.device)
    elif isinstance(xmax, (list, tuple)):
        xmax = torch.tensor(xmax, device=query_pts.device)

    corner_values, weights, x_relative = regular_nd_grid_interpolation_coefficients(
        latent_grid, query_pts, xmin, xmax)

    # Expand x_relative to match corner_values dimensions
    # x_relative: [batch, num_points, spatial_dim] -> [batch, num_points, num_corners, spatial_dim]
    num_corners = corner_values.shape[2]
    x_relative_expanded = x_relative.unsqueeze(2).expand(-1, -1, num_corners, -1)

    concat_features = torch.cat([x_relative_expanded, corner_values], dim=-1)  # [b, num_points, 2**d,  d+c]
    input_shape = concat_features.shape

    # flatten and feed through model
    output = model(concat_features.reshape([-1, input_shape[-1]]))

    # reshape output
    output = output.reshape([input_shape[0], input_shape[1], input_shape[2], -1])  # [b, p, 2**d, o]

    # interpolate the output values
    output = torch.sum(output * weights.unsqueeze(-1), dim=-2)  # [b, p, o]

    return output