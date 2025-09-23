from __future__ import annotations

import xarray as xr
import numpy as np

def get_scale_from_coords(da: xr.DataArray, dims: list[str]) -> tuple[float, ...]:
    scale = []
    for dim in dims:
        # get scalar value if 0D array
        if da.coords[dim].values.ndim == 0:
            scalar = da.coords[dim].values
        # get scalar value from finding the mean difference between coords and data shape
        else:
            scalar = np.mean(np.diff(da.coords[dim].values))
        scale.append(scalar)
    return tuple(scale)

def get_layerdatatuple_from_xarray(
    da: "xr.DataArray",
    dim: str,
    label: str,
    layer_type: str = "image",
    blending: str | None = None,
) -> tuple:
    """Convert a DataArray slice to a napari LayerDataTuple.

    Parameters
    ----------
    da : xr.DataArray
        The input xarray DataArray.
    dim : str
        The dimension to slice on (e.g., 'C' for channels).
    label : str
        The specific label in the dimension to select (e.g., 'membrane').
    blending : str | None, optional
        The blending mode for the layer (e.g., 'additive'), by default None.
    layer_type : str, optional
        The type of napari layer (default is 'image').

    Returns
    -------
    tuple
        A LayerDataTuple compatible with napari.
    """
    
    if "colormaps" not in da.attrs or label not in da.attrs["colormaps"]:
        raise ValueError(f"Colormap for label '{label}' not found in DataArray attributes.")

    if "contrast_limits" not in da.attrs or label not in da.attrs["contrast_limits"]:
        raise ValueError(f"Contrast limits for label '{label}' not found in DataArray attributes.")

    # Determine spatial dimensions (excluding the sliced dimension)
    spatial_dims = [str(d) for d in da.dims if d != dim]
    
    scale = get_scale_from_coords(da, dims=spatial_dims)

    layer_dict = {
        "name": label,
        "colormap": da.attrs["colormaps"][label],
        "contrast_limits": da.attrs["contrast_limits"][label],
        "scale": scale,
    }

    if blending is not None:
        layer_dict["blending"] = blending

    return (da.sel({dim: label}).data, layer_dict, layer_type)