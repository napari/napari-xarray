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
    **kwargs,
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
    layer_type : str, optional
        The type of napari layer (default is 'image').
    **kwargs
        Additional keyword arguments to pass to the napari layer
        (e.g., blending='additive', opacity=0.8, visible=False, etc.).

    Returns
    -------
    tuple
        A LayerDataTuple compatible with napari.
    """
    
    # Determine spatial dimensions (excluding the sliced dimension)
    spatial_dims = [str(d) for d in da.dims if d != dim]
    
    scale = get_scale_from_coords(da, dims=spatial_dims)

    # Start with base metadata from xarray attributes
    metadata = {
        "name": label,
        "colormap": da.attrs.get("colormaps", {}).get(label),
        "contrast_limits": da.attrs.get("contrast_limits", {}).get(label),
        "scale": scale,
    }
    
    # Add any additional kwargs (this will override base metadata if there are conflicts)
    metadata.update(kwargs)

    return (da.sel({dim: label}).data, metadata, layer_type)