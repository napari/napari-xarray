"""Sample data for napari-xarray."""

import numpy as np
import xarray as xr
from skimage import data

from .utils import get_layerdatatuple_from_xarray

def cells3d() -> list[tuple]:
    """Create a 4D cells3d xarray with rich metadata for napari visualization.
    
    This function creates an xarray.DataArray from scikit-image's cells3d dataset
    with comprehensive metadata including:
    - Named dimensions and coordinates
    - Channel-specific colormaps and contrast limits
    - Scale information with units
    - Proper coordinate scaling for spatial dimensions

    Returns
    -------
    list of tuple
        LayerDataTuple suitable for direct use in napari.
    """
    # Physical scaling parameters
    PIXEL_SCALE = 0.26  # μm per slice/pixel
    
    # Load the cells3d dataset from scikit-image
    raw_data = data.cells3d()
    print(raw_data.shape)
    
    # Create the xarray with rich metadata
    cells3d_xarray = xr.DataArray(
        data=raw_data,
        name='cells3d',
        dims=['Z', 'C', 'Y', 'X'],
        coords={
            'Z': np.arange(raw_data.shape[0]) * PIXEL_SCALE,  # scaled z coordinates
            'C': ['membrane', 'nuclei'],  # channel names
            'Y': np.arange(raw_data.shape[2]) * PIXEL_SCALE,  # scaled y coordinates
            'X': np.arange(raw_data.shape[3]) * PIXEL_SCALE,  # scaled x coordinates
        },
        attrs={
            'scale_units': {
                'Z': 'μm', 
                'Y': 'μm', 
                'X': 'μm'
            },
            'colormaps': {
                'membrane': 'orange',
                'nuclei': 'cyan'
            },
            'contrast_limits': {
                'membrane': (0, 28000),
                'nuclei': (0, 60000),
            },
            'description': 'A 3D fluorescence microscopy dataset of cells with membrane and nuclei channels',
            'source': 'scikit-image cells3d dataset',
        }
    )

    # Use the utility function to create LayerDataTuples
    membrane_tuple = get_layerdatatuple_from_xarray(
        cells3d_xarray, 
        dim='C', 
        label='membrane'
    )

    nuclei_tuple = get_layerdatatuple_from_xarray(
        cells3d_xarray, 
        dim='C', 
        label='nuclei',
        blending='additive'
    )

    return [membrane_tuple, nuclei_tuple]
