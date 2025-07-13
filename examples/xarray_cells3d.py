# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "napari[all]",
#     "numpy",
#     "xarray",
#     "scikit-image",
# ]
# ///

import napari
import numpy as np
import xarray as xr
from skimage import data

CHANNEL_DIM = 'C'
Z_SCALE = 0.29
Y_SCALE = 0.26
X_SCALE = 0.22

data = data.cells3d()

cells3d = xr.DataArray(
    data=data,
    name='cells3d',
    dims=['Z', 'C', 'Y', 'X'],
    coords={
        'Z': Z_SCALE, # scalar
        'C': ['membrane', 'nuclei'],
        'Y': np.arange(data.shape[2]) * Y_SCALE, # 1D array of scaled coordinates
        'X': np.arange(data.shape[3]) * X_SCALE,
    },
    attrs={
        'scale_units': {'Z': 'μm', 'Y': 'μm', 'X': 'μm'},
        'colormaps': {
            'membrane': 'orange',
            'nuclei': 'cyan'
        },
        'contrast_limits': {
            'membrane': (0, 28000),
            'nuclei': (0, 60000),
        }
    }
)# Should print 1, indicating scalar value

channel_axis = cells3d.dims.index(CHANNEL_DIM)
dims_no_channel_axis = [dim for dim in cells3d.dims if dim != CHANNEL_DIM]
channels = cells3d.coords[CHANNEL_DIM].values.tolist()
layer_names = [str(cells3d.name) + '_' + C for C in channels]
colormaps = [cells3d.attrs['colormaps'][C] for C in channels]
contrast_limits = [cells3d.attrs['contrast_limits'][C] for C in channels]

def calc_scale_from_coords(da, dims):
    scale = []
    for dim in dims:
        # get scalar value if 0D array
        if da.coords[dim].values.ndim == 0:
            scalar = da.coords[dim].values
        # calculate scalar value from finding the mean difference between coords and data shape
        else:
            scalar = np.mean(np.diff(da.coords[dim].values))
        scale.append(scalar)
    return scale

scale = calc_scale_from_coords(cells3d, dims_no_channel_axis)

scale_units = [cells3d.attrs['scale_units'][dim] for dim in dims_no_channel_axis]

viewer = napari.Viewer()

viewer.add_image(
    cells3d.data,
    name=layer_names,
    channel_axis=channel_axis,
    colormap=colormaps,
    contrast_limits=contrast_limits,
    scale=scale,
    units=scale_units,
    metadata=cells3d.attrs
)

viewer.dims.axis_labels = dims_no_channel_axis
viewer.axes.visible = True
viewer.scale_bar.visible = True
viewer.scale_bar.unit = cells3d.attrs['scale_units']['Z']


napari.run()