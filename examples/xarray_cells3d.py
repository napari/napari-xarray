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
import xarray as xr
from skimage import data

CHANNEL_DIM = 'C'

cells3d = xr.DataArray(
    data=data.cells3d(),
    name='cells3d',
    dims=['Z', 'C', 'Y', 'X'],
    coords={
        'C': ['membrane', 'nuclei'],
    },
    attrs={
        'scale': {'Z': 0.5, 'Y': 0.51, 'X': 0.52},
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
)

viewer = napari.Viewer()

channel_axis = cells3d.dims.index(CHANNEL_DIM)
dims_no_channel_axis = [dim for dim in cells3d.dims if dim != CHANNEL_DIM]
channels = cells3d.coords[CHANNEL_DIM].values.tolist()
layer_names = [str(cells3d.name) + '_' + C for C in channels]
colormaps = [cells3d.attrs['colormaps'][C] for C in channels]
contrast_limits = [cells3d.attrs['contrast_limits'][C] for C in channels]
scale = [cells3d.attrs['scale'][dim] for dim in dims_no_channel_axis]
scale_units = [cells3d.attrs['scale_units'][dim] for dim in dims_no_channel_axis]


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