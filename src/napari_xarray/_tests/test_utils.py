"""Tests for utility functions."""

import numpy as np
import pytest
import xarray as xr

from napari_xarray.utils import get_scale_from_coords, get_layerdatatuple_from_xarray


class TestGetScaleFromCoords:
    """Tests for get_scale_from_coords function."""

    def test_uniform_spacing_1d(self):
        """Test with uniform spacing in 1D."""
        data = xr.DataArray(
            np.random.rand(10),
            dims=['X'],
            coords={'X': np.arange(10) * 0.5}  # 0.5 spacing
        )
        scale = get_scale_from_coords(data, ['X'])
        assert len(scale) == 1
        assert scale[0] == pytest.approx(0.5)

    def test_uniform_spacing_3d(self):
        """Test with uniform spacing in 3D."""
        data = xr.DataArray(
            np.random.rand(5, 10, 15),
            dims=['Z', 'Y', 'X'],
            coords={
                'Z': np.arange(5) * 0.3,   # 0.3 spacing
                'Y': np.arange(10) * 0.2,  # 0.2 spacing
                'X': np.arange(15) * 0.1,  # 0.1 spacing
            }
        )
        scale = get_scale_from_coords(data, ['Z', 'Y', 'X'])
        assert len(scale) == 3
        assert scale[0] == pytest.approx(0.3)
        assert scale[1] == pytest.approx(0.2)
        assert scale[2] == pytest.approx(0.1)

    def test_scalar_coordinates(self):
        """Test with scalar (0D) coordinates."""
        data = xr.DataArray(
            np.random.rand(10),
            dims=['X'],
            coords={'X': 1.5}  # scalar coordinate
        )
        scale = get_scale_from_coords(data, ['X'])
        assert len(scale) == 1
        assert scale[0] == 1.5

    def test_mixed_uniform_nonuniform_spacing(self):
        """Test with non-uniform spacing (should use mean difference)."""
        data = xr.DataArray(
            np.random.rand(5),
            dims=['X'],
            coords={'X': [0, 1, 3, 6, 10]}  # non-uniform: diffs are [1, 2, 3, 4]
        )
        scale = get_scale_from_coords(data, ['X'])
        expected_mean = np.mean([1, 2, 3, 4])  # 2.5
        assert scale[0] == pytest.approx(expected_mean)

    def test_subset_of_dimensions(self):
        """Test selecting only some dimensions."""
        data = xr.DataArray(
            np.random.rand(5, 10, 15),
            dims=['Z', 'Y', 'X'],
            coords={
                'Z': np.arange(5) * 0.3,
                'Y': np.arange(10) * 0.2,
                'X': np.arange(15) * 0.1,
            }
        )
        scale = get_scale_from_coords(data, ['Y', 'X'])  # only Y and X
        assert len(scale) == 2
        assert scale[0] == pytest.approx(0.2)  # Y
        assert scale[1] == pytest.approx(0.1)  # X


class TestGetLayerDataTupleFromXarray:
    """Tests for get_layerdatatuple_from_xarray function."""

    @pytest.fixture
    def sample_xarray(self):
        """Create a sample xarray for testing."""
        data = np.random.rand(2, 10, 15)  # 2 channels, 10x15 spatial
        return xr.DataArray(
            data=data,
            dims=['C', 'Y', 'X'],
            coords={
                'C': ['channel1', 'channel2'],
                'Y': np.arange(10) * 0.5,
                'X': np.arange(15) * 0.2,
            },
            attrs={
                'colormaps': {
                    'channel1': 'red',
                    'channel2': 'green'
                },
                'contrast_limits': {
                    'channel1': (0, 100),
                    'channel2': (10, 200),
                },
            }
        )

    def test_basic_functionality(self, sample_xarray):
        """Test basic LayerDataTuple creation."""
        layer_tuple = get_layerdatatuple_from_xarray(
            sample_xarray, 
            dim='C', 
            label='channel1'
        )
        
        # Check tuple structure
        assert isinstance(layer_tuple, tuple)
        assert len(layer_tuple) == 3
        
        data, metadata, layer_type = layer_tuple
        
        # Check data
        assert isinstance(data, np.ndarray)
        assert data.shape == (10, 15)  # C dimension removed
        
        # Check metadata
        assert isinstance(metadata, dict)
        assert metadata['name'] == 'channel1'
        assert metadata['colormap'] == 'red'
        assert metadata['contrast_limits'] == (0, 100)
        assert len(metadata['scale']) == 2  # Y and X dimensions
        
        # Check layer type
        assert layer_type == 'image'

    def test_with_kwargs(self, sample_xarray):
        """Test with additional kwargs."""
        layer_tuple = get_layerdatatuple_from_xarray(
            sample_xarray, 
            dim='C', 
            label='channel2',
            blending='additive',
            opacity=0.7,
            visible=False
        )
        
        data, metadata, layer_type = layer_tuple
        
        # Check that kwargs were added to metadata
        assert metadata['blending'] == 'additive'
        assert metadata['opacity'] == 0.7
        assert metadata['visible'] == False
        
        # Check original metadata still present
        assert metadata['name'] == 'channel2'
        assert metadata['colormap'] == 'green'

    def test_kwargs_override_defaults(self, sample_xarray):
        """Test that kwargs override default metadata."""
        layer_tuple = get_layerdatatuple_from_xarray(
            sample_xarray, 
            dim='C', 
            label='channel1',
            name='custom_name',  # override default name
            colormap='blue'      # override default colormap
        )
        
        data, metadata, layer_type = layer_tuple
        
        # Check overrides
        assert metadata['name'] == 'custom_name'
        assert metadata['colormap'] == 'blue'
        
        # Check non-overridden values remain
        assert metadata['contrast_limits'] == (0, 100)

    def test_missing_metadata(self):
        """Test with missing colormap and contrast_limits metadata."""
        data = np.random.rand(2, 5, 8)
        xarray_no_metadata = xr.DataArray(
            data=data,
            dims=['C', 'Y', 'X'],
            coords={
                'C': ['ch1', 'ch2'],
                'Y': np.arange(5) * 0.1,
                'X': np.arange(8) * 0.1,
            }
            # No attrs with colormaps or contrast_limits
        )
        
        layer_tuple = get_layerdatatuple_from_xarray(
            xarray_no_metadata, 
            dim='C', 
            label='ch1'
        )
        
        data, metadata, layer_type = layer_tuple
        
        # Should have None for missing metadata
        assert metadata['colormap'] is None
        assert metadata['contrast_limits'] is None
        
        # But should still have name and scale
        assert metadata['name'] == 'ch1'
        assert 'scale' in metadata

    def test_different_layer_type(self, sample_xarray):
        """Test with different layer type."""
        layer_tuple = get_layerdatatuple_from_xarray(
            sample_xarray, 
            dim='C', 
            label='channel1',
            layer_type='labels'
        )
        
        data, metadata, layer_type = layer_tuple
        assert layer_type == 'labels'

    def test_scale_calculation(self, sample_xarray):
        """Test that scale is calculated correctly."""
        layer_tuple = get_layerdatatuple_from_xarray(
            sample_xarray, 
            dim='C', 
            label='channel1'
        )
        
        data, metadata, layer_type = layer_tuple
        scale = metadata['scale']
        
        # Should have scale for Y and X (excluding C)
        assert len(scale) == 2
        assert scale[0] == pytest.approx(0.5)  # Y spacing
        assert scale[1] == pytest.approx(0.2)  # X spacing