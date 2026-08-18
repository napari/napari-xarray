"""Tests for sample data functions."""

import numpy as np
import pytest
import xarray as xr

from napari_xarray._sample_data import cells3d


def test_cells3d():
    """Test the cells3d sample data function."""
    layer_tuples = cells3d()
    
    # Check it returns a list of 2 LayerDataTuples
    assert isinstance(layer_tuples, list)
    assert len(layer_tuples) == 2
    
    # Check each tuple has the correct structure (data, metadata, layer_type)
    for layer_tuple in layer_tuples:
        assert isinstance(layer_tuple, tuple)
        assert len(layer_tuple) == 3
        
        data, metadata, layer_type = layer_tuple
        
        # Check data is numpy array with correct shape (60, 256, 256) - 3D after channel selection
        assert isinstance(data, np.ndarray)
        assert data.shape == (60, 256, 256)
        
        # Check metadata is dict with required keys
        assert isinstance(metadata, dict)
        assert "name" in metadata
        assert "scale" in metadata
        
        # Check layer type
        assert layer_type == "image"
    
    # Check specific layer names
    layer_names = [layer[1]["name"] for layer in layer_tuples]
    assert "membrane" in layer_names
    assert "nuclei" in layer_names