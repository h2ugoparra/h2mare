"""Tests for utils/spatial.py."""

import numpy as np
import pytest
import xarray as xr

from h2mare.types import BBox
from h2mare.utils.spatial import (
    GridBuilder,
    clip_land_data,
    haversine_min_distance_kdtree,
)


class TestHaversineMinDistance:
    def test_basic_distance(self):
        coords1 = np.array([[40.0, -10.0]])
        coords2 = np.array([[40.0, -10.0]])
        result = haversine_min_distance_kdtree(coords1, coords2)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        coords1 = np.array([[0.0, 0.0]])
        coords2 = np.array([[0.0, 1.0]])
        result = haversine_min_distance_kdtree(coords1, coords2)
        assert result[0] == pytest.approx(111.19, abs=1.0)

    def test_invalid_shape_coords1(self):
        with pytest.raises(ValueError, match="coords1"):
            haversine_min_distance_kdtree(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]))

    def test_invalid_shape_coords2(self):
        with pytest.raises(ValueError, match="coords2"):
            haversine_min_distance_kdtree(np.array([[1.0, 2.0]]), np.array([1.0, 2.0]))

    def test_wrong_columns_coords1(self):
        with pytest.raises(ValueError, match="coords1"):
            haversine_min_distance_kdtree(
                np.array([[1.0, 2.0, 3.0]]), np.array([[1.0, 2.0]])
            )


class TestGridBuilder:
    def test_generate_grid_shape(self):
        bbox = BBox(-10, 30, 10, 50)
        grid = GridBuilder(bbox, dx=1.0, dy=1.0).generate_grid()
        assert "lat" in grid.coords
        assert "lon" in grid.coords
        assert len(grid.lat) == 20
        assert len(grid.lon) == 20

    def test_generate_grid_with_attributes(self):
        bbox = BBox(-10, 30, 10, 50)
        attrs = {"title": "test grid"}
        grid = GridBuilder(
            bbox, dx=1.0, dy=1.0, attributes=attrs
        ).generate_grid_with_attributes()
        assert grid.attrs["title"] == "test grid"

    def test_generate_grid_with_none_attributes(self):
        bbox = BBox(-10, 30, 10, 50)
        grid = GridBuilder(
            bbox, dx=1.0, dy=1.0, attributes=None
        ).generate_grid_with_attributes()
        assert isinstance(grid, xr.Dataset)


class TestClipLandData:
    def test_returns_dataset(self):
        ds = xr.Dataset(
            {"sst": (["lat", "lon"], np.ones((4, 4)))},
            coords={"lat": [0.0, 1.0, 2.0, 3.0], "lon": [0.0, 1.0, 2.0, 3.0]},
        )
        result = clip_land_data(ds)
        assert isinstance(result, xr.Dataset)
        assert "sst" in result.data_vars
