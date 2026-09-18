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

    @pytest.mark.parametrize(
        ("lat", "expected_km"),
        [(0.0, 111.19), (40.0, 85.17), (60.0, 55.60), (70.0, 38.03)],
    )
    def test_a_degree_of_longitude_shrinks_with_latitude(self, lat, expected_km):
        """
        Regression: the search ran on (lat, lon) in radians, which is Euclidean
        in a plane. A degree of longitude measured 111 km at every latitude —
        twice the truth at 60°N — and the neighbour it picked was wrong with it.
        """
        result = haversine_min_distance_kdtree(
            np.array([[lat, 0.0]]), np.array([[lat, 1.0]])
        )
        assert result[0] == pytest.approx(expected_km, rel=1e-3)

    def test_matches_a_reference_haversine(self):
        rng = np.random.default_rng(0)
        query = np.column_stack(
            (rng.uniform(-80, 80, 200), rng.uniform(-180, 180, 200))
        )
        target = np.column_stack(
            (rng.uniform(-80, 80, 200), rng.uniform(-180, 180, 200))
        )

        result = haversine_min_distance_kdtree(query, target)

        p1 = np.radians(query[:, 0])[:, None]
        l1 = np.radians(query[:, 1])[:, None]
        p2 = np.radians(target[:, 0])[None, :]
        l2 = np.radians(target[:, 1])[None, :]
        a = (
            np.sin((p2 - p1) / 2) ** 2
            + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
        )
        expected = (2 * 6371.0 * np.arcsin(np.sqrt(a))).min(axis=1)

        np.testing.assert_allclose(result, expected, rtol=1e-9)

    def test_picks_the_same_eddy_as_the_attribute_lookup(self):
        """
        The eddy processor measures the distance with this and reads the
        attributes with ``find_nearest_vectorized``. On the old metric the two
        named different eddies for ~13% of cells, so a cell's ``dist_km``
        described one eddy and its ``track`` another.
        """
        from h2mare.processing.core.aviso import find_nearest_vectorized

        # At 60°N a 1.5° zonal step is ~83 km and a 1.0° meridional step is
        # ~111 km, but in raw degrees the first looks the larger of the two.
        query = np.array([[60.0, 0.0]])
        targets = np.array([[60.0, 1.5], [61.0, 0.0]])

        distance = haversine_min_distance_kdtree(query, targets)
        chosen = find_nearest_vectorized(
            query[:, 0], query[:, 1], targets[:, 0], targets[:, 1]
        )

        assert chosen[0] == 0
        assert distance[0] == pytest.approx(83.4, rel=1e-2)

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
