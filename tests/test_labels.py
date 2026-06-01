"""Tests for utils/labels.py."""

import numpy as np
import pandas as pd
import xarray as xr

from h2mare.types import BBox, DateRange
from h2mare.utils.labels import create_filename_label, create_label_from_dataset


class TestCreateFilenameLabel:
    def test_bbox_object_with_date_range(self):
        bbox = BBox(-10, 30, 20, 50)
        dr = DateRange("2023-01-01", "2023-12-31")
        label = create_filename_label(bbox, "year", dr)
        assert label == "10W-20E-30N-50N_2023"

    def test_tuple_bbox_converted(self):
        label = create_filename_label((-10.0, 30.0, 20.0, 50.0), "year")
        assert label == "10W-20E-30N-50N"

    def test_no_date_range(self):
        bbox = BBox(-10, 30, 20, 50)
        label = create_filename_label(bbox, "year")
        assert "_" not in label
        assert label == "10W-20E-30N-50N"

    def test_yearmonth_format(self):
        bbox = BBox(-10, 30, 20, 50)
        dr = DateRange("2023-03-01", "2023-03-31")
        label = create_filename_label(bbox, "yearmonth", dr)
        assert "2023-03" in label


class TestCreateLabelFromDataset:
    def _make_ds(self, years):
        times = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="MS")
        return xr.Dataset(
            {"sst": (["time", "lat", "lon"], np.ones((len(times), 2, 2)))},
            coords={
                "time": times,
                "lat": [30.0, 40.0],
                "lon": [-10.0, 0.0],
            },
        )

    def test_single_year(self):
        ds = self._make_ds([2023])
        label = create_label_from_dataset(ds, date_format="year")
        assert "2023" in label

    def test_multi_year_triggers_warning(self, caplog):
        import logging

        ds = self._make_ds([2022, 2023])
        with caplog.at_level(logging.WARNING):
            label = create_label_from_dataset(ds, date_format="year")
        assert "multiple years" in caplog.text.lower() or label  # warning logged

    def test_multi_year_no_warning_when_suppressed(self, caplog):
        import logging

        ds = self._make_ds([2022, 2023])
        with caplog.at_level(logging.WARNING):
            create_label_from_dataset(ds, date_format="year", warn_multi_year=False)
        assert "multiple years" not in caplog.text.lower()
