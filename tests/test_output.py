import numpy as np
import pytest
import netCDF4 as nc
import datetime
from cosmo14C.output import write_year


def make_aux(nlat, nlon):
    return {
        'phi':       np.full(12, 650.0),
        'Pc':        np.full((nlat, nlon), 5.0),
        'q_col':     np.full((12, nlat, nlon), 1.7),
        'geomag_lat': np.zeros((nlat, nlon)),
    }


class TestWriteYear:
    def test_file_is_created(self, tmp_path):
        data = np.zeros((12, 4, 8))
        lat  = np.arange(-89.5, -85.5, 1.0)   # 4 values
        lon  = np.arange(-179.5, -171.5, 1.0) # 8 values
        cfg  = {'output': {'directory': str(tmp_path), 'compress': False}}
        write_year(2020, data, lat, lon, make_aux(4, 8), cfg)
        assert (tmp_path / "flux_c14.cosmogenic.2020.nc").exists()

    def test_dimensions_correct(self, tmp_path):
        data = np.zeros((12, 4, 8))
        # lat has 3 values but data nlat=4 — exercises the NaN-fallback path in write_year
        lat  = np.arange(-89.5, -86.5, 1.0)
        lon  = np.arange(-179.5, -171.5, 1.0)
        cfg  = {'output': {'directory': str(tmp_path), 'compress': False}}
        write_year(2020, data, lat, lon, make_aux(3, 8), cfg)
        with nc.Dataset(tmp_path / "flux_c14.cosmogenic.2020.nc") as ds:
            assert ds.dimensions['time'].size == 12
            assert ds.dimensions['lat'].size  == 4
            assert ds.dimensions['lon'].size  == 8

    def test_no_vertical_dimension(self, tmp_path):
        """TM5 does the vertical distribution, so no lev axis is written."""
        data = np.zeros((12, 4, 8))
        lat  = np.arange(-89.5, -85.5, 1.0)
        lon  = np.arange(-179.5, -171.5, 1.0)
        cfg  = {'output': {'directory': str(tmp_path), 'compress': False}}
        write_year(2020, data, lat, lon, make_aux(4, 8), cfg)
        with nc.Dataset(tmp_path / "flux_c14.cosmogenic.2020.nc") as ds:
            assert 'lev' not in ds.dimensions
            assert 'lev' not in ds.variables
            assert ds.variables['c14flux'].dimensions == ('time', 'lat', 'lon')

    def test_time_axis_is_month_starts(self, tmp_path):
        """Monthly fluxes are stamped at the START of the month, matching the
        rest of the input archive. A mismatch here is silent: xarray aligns on
        the union of time axes and fills the gaps with NaN."""
        data = np.zeros((12, 4, 8))
        lat  = np.arange(-89.5, -85.5, 1.0)
        lon  = np.arange(-179.5, -171.5, 1.0)
        cfg  = {'output': {'directory': str(tmp_path), 'compress': False}}
        write_year(2020, data, lat, lon, make_aux(4, 8), cfg)
        with nc.Dataset(tmp_path / "flux_c14.cosmogenic.2020.nc") as ds:
            t = ds.variables['time']
            assert t.units == 'days since 2020-01-01 00:00:00'
            # 2020 is a leap year: 1 March is day 60, not 59.
            assert list(t[:]) == [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

    def test_time_axis_common_year(self, tmp_path):
        data = np.zeros((12, 4, 8))
        lat  = np.arange(-89.5, -85.5, 1.0)
        lon  = np.arange(-179.5, -171.5, 1.0)
        cfg  = {'output': {'directory': str(tmp_path), 'compress': False}}
        write_year(2021, data, lat, lon, make_aux(4, 8), cfg)
        with nc.Dataset(tmp_path / "flux_c14.cosmogenic.2021.nc") as ds:
            assert list(ds.variables['time'][:]) == [
                0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334
            ]

    def test_primary_variable_present(self, tmp_path):
        data = np.random.rand(12, 4, 8)
        lat  = np.arange(-89.5, -85.5, 1.0)
        lon  = np.arange(-179.5, -171.5, 1.0)
        cfg  = {'output': {'directory': str(tmp_path), 'compress': False}}
        write_year(2020, data, lat, lon, make_aux(4, 8), cfg)
        with nc.Dataset(tmp_path / "flux_c14.cosmogenic.2020.nc") as ds:
            assert 'c14flux' in ds.variables
            assert ds.variables['c14flux'].units == 'umol m-2 s-1'

    def test_data_values_preserved(self, tmp_path):
        data = np.random.rand(12, 4, 8)
        lat  = np.arange(-89.5, -85.5, 1.0)
        lon  = np.arange(-179.5, -171.5, 1.0)
        cfg  = {'output': {'directory': str(tmp_path), 'compress': False}}
        write_year(2020, data, lat, lon, make_aux(4, 8), cfg)
        with nc.Dataset(tmp_path / "flux_c14.cosmogenic.2020.nc") as ds:
            stored = ds.variables['c14flux'][:]
            assert np.allclose(stored, data, rtol=1e-6)

    def test_diagnostic_variables_present(self, tmp_path):
        data = np.zeros((12, 4, 8))
        lat  = np.arange(-89.5, -85.5, 1.0)
        lon  = np.arange(-179.5, -171.5, 1.0)
        cfg  = {'output': {'directory': str(tmp_path), 'compress': False}}
        write_year(2020, data, lat, lon, make_aux(4, 8), cfg)
        with nc.Dataset(tmp_path / "flux_c14.cosmogenic.2020.nc") as ds:
            for vname in ('phi', 'Pc', 'q_col', 'geomag_lat'):
                assert vname in ds.variables, f"Missing variable: {vname}"
