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
        data = np.zeros((12, 3, 4, 8))
        lat  = np.arange(-89.5, -86.5, 1.0)  # 3 values
        lon  = np.arange(-179.5, -171.5, 1.0) # 8 values
        cfg  = {'output': {'directory': str(tmp_path), 'compress': False}}
        write_year(2020, data, lat, lon, make_aux(3, 8), cfg)
        assert (tmp_path / "cosmo14C_2020.nc").exists()

    def test_dimensions_correct(self, tmp_path):
        data = np.zeros((12, 3, 4, 8))
        # lat has 3 values but data nlat=4 — exercises the NaN-fallback path in write_year
        lat  = np.arange(-89.5, -86.5, 1.0)
        lon  = np.arange(-179.5, -171.5, 1.0)
        cfg  = {'output': {'directory': str(tmp_path), 'compress': False}}
        write_year(2020, data, lat, lon, make_aux(3, 8), cfg)
        with nc.Dataset(tmp_path / "cosmo14C_2020.nc") as ds:
            assert ds.dimensions['time'].size == 12
            assert ds.dimensions['lev'].size  == 3
            assert ds.dimensions['lat'].size  == 4
            assert ds.dimensions['lon'].size  == 8

    def test_primary_variable_present(self, tmp_path):
        data = np.random.rand(12, 2, 4, 8)
        lat  = np.arange(-89.5, -85.5, 1.0)
        lon  = np.arange(-179.5, -171.5, 1.0)
        cfg  = {'output': {'directory': str(tmp_path), 'compress': False}}
        write_year(2020, data, lat, lon, make_aux(4, 8), cfg)
        with nc.Dataset(tmp_path / "cosmo14C_2020.nc") as ds:
            assert 'C14_isoflux' in ds.variables
            assert ds.variables['C14_isoflux'].units == 'TgC permil yr-1'

    def test_data_values_preserved(self, tmp_path):
        data = np.random.rand(12, 2, 4, 8)
        lat  = np.arange(-89.5, -85.5, 1.0)
        lon  = np.arange(-179.5, -171.5, 1.0)
        cfg  = {'output': {'directory': str(tmp_path), 'compress': False}}
        write_year(2020, data, lat, lon, make_aux(4, 8), cfg)
        with nc.Dataset(tmp_path / "cosmo14C_2020.nc") as ds:
            stored = ds.variables['C14_isoflux'][:]
            assert np.allclose(stored, data, rtol=1e-6)

    def test_diagnostic_variables_present(self, tmp_path):
        data = np.zeros((12, 2, 4, 8))
        lat  = np.arange(-89.5, -85.5, 1.0)
        lon  = np.arange(-179.5, -171.5, 1.0)
        cfg  = {'output': {'directory': str(tmp_path), 'compress': False}}
        write_year(2020, data, lat, lon, make_aux(4, 8), cfg)
        with nc.Dataset(tmp_path / "cosmo14C_2020.nc") as ds:
            for vname in ('phi', 'Pc', 'q_col', 'geomag_lat'):
                assert vname in ds.variables, f"Missing variable: {vname}"
