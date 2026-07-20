"""
Smoke test: run a single month with real physics, verify output structure and
basic physical constraints. Uses real data stubs from data/ — requires that
Task 0 data files exist with at least 2020-01 entries.
"""
import numpy as np
import pytest
import netCDF4 as nc
import os
import yaml


import pathlib as _pathlib
_OULU_PHI_PATH = _pathlib.Path(__file__).parent.parent / 'data/solar_phi/oulu_phi.csv'


# Skip entire module if Oulu data stubs have only placeholder content
def _has_real_phi_data():
    try:
        import pandas as pd
        path = _pathlib.Path(__file__).parent.parent / 'data/solar_phi/oulu_phi.csv'
        df = pd.read_csv(path)
        return len(df) >= 12
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_real_phi_data(),
    reason="Oulu phi data not yet populated (run Task 0 first)"
)


class TestIntegration:
    @pytest.fixture(autouse=True)
    def run_single_year(self, tmp_path):
        """Run the pipeline for a single year and store output path."""
        from cosmo14C.run import run_pipeline
        with open('config/run.yaml') as f:
            cfg = yaml.safe_load(f)
        # Override to run just one year and write to tmp
        cfg['period']['start'] = 2020
        cfg['period']['end']   = 2020
        cfg['output']['directory'] = str(tmp_path)
        run_pipeline(cfg)
        self.nc_path = tmp_path / 'cosmo14C_2020.nc'

    def test_output_file_exists(self):
        assert self.nc_path.exists()

    def test_dimensions(self):
        with nc.Dataset(self.nc_path) as ds:
            assert ds.dimensions['time'].size == 12
            assert ds.dimensions['lev'].size  == 34
            assert ds.dimensions['lat'].size  == 180
            assert ds.dimensions['lon'].size  == 360

    def test_isoflux_all_positive(self):
        with nc.Dataset(self.nc_path) as ds:
            data = ds.variables['C14_isoflux'][:]
            assert np.all(data >= 0.0)

    def test_isoflux_not_all_zero(self):
        with nc.Dataset(self.nc_path) as ds:
            data = ds.variables['C14_isoflux'][:]
            assert data.sum() > 0.0

    def test_global_sum_reasonable(self):
        """
        Summing over all cells and levels for one month should give O(1e4) TgC permil yr^-1.
        Global production ~2 atoms cm^-2 s^-1 × 5.1e18 cm^2 = 1e19 atoms s^-1
        × 3.16e7 s/yr / 6e23 * 12 / 1.2e-12 ≈ 1e4–1e5 TgC permil yr^-1 globally.
        """
        with nc.Dataset(self.nc_path) as ds:
            data = ds.variables['C14_isoflux'][:]
            month_total = float(data[0].sum())
        assert 1e3 < month_total < 1e6

    def test_phi_diagnostic_present_and_reasonable(self):
        with nc.Dataset(self.nc_path) as ds:
            phi = ds.variables['phi'][:]
            assert phi.shape == (12,)
            assert np.all(phi > 0)
            assert np.all(phi < 2000)
