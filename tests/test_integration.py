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
        self.nc_path = tmp_path / 'flux_c14.cosmogenic.2020.nc'

    def test_output_file_exists(self):
        assert self.nc_path.exists()

    def test_dimensions(self):
        with nc.Dataset(self.nc_path) as ds:
            assert ds.dimensions['time'].size == 12
            assert ds.dimensions['lat'].size  == 180
            assert ds.dimensions['lon'].size  == 360
            # No vertical axis: TM5 distributes the column field itself
            assert 'lev' not in ds.dimensions

    def test_flux_all_positive(self):
        with nc.Dataset(self.nc_path) as ds:
            data = ds.variables['c14flux'][:]
            assert np.all(data >= 0.0)

    def test_flux_not_all_zero(self):
        with nc.Dataset(self.nc_path) as ds:
            data = ds.variables['c14flux'][:]
            assert data.sum() > 0.0

    def test_flux_magnitude_reasonable(self):
        """
        Global-mean flux should be O(1e-2) umol C_D14C m^-2 s^-1.
        Global-mean Q ~1.5-2.5 atoms cm^-2 s^-1
          x 1e4 / N_A / R_std x 1e6 -> ~2-3.5e-2 umol m^-2 s^-1.
        """
        with nc.Dataset(self.nc_path) as ds:
            data = ds.variables['c14flux'][:]
            mean_flux = float(np.mean(data))
        assert 5e-3 < mean_flux < 1e-1

    def test_global_annual_production_matches_expectation(self):
        """
        Area-weighted global annual total, converted back to PgC permil yr^-1,
        must land in the range the physics implies (~3500-6500 depending on
        solar activity). This is the end-to-end guard on the unit conversion.
        """
        M_C, s_per_yr, R_e = 12.011, 31_557_600.0, 6.371e6   # m
        with nc.Dataset(self.nc_path) as ds:
            data = ds.variables['c14flux'][:]               # (12, 180, 360)
            lat  = ds.variables['lat'][:]
        dlat = dlon = np.radians(1.0)
        area = R_e**2 * np.cos(np.radians(lat)) * dlat * dlon   # m^2, (180,)
        # umol m-2 s-1 -> mol s-1 -> gC yr-1 -> TgC yr-1 (== PgC permil yr-1)
        per_month = (data * 1e-6 * area[None, :, None]).sum(axis=(1, 2))
        annual = float(per_month.mean()) * s_per_yr * M_C * 1e-12
        assert 3000 < annual < 7000, f"global annual total {annual:.0f} out of range"

    def test_phi_diagnostic_present_and_reasonable(self):
        with nc.Dataset(self.nc_path) as ds:
            phi = ds.variables['phi'][:]
            assert phi.shape == (12,)
            assert np.all(phi > 0)
            assert np.all(phi < 2000)
