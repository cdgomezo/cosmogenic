import numpy as np
import pytest
from cosmo14C.geomagnetic import get_geomag_lat_grid


LAT = np.arange(-89.5, 90.5, 1.0)   # 180 values
LON = np.arange(-179.5, 180.5, 1.0) # 360 values


class TestGeomagLatGrid:
    def test_output_shape(self):
        grid = get_geomag_lat_grid(2020, LAT, LON)
        assert grid.shape == (180, 360)

    def test_values_in_range(self):
        grid = get_geomag_lat_grid(2020, LAT, LON)
        assert np.all(grid >= -90.0)
        assert np.all(grid <= 90.0)

    def test_symmetry_roughly_antisymmetric_about_equator(self):
        # Geomagnetic pole is offset but grid should have roughly symmetric magnitudes
        grid = get_geomag_lat_grid(2020, LAT, LON)
        assert np.max(grid) > 60.0   # some high-lat cells near geomagnetic pole
        assert np.min(grid) < -60.0  # southern hemisphere counterpart

    def test_polar_cells_have_high_geomag_lat(self):
        grid = get_geomag_lat_grid(2020, LAT, LON)
        # Geographic North Pole (last row, lat=89.5) should have high geomag lat
        north_pole_row = grid[-1, :]
        assert np.mean(np.abs(north_pole_row)) > 60.0

    def test_different_years_give_slightly_different_grids(self):
        g2000 = get_geomag_lat_grid(2000, LAT, LON)
        g2020 = get_geomag_lat_grid(2020, LAT, LON)
        # Pole drifts ~0.1 deg/yr → 2 deg over 20 years → grids differ slightly
        assert not np.allclose(g2000, g2020, atol=0.01)
