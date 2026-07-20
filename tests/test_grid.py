import numpy as np
import pytest
from cosmo14C.grid import make_lat_lon_arrays, make_Pc_grid, make_cell_area
from cosmo14C.config import R_earth


class TestMakeLatLonArrays:
    def test_default_1deg_shape(self):
        lat, lon = make_lat_lon_arrays(1.0, 1.0)
        assert lat.shape == (180,)
        assert lon.shape == (360,)

    def test_lat_range(self):
        lat, _ = make_lat_lon_arrays(1.0, 1.0)
        assert lat[0]  == pytest.approx(-89.5)
        assert lat[-1] == pytest.approx(89.5)

    def test_lon_range(self):
        _, lon = make_lat_lon_arrays(1.0, 1.0)
        assert lon[0]  == pytest.approx(-179.5)
        assert lon[-1] == pytest.approx(179.5)


class TestMakePcGrid:
    def test_shape(self):
        geomag_lat = np.zeros((180, 360))
        Pc = make_Pc_grid(geomag_lat, M_1e22=7.8)
        assert Pc.shape == (180, 360)

    def test_equator_maximum(self):
        # geomag_lat=0 → Pc = 1.9 * 7.8 * cos^4(0) = 1.9 * 7.8 * 1 = 14.82
        geomag_lat = np.zeros((180, 360))
        Pc = make_Pc_grid(geomag_lat, M_1e22=7.8)
        assert np.all(Pc == pytest.approx(14.82, rel=1e-10))

    def test_pole_zero(self):
        geomag_lat = np.full((180, 360), 90.0)
        Pc = make_Pc_grid(geomag_lat, M_1e22=7.8)
        assert np.all(Pc == pytest.approx(0.0, abs=1e-10))

    def test_decreases_with_geomag_lat(self):
        geomag_lat = np.array([[0.0], [30.0], [60.0], [90.0]])  # shape (4, 1)
        Pc = make_Pc_grid(geomag_lat, M_1e22=7.8)
        assert Pc[0, 0] > Pc[1, 0] > Pc[2, 0] > Pc[3, 0]


class TestMakeCellArea:
    def test_shape(self):
        lat, _ = make_lat_lon_arrays(1.0, 1.0)
        area = make_cell_area(lat, 1.0, 1.0)
        assert area.shape == (180,)

    def test_total_area_equals_earth_surface(self):
        lat, _ = make_lat_lon_arrays(1.0, 1.0)
        area = make_cell_area(lat, 1.0, 1.0)
        # 360 cells per latitude band
        total = area.sum() * 360
        expected = 4 * np.pi * R_earth**2  # cm^2
        assert abs(total - expected) / expected < 0.001  # within 0.1%

    def test_equatorial_cells_larger_than_polar(self):
        lat, _ = make_lat_lon_arrays(1.0, 1.0)
        area = make_cell_area(lat, 1.0, 1.0)
        assert area[90] > area[0]   # equator (index 90) vs south pole (index 0)
