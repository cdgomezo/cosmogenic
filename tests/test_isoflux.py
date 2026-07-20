import numpy as np
import pytest
from cosmo14C.isoflux import to_isoflux
from cosmo14C.config import N_A, M_C, R_std, s_per_yr


class TestToIsoflux:
    def setup_method(self):
        # Minimal: 2 layers, 3 lat, 4 lon
        self.nlev, self.nlat, self.nlon = 2, 3, 4
        # Uniform columnar production [atoms cm^-2 s^-1]
        self.q_col = np.full((self.nlat, self.nlon), 1.7)
        # Equal shape weights summing to 1
        self.shape = np.full((self.nlev, self.nlat, self.nlon), 0.5)
        # Cell area [cm^2] — uniform for simplicity
        self.cell_area = np.full(self.nlat, 1.0e10)

    def test_output_shape(self):
        result = to_isoflux(self.q_col, self.shape, self.cell_area)
        assert result.shape == (self.nlev, self.nlat, self.nlon)

    def test_sum_over_levels_recovers_columnar(self):
        # sum_k(flux_k) should equal columnar isoflux
        result = to_isoflux(self.q_col, self.shape, self.cell_area)
        # Compute expected columnar isoflux manually
        area = self.cell_area[:, None]   # (nlat, 1)
        scalar = (s_per_yr * M_C * 1e-12) / (N_A * R_std)
        expected_col = self.q_col * area * scalar
        # Sum over levels
        assert np.allclose(result.sum(axis=0), expected_col, rtol=1e-10)

    def test_shape_weights_distribute_correctly(self):
        # Layer 0 gets 70%, layer 1 gets 30%
        shape = np.zeros((2, self.nlat, self.nlon))
        shape[0] = 0.7
        shape[1] = 0.3
        result = to_isoflux(self.q_col, shape, self.cell_area)
        assert np.allclose(result[0] / result[1], 0.7 / 0.3, rtol=1e-10)

    def test_units_order_of_magnitude(self):
        # With realistic inputs: q~1.7, cell area ~1e16 cm^2 (1° × 1° at equator)
        # Expected isoflux per layer ~O(1) TgC permil yr^-1 per grid cell
        area_eq = np.full(self.nlat, 1.23e16)   # ~1°×1° at equator
        result = to_isoflux(self.q_col, self.shape, area_eq)
        # Each value should be positive and in a reasonable range
        # With q~1.7, area~1.23e16, shape~0.5 → ~5.6 TgC permil yr^-1 per layer
        assert np.all(result > 0)
        assert np.all(result < 1e5)   # reasonable upper bound for cell isoflux

    def test_zero_production_gives_zero_isoflux(self):
        q_zero = np.zeros((self.nlat, self.nlon))
        result = to_isoflux(q_zero, self.shape, self.cell_area)
        assert np.all(result == 0.0)
