import numpy as np
import pytest
from cosmo14C.yield_function import Y_proton, Y_alpha

# Kovaltsov (2012) Table 1 — ground truth nodes
_E_nodes = [0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 7.0, 10.0, 19.0, 49.0, 99.0, 499.0, 999.0]
_Yp_over_pi = [0.025, 0.26, 0.72, 1.29, 2.07, 5.19, 8.32, 9.72,
               12.40, 17.45, 23.24, 48.30, 72.73]
_Ya_over_pi = [0.036, 0.38, 0.89, 1.55, 2.16, 4.18, 7.17, 8.67,
               12.40, 17.45, 23.24, 48.30, 72.73]


class TestYProton:
    @pytest.mark.parametrize("E, Y_over_pi", list(zip(_E_nodes, _Yp_over_pi)))
    def test_at_table_nodes(self, E, Y_over_pi):
        expected = Y_over_pi * np.pi
        assert abs(Y_proton(E) - expected) / expected < 1e-10

    def test_increases_with_energy(self):
        assert Y_proton(0.1) < Y_proton(1.0) < Y_proton(100.0)

    def test_positive(self):
        for E in [0.1, 1.0, 10.0, 100.0]:
            assert Y_proton(E) > 0.0


class TestYAlpha:
    @pytest.mark.parametrize("E, Y_over_pi", list(zip(_E_nodes, _Ya_over_pi)))
    def test_at_table_nodes(self, E, Y_over_pi):
        expected = Y_over_pi * np.pi
        assert abs(Y_alpha(E) - expected) / expected < 1e-10

    def test_higher_than_proton_at_low_energy(self):
        # Alpha yield > proton yield at low energies (nuclear interaction differences)
        assert Y_alpha(0.1) > Y_proton(0.1)
        assert Y_alpha(0.5) > Y_proton(0.5)

    def test_converges_with_proton_at_19_GeV(self):
        # Table values are equal from 19 GeV/nuc onward
        assert abs(Y_alpha(19.0) - Y_proton(19.0)) / Y_proton(19.0) < 1e-10
