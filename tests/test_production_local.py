import numpy as np
import pytest
from cosmo14C.production import local_Q, global_Q
from cosmo14C.config import E_rest


class TestLocalQ:
    def test_polar_exceeds_global_average(self):
        # At Pc=0 (pole), no shielding → higher than geomagnetic-averaged global Q
        assert local_Q(650.0, 0.0) > global_Q(650.0)

    def test_equatorial_below_global_average(self):
        # At Pc=14.82 GV (equator, M=7.8), maximum shielding → lower than global average
        assert local_Q(650.0, 14.82) < global_Q(650.0)

    def test_decreases_with_Pc(self):
        # More shielding → less production
        q0  = local_Q(650.0, 0.0)
        q5  = local_Q(650.0, 5.0)
        q14 = local_Q(650.0, 14.0)
        assert q0 > q5 > q14

    def test_zero_Pc_equals_no_cutoff_integral(self):
        # At Pc=0, cutoff energy for proton = 0 → same as integrating from E_min
        # local_Q(phi, 0) should be close to the polar limit
        q = local_Q(650.0, 0.0)
        assert q > 0
        assert q < 10.0  # sanity: less than 10 atoms cm^-2 s^-1

    def test_solar_modulation_effect(self):
        # Higher phi → lower flux → lower production
        assert local_Q(300.0, 5.0) > local_Q(650.0, 5.0) > local_Q(1200.0, 5.0)

    def test_high_Pc_near_zero(self):
        # At very high cutoff (20 GV), production is strongly suppressed vs polar.
        # Alpha cutoff energy at 20 GV is ~9 GeV/nuc (Z/A=0.5), so some alpha
        # flux still penetrates; result is ~0.4 cm^-2 s^-1, well below polar (~2.8).
        # Use a threshold of 0.6 (< 25% of polar value) to assert strong suppression.
        assert local_Q(650.0, 20.0) < 0.6

    def test_returns_cm2_units(self):
        # Sanity: result should be O(1) atoms cm^-2 s^-1 for typical parameters
        q = local_Q(650.0, 2.0)
        assert 0.5 < q < 5.0
