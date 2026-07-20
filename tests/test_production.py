import numpy as np
import pytest
from cosmo14C.production import rigidity_cutoff, cutoff_energy
from cosmo14C.config import E_rest
from cosmo14C.production import global_Q
from cosmo14C.config import Q_KOVALTSOV_PHI650


class TestRigidityCutoff:
    def test_equator(self):
        # Pc = 1.9 * 7.8 * cos^4(0°) = 1.9 * 7.8 * 1 = 14.82 GV
        assert rigidity_cutoff(0.0, M_1e22=7.8) == pytest.approx(14.82, rel=1e-10)

    def test_pole(self):
        # cos(90°) = 0 → Pc = 0
        assert rigidity_cutoff(90.0, M_1e22=7.8) == pytest.approx(0.0, abs=1e-10)

    def test_negative_latitude_symmetric(self):
        # cos^4 is even, so southern latitudes give same Pc
        assert rigidity_cutoff(-45.0) == pytest.approx(rigidity_cutoff(45.0), rel=1e-10)

    def test_decreases_toward_pole(self):
        assert rigidity_cutoff(0) > rigidity_cutoff(30) > rigidity_cutoff(60) > rigidity_cutoff(90)

    def test_scales_with_dipole_moment(self):
        # Doubling M should double Pc
        assert rigidity_cutoff(30.0, M_1e22=7.8) == pytest.approx(
            2 * rigidity_cutoff(30.0, M_1e22=3.9), rel=1e-10
        )


class TestCutoffEnergy:
    def test_zero_rigidity_gives_zero_energy(self):
        # Pc=0: Eic = sqrt(E_rest^2 + 0) - E_rest = 0
        assert cutoff_energy(0.0, 'p') == pytest.approx(0.0, abs=1e-10)

    def test_proton_at_14_GV(self):
        # p/nuc = Pc * Z/A = 14 * 1 = 14 GeV/c; Eic = sqrt(0.938^2 + 14^2) - 0.938
        expected = np.sqrt(E_rest**2 + 14.0**2) - E_rest
        assert cutoff_energy(14.0, 'p') == pytest.approx(expected, rel=1e-10)

    def test_alpha_lower_than_proton(self):
        # For alpha: p/nuc = Pc * (2/4) = 0.5*Pc — lower momentum, lower cutoff energy
        assert cutoff_energy(10.0, 'a') < cutoff_energy(10.0, 'p')

    def test_alpha_at_10_GV(self):
        # p/nuc = 10 * (2/4) = 5 GeV/c; Eic = sqrt(0.938^2 + 5^2) - 0.938
        expected = np.sqrt(E_rest**2 + 5.0**2) - E_rest
        assert cutoff_energy(10.0, 'a') == pytest.approx(expected, rel=1e-10)


class TestGlobalQ:
    def test_phi650_matches_kovaltsov(self):
        # Primary validation gate: must be within 5% of 1.66 atoms cm^-2 s^-1
        Q = global_Q(650.0)
        deviation = abs(Q - Q_KOVALTSOV_PHI650) / Q_KOVALTSOV_PHI650
        assert deviation < 0.05, (
            f"Q = {Q:.4f}, target = {Q_KOVALTSOV_PHI650:.4f}, "
            f"deviation = {deviation:.1%}"
        )

    def test_increases_at_solar_minimum(self):
        # Lower phi → less modulation → more GCR → higher production
        assert global_Q(300.0) > global_Q(650.0) > global_Q(1200.0)

    def test_higher_M_reduces_production(self):
        # Stronger dipole → more shielding → lower production
        assert global_Q(650.0, M_1e22=3.0) > global_Q(650.0, M_1e22=7.8)
