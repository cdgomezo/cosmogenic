import numpy as np
import pytest
from cosmo14C.gcr_spectrum import J_LIS_proton, J_modulated
from cosmo14C.config import E_rest

class TestJLISProton:
    def test_formula_at_1_GeV(self):
        # Verify formula directly: P = sqrt(E*(E+2*Er))
        E = 1.0
        P = np.sqrt(E * (E + 2 * E_rest))
        expected = 1.9e4 * P**(-2.78) / (1.0 + 0.4866 * P**(-2.51))
        assert abs(J_LIS_proton(E) - expected) / expected < 1e-12

    def test_decreases_with_energy(self):
        # LIS is a falling power-law spectrum
        assert J_LIS_proton(1.0) > J_LIS_proton(10.0) > J_LIS_proton(100.0)

    def test_positive_for_valid_energies(self):
        for E in [0.1, 0.5, 1.0, 10.0, 100.0, 999.0]:
            assert J_LIS_proton(E) > 0.0


class TestJModulated:
    def test_zero_phi_equals_LIS_proton(self):
        # At phi=0, the force-field shift is zero: J_mod = J_LIS exactly
        for E in [0.5, 1.0, 5.0, 10.0]:
            ratio = J_modulated(E, 0.0, 'p') / J_LIS_proton(E)
            assert abs(ratio - 1.0) < 1e-12

    def test_zero_phi_alpha_equals_proton_LIS(self):
        # At phi=0, alpha Phi_GeV = 0*1e-3*(2/4)=0, so same as proton LIS
        # (ALPHA_RATIO NOT applied inside J_modulated)
        for E in [1.0, 10.0]:
            ratio = J_modulated(E, 0.0, 'a') / J_LIS_proton(E)
            assert abs(ratio - 1.0) < 1e-12

    def test_high_phi_suppresses_low_energy(self):
        # Solar modulation attenuates low-energy particles
        assert J_modulated(0.3, 1000.0, 'p') < J_LIS_proton(0.3)

    def test_high_phi_negligible_effect_at_high_energy(self):
        # At 100 GeV/nuc, phi=1000 MV has reduced effect (~5%) compared to low energy
        # Force-field approximation: effect scales as phi*Z/A / E, so diminishes at high E
        ratio = J_modulated(100.0, 1000.0, 'p') / J_LIS_proton(100.0)
        assert abs(ratio - 1.0) < 0.05

    def test_alpha_shift_smaller_than_proton(self):
        # Alphas: Phi_GeV = phi*1e-3*(2/4) = phi*1e-3*0.5 — half the proton shift
        # So alpha modulation is weaker; at E=0.5, J_mod_alpha > J_mod_proton
        E, phi = 0.5, 1000.0
        assert J_modulated(E, phi, 'a') > J_modulated(E, phi, 'p')
