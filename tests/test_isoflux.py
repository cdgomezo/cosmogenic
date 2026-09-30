"""Tests for the conversion of columnar production into a C_D14C flux.

There is one conversion in this module and one constant behind it, so these
check the constant against the chain written out longhand, and the three
properties a linear conversion has to have.
"""

import numpy as np
import pytest

from cosmo14C.config import N_A, R_std
from cosmo14C.isoflux import to_umol_flux


def longhand(q_col):
    """The conversion written out step by step, as the module docstring gives it."""
    atoms_per_m2 = q_col * 1e4          # atoms cm-2 s-1 -> atoms m-2 s-1
    mol_14c = atoms_per_m2 / N_A        # -> mol 14C m-2 s-1
    mol_cd14c = mol_14c / R_std         # -> mol C_D14C m-2 s-1
    return mol_cd14c * 1e6              # -> umol C_D14C m-2 s-1


class TestToUmolFlux:
    def test_matches_the_chain_written_longhand(self):
        q_col = np.array([[0.5, 1.0], [1.7, 3.4]])
        np.testing.assert_allclose(to_umol_flux(q_col), longhand(q_col), rtol=1e-12)

    def test_shape_is_preserved(self):
        assert to_umol_flux(np.ones((3, 4))).shape == (3, 4)
        assert to_umol_flux(np.ones((12, 180, 360))).shape == (12, 180, 360)

    def test_a_scalar_works(self):
        assert float(to_umol_flux(np.array(1.8))) == pytest.approx(longhand(1.8))

    def test_zero_production_is_zero_flux(self):
        assert float(to_umol_flux(np.array(0.0))) == 0.0

    def test_it_is_linear(self):
        """Doubling the production doubles the flux, with no offset."""
        single = float(to_umol_flux(np.array(1.7)))
        assert float(to_umol_flux(np.array(3.4))) == pytest.approx(2.0 * single,
                                                                   rel=1e-12)

    def test_the_magnitude_is_the_documented_one(self):
        """1 atom cm-2 s-1 is 0.0141202 umol C_D14C m-2 s-1.

        Pins the factor, so a change to R_std or a stray 1000 shows up here.
        """
        assert float(to_umol_flux(np.array(1.0))) == pytest.approx(0.01412023, rel=1e-6)

    def test_a_list_is_accepted(self):
        """asarray, not an ndarray-only interface."""
        np.testing.assert_allclose(to_umol_flux([1.0, 2.0]),
                                   longhand(np.array([1.0, 2.0])), rtol=1e-12)
