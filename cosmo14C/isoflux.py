import numpy as np
from .config import N_A, R_std

# ---------------------------------------------------------------------------
# Primary conversion: columnar production -> C_D14C flux for TM5
# ---------------------------------------------------------------------------
#
# TM5's radio_co2 setup (Basu et al. 2016) carries the composite tracer
#     C_D14C = CO2 x Delta14C,
# with Delta14C as a DIMENSIONLESS fraction (i.e. permil / 1000), not permil.
# C_D14C therefore has the units of a CO2 amount.
#
# Cosmogenic production injects bare 14C atoms with no accompanying total
# carbon. Writing Delta = R/R_std - 1 and C_D14C = C * Delta = N_14/R_std - C,
# a source of dN_14 with dC = 0 contributes
#     d(C_D14C) = dN_14 / R_std
# so one mole of 14C produced is equivalent to 1/R_std moles of C_D14C.
#
# Conversion chain, per unit area:
#     q_col [atoms cm^-2 s^-1]
#       x 1e4    -> atoms m^-2 s^-1
#       / N_A    -> mol 14C m^-2 s^-1
#       / R_std  -> mol C_D14C m^-2 s^-1   (dimensionless-Delta convention)
#       x 1e6    -> umol C_D14C m^-2 s^-1
#
# NOTE on the factor of 1000: no explicit /1000 appears here, and none is
# needed. Dividing by R_std without multiplying by 1000 already puts the result
# in the dimensionless-Delta convention, so the quantity this package has always
# computed was correct even when it was labelled "TgC permil" -- the label was
# the error, not the arithmetic. Miller et al. (2025) label the same number
# "Pg C per mil", which reads it correctly, since 1 TgC x dimensionless equals
# 1 PgC x permil. A further /1000 here would break that agreement by exactly
# 1000x, and scripts/make_miller_nc.py is what guards against it.
_ATOMS_CM2_TO_UMOL_M2 = 1e4 / N_A / R_std * 1e6


def to_umol_flux(q_col_2d):
    """
    Convert columnar production to a C_D14C flux [umol C_D14C m^-2 s^-1].

    This is the field TM5 consumes. It is deliberately column-integrated
    (no vertical dimension): TM5 distributes it in the vertical itself, from
    the live meteorological fields.

    q_col_2d : columnar production [atoms cm^-2 s^-1], any shape

    Returns flux [umol C_D14C m^-2 s^-1], same shape as q_col_2d.
    """
    return np.asarray(q_col_2d) * _ATOMS_CM2_TO_UMOL_M2
