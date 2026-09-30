"""Tests for the regional (arbitrary-resolution) path.

The global 1x1 path is covered by test_integration.py; these cover only what
cosmo14C.regional adds: the grid definition, the permil convention, and the
claim in its docstring that a regional field is computed rather than
interpolated -- i.e. that it agrees with the global field where the two share
a point.
"""

import numpy as np
import pytest

from cosmo14C.config import R_std
from cosmo14C.isoflux import to_umol_flux
from cosmo14C.regional import PERMIL_PER_UNIT, Grid, domain_budget


EUROPE = Grid(33.0, 73.0, -15.0, 35.0, 0.1)


def test_a_europe_wide_grid_at_0p1_degree():
    """400 x 500 cells, centres 33.05..72.95 and -14.95..34.95."""
    assert EUROPE.nlat == 400
    assert EUROPE.nlon == 500
    assert EUROPE.lat[0] == pytest.approx(33.05)
    assert EUROPE.lat[-1] == pytest.approx(72.95)
    assert EUROPE.lon[0] == pytest.approx(-14.95)
    assert EUROPE.lon[-1] == pytest.approx(34.95)


def test_cell_area_sums_to_the_spherical_cap_segment():
    """Analytic check: the domain is (sin lat1 - sin lat0)/2 x dlon/360 of the
    sphere, about 2.86 % for the window."""
    total = float(np.sum(EUROPE.cell_area_cm2())) * EUROPE.nlon
    earth = 4.0 * np.pi * (6.371e8 ** 2)
    expected = (np.sin(np.radians(73.0)) - np.sin(np.radians(33.0))) / 2.0 * (50.0 / 360.0)
    assert total / earth == pytest.approx(expected, rel=1e-4)


def test_permil_convention_is_exactly_a_factor_of_1000():
    assert PERMIL_PER_UNIT == 1000.0


def test_one_mole_of_14c_is_one_over_rstd_moles_of_cd14c():
    """The conversion the whole product rests on: a source of bare 14C atoms
    with no accompanying carbon contributes d(C_D14C) = dN_14 / R_std."""
    q_col = np.array([[1.0]])            # atoms cm-2 s-1
    flux = to_umol_flux(q_col)[0, 0]     # umol C_D14C m-2 s-1
    expected = 1e4 / 6.02214076e23 / R_std * 1e6
    assert flux == pytest.approx(expected, rel=1e-12)


def test_regional_field_agrees_with_the_global_one():
    """Nothing is interpolated: evaluated on the same points, the regional and
    global reconstructions must be identical to floating-point noise."""
    pytest.importorskip("ppigrf")
    from cosmo14C.regional import global_reference, reconstruct_regional

    cfg = _config()
    world, world_result = global_reference(2024, cfg, resolution=1.0)

    # A 1-degree regional window lands exactly on a subset of the global cells
    window = Grid(33.0, 73.0, -15.0, 35.0, 1.0)
    result = reconstruct_regional(2024, window, cfg)

    lat_in = np.isin(world.lat, window.lat)
    lon_in = np.isin(world.lon, window.lon)
    expected = world_result["q_col"][:, lat_in, :][:, :, lon_in]
    np.testing.assert_allclose(result["q_col"], expected, rtol=1e-12)


def test_domain_budget_is_a_share_of_the_global_one():
    pytest.importorskip("ppigrf")
    from cosmo14C.regional import global_reference, reconstruct_regional

    cfg = _config()
    lengths = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]   # 2024, a leap year

    world, world_result = global_reference(2024, cfg, resolution=1.0)
    _, pgc_global = domain_budget(world_result["q_col"], world, lengths)

    window = Grid(33.0, 73.0, -15.0, 35.0, 1.0)
    _, pgc_region = domain_budget(reconstruct_regional(2024, window, cfg)["q_col"],
                                  window, lengths)

    # The global production is a few thousand PgC permil/yr; the window holds a
    # larger share of it than of the Earth's area, because production rises
    # towards the poles.
    assert 3000.0 < pgc_global < 5500.0
    assert 0.0286 < pgc_region / pgc_global < 0.10


def _config():
    """The shipped run config, with its relative paths resolved."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    with open(root / "config" / "run.yaml") as handle:
        cfg = yaml.safe_load(handle)
    cfg["production"]["lookup_table"] = str(root / cfg["production"]["lookup_table"])
    for key in ("oulu_phi_file", "nm_rate_file"):
        cfg["solar_modulation"][key] = str(root / cfg["solar_modulation"][key])
    return cfg
