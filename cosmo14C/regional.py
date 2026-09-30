"""Cosmogenic 14C production on a regional, arbitrary-resolution grid.

The global 1x1 path (cosmo14C.run) still produces the validated product; this
module is its regional counterpart, for a bounded domain at any resolution.

WHY THERE IS NO REGRIDDING HERE. The columnar production rate is an analytic
function of exactly two things,

    q_col = Q(phi(t), Pc(lat, lon)),      Pc = 1.9 * M * cos^4(geomagnetic lat)

with phi the monthly solar modulation potential and Pc the vertical cutoff
rigidity from the IGRF dipole. Neither has a native horizontal resolution: the
lookup table Q(phi, Pc) is a physics calculation, and the geomagnetic latitude
is evaluated from the IGRF Gauss coefficients at whatever points it is asked
for. So a 0.1 degree field is computed directly at 0.1 degree, not interpolated
from 1 degree. That is the one respect in which this product is better behaved
than the ocean and terrestrial ones, which do interpolate coarse inputs.

What a fine grid does not buy. The field is smooth on the scale of the
geomagnetic dipole: over a domain the size of Europe q_col varies by roughly a
factor of two, monotonically with geomagnetic latitude, with no structure below
a few hundred kilometres. Resolving it finely costs nothing and adds nothing.
The resolution is there to match whatever the field is used alongside, not
because the signal needs it.

WHAT THE FIELD IS, VERTICALLY. It is column-integrated. Roughly two thirds of
cosmogenic 14C production happens in the stratosphere and most of the rest in
the upper troposphere; essentially none of it is a surface flux. TM5 receives
the column and distributes it in the vertical itself from its own air mass. A model that
ingests this field as if it were a surface emission will put the production in
the wrong place, and, because a regional domain is a window on a global source,
will also be missing the production transported in from outside it.
"""

from dataclasses import dataclass

import numpy as np

from .build_lookup import load_local_Q_table
from .config import M_C, N_A, R_earth, R_std, s_per_yr
from .geomagnetic import get_geomag_lat_grid
from .grid import make_Pc_grid
from .isoflux import to_umol_flux
from .solar_modulation import get_phi_series

# C_D14C carries Delta14C as a dimensionless fraction (permil / 1000); the
# D14CO2 convention carries it in permil. The two fluxes differ by exactly this.
PERMIL_PER_UNIT = 1000.0

# For the Delta14C tendency (see delta_tendency below)
G = 9.80665              # m s-2, standard gravity
M_AIR = 0.0289644        # kg mol-1, molar mass of dry air
P_STANDARD = 101325.0    # Pa, standard sea-level pressure
XCO2_2024 = 422.8e-6     # mol mol-1, global annual mean CO2 for 2024


def carbon_column(p_surface_pa=P_STANDARD, xco2=XCO2_2024):
    """Carbon in the air column above one square metre [mol C m-2].

    p_surface / (g M_air) is the moles of air in the column; multiplying by the
    CO2 mole fraction gives the moles of carbon, since CO2 carries one carbon
    atom each.

    WHY A CONSTANT PRESSURE IS THE CONSISTENT CHOICE. It is tempting to use a
    real surface pressure field and give the result orographic structure, but
    the numerator would not match: q_col from this package is the production
    integrated over the WHOLE atmospheric column, computed with no reference to
    the surface at all. Dividing a full-column production by a
    terrain-following carbon column would put mountains in the answer that are
    an artefact of mixing the two conventions. The standard column keeps
    numerator and denominator on the same footing, and the resulting tendency
    therefore varies across the domain exactly as the production does.
    """
    return p_surface_pa / (G * M_AIR) * xco2


def delta_tendency(q_col, carbon_col):
    """Rate at which cosmogenic production enriches the air column [permil yr-1].

    Cosmogenic production has no Delta14C of its own -- it injects bare 14C
    atoms with no accompanying carbon, so the ratio that Delta is defined from
    has nothing in its denominator and is formally infinite. What it does have
    is a rate of change it imparts to the Delta14C of the air it is injected
    into:

        dDelta/dt = 1000 * (dN_14/dt) / (R_std * C_column)

    with dN_14/dt the production in mol 14C m-2 s-1 and C_column the carbon in
    the air column in mol C m-2. This is the per-mil quantity that accompanies
    the production the way delta_oce accompanies the ocean flux.

    IT IS A DIAGNOSTIC, NOT A FORCING. It is what production alone would do to
    the column if nothing removed 14C from it -- no decay, no transport, no
    stratosphere-troposphere exchange, and no mixing with the rest of the
    carbon cycle. A transport model forms the real tendency from its own live
    air mass and does not want this field; it wants the flux.

    q_col      : columnar production [atoms cm-2 s-1], any shape
    carbon_col : carbon in the air column [mol C m-2], scalar or same shape
    """
    mol_14c = np.asarray(q_col) * 1e4 / N_A      # atoms cm-2 s-1 -> mol m-2 s-1
    return PERMIL_PER_UNIT * mol_14c / (R_std * carbon_col) * s_per_yr


@dataclass(frozen=True)
class Grid:
    """A regional cell-centred lat/lon grid.

    lat0/lat1, lon0/lon1 are the domain EDGES, so (33, 73, -15, 35) at 0.1
    degree gives cell centres 33.05..72.95 and -14.95..34.95.
    """

    lat0: float
    lat1: float
    lon0: float
    lon1: float
    resolution: float

    @property
    def nlat(self):
        return int(round((self.lat1 - self.lat0) / self.resolution))

    @property
    def nlon(self):
        return int(round((self.lon1 - self.lon0) / self.resolution))

    @property
    def lat(self):
        return self.lat0 + self.resolution * (np.arange(self.nlat) + 0.5)

    @property
    def lon(self):
        return self.lon0 + self.resolution * (np.arange(self.nlon) + 0.5)

    def cell_area_cm2(self):
        """Area [cm^2] of one cell at each latitude, shape (nlat,)."""
        dlat = np.radians(self.resolution)
        dlon = np.radians(self.resolution)
        return R_earth**2 * np.cos(np.radians(self.lat)) * dlat * dlon


def reconstruct_regional(year, grid, cfg, xco2=XCO2_2024, p_surface=P_STANDARD):
    """Monthly cosmogenic 14C production on `grid` for `year`.

    year      : calendar year (int)
    grid      : Grid
    cfg       : the run.yaml dict (production.lookup_table, geomagnetic.M_1e22,
                solar_modulation.*, production.Pc_grid_GV bounds)
    xco2      : CO2 mole fraction used for the Delta14C tendency [mol mol-1]
    p_surface : surface pressure used for the same [Pa]

    Returns a dict:
      phi        (12,)                solar modulation potential [MV]
      geomag_lat (nlat, nlon)         geomagnetic latitude [degrees]
      Pc         (nlat, nlon)         vertical cutoff rigidity [GV]
      q_col      (12, nlat, nlon)     columnar production [atoms cm-2 s-1]
      flux_cd14c (12, nlat, nlon)     C_D14C flux [umol m-2 s-1]
      flux_d14co2(12, nlat, nlon)     D14CO2 flux [umol m-2 s-1], permil
                                      convention = flux_cd14c * 1000
      delta_tendency (12, nlat, nlon) Delta14C tendency [permil yr-1]
      carbon_column  scalar           the column the tendency divides by [mol C m-2]
    """
    lat = grid.lat
    lon = grid.lon

    interp_Q = load_local_Q_table(cfg["production"]["lookup_table"])
    phi_series = get_phi_series(year, year, cfg["solar_modulation"])

    geomag_lat = get_geomag_lat_grid(year, lat, lon)
    Pc = np.clip(
        make_Pc_grid(geomag_lat, cfg["geomagnetic"]["M_1e22"]),
        cfg["production"]["Pc_grid_GV"]["min"],
        cfg["production"]["Pc_grid_GV"]["max"],
    )

    phi = np.empty(12)
    q_col = np.empty((12, grid.nlat, grid.nlon))
    flux = np.empty((12, grid.nlat, grid.nlon))

    for mi, month in enumerate(range(1, 13)):
        value = phi_series.get((year, month))
        if value is None:
            raise ValueError(f"No solar modulation potential for ({year}, {month})")
        phi[mi] = value

        pts = np.column_stack([np.full(Pc.size, value), Pc.ravel()])
        q_col[mi] = interp_Q(pts).reshape(Pc.shape)
        flux[mi] = to_umol_flux(q_col[mi])

    carbon_col = carbon_column(p_surface, xco2)

    return {
        "phi": phi,
        "geomag_lat": geomag_lat,
        "Pc": Pc,
        "q_col": q_col,
        "flux_cd14c": flux,
        "flux_d14co2": flux * PERMIL_PER_UNIT,
        "delta_tendency": delta_tendency(q_col, carbon_col),
        "carbon_column": carbon_col,
        "xco2": xco2,
        "p_surface": p_surface,
    }


def domain_budget(q_col, grid, month_lengths_days):
    """Annual 14C production integrated over the domain.

    Returns (atoms_per_year, pgc_permil_per_year). The second is the number the
    rest of the 14C budget is quoted in: dividing the atom flux by R_std turns
    bare 14C into C_D14C, and 1 TgC x dimensionless == 1 PgC x permil, which is
    the identity isoflux.py documents.

    q_col              : (12, nlat, nlon) [atoms cm-2 s-1]
    month_lengths_days : (12,) length of each month, so a leap year is exact
    """
    area = grid.cell_area_cm2()[:, None]          # cm^2, (nlat, 1)
    seconds = np.asarray(month_lengths_days, dtype=float) * 86400.0
    atoms = float(np.sum(q_col * area[None, :, :] * seconds[:, None, None]))
    # atoms/yr -> mol 14C -> mol C_D14C -> gC -> TgC (== PgC permil)
    pgc_permil = atoms / N_A / R_std * M_C * 1e-12
    return atoms, pgc_permil


def global_reference(year, cfg, resolution=1.0):
    """The same quantity on the global 1 degree grid, for the sanity check.

    Used only by the consistency check in scripts/make_regional_cosmogenic.py:
    the regional field must agree with the global one where they overlap, since
    both come from the same analytic Q(phi, Pc).
    """
    world = Grid(-90.0, 90.0, -180.0, 180.0, resolution)
    return world, reconstruct_regional(year, world, cfg)


__all__ = [
    "Grid",
    "PERMIL_PER_UNIT",
    "XCO2_2024",
    "P_STANDARD",
    "carbon_column",
    "delta_tendency",
    "domain_budget",
    "global_reference",
    "reconstruct_regional",
    "s_per_yr",
]
