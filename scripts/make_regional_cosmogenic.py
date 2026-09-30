"""Write the cosmogenic 14C production term on a regional grid, in both the
C_D14C and the per-mil convention, plus the Delta14C tendency.

    python scripts/make_regional_cosmogenic.py --year 2024 \
        --resolution 0.1 --domain 33 73 -15 35 --tag eur010x010 \
        --config config/run.yaml --out-dir output/regional

Nothing here replaces the global path: cosmo14C.run still produces the global
1x1 C_D14C product that scripts/make_miller_nc.py validates against Miller et
al. (2025). This is its regional counterpart, built on cosmo14C.regional; read
that module's docstring for what a fine grid does and does not buy, and for the
vertical caveat.

TWO CONVENTIONS, ONE FIELD. The package computes the flux of the composite
tracer C_D14C = CO2 x Delta14C with Delta14C DIMENSIONLESS (permil/1000), which
is what TM5 reads. The delivery directory D14CO2/ carries the permil
convention, so the shipped field is that same flux x 1000. Both are written:
the D14CO2 one is the delivery, the C_D14C one is written beside it so the
factor is inspectable rather than asserted. --convention selects which is
staged for the preprocessor.

It writes into --out-dir:

  cosmo_d14cflux.<tag>.<year>.nc    monthly, the D14CO2 (permil) flux in
      umol m-2 s-1, stamped at the interval START, which is the convention
      throughout: a consumer that wants interval mid-points derives them from
      the bounds, so staging mid-points here would shift the whole axis.
  cosmo_c14flux.<tag>.<year>.nc     the same field in the C_D14C convention,
      i.e. divided by 1000.
  cosmo_delta_tendency.<tag>.<year>.nc  monthly, the Delta14C TENDENCY in
      permil yr-1: the rate at which production enriches the air column,
      1000 q / (R_std C_column). This is the per-mil field that accompanies the
      production the way delta_oce accompanies the ocean flux -- production
      itself has no Delta14C, because it injects 14C with no carbon. It is a
      diagnostic: a transport model forms the real tendency from its own live
      air mass. See cosmo14C.regional.delta_tendency.
  cosmo_consistency.<tag>.<year>.txt   the checks described below.

THE CHECKS. Three, all measured rather than asserted:
  A. regional vs global. q_col is an analytic function of (phi, Pc) alone, so
     the 0.1 degree field and the global 1 degree field must agree where they
     overlap, up to the sampling of Pc. Reported as the domain-mean relative
     difference.
  B. the permil factor. mean(d14cflux) / mean(c14flux) must be 1000 exactly.
  C. the budget. Domain-integrated annual production in PgC permil/yr, beside
     the global total, so the reader can see what fraction of a global source
     this window holds (a few per mil of the planet -- the domain is 2.6% of
     the Earth's surface and sits at high cutoff-rigidity-favourable latitudes).
  D. the tendency, against the one number that can be checked from outside:
     production alone would enrich the atmospheric column by about 6 permil per
     year, which is the same order as the observed ~5 permil per year DECLINE
     of atmospheric Delta14CO2. The two nearly cancelling is the whole reason
     the 14C budget is worth closing carefully.
"""

import argparse
import calendar
import datetime
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosmo14C.regional import (  # noqa: E402
    PERMIL_PER_UNIT,
    P_STANDARD,
    XCO2_2024,
    Grid,
    domain_budget,
    global_reference,
    reconstruct_regional,
)

logger = logging.getLogger("make_regional_cosmogenic")


def _read_attrs(path):
    """Descriptive global attributes, read from a YAML file and merged into
    every output, so the metadata travels with the data."""
    if path is None:
        return {}
    with open(path) as handle:
        loaded = yaml.safe_load(handle) or {}
    return {key: str(value).strip() for key, value in loaded.items()}


def _month_starts(year):
    """The 12 month starts as datetime64 -- the INTERVAL START convention."""
    return pd.to_datetime([f"{year}-{m:02d}-01" for m in range(1, 13)]).values


def _month_lengths(year):
    return [calendar.monthrange(year, m)[1] for m in range(1, 13)]


def _common_attrs(year, grid, result, args):
    return {
        "Conventions": "CF-1.8",
        "year": str(year),
        "geospatial_lat_resolution": f"{grid.resolution} degree",
        "geospatial_lon_resolution": f"{grid.resolution} degree",
        "domain": (
            f"{grid.lat0} to {grid.lat1} N, {grid.lon0} to {grid.lon1} E "
            f"({grid.nlat} x {grid.nlon} cells)"
        ),
        "frequency": "monthly",
        "vertical_distribution": (
            "none -- this is the column-integrated production; the transport "
            "model distributes it in the vertical from its own air mass"
        ),
        "geomagnetic_model": f"IGRF dipole via ppigrf, evaluated at {year}.0",
        "geomagnetic_dipole_moment": f"{args.dipole_moment} x 1e22 A m2",
        "solar_modulation": (
            "monthly Oulu neutron-monitor phi [MV]: "
            + ", ".join(f"{v:.0f}" for v in result["phi"])
        ),
        "yield_function": (
            "Kovaltsov, Mishev & Usoskin (2012) 14C yield function integrated "
            "against the Burger/Usoskin local interstellar GCR spectrum"
        ),
        "geomagnetic_approximation": (
            "dipole cutoff rigidity Pc = 1.9 M cos^4(geomagnetic lat), not a "
            "full IGRF/MAGNETOCOSMICS trajectory computation. Over this domain "
            "that is the dominant known bias: against Miller et al. (2025) the "
            "dipole gives 20-60 % MORE production poleward of 45 deg and "
            "30-50 % less equatorward of 40 deg"
        ),
        "reconstruction_code": "Cosmogenic/scripts/make_regional_cosmogenic.py (cosmo14C.regional)",
        "history": (
            f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} written by "
            "Cosmogenic/scripts/make_regional_cosmogenic.py"
        ),
        "creation_date": f"{datetime.date.today():%Y-%m-%d}",
    }


def _save(ds, path, compress=True):
    encoding = {}
    for name, var in ds.data_vars.items():
        encoding[name] = {"zlib": bool(compress), "complevel": 4}
    ds.to_netcdf(path, encoding=encoding)
    logger.info("wrote %s", path)


def write_flux(result, key, year, grid, args, out_path, long_name, units, comment):
    """One flux file: (time, lat, lon), monthly, stamped at the month START."""
    ds = xr.Dataset(
        {
            "c14flux": (
                ("time", "lat", "lon"),
                result[key].astype("float32"),
                {
                    "units": units,
                    "long_name": long_name,
                    "comment": comment,
                },
            )
        },
        coords={
            "time": _month_starts(year),
            "lat": grid.lat,
            "lon": grid.lon,
        },
    )
    ds["lat"].attrs = {
        "units": "degrees_north",
        "long_name": "latitude",
        "standard_name": "latitude",
    }
    ds["lon"].attrs = {
        "units": "degrees_east",
        "long_name": "longitude",
        "standard_name": "longitude",
    }
    ds["time"].attrs = {"long_name": "start of the averaging interval"}

    # Diagnostics, so the field can be re-derived from the file itself
    ds["phi"] = ("time", result["phi"].astype("float32"),
                 {"units": "MV", "long_name": "solar modulation potential"})
    ds["Pc"] = (("lat", "lon"), result["Pc"].astype("float32"),
                {"units": "GV",
                 "long_name": "vertical geomagnetic cutoff rigidity"})
    ds["geomag_lat"] = (("lat", "lon"), result["geomag_lat"].astype("float32"),
                        {"units": "degrees",
                         "long_name": "geomagnetic latitude"})
    ds["q_col"] = (("time", "lat", "lon"), result["q_col"].astype("float32"),
                   {"units": "atoms cm-2 s-1",
                    "long_name": "columnar 14C production rate"})

    ds.attrs.update(_common_attrs(year, grid, result, args))
    ds.attrs.update(_read_attrs(args.attrs))
    _save(ds, out_path, compress=not args.no_compress)
    return ds


def write_tendency(result, year, grid, args, out_path):
    """The Delta14C tendency, permil yr-1, monthly, stamped at the month START."""
    ds = xr.Dataset(
        {
            "delta_tendency": (
                ("time", "lat", "lon"),
                result["delta_tendency"].astype("float32"),
                {
                    "units": "permil yr-1",
                    "long_name": (
                        "Delta14C tendency of the air column from cosmogenic "
                        "14C production"
                    ),
                    "cell_methods": "time: mean area: mean",
                    "comment": (
                        "1000 x q / (R_std x C_column): the rate at which "
                        "cosmogenic production alone would enrich the Delta14C "
                        "of the air column above each cell. Production has no "
                        "Delta14C of its own -- it injects bare 14C atoms with "
                        "no accompanying carbon -- so this rate is the per-mil "
                        "quantity that accompanies it. DIAGNOSTIC: it excludes "
                        "decay, transport, stratosphere-troposphere exchange "
                        "and exchange with the ocean and biosphere."
                    ),
                },
            )
        },
        coords={
            "time": _month_starts(year),
            "lat": grid.lat,
            "lon": grid.lon,
        },
    )
    ds["lat"].attrs = {
        "units": "degrees_north",
        "long_name": "latitude",
        "standard_name": "latitude",
    }
    ds["lon"].attrs = {
        "units": "degrees_east",
        "long_name": "longitude",
        "standard_name": "longitude",
    }
    ds["time"].attrs = {"long_name": "start of the averaging interval"}

    ds.attrs.update(_common_attrs(year, grid, result, args))
    ds.attrs.update(
        {
            "carbon_column": f"{result['carbon_column']:.2f} mol C m-2",
            "xco2": f"{result['xco2'] * 1e6:.1f} ppm",
            "surface_pressure": f"{result['p_surface'] / 100:.2f} hPa (standard)",
        }
    )
    ds.attrs.update(_read_attrs(args.attrs_tendency or args.attrs))
    _save(ds, out_path, compress=not args.no_compress)
    return ds


def consistency_check(result, year, grid, args, out_path):
    """The three checks in the module docstring, written to a text file."""
    lengths = _month_lengths(year)

    # C. budget -- regional, then global for context
    _, pgc_permil_region = domain_budget(result["q_col"], grid, lengths)
    world, world_result = global_reference(year, args._cfg, resolution=1.0)
    _, pgc_permil_global = domain_budget(world_result["q_col"], world, lengths)

    # A. regional vs global, over the domain only
    lat_in = (world.lat >= grid.lat0) & (world.lat <= grid.lat1)
    lon_in = (world.lon >= grid.lon0) & (world.lon <= grid.lon1)
    q_world = world_result["q_col"][:, lat_in, :][:, :, lon_in]
    area_w = world.cell_area_cm2()[lat_in][:, None]
    area_r = grid.cell_area_cm2()[:, None]
    # area_* holds one cell area per latitude, so the total area of the window
    # is that column summed and multiplied by the number of longitudes.
    total_w = float(np.sum(area_w)) * int(q_world.shape[2])
    total_r = float(np.sum(area_r)) * grid.nlon
    mean_world = float(np.sum(q_world * area_w[None]) / (total_w * q_world.shape[0]))
    mean_region = float(
        np.sum(result["q_col"] * area_r[None])
        / (total_r * result["q_col"].shape[0])
    )
    rel_a = 100.0 * (mean_region - mean_world) / mean_world

    # The window's share of the Earth's surface, for the budget line below
    earth_area = 4.0 * np.pi * (6.371e8 ** 2)      # cm^2
    area_fraction = 100.0 * total_r / earth_area

    # D. the tendency, area-weighted like check A
    mean_tendency = float(
        np.sum(result["delta_tendency"] * area_r[None])
        / (total_r * result["delta_tendency"].shape[0])
    )

    # B. the permil factor
    ratio = float(
        np.mean(result["flux_d14co2"]) / np.mean(result["flux_cd14c"])
    )

    lines = [
        f"cosmogenic 14C production, {args.tag} {year} -- consistency",
        "=" * 62,
        "",
        f"A. REGIONAL ({grid.resolution} deg) vs GLOBAL (1 deg), same year, same domain",
        f"   area-weighted mean q_col, {grid.resolution:<5} deg : {mean_region:.6f} atoms cm-2 s-1",
        f"   area-weighted mean q_col, 1.0   deg : {mean_world:.6f} atoms cm-2 s-1",
        f"   difference                        : {rel_a:+.4f} %",
        "   q_col = Q(phi, Pc) is analytic, so this is the sampling of the",
        "   cutoff rigidity alone -- there is no interpolation of a coarse",
        "   input field anywhere in this product.",
        "",
        "B. THE PERMIL FACTOR",
        f"   mean(D14CO2 flux) / mean(C_D14C flux) = {ratio:.6f}",
        f"   expected exactly {PERMIL_PER_UNIT:.0f}: C_D14C carries Delta14C as a",
        "   dimensionless fraction, D14CO2 carries it in permil.",
        "",
        "C. BUDGET",
        f"   domain-integrated production, {year} : {pgc_permil_region:9.3f} PgC permil/yr",
        f"   global production,            {year} : {pgc_permil_global:9.3f} PgC permil/yr",
        f"   the domain holds                    : "
        f"{100.0 * pgc_permil_region / pgc_permil_global:.3f} % of the global source",
        f"   the domain is                       : {area_fraction:.3f} % of the"
        " Earth's surface",
        "   It holds more of the source than of the area because production",
        "   rises towards the poles as the cutoff rigidity falls.",
        "",
        "D. THE TENDENCY",
        f"   domain-mean dDelta/dt              : {mean_tendency:.3f} permil yr-1"
        "  (area-weighted)",
        f"   carbon column it divides by        : {result['carbon_column']:.2f} mol C m-2",
        f"   from                               : {result['xco2'] * 1e6:.1f} ppm CO2, "
        f"{result['p_surface'] / 100:.2f} hPa",
        "   Production alone would enrich the column by about 6 permil per",
        "   year. Atmospheric Delta14CO2 is instead FALLING at roughly 5 permil",
        "   per year: fossil dilution and ocean uptake together slightly",
        "   outweigh production. That the two are the same order is the reason",
        "   this term has to be got right, and it is the only check on this",
        "   number available from outside the calculation.",
        "",
        "WHAT THIS DOES NOT CHECK. That the delivered field is usable as a",
        "surface flux -- it is not. Cosmogenic production is overwhelmingly",
        "stratospheric and upper-tropospheric, and this is its column integral.",
        "A model that places it at the surface, or that ignores production",
        "outside the domain, will be wrong for reasons no check in this file",
        "can see.",
        "",
    ]
    Path(out_path).write_text("\n".join(lines))
    logger.info("wrote %s", out_path)
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--resolution", type=float, default=0.1,
                   help="grid spacing in degrees (default 0.1)")
    p.add_argument("--domain", type=float, nargs=4, default=[33, 73, -15, 35],
                   metavar=("LAT0", "LAT1", "LON0", "LON1"),
                   help="domain EDGES (default: the EUROCOM domain)")
    p.add_argument("--tag", default="eur010x010")
    p.add_argument("--config", default="config/run.yaml",
                   help="the cosmo14C run config, for the lookup table, the "
                        "dipole moment and the solar modulation files")
    p.add_argument("--out-dir", default="output/regional")
    p.add_argument("--attrs", default=None, metavar="FILE",
                   help="YAML of descriptive global attributes, merged into "
                        "both flux files, and into the tendency unless "
                        "--attrs-tendency is given.")
    p.add_argument("--attrs-tendency", default=None, metavar="FILE",
                   help="YAML of descriptive global attributes for the "
                        "Delta14C tendency file, which describes a different "
                        "quantity from the two flux files.")
    p.add_argument("--xco2", type=float, default=XCO2_2024 * 1e6, metavar="PPM",
                   help="CO2 mole fraction of the air column the Delta14C "
                        f"tendency divides by, in ppm (default {XCO2_2024 * 1e6:g}, "
                        "the 2024 global annual mean).")
    p.add_argument("--surface-pressure", type=float, default=P_STANDARD / 100.0,
                   metavar="HPA",
                   help="Surface pressure of that column, in hPa (default "
                        f"{P_STANDARD / 100:g}). Constant on purpose: q_col is a "
                        "whole-atmosphere column, so a terrain-following "
                        "denominator would not match it.")
    p.add_argument("--no-compress", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    with open(cfg_path) as handle:
        cfg = yaml.safe_load(handle)

    # Paths inside the config are relative to the package root
    lookup = Path(cfg["production"]["lookup_table"])
    if not lookup.is_absolute():
        cfg["production"]["lookup_table"] = str(root / lookup)
    for key in ("oulu_phi_file", "nm_rate_file"):
        path = Path(cfg["solar_modulation"][key])
        if not path.is_absolute():
            cfg["solar_modulation"][key] = str(root / path)

    args._cfg = cfg
    args.dipole_moment = cfg["geomagnetic"]["M_1e22"]

    grid = Grid(args.domain[0], args.domain[1], args.domain[2], args.domain[3],
                args.resolution)
    logger.info("grid %d x %d cells, %.2f deg", grid.nlat, grid.nlon,
                grid.resolution)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = reconstruct_regional(args.year, grid, cfg,
                                  xco2=args.xco2 * 1e-6,
                                  p_surface=args.surface_pressure * 100.0)

    write_flux(
        result, "flux_d14co2", args.year, grid, args,
        out_dir / f"cosmo_d14cflux.{args.tag}.{args.year}.nc",
        long_name="Cosmogenic 14C production as a D14CO2 flux",
        units="umol m-2 s-1",
        comment=(
            "D14CO2 convention: the flux of CO2 x Delta14C with Delta14C in "
            "PERMIL. Cosmogenic production injects bare 14C atoms with no "
            "accompanying total carbon, so one mole of 14C produced is "
            "1000/R_std moles of this quantity (R_std = 1.176e-12). "
            "Column-integrated; the transport model distributes it vertically."
        ),
    )
    write_flux(
        result, "flux_cd14c", args.year, grid, args,
        out_dir / f"cosmo_c14flux.{args.tag}.{args.year}.nc",
        long_name="Cosmogenic 14C production as a C_D14C flux",
        units="umol m-2 s-1",
        comment=(
            "C_D14C convention: Delta14C DIMENSIONLESS (permil/1000), after "
            "Basu et al. (2016) and what TM5 reads. Exactly 1/1000 of the "
            "D14CO2 file beside it."
        ),
    )

    write_tendency(result, args.year, grid, args,
                   out_dir / f"cosmo_delta_tendency.{args.tag}.{args.year}.nc")

    consistency_check(result, args.year, grid, args,
                      out_dir / f"cosmo_consistency.{args.tag}.{args.year}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
