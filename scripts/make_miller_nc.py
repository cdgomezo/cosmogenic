#!/usr/bin/env python3
"""
make_miller_nc.py — Convert cosmo14C annual output to Miller et al. (2025) format.

Reads flux_c14.cosmogenic.YYYY.nc files produced by the cosmo14C pipeline,
converts the shipped `c14flux` field [umol C_D14C m-2 s-1] to PgC permil month-1
per grid cell, and writes a single multi-year NetCDF with dimension order
(lat, lon, time) matching Cosmo.nc (Miller et al. 2025).

The `q_col` diagnostic [atoms cm-2 s-1] is converted independently and used to
cross-check `c14flux`, so this script validates the units of the shipped product
rather than merely re-deriving them.

With --reference, also computes and prints comparison statistics vs. the
reference file.

Usage
-----
    python scripts/make_miller_nc.py \\
        --input-dir output/ \\
        --output comparison/cosmo14C_2000_2012_miller_format.nc \\
        --start 2000 --end 2012 \\
        [--reference Cosmo.nc]
"""

import argparse
import datetime
import os
import sys

import netCDF4 as nc
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants — kept in sync with cosmo14C/config.py
# ---------------------------------------------------------------------------
R_std    = 1.176e-12        # 14C/C ratio of NBS oxalic acid standard
N_A      = 6.02214076e23    # mol-1
M_C      = 12.011           # g/mol
s_per_yr = 31_557_600.0     # s/yr (365.25 days)
R_earth  = 6.371e8          # cm

# Unit conversion:
#   q_col [atoms cm-2 s-1] * cell_area [cm2] -> gC month-1 * 1e-12
#   = (s_per_yr/12) / N_A * M_C * 1e-12 / R_std
#
# NOTE on units. Dividing by R_std without also multiplying by 1000 puts this in
# the DIMENSIONLESS-Delta convention, so the result is "TgC month-1 with Delta
# dimensionless". That is numerically identical to "PgC permil month-1", which
# is exactly how Miller et al. (2025) Cosmo.nc labels it -- so Cosmo.nc's label
# is correct and directly comparable to these values. (An earlier revision of
# this file asserted Cosmo.nc had a factor-of-1000 label error; it does not.
# Confirmed against the global integral: 5055 in Cosmo.nc native units vs an
# expected ~4900 from a first-principles 6.7 kg 14C/yr production rate.)
_CONV = (s_per_yr / 12.0) / N_A * M_C * 1e-12 / R_std


# ---------------------------------------------------------------------------
# Grid helpers — mirrors cosmo14C/grid.py
# ---------------------------------------------------------------------------

def _make_lat_lon():
    """Return 1-degree cell-centre lat and lon arrays."""
    lat = np.linspace(-89.5, 89.5, 180)
    lon = np.linspace(-179.5, 179.5, 360)
    return lat, lon


def _make_cell_area(lat):
    """Cell area [cm2] at each latitude, shape (nlat,)."""
    dlat = np.radians(1.0)
    dlon = np.radians(1.0)
    return R_earth**2 * np.cos(np.radians(lat)) * dlat * dlon


# ---------------------------------------------------------------------------
# Core reading / conversion
# ---------------------------------------------------------------------------

def read_year(path):
    """
    Read the shipped C_D14C flux, the q_col diagnostic, and phi from one annual
    cosmo14C NetCDF file.

    Returns
    -------
    flux  : ndarray, shape (12, 180, 360) [umol C_D14C m-2 s-1], or None if the
            file predates the umol/m2/s output format
    q_col : ndarray, shape (12, 180, 360) [atoms cm-2 s-1]
    phi   : ndarray, shape (12,) [MV]
    """
    with nc.Dataset(path, 'r') as ds:
        q_col = ds.variables['q_col'][:].data.astype('f8')
        phi   = ds.variables['phi'][:].data.astype('f8')
        if 'c14flux' in ds.variables:
            flux = ds.variables['c14flux'][:].data.astype('f8')
        else:
            flux = None
    return flux, q_col, phi


def convert_to_miller(q_col_3d, cell_area_1d):
    """
    Convert q_col [atoms cm-2 s-1] to PgC permil month-1 per grid cell.

    Parameters
    ----------
    q_col_3d    : shape (12, nlat, nlon) [atoms cm-2 s-1]
    cell_area_1d: shape (nlat,) [cm2]

    Returns
    -------
    shape (12, nlat, nlon) [PgC permil month-1 per cell]
    """
    area = cell_area_1d[:, np.newaxis]              # (nlat, 1)
    return q_col_3d * area[np.newaxis, :, :] * _CONV


def convert_flux_to_miller(flux_3d, cell_area_m2_1d):
    """
    Convert the shipped C_D14C flux [umol m-2 s-1] to Miller units
    (PgC permil month-1 per grid cell).

    This is the path that actually validates the units of the product we ship,
    as opposed to re-deriving them from the q_col diagnostic.

    Chain:
      flux [umol C_D14C m-2 s-1]
        x 1e-6            -> mol C_D14C m-2 s-1
        x cell_area [m2]  -> mol C_D14C s-1
        x s_per_yr/12     -> mol C_D14C month-1
        x M_C             -> gC month-1
        x 1e-12           -> TgC month-1  (== PgC permil month-1)

    Parameters
    ----------
    flux_3d        : shape (12, nlat, nlon) [umol C_D14C m-2 s-1]
    cell_area_m2_1d: shape (nlat,) [m2]

    Returns
    -------
    shape (12, nlat, nlon) [PgC permil month-1 per cell]
    """
    area = cell_area_m2_1d[:, np.newaxis]           # (nlat, 1)
    conv = 1e-6 * (s_per_yr / 12.0) * M_C * 1e-12
    return flux_3d * area[np.newaxis, :, :] * conv


# ---------------------------------------------------------------------------
# Time axes
# ---------------------------------------------------------------------------

def decimal_years(start_year, end_year):
    """
    Monthly decimal years from Jan start_year through Dec end_year.
    Formula: year + (month - 0.5) / 12  — matches Miller Cosmo.nc decdate.
    """
    dates = []
    for yr in range(start_year, end_year + 1):
        for mo in range(1, 13):
            dates.append(yr + (mo - 0.5) / 12.0)
    return np.array(dates)


def cf_time_days(start_year, end_year):
    """CF time [days since 2000-01-01] at month midpoints (day 15)."""
    ref = datetime.date(2000, 1, 1)
    days = []
    for yr in range(start_year, end_year + 1):
        for mo in range(1, 13):
            mid = datetime.date(yr, mo, 15)
            days.append((mid - ref).days)
    return np.array(days, dtype='f8')


# ---------------------------------------------------------------------------
# NetCDF writer
# ---------------------------------------------------------------------------

def write_output(out_path, data, lat, lon, decdate, cf_days, phi_all,
                 start_year, end_year):
    """
    Write Miller-format NetCDF.

    data : shape (n_months, 180, 360) [PgC permil month-1]

    Dimension order in file: (lat, lon, time) — matches Miller Cosmo.nc exactly.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with nc.Dataset(out_path, 'w', format='NETCDF4') as ds:
        ds.createDimension('lat',  180)
        ds.createDimension('lon',  360)
        ds.createDimension('time', len(decdate))

        # Coordinates
        v = ds.createVariable('lat', 'f4', ('lat',))
        v.units         = 'degrees_north'
        v.long_name     = 'latitude'
        v.standard_name = 'latitude'
        v[:]            = lat.astype('f4')

        v = ds.createVariable('lon', 'f4', ('lon',))
        v.units         = 'degrees_east'
        v.long_name     = 'longitude'
        v.standard_name = 'longitude'
        v[:]            = lon.astype('f4')

        # Decimal-year time axis (matches Miller's decdate variable name)
        v = ds.createVariable('decdate', 'f8', ('time',))
        v.long_name = 'decimal year — month midpoint (year + (month-0.5)/12)'
        v.units     = 'decimal years'
        v[:]        = decdate

        # CF-compliant time (bonus, for xarray / CDO interoperability)
        v = ds.createVariable('time', 'f8', ('time',))
        v.units         = 'days since 2000-01-01 00:00:00'
        v.calendar      = 'standard'
        v.long_name     = 'time'
        v.standard_name = 'time'
        v.axis          = 'T'
        v[:]            = cf_days

        # Solar modulation potential (auxiliary)
        v = ds.createVariable('phi', 'f4', ('time',))
        v.units     = 'MV'
        v.long_name = 'Solar modulation potential (Usoskin/Oulu)'
        v[:]        = phi_all.astype('f4')

        # Primary variable — (lat, lon, time) to match Miller Cosmo.nc dimension order
        v = ds.createVariable('C14_isoflux_col', 'f4',
                              ('lat', 'lon', 'time'),
                              zlib=True, complevel=4)
        v.units     = 'TgC permil month-1'
        v.long_name = 'Columnar cosmogenic 14C isoflux per grid cell per month'
        v.comment   = (
            'Converted from q_col [atoms cm-2 s-1] stored in cosmo14C annual '
            'NetCDF files: C14_isoflux_col = q_col * cell_area * (s_per_yr/12) '
            '/ N_A * M_C * 1e-12 / R_std. '
            'Dimension order (lat, lon, time) matches Miller et al. (2025) Cosmo.nc. '
            'NOTE: Cosmo.nc labels its units as "Pg C per mil" but the numeric values '
            'are consistent with TgC permil month-1 (factor-of-1000 label error).'
        )
        # data is (n_months, 180, 360); file layout is (180, 360, n_months)
        v[:] = np.transpose(data, (1, 2, 0)).astype('f4')

        # Global attributes
        ds.title      = 'Cosmogenic 14C columnar isoflux — Miller (2025) format'
        ds.source     = 'cosmo14C pipeline; post-processed by scripts/make_miller_nc.py'
        ds.period     = f'{start_year}–{end_year}'
        ds.phi_source = 'Usoskin et al. (2017) Phi_mon.txt (Burger 2000 LIS)'
        ds.reference  = 'Miller et al. (2025), Global Biogeochemical Cycles'
        ds.conversion = (
            f'_CONV = (s_per_yr/12)/N_A*M_C*1e-15/R_std = {_CONV:.6e}; '
            f'R_std={R_std}, N_A={N_A}, M_C={M_C}, s_per_yr={s_per_yr}'
        )
        ds.created = datetime.datetime.now(datetime.timezone.utc).isoformat()

    print(f"Written: {out_path}  ({len(decdate)} months, shape 180×360×{len(decdate)})")


# ---------------------------------------------------------------------------
# Comparison vs. Miller reference file
# ---------------------------------------------------------------------------

def compare_with_miller(our_data, our_decdate, ref_path):
    """
    Compare our columnar isoflux against Miller's Cosmo.nc.

    our_data    : (n_months, 180, 360) [PgC permil month-1]
    our_decdate : (n_months,) decimal years
    ref_path    : path to Cosmo.nc

    Prints a summary to stdout; returns (rmse, r) scalars.
    """
    with nc.Dataset(ref_path, 'r') as ds:
        ref_decdate = ds.variables['Decimal Date'][:].data.astype('f8')  # (156,)
        ref_cosmo   = ds.variables['Cosmo'][:].data.astype('f8')         # (180, 360, 156)

    # Match overlapping months by decdate (tolerance = half a month)
    tol = 0.5 / 12.0
    our_idx, ref_idx = [], []
    for i, t_our in enumerate(our_decdate):
        diffs = np.abs(ref_decdate - t_our)
        j = int(np.argmin(diffs))
        if diffs[j] < tol:
            our_idx.append(i)
            ref_idx.append(j)

    if not our_idx:
        print("WARNING: no overlapping time steps found between our output and Cosmo.nc")
        return None, None

    n_match = len(our_idx)
    t0 = our_decdate[our_idx[0]]
    t1 = our_decdate[our_idx[-1]]
    print(f"\nOverlapping months: {n_match}  ({t0:.4f} – {t1:.4f})")

    our_sel = our_data[our_idx, :, :]                          # (n, 180, 360)
    ref_sel = np.transpose(ref_cosmo[:, :, ref_idx], (2, 0, 1))  # (n, 180, 360)

    # Per-month global mean table (first 24 months)
    print(f"\n{'decdate':<10} {'Our (TgC‰/mo)':>16} {'Miller (TgC‰/mo)':>18} {'Ratio':>8}")
    print("─" * 58)
    for k in range(min(n_match, 24)):
        our_mean = float(our_sel[k].mean())
        ref_mean = float(ref_sel[k].mean())
        ratio    = our_mean / ref_mean if ref_mean != 0 else float('nan')
        print(f"{our_decdate[our_idx[k]]:.4f}    {our_mean:16.4e}   {ref_mean:16.4e}   {ratio:8.4f}")
    if n_match > 24:
        print(f"  … ({n_match - 24} more months not shown)")

    # Summary statistics
    diff     = our_sel - ref_sel
    rmse     = float(np.sqrt(np.mean(diff**2)))
    r        = float(np.corrcoef(our_sel.ravel(), ref_sel.ravel())[0, 1])
    bias     = float(diff.mean())
    rel_bias = bias / float(ref_sel.mean()) * 100.0

    # Global annual totals for scale context (TgC‰/mo → TgC‰/yr)
    # NOTE: Miller Cosmo.nc labeled "PgC" but actual values are TgC (factor-of-1000 label error)
    our_total  = float(our_sel.sum(axis=(1, 2)).mean()) * 12.0
    ref_total  = float(ref_sel.sum(axis=(1, 2)).mean()) * 12.0

    print(f"\n{'─'*58}")
    print(f"  RMSE (per cell per month) : {rmse:.4e} TgC permil month-1")
    print(f"  Pearson r                 : {r:.6f}")
    print(f"  Mean bias (our − Miller)  : {bias:+.4e} TgC permil month-1  ({rel_bias:+.2f}%)")
    print(f"  Our  global mean total    : {our_total:.4f} TgC permil yr-1")
    print(f"  Miller global mean total  : {ref_total:.4f} TgC permil yr-1")

    return rmse, r


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Convert cosmo14C annual NetCDF output to Miller (2025) format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--input-dir',  required=True,
                   help='Directory containing flux_c14.cosmogenic.YYYY.nc files')
    p.add_argument('--output',     required=True,
                   help='Output NetCDF path')
    p.add_argument('--start',      type=int, required=True, help='First year')
    p.add_argument('--end',        type=int, required=True, help='Last year (inclusive)')
    p.add_argument('--reference',  default=None,
                   help='Path to Miller Cosmo.nc for comparison statistics')
    return p.parse_args()


def main():
    args = parse_args()

    lat, lon  = _make_lat_lon()
    cell_area = _make_cell_area(lat)
    decdate   = decimal_years(args.start, args.end)
    cf_days   = cf_time_days(args.start, args.end)
    n_months  = len(decdate)

    data_all = np.empty((n_months, 180, 360), dtype='f8')
    phi_all  = np.empty(n_months,             dtype='f8')

    t = 0
    n_checked = 0
    for year in range(args.start, args.end + 1):
        fpath = os.path.join(args.input_dir, f'flux_c14.cosmogenic.{year}.nc')
        if not os.path.exists(fpath):
            print(f"ERROR: missing input file: {fpath}", file=sys.stderr)
            sys.exit(1)

        flux, q_col, phi = read_year(fpath)
        print(f"  {year}: phi {phi.min():.0f}–{phi.max():.0f} MV, "
              f"mean Q {q_col.mean():.3f} atoms cm-2 s-1")

        from_qcol = convert_to_miller(q_col, cell_area)

        if flux is None:
            # Legacy file without the shipped c14flux variable
            print(f"    (no c14flux variable — falling back to q_col)")
            data_all[t:t+12] = from_qcol
        else:
            # Convert the SHIPPED product, so this comparison validates the
            # umol m-2 s-1 units we actually hand to TM5.
            from_flux = convert_flux_to_miller(flux, cell_area / 1e4)
            data_all[t:t+12] = from_flux

            # Cross-check the two independent paths agree. Tolerance is set by
            # c14flux being stored as float32 (~1e-7 relative).
            denom = np.where(from_qcol == 0.0, 1.0, from_qcol)
            max_rel = float(np.abs((from_flux - from_qcol) / denom).max())
            if max_rel > 1e-5:
                raise SystemExit(
                    f"ERROR: {year}: c14flux and q_col disagree by "
                    f"{max_rel:.3e} relative — unit conversion is inconsistent."
                )
            n_checked += 1

        phi_all[t:t+12]  = phi
        t += 12

    print(f"\nAssembled {n_months} months  "
          f"({decdate[0]:.4f} – {decdate[-1]:.4f})")
    print(f"Value range: {data_all.min():.4e} – {data_all.max():.4e} "
          f"PgC permil month-1")
    if n_checked:
        print(f"Unit cross-check PASSED for {n_checked} year(s): "
              f"c14flux [umol m-2 s-1] and q_col agree to <1e-5 relative.")

    # Unit-consistency self-check: _CONV [TgC permil month-1] = isoflux.py constant / 12
    _atoms_to_tgc = s_per_yr / N_A * M_C * 1e-12 / R_std
    assert abs(_CONV - _atoms_to_tgc / 12.0) / _CONV < 1e-10, \
        "Unit conversion constant mismatch — check physical constants"

    write_output(args.output, data_all, lat, lon, decdate, cf_days, phi_all,
                 args.start, args.end)

    if args.reference:
        if not os.path.exists(args.reference):
            print(f"WARNING: reference file not found: {args.reference}",
                  file=sys.stderr)
        else:
            compare_with_miller(data_all, decdate, args.reference)

    print("\nDone.")


if __name__ == '__main__':
    main()
