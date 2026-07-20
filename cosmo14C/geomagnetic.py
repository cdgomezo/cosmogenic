import numpy as np
import pandas as pd
import ppigrf.ppigrf as _pp


def _get_dipole_coeffs(year):
    """
    Return interpolated IGRF dipole Gauss coefficients g10, g11, h11 for the
    given calendar year (int or float).

    Uses ppigrf's read_shc() to load the IGRF coefficient file and interpolates
    linearly in time to the requested year.

    Returns
    -------
    g10, g11, h11 : floats  [nT]
    """
    g, h = _pp.read_shc()

    # Build a target Timestamp at Jan 1 of the requested year
    year_int = int(year)
    year_frac = float(year) - year_int
    # Fraction of the year as days
    days_in_year = 366 if _pp.is_leapyear(year_int) else 365
    day_offset = year_frac * days_in_year
    target_date = pd.Timestamp(str(year_int)) + pd.to_timedelta(day_offset, unit='D')

    # Reindex with the target date and interpolate
    index = g.index.union([target_date])
    g_interp = (g.reindex(index)
                  .groupby(index).first()
                  .interpolate(method='time')
                  .loc[target_date])
    h_interp = (h.reindex(index)
                  .groupby(index).first()
                  .interpolate(method='time')
                  .loc[target_date])

    g10 = float(g_interp[(1, 0)])
    g11 = float(g_interp[(1, 1)])
    h11 = float(h_interp[(1, 1)])
    return g10, g11, h11


def get_geomag_lat_grid(year, lat_arr, lon_arr):
    """
    Compute geomagnetic latitude [degrees] for every (lat, lon) cell.

    Uses IGRF via ppigrf to find the dipole axis direction (from g10, g11, h11
    Gauss coefficients), then computes the angle between each geographic position
    and that axis.

    Parameters
    ----------
    year    : int or float
        Calendar year for the IGRF model evaluation.
    lat_arr : array-like, shape (nlat,)
        Geographic latitudes [degrees].
    lon_arr : array-like, shape (nlon,)
        Geographic longitudes [degrees].

    Returns
    -------
    geomag_lat : ndarray, shape (nlat, nlon)
        Geomagnetic latitude [degrees] at each grid cell.
    """
    lat_arr = np.asarray(lat_arr, dtype=float)
    lon_arr = np.asarray(lon_arr, dtype=float)

    # Get IGRF dipole coefficients at the requested year
    g10, g11, h11 = _get_dipole_coeffs(year)

    # Dipole axis: geographic colatitude and longitude of the north geomagnetic pole
    pole_colat = np.degrees(np.arctan2(np.sqrt(g11**2 + h11**2), g10))
    pole_lat   = 90.0 - pole_colat
    pole_lon   = np.degrees(np.arctan2(h11, g11))

    pole_lat_r = np.radians(pole_lat)
    pole_lon_r = np.radians(pole_lon)

    lat_r = np.radians(lat_arr)   # (nlat,)
    lon_r = np.radians(lon_arr)   # (nlon,)

    # Broadcast to (nlat, nlon)
    sin_lat = np.sin(lat_r)[:, None]
    cos_lat = np.cos(lat_r)[:, None]
    dlon    = lon_r[None, :] - pole_lon_r

    # Spherical law of cosines: sin(geomag_lat) = dot product of unit vectors
    sin_geomag = (sin_lat * np.sin(pole_lat_r)
                  + cos_lat * np.cos(pole_lat_r) * np.cos(dlon))
    sin_geomag = np.clip(sin_geomag, -1.0, 1.0)
    return np.degrees(np.arcsin(sin_geomag))
