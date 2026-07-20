import numpy as np
from .config import R_earth
from .production import rigidity_cutoff


def make_lat_lon_arrays(lat_res_deg, lon_res_deg):
    """
    Cell-centre latitude and longitude arrays.
    lat_res_deg=1.0 → lat shape (180,), range -89.5..89.5
    lon_res_deg=1.0 → lon shape (360,), range -179.5..179.5
    """
    nlat = int(round(180.0 / lat_res_deg))
    nlon = int(round(360.0 / lon_res_deg))
    lat = np.linspace(-90.0 + lat_res_deg / 2.0,  90.0 - lat_res_deg / 2.0, nlat)
    lon = np.linspace(-180.0 + lon_res_deg / 2.0, 180.0 - lon_res_deg / 2.0, nlon)
    return lat, lon


def make_Pc_grid(geomag_lat_2d, M_1e22=7.8):
    """
    Cutoff rigidity [GV] for every cell.
    geomag_lat_2d : 2D array [nlat, nlon] of geomagnetic latitudes [degrees]
    Returns Pc [GV], same shape.
    """
    return rigidity_cutoff(geomag_lat_2d, M_1e22)


def make_cell_area(lat_arr, lat_res_deg, lon_res_deg):
    """
    Area [cm^2] of a single longitude cell at each latitude.
    Returns shape (nlat,) — broadcast over longitude with [:, None].

    A(lat) = R_earth^2 * cos(lat) * dlat_rad * dlon_rad
    """
    dlat = np.radians(lat_res_deg)
    dlon = np.radians(lon_res_deg)
    return R_earth**2 * np.cos(np.radians(lat_arr)) * dlat * dlon
