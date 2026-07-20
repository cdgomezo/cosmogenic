import numpy as np
from .config import N_A, M_C, R_std, s_per_yr


def to_isoflux(q_col_2d, shape_3d, cell_area_1d):
    """
    Convert columnar production rates to 3D isoflux [TgC permil yr^-1].

    q_col_2d    : columnar production [atoms cm^-2 s^-1], shape (nlat, nlon)
    shape_3d    : normalized vertical shape weights, shape (nlev, nlat, nlon),
                  must sum to 1 along axis=0 for each (lat, lon) cell
    cell_area_1d: cell area [cm^2] per latitude band, shape (nlat,)
                  (one value per lat, broadcast over lon)

    Returns isoflux [TgC permil yr^-1], shape (nlev, nlat, nlon)

    Conversion chain per layer k per cell:
      q_layer = q_col * shape_k              [atoms cm^-2 s^-1]
      × cell_area                            → atoms s^-1
      × s_per_yr                             → atoms yr^-1
      / N_A                                  → mol yr^-1
      × M_C                                  → gC yr^-1
      × 1e-12                                → TgC yr^-1
      / R_std                                → TgC permil yr^-1
    """
    area = cell_area_1d[:, None]   # (nlat, 1)

    # Columnar production per cell [atoms s^-1]: shape (nlat, nlon)
    q_cell = q_col_2d * area

    # Distribute over layers: shape (nlev, nlat, nlon)
    q_layer = shape_3d * q_cell[None, :, :]

    # Unit conversion
    return q_layer * s_per_yr / N_A * M_C * 1e-12 / R_std
