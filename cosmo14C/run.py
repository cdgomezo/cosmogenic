import os
import hashlib
import json
import yaml
import numpy as np
from .build_lookup import build_local_Q_table, load_local_Q_table
from .solar_modulation import get_phi_series
from .geomagnetic import get_geomag_lat_grid
from .vertical_profile import load_tm5_coeffs, get_shape_weights
from .grid import make_lat_lon_arrays, make_Pc_grid, make_cell_area
from .isoflux import to_isoflux
from .output import write_year


def _config_hash(cfg):
    """Short hash of production config — used to detect when lookup table needs rebuild."""
    prod_cfg = cfg.get('production', {})
    return hashlib.md5(json.dumps(prod_cfg, sort_keys=True).encode()).hexdigest()[:8]


def _ensure_lookup_table(cfg):
    """Build local_Q lookup table if absent or config has changed."""
    path = cfg['production']['lookup_table']
    hash_path = path + '.hash'
    current_hash = _config_hash(cfg)

    if os.path.exists(path) and os.path.exists(hash_path):
        with open(hash_path) as f:
            if f.read().strip() == current_hash:
                print(f"Using cached lookup table: {path}")
                return

    print("Building local_Q lookup table (takes ~30 s)...")
    dir_part = os.path.dirname(path)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)
    build_local_Q_table(path, cfg['production']['phi_grid_MV'],
                        cfg['production']['Pc_grid_GV'])
    with open(hash_path, 'w') as f:
        f.write(current_hash)


def _build_shape_volume(Pc_2d, a_arr, b_arr, mb_file, p_surf_hPa, cache):
    """
    For each unique Pc value in Pc_2d, compute shape weights.
    Uses a dict cache keyed by Pc rounded to 0.1 GV to avoid redundant computation.
    Returns shape (nlev, nlat, nlon).
    """
    nlat, nlon = Pc_2d.shape
    # Probe number of levels from a dummy call
    dummy_w = get_shape_weights(0.0, a_arr, b_arr, mb_file=mb_file, p_surf_hPa=p_surf_hPa)
    nlev = len(dummy_w)

    shape_vol = np.empty((nlev, nlat, nlon))
    unique_Pc = np.unique(np.round(Pc_2d, 1))
    weight_map = {}
    for pc in unique_Pc:
        key = round(float(pc), 1)
        if key not in cache:
            cache[key] = get_shape_weights(pc, a_arr, b_arr,
                                           mb_file=mb_file, p_surf_hPa=p_surf_hPa)
        weight_map[key] = cache[key]

    for i in range(nlat):
        for j in range(nlon):
            key = round(float(Pc_2d[i, j]), 1)
            shape_vol[:, i, j] = weight_map[key]

    return shape_vol


def run_pipeline(cfg):
    """Execute the full Phase 2 pipeline for all years in cfg['period']."""
    year_start = cfg['period']['start']
    year_end   = cfg['period']['end']
    lat_res    = cfg['grid']['lat_res_deg']
    lon_res    = cfg['grid']['lon_res_deg']
    M_1e22     = cfg['geomagnetic']['M_1e22']
    p_surf     = cfg['vertical']['p_surf_hPa']
    tm5_file   = cfg['vertical']['tm5_coeffs']
    mb_file    = cfg['vertical']['masarik_beer_table']

    # Step 1: Ensure lookup table
    _ensure_lookup_table(cfg)
    interp_Q = load_local_Q_table(cfg['production']['lookup_table'])

    # Step 2: Grid
    lat, lon = make_lat_lon_arrays(lat_res, lon_res)
    cell_area = make_cell_area(lat, lat_res, lon_res)
    a_arr, b_arr = load_tm5_coeffs(tm5_file)

    # Step 3: Solar phi for entire period
    phi_series = get_phi_series(year_start, year_end, cfg['solar_modulation'])

    _shape_cache = {}
    for year in range(year_start, year_end + 1):
        print(f"\n--- Processing {year} ---")

        # Geomagnetic lat and Pc (updated annually)
        geomag_lat = get_geomag_lat_grid(year, lat, lon)
        Pc_grid = np.clip(
            make_Pc_grid(geomag_lat, M_1e22),
            cfg['production']['Pc_grid_GV']['min'],
            cfg['production']['Pc_grid_GV']['max'],
        )

        # Vertical shape volume (cached per unique Pc)
        print("  Computing vertical shape weights...")
        shape_vol = _build_shape_volume(Pc_grid, a_arr, b_arr, mb_file, p_surf, _shape_cache)
        nlev = shape_vol.shape[0]

        phi_arr  = np.empty(12)
        q_col_3d = np.empty((12, len(lat), len(lon)))
        data_4d  = np.empty((12, nlev, len(lat), len(lon)))

        for mi, month in enumerate(range(1, 13)):
            phi = phi_series.get((year, month), None)
            if phi is None:
                raise ValueError(f"No phi value for ({year}, {month})")
            phi_arr[mi] = phi

            # Columnar production via lookup table
            pts = np.column_stack([
                np.full(Pc_grid.size, phi),
                Pc_grid.ravel()
            ])
            q_col = interp_Q(pts).reshape(Pc_grid.shape)
            q_col_3d[mi] = q_col

            # 3D isoflux
            data_4d[mi] = to_isoflux(q_col, shape_vol, cell_area)
            print(f"  Month {month:02d}: phi={phi:.0f} MV, "
                  f"mean Q={q_col.mean():.3f} atoms cm^-2 s^-1")

        aux = {
            'phi':        phi_arr,
            'Pc':         Pc_grid,
            'q_col':      q_col_3d,
            'geomag_lat': geomag_lat,
        }
        write_year(year, data_4d, lat, lon, aux, cfg)
        print(f"  Written: {cfg['output']['directory']}/cosmo14C_{year}.nc")


def main():
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config/run.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    run_pipeline(cfg)


if __name__ == '__main__':
    main()
