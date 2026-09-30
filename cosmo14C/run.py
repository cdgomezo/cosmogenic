import os
import hashlib
import json
import yaml
import numpy as np
from .build_lookup import build_local_Q_table, load_local_Q_table
from .solar_modulation import get_phi_series
from .geomagnetic import get_geomag_lat_grid
from .grid import make_lat_lon_arrays, make_Pc_grid
from .isoflux import to_umol_flux
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


def run_pipeline(cfg):
    """
    Execute the full pipeline for all years in cfg['period'].

    Produces one (time, lat, lon) file per year holding the column-integrated
    C_D14C flux in umol m^-2 s^-1. There is no vertical dimension: TM5 does the
    vertical distribution from its own meteorological fields, so the
    Masarik & Beer profile is not applied here. The `vertical:` block in the
    config is consequently unused and ignored.
    """
    year_start = cfg['period']['start']
    year_end   = cfg['period']['end']
    lat_res    = cfg['grid']['lat_res_deg']
    lon_res    = cfg['grid']['lon_res_deg']
    M_1e22     = cfg['geomagnetic']['M_1e22']

    # Step 1: Ensure lookup table
    _ensure_lookup_table(cfg)
    interp_Q = load_local_Q_table(cfg['production']['lookup_table'])

    # Step 2: Grid
    lat, lon = make_lat_lon_arrays(lat_res, lon_res)

    # Step 3: Solar phi for entire period
    phi_series = get_phi_series(year_start, year_end, cfg['solar_modulation'])

    for year in range(year_start, year_end + 1):
        print(f"\n--- Processing {year} ---")

        # Geomagnetic lat and Pc (updated annually)
        geomag_lat = get_geomag_lat_grid(year, lat, lon)
        Pc_grid = np.clip(
            make_Pc_grid(geomag_lat, M_1e22),
            cfg['production']['Pc_grid_GV']['min'],
            cfg['production']['Pc_grid_GV']['max'],
        )

        phi_arr  = np.empty(12)
        q_col_3d = np.empty((12, len(lat), len(lon)))
        data_3d  = np.empty((12, len(lat), len(lon)))

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

            # Column C_D14C flux [umol m^-2 s^-1]
            data_3d[mi] = to_umol_flux(q_col)
            print(f"  Month {month:02d}: phi={phi:.0f} MV, "
                  f"mean Q={q_col.mean():.3f} atoms cm^-2 s^-1, "
                  f"mean flux={data_3d[mi].mean():.4e} umol m^-2 s^-1")

        aux = {
            'phi':        phi_arr,
            'Pc':         Pc_grid,
            'q_col':      q_col_3d,
            'geomag_lat': geomag_lat,
        }
        write_year(year, data_3d, lat, lon, aux, cfg)
        print(f"  Written: {cfg['output']['directory']}/flux_c14.cosmogenic.{year}.nc")


def main():
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config/run.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    run_pipeline(cfg)


if __name__ == '__main__':
    main()
