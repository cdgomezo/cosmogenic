import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .production import local_Q


def build_local_Q_table(output_path, phi_cfg, Pc_cfg):
    """
    Build the local_Q(phi, Pc) lookup table and save to output_path (.npz).

    phi_cfg : dict with keys min, max, n, scale ('log' or 'linear')
    Pc_cfg  : dict with keys min, max, n, scale ('log' or 'linear')
    """
    if phi_cfg['scale'] == 'log':
        phi_grid = np.logspace(np.log10(phi_cfg['min']),
                               np.log10(phi_cfg['max']),
                               phi_cfg['n'])
    else:
        phi_grid = np.linspace(phi_cfg['min'], phi_cfg['max'], phi_cfg['n'])

    if Pc_cfg['scale'] == 'log':
        Pc_grid = np.logspace(np.log10(max(Pc_cfg['min'], 1e-6)),
                              np.log10(Pc_cfg['max']),
                              Pc_cfg['n'])
    else:
        Pc_grid = np.linspace(Pc_cfg['min'], Pc_cfg['max'], Pc_cfg['n'])

    Q = np.empty((len(phi_grid), len(Pc_grid)))
    total = len(phi_grid) * len(Pc_grid)
    done = 0
    for i, phi in enumerate(phi_grid):
        for j, Pc in enumerate(Pc_grid):
            Q[i, j] = local_Q(phi, Pc)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{total} integrations complete")

    np.savez(output_path, Q=Q, phi_grid=phi_grid, Pc_grid=Pc_grid)
    print(f"Saved lookup table to {output_path}")


def load_local_Q_table(path):
    """
    Load precomputed local_Q table and return a callable interpolator.

    Returns a RegularGridInterpolator: interp([[phi, Pc]]) -> Q [atoms cm^-2 s^-1]
    Inputs are clipped to the table bounds (no extrapolation).
    """
    data = np.load(path)
    return RegularGridInterpolator(
        (data['phi_grid'], data['Pc_grid']),
        data['Q'],
        method='linear',
        bounds_error=False,
        fill_value=None  # extrapolate (clips to boundary)
    )
