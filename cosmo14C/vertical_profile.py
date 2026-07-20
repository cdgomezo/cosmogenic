import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def load_tm5_coeffs(path):
    """
    Load TM5 tropo34 hybrid sigma-pressure coefficients.

    CSV columns: half_level, a_Pa, b
    Returns (a_arr, b_arr) sorted surface-first (half_level descending).
    a_arr [Pa], b_arr [dimensionless], each shape (n_half_levels,).
    """
    df = pd.read_csv(path)
    required = {'half_level', 'a_Pa', 'b'}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"TM5 CSV missing columns: {missing_cols}")
    df = df.sort_values('half_level', ascending=False).reset_index(drop=True)
    return df['a_Pa'].to_numpy(dtype=float), df['b'].to_numpy(dtype=float)


def get_shape_weights(Pc_GV, a_arr, b_arr, mb_file, p_surf_hPa=1013.25):
    """
    Normalized vertical production shape weights for n_layers = len(a_arr) - 1 layers.

    Uses Masarik & Beer (2009) Table 1 (mb_file) as the vertical shape function,
    interpolated to Pc_GV. The shape is integrated over each TM5 layer's depth
    interval and normalized to sum to 1.

    Pc_GV     : geomagnetic cutoff rigidity [GV]
    a_arr     : hybrid coefficient a [Pa], shape (n_half_levels,), surface-first
    b_arr     : hybrid coefficient b [dimensionless], shape (n_half_levels,)
    mb_file   : path to Masarik & Beer (2009) Table 1 CSV
    p_surf_hPa: standard surface pressure [hPa] (default 1013.25)

    Returns shape (n_layers,) array of normalized weights, summing to 1.
    """
    g = 9.80665  # m/s^2

    # Half-level pressures [Pa]
    p_iface_Pa = a_arr + b_arr * (p_surf_hPa * 100.0)   # convert hPa→Pa

    # Atmospheric depth at each half-level [g/cm^2]
    # d = p / (g * 10) where p in Pa, g in m/s^2 → d in kg/m^2 / 10 → g/cm^2
    depth_iface = p_iface_Pa / (g * 10.0)  # g/cm^2

    # Load and interpolate M&B table
    df = pd.read_csv(mb_file)
    Pc_cols = {float(c.replace('Pc', '')): c for c in df.columns if c.startswith('Pc')}
    Pc_vals = np.array(sorted(Pc_cols.keys()))

    # Clip Pc to table range
    Pc_clipped = np.clip(Pc_GV, Pc_vals[0], Pc_vals[-1])

    # Linear interpolation between two nearest Pc columns
    if Pc_clipped <= Pc_vals[0]:
        q_profile = df[Pc_cols[Pc_vals[0]]].to_numpy(dtype=float)
    elif Pc_clipped >= Pc_vals[-1]:
        q_profile = df[Pc_cols[Pc_vals[-1]]].to_numpy(dtype=float)
    else:
        idx = np.searchsorted(Pc_vals, Pc_clipped) - 1
        frac = (Pc_clipped - Pc_vals[idx]) / (Pc_vals[idx + 1] - Pc_vals[idx])
        q_lo = df[Pc_cols[Pc_vals[idx]]].to_numpy(dtype=float)
        q_hi = df[Pc_cols[Pc_vals[idx + 1]]].to_numpy(dtype=float)
        q_profile = q_lo + frac * (q_hi - q_lo)

    depths_table = df['depth_gcm2'].to_numpy(dtype=float)

    # Build continuous q(depth) interpolator
    q_interp = interp1d(depths_table, q_profile, kind='linear',
                        bounds_error=False, fill_value=(q_profile[0], q_profile[-1]))

    # Integrate q over each layer's depth interval
    n_layers = len(a_arr) - 1
    weights = np.empty(n_layers)
    for k in range(n_layers):
        d_top = depth_iface[k + 1]   # smaller depth = higher altitude
        d_bot = depth_iface[k]       # larger depth = lower altitude
        if d_bot <= d_top:
            weights[k] = 0.0
            continue
        # Simple trapezoidal integration over layer depth interval
        n_pts = max(10, int((d_bot - d_top) / 5))
        dd = np.linspace(d_top, d_bot, n_pts)
        weights[k] = np.trapezoid(q_interp(dd), dd)

    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights
