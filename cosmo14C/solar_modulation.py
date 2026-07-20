import warnings
import pandas as pd
import numpy as np


def get_phi_series(year_start, year_end, config):
    """
    Return monthly solar modulation potential φ [MV] for year_start..year_end.

    Uses Oulu published φ series when available; applies count-rate regression
    for any months not covered.

    config keys:
      oulu_phi_file       : path to CSV with columns year, month, phi_MV
      nm_rate_file        : path to CSV with columns year, month, count_rate
      regression_coeffs   : dict with keys a, b  (phi = a * rate + b)
      regression_start_year : first year of overlap window used for calibration
                              (informational — coefficients are pre-computed)

    Returns dict: {(year, month): phi_MV}
    """
    phi_df = pd.read_csv(config['oulu_phi_file'])
    nm_df  = pd.read_csv(config['nm_rate_file'])

    phi_df = phi_df.set_index(['year', 'month'])['phi_MV']
    nm_df  = nm_df.set_index(['year', 'month'])['count_rate']

    a = config['regression_coeffs']['a']
    b = config['regression_coeffs']['b']

    result = {}
    regression_used = False

    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            key = (year, month)
            if key in phi_df.index:
                result[key] = float(phi_df[key])
            elif key in nm_df.index:
                regression_used = True
                result[key] = float(a * nm_df[key] + b)
            # else: month simply absent (tests may not provide all months)

    if regression_used:
        warnings.warn(
            "Solar modulation: NM count-rate regression used for one or more months "
            "(uncertainty ~20-30 MV). Check oulu_phi_file coverage.",
            UserWarning,
            stacklevel=2,
        )

    return result
