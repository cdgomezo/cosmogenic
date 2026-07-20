import numpy as np
from scipy.interpolate import interp1d

# Kovaltsov et al. (2012) Table 1 — energy nodes [GeV/nuc]
_E_Y = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 7.0, 10.0,
                  19.0, 49.0, 99.0, 499.0, 999.0])

# Y/π [atoms per incident nucleon] — proton and alpha
_Yp_over_pi = np.array([0.025, 0.26, 0.72, 1.29, 2.07, 5.19, 8.32, 9.72,
                          12.40, 17.45, 23.24, 48.30, 72.73])
_Ya_over_pi = np.array([0.036, 0.38, 0.89, 1.55, 2.16, 4.18, 7.17, 8.67,
                          12.40, 17.45, 23.24, 48.30, 72.73])

# Build log-log interpolators: Y = (Y/π) × π, interpolated in log space
_log_Yp = interp1d(np.log(_E_Y), np.log(_Yp_over_pi * np.pi),
                    kind='linear', fill_value='extrapolate')
_log_Ya = interp1d(np.log(_E_Y), np.log(_Ya_over_pi * np.pi),
                    kind='linear', fill_value='extrapolate')


def Y_proton(E):
    """
    Proton 14C yield function [atoms per incident nucleon], omnidirectional (π already included).
    E: kinetic energy per nucleon [GeV/nuc]
    Returns Y = (Y/π) × π, where Y/π values are from Kovaltsov et al. (2012) Table 1.
    The π factor converts from the tabulated differential yield to the full solid-angle yield.
    """
    return np.exp(_log_Yp(np.log(np.asarray(E, dtype=float))))


def Y_alpha(E):
    """
    Alpha 14C yield function [atoms per incident nucleon], omnidirectional (π already included).
    E: kinetic energy per nucleon [GeV/nuc]
    Returns Y = (Y/π) × π, where Y/π values are from Kovaltsov et al. (2012) Table 1.
    The π factor converts from the tabulated differential yield to the full solid-angle yield.
    """
    return np.exp(_log_Ya(np.log(np.asarray(E, dtype=float))))
