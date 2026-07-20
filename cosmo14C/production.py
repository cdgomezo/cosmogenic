import numpy as np
from .config import E_rest


def rigidity_cutoff(geomag_lat_deg, M_1e22=7.8):
    """
    Vertical cutoff rigidity [GV] using the dipole approximation.
    geomag_lat_deg : geomagnetic latitude [degrees]
    M_1e22         : geomagnetic dipole moment [10^22 A m^2]
    """
    cos_lam = np.cos(np.radians(geomag_lat_deg))
    return 1.9 * M_1e22 * cos_lam**4


def cutoff_energy(Pc_GV, species='p'):
    """
    Minimum kinetic energy per nucleon [GeV/nuc] needed to penetrate
    cutoff rigidity Pc_GV [GV].
    species : 'p' (proton, Z=1 A=1) or 'a' (alpha, Z=2 A=4)
    """
    Zi = 1 if species == 'p' else 2
    Ai = 1 if species == 'p' else 4
    p_per_nuc = Pc_GV * (Zi / Ai)   # momentum per nucleon [GeV/c per nuc]
    return np.sqrt(E_rest**2 + p_per_nuc**2) - E_rest
