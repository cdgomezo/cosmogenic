import numpy as np
from scipy.integrate import quad
from .config import E_rest, ALPHA_RATIO
from .gcr_spectrum import J_modulated
from .yield_function import Y_proton, Y_alpha


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


def global_Q(phi_MV, M_1e22=7.8):
    """
    Global columnar 14C production rate [atoms cm^-2 s^-1].

    Uses the Kovaltsov (2012) analytical geomagnetic averaging formula:
        Q = ∫ Y_p(E)·J_p(E,φ)·f_p(E) dE
          + ALPHA_RATIO × ∫ Y_a(E)·J_a(E,φ)·f_a(E) dE

    where f_s(E) is the fraction of Earth's surface accessible to species s
    with rigidity P_s(E) under a dipole field of moment M_1e22:
        f(E) = 1 − sqrt(1 − sqrt(P_s / Pc_max))  for P_s ≤ Pc_max
        f(E) = 1                                   for P_s > Pc_max

    Geomagnetic rigidity P = (A/Z) · sqrt(E·(E+2·m)) [GV]:
        proton (A=1, Z=1): P_p = sqrt(E·(E+2·m))
        alpha  (A=4, Z=2): P_a = 2 · sqrt(E·(E+2·m))

    phi_MV : solar modulation potential [MV]
    M_1e22 : geomagnetic dipole moment [10^22 A m^2]
    """
    E_min   = 0.1     # GeV/nuc — lowest Kovaltsov table node
    E_max   = 1000.0  # GeV/nuc — contribution above this is negligible
    Pc_max  = 1.9 * M_1e22   # maximum cutoff rigidity [GV] (at equator)

    def accessible_fraction(E, A_over_Z=1.0):
        """Fraction of Earth's surface where particles can penetrate.

        A_over_Z : mass-to-charge ratio (1 for proton, 2 for alpha He-4)
                   scales the geomagnetic rigidity P = (A/Z) · p_per_nuc.
        """
        P = A_over_Z * np.sqrt(E * (E + 2.0 * E_rest))   # rigidity [GV]
        if P >= Pc_max:
            return 1.0
        inner = max(0.0, 1.0 - np.sqrt(P / Pc_max))   # guard against float noise
        return 1.0 - np.sqrt(inner)

    def integrand_p(E):
        return Y_proton(E) * J_modulated(E, phi_MV, 'p') * accessible_fraction(E, 1.0)

    def integrand_a(E):
        return Y_alpha(E) * J_modulated(E, phi_MV, 'a') * accessible_fraction(E, 2.0)

    Ip, _ = quad(integrand_p, E_min, E_max, limit=200, epsrel=1e-4)
    Ia, _ = quad(integrand_a, E_min, E_max, limit=200, epsrel=1e-4)

    q_m2 = Ip + ALPHA_RATIO * Ia   # [atoms m^-2 s^-1]
    return q_m2 * 1e-4              # convert m^-2 → cm^-2
