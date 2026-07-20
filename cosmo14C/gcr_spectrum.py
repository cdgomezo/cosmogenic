import numpy as np
from .config import E_rest

def J_LIS_proton(E):
    """
    Proton Local Interstellar Spectrum [particles / (m^2 sr s GeV/nuc)].
    E: kinetic energy per nucleon [GeV/nuc]
    Source: Burger et al. (2000) / Usoskin et al. (2005)
    """
    P = np.sqrt(E * (E + 2.0 * E_rest))  # magnetic rigidity [GV]
    return 1.9e4 * P**(-2.78) / (1.0 + 0.4866 * P**(-2.51))


def J_modulated(E, phi_MV, species='p'):
    """
    Force-field modulated GCR spectrum at 1 AU [particles / (m^2 sr s GeV/nuc)].
    E       : kinetic energy per nucleon [GeV/nuc]
    phi_MV  : solar modulation potential [MV]
    species : 'p' (proton, Z=1 A=1) or 'a' (alpha, Z=2 A=4)

    NOTE: ALPHA_RATIO is NOT applied here. Apply it in production integrands
    when summing species contributions.
    """
    if species not in ('p', 'a'):
        raise ValueError(f"species must be 'p' or 'a', got {species!r}")
    Zi = 1 if species == 'p' else 2
    Ai = 1 if species == 'p' else 4
    phi_GeV = phi_MV * 1e-3 * (Zi / Ai)   # energy loss potential [GeV/nuc]
    E_LIS = E + phi_GeV                     # equivalent LIS energy [GeV/nuc]

    numerator   = E     * (E     + 2.0 * E_rest)
    denominator = E_LIS * (E_LIS + 2.0 * E_rest)
    return J_LIS_proton(E_LIS) * (numerator / denominator)
