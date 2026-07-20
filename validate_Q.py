#!/usr/bin/env python3
"""
Validation script: global 14C production rate at phi=650 MV.

Run from the Cosmogenic/ directory:
    conda run -n carlos python validate_Q.py

Expected result: Q ≈ 1.66 atoms cm^-2 s^-1 (Kovaltsov et al. 2012)
"""
from cosmo14C.production import global_Q
from cosmo14C.config import Q_KOVALTSOV_PHI650, Q_MILLER_PREFERRED

PHI_TEST  = 650.0   # MV
TOLERANCE = 0.05    # 5% acceptance band


def main():
    print(f"Computing global Q at phi = {PHI_TEST} MV ...")
    Q = global_Q(PHI_TEST)

    dev_kov  = (Q - Q_KOVALTSOV_PHI650) / Q_KOVALTSOV_PHI650 * 100.0
    dev_mill = (Q - Q_MILLER_PREFERRED)  / Q_MILLER_PREFERRED  * 100.0

    print()
    print(f"  Computed Q           : {Q:.4f} atoms cm⁻² s⁻¹")
    print(f"  Kovaltsov (2012)     : {Q_KOVALTSOV_PHI650:.4f}  ({dev_kov:+.1f}%)")
    print(f"  Miller et al. (2025) : {Q_MILLER_PREFERRED:.4f}  ({dev_mill:+.1f}%)")

    if abs(dev_kov / 100.0) <= TOLERANCE:
        print(f"\n  PASS — within {TOLERANCE*100:.0f}% of Kovaltsov target")
    else:
        print(f"\n  FAIL — deviation {dev_kov:+.1f}% exceeds {TOLERANCE*100:.0f}% tolerance")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
