from cosmo14C import config

def test_E_rest():
    assert config.E_rest == 0.938

def test_R_std():
    assert config.R_std == 1.176e-12

def test_N_A():
    assert config.N_A == 6.02214076e23

def test_M_C():
    assert config.M_C == 12.011

def test_R_earth():
    assert config.R_earth == 6.371e8

def test_s_per_yr():
    assert config.s_per_yr == 31_557_600.0

def test_ALPHA_RATIO():
    assert config.ALPHA_RATIO == 0.3

def test_calibration_targets():
    assert config.Q_KOVALTSOV_PHI650 == 1.66
    assert config.Q_MILLER_PREFERRED == 1.64
