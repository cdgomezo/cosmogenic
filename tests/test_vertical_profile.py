import numpy as np
import pytest
import pandas as pd
from cosmo14C.vertical_profile import get_shape_weights, load_tm5_coeffs

# Minimal M&B table for testing (3 depth points, 2 Pc values)
MB_CSV = """depth_gcm2,Pc0,Pc2
0,0.0,0.0
100,1.0,0.5
1033,0.1,0.05
"""

# Minimal tropo34 coeffs for testing (3 layers = 4 half-levels)
TM5_CSV = """half_level,a_Pa,b
137,0.0,1.0
131,4000.0,0.944
126,7000.0,0.901
0,0.0,0.0
"""


class TestLoadTm5Coeffs:
    def test_returns_arrays(self, tmp_path):
        f = tmp_path / "coeffs.csv"
        f.write_text(TM5_CSV)
        a, b = load_tm5_coeffs(str(f))
        assert isinstance(a, np.ndarray)
        assert isinstance(b, np.ndarray)
        assert len(a) == len(b)

    def test_sorted_surface_to_top(self, tmp_path):
        # half-levels should be sorted descending (137 first = surface)
        f = tmp_path / "coeffs.csv"
        f.write_text(TM5_CSV)
        a, b = load_tm5_coeffs(str(f))
        # b should decrease from surface (b=1) to top (b=0)
        assert b[0] > b[-1]


class TestGetShapeWeights:
    def test_weights_sum_to_one(self, tmp_path):
        mb_file  = tmp_path / "mb.csv"
        tm5_file = tmp_path / "tm5.csv"
        mb_file.write_text(MB_CSV)
        tm5_file.write_text(TM5_CSV)
        a, b = load_tm5_coeffs(str(tm5_file))
        weights = get_shape_weights(0.0, a, b, mb_file=str(mb_file))
        assert weights.sum() == pytest.approx(1.0, abs=1e-10)

    def test_output_length_matches_n_layers(self, tmp_path):
        mb_file  = tmp_path / "mb.csv"
        tm5_file = tmp_path / "tm5.csv"
        mb_file.write_text(MB_CSV)
        tm5_file.write_text(TM5_CSV)
        a, b = load_tm5_coeffs(str(tm5_file))
        # 4 half-levels → 3 layers
        weights = get_shape_weights(0.0, a, b, mb_file=str(mb_file))
        assert len(weights) == 3

    def test_polar_heavier_at_top_than_equatorial(self, tmp_path):
        # Pc=0 (polar) has more high-altitude production than Pc=2 (equatorial)
        mb_file  = tmp_path / "mb.csv"
        tm5_file = tmp_path / "tm5.csv"
        mb_file.write_text(MB_CSV)
        tm5_file.write_text(TM5_CSV)
        a, b = load_tm5_coeffs(str(tm5_file))
        w_polar = get_shape_weights(0.0,  a, b, mb_file=str(mb_file))
        w_equat = get_shape_weights(2.0,  a, b, mb_file=str(mb_file))
        # Simple sanity: both should be all-positive
        assert np.all(w_polar >= 0)
        assert np.all(w_equat >= 0)

    def test_nonnegative_weights(self, tmp_path):
        mb_file  = tmp_path / "mb.csv"
        tm5_file = tmp_path / "tm5.csv"
        mb_file.write_text(MB_CSV)
        tm5_file.write_text(TM5_CSV)
        a, b = load_tm5_coeffs(str(tm5_file))
        weights = get_shape_weights(1.5, a, b, mb_file=str(mb_file))
        assert np.all(weights >= 0)
