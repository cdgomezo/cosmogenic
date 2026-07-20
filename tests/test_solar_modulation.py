import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from cosmo14C.solar_modulation import get_phi_series


# Minimal CSV content for mocking
OULU_PHI_CSV = "year,month,phi_MV\n2020,1,600.0\n2020,2,610.0\n2020,3,620.0\n"
OULU_NM_CSV  = "year,month,count_rate\n2020,1,6800.0\n2020,2,6750.0\n2023,6,6700.0\n"


def _make_config(oulu_phi_path, nm_path):
    return {
        'oulu_phi_file': oulu_phi_path,
        'nm_rate_file': nm_path,
        'regression_coeffs': {'a': -0.85, 'b': 1420.0},
        'regression_start_year': 2020,
    }


class TestGetPhiSeries:
    def test_returns_dict_with_month_keys(self, tmp_path):
        phi_file = tmp_path / "oulu_phi.csv"
        nm_file  = tmp_path / "nm_rates.csv"
        phi_file.write_text(OULU_PHI_CSV)
        nm_file.write_text(OULU_NM_CSV)
        cfg = _make_config(str(phi_file), str(nm_file))
        result = get_phi_series(2020, 2020, cfg)
        assert (2020, 1) in result
        assert (2020, 3) in result

    def test_published_phi_values_used_when_available(self, tmp_path):
        phi_file = tmp_path / "oulu_phi.csv"
        nm_file  = tmp_path / "nm_rates.csv"
        phi_file.write_text(OULU_PHI_CSV)
        nm_file.write_text(OULU_NM_CSV)
        cfg = _make_config(str(phi_file), str(nm_file))
        result = get_phi_series(2020, 2020, cfg)
        assert result[(2020, 1)] == pytest.approx(600.0)
        assert result[(2020, 2)] == pytest.approx(610.0)

    def test_regression_applied_for_missing_months(self, tmp_path):
        # oulu_phi only has 2020; nm_rates has 2023 entry → regression used
        phi_file = tmp_path / "oulu_phi.csv"
        nm_file  = tmp_path / "nm_rates.csv"
        phi_file.write_text(OULU_PHI_CSV)
        nm_file.write_text(OULU_NM_CSV)
        cfg = _make_config(str(phi_file), str(nm_file))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = get_phi_series(2023, 2023, cfg)
        # phi = a * count_rate + b = -0.85 * 6700 + 1420 = -4275 → clamped to 10.0
        assert (2023, 6) in result
        assert result[(2023, 6)] == pytest.approx(10.0)

    def test_all_12_months_returned_for_complete_year(self, tmp_path):
        # Build a full-year CSV
        rows = "\n".join(f"2020,{m},600.0" for m in range(1, 13))
        phi_file = tmp_path / "oulu_phi.csv"
        nm_file  = tmp_path / "nm_rates.csv"
        phi_file.write_text("year,month,phi_MV\n" + rows)
        nm_file.write_text(OULU_NM_CSV)
        cfg = _make_config(str(phi_file), str(nm_file))
        result = get_phi_series(2020, 2020, cfg)
        for m in range(1, 13):
            assert (2020, m) in result

    def test_warns_when_regression_used(self, tmp_path, recwarn):
        phi_file = tmp_path / "oulu_phi.csv"
        nm_file  = tmp_path / "nm_rates.csv"
        phi_file.write_text(OULU_PHI_CSV)
        nm_file.write_text(OULU_NM_CSV)
        cfg = _make_config(str(phi_file), str(nm_file))
        get_phi_series(2023, 2023, cfg)
        warning_messages = [str(w.message) for w in recwarn.list]
        assert any('regression' in m.lower() for m in warning_messages)

    def test_raises_on_inverted_year_range(self, tmp_path):
        phi_file = tmp_path / "oulu_phi.csv"
        nm_file  = tmp_path / "nm_rates.csv"
        phi_file.write_text(OULU_PHI_CSV)
        nm_file.write_text(OULU_NM_CSV)
        cfg = _make_config(str(phi_file), str(nm_file))
        with pytest.raises(ValueError, match="year_start"):
            get_phi_series(2025, 2020, cfg)
