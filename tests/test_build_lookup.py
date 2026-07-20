import numpy as np
import pytest
from cosmo14C.build_lookup import build_local_Q_table, load_local_Q_table


class TestBuildLocalQTable:
    def test_table_has_correct_shape(self, tmp_path):
        path = str(tmp_path / "test_table.npz")
        phi_cfg = dict(min=100, max=1000, n=5, scale='log')
        Pc_cfg  = dict(min=0,   max=15,   n=8, scale='linear')
        build_local_Q_table(path, phi_cfg, Pc_cfg)
        data = np.load(path)
        assert data['Q'].shape == (5, 8)
        assert data['phi_grid'].shape == (5,)
        assert data['Pc_grid'].shape == (8,)

    def test_phi_grid_is_log_spaced(self, tmp_path):
        path = str(tmp_path / "test_table.npz")
        phi_cfg = dict(min=100, max=1000, n=5, scale='log')
        Pc_cfg  = dict(min=0,   max=15,   n=4, scale='linear')
        build_local_Q_table(path, phi_cfg, Pc_cfg)
        data = np.load(path)
        log_diffs = np.diff(np.log(data['phi_grid']))
        assert np.allclose(log_diffs, log_diffs[0], rtol=1e-6)

    def test_Q_decreases_with_phi(self, tmp_path):
        path = str(tmp_path / "test_table.npz")
        phi_cfg = dict(min=300, max=1200, n=4, scale='log')
        Pc_cfg  = dict(min=0,   max=5,    n=3, scale='linear')
        build_local_Q_table(path, phi_cfg, Pc_cfg)
        data = np.load(path)
        # For any fixed Pc column, Q should decrease with increasing phi
        for j in range(data['Q'].shape[1]):
            assert np.all(np.diff(data['Q'][:, j]) < 0)

    def test_Q_decreases_with_Pc(self, tmp_path):
        path = str(tmp_path / "test_table.npz")
        phi_cfg = dict(min=650, max=650, n=2, scale='log')
        Pc_cfg  = dict(min=0,   max=14,  n=5, scale='linear')
        build_local_Q_table(path, phi_cfg, Pc_cfg)
        data = np.load(path)
        # For fixed phi, Q decreases as Pc increases
        assert np.all(np.diff(data['Q'][0, :]) < 0)

    def test_interpolator_agrees_with_direct_local_Q(self, tmp_path):
        from cosmo14C.production import local_Q
        path = str(tmp_path / "test_table.npz")
        phi_cfg = dict(min=200, max=1500, n=20, scale='log')
        Pc_cfg  = dict(min=0,   max=16,   n=40, scale='linear')
        build_local_Q_table(path, phi_cfg, Pc_cfg)
        interp = load_local_Q_table(path)
        # Test at an interior point (not a grid node)
        q_interp = interp([[650.0, 5.0]])[0]
        q_direct = local_Q(650.0, 5.0)
        assert abs(q_interp - q_direct) / q_direct < 0.02  # within 2%
