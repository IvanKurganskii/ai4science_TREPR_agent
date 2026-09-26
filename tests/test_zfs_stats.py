"""Verification tests for zfs_stats metrics."""
import numpy as np

from src.zfs_stats import mae, mape, rmse, r2

# Hand-computed from DMRG_table.csv (16 molecules)
D_CALC = np.array([0.0634, 0.1250, 0.0502, 0.0694, 0.0813, 0.0861, 0.0888,
                   0.0386, 0.0933, 0.1019, 0.0948, 0.0668, 0.0582, 0.0330,
                   0.0330, 0.0306])
D_EXP = np.array([0.0702, 0.121, 0.0573, 0.079, 0.095, 0.1367, 0.081, 0.0467,
                  0.090, 0.1016, 0.091, 0.068, 0.0967, 0.044, 0.0333, 0.036])


def test_mae_D():
    assert abs(mae(D_CALC, D_EXP) - 0.0107) < 1e-4


def test_rmse_D():
    assert abs(rmse(D_CALC, D_EXP) - 0.0172) < 1e-4


def test_r2_D():
    assert abs(r2(D_CALC, D_EXP) - 0.6437) < 1e-3


def test_mape_D():
    assert abs(mape(D_CALC, D_EXP) - 12.91) < 0.1


# |E|: theoretical |E| = |E_theory|, experimental |E|_exp
E_CALC = np.array([0.0118, 0.0144, 0.0065, 0.0186, 0.0224, 0.0038, 0.0092,
                   0.0002, 0.0143, 0.0128, 0.0105, 0.0092, 0.0064, 0.0108,
                   0.0094, 0.0088])
E_EXP = np.array([0.0081, 0.027, 0.0043, 0.014, 0.025, 0.0026, 0.017, 0.0018,
                  0.025, 0.0286, 0.023, 0.018, 0.0030, 0.007, 0.0099, 0.009])


def test_mae_E():
    assert abs(mae(E_CALC, E_EXP) - 0.0057) < 1e-4


def test_rmse_E():
    assert abs(rmse(E_CALC, E_EXP) - 0.0075) < 1e-4


def test_r2_E():
    assert abs(r2(E_CALC, E_EXP) - 0.3375) < 1e-3


def test_mape_E():
    assert abs(mape(E_CALC, E_EXP) - 46.49) < 0.1


# |E/D|: theoretical |E/D|, experimental |E/D|_exp
ED_CALC = np.array([0.19, 0.12, 0.13, 0.27, 0.28, 0.04, 0.10, 0.01, 0.15,
                    0.13, 0.11, 0.14, 0.11, 0.33, 0.29, 0.29])
ED_EXP = np.array([0.12, 0.22, 0.08, 0.18, 0.26, 0.02, 0.21, 0.04, 0.28,
                   0.28, 0.26, 0.26, 0.03, 0.16, 0.30, 0.25])


def test_mae_ED():
    assert abs(mae(ED_CALC, ED_EXP) - 0.0838) < 1e-4


def test_rmse_ED():
    assert abs(rmse(ED_CALC, ED_EXP) - 0.0977) < 1e-4


def test_r2_ED():
    assert abs(r2(ED_CALC, ED_EXP) - (-0.0717)) < 1e-3


def test_mape_ED():
    assert abs(mape(ED_CALC, ED_EXP) - 65.47) < 0.1