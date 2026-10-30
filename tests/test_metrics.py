import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pml.metrics import dm_test, gw_test, mae, rmae

rng = np.random.default_rng(0)
Y = rng.normal(500, 100, (300, 24))
F1 = Y + rng.normal(0, 20, Y.shape)


def test_identicos():
    r = dm_test(Y, F1, F1 + 1e-9 * rng.normal(size=Y.shape))
    assert r["p"] > 0.5 or np.isnan(r["p"])


def test_pares_independientes_misma_calidad():
    F2 = Y + rng.normal(0, 20, Y.shape)
    assert dm_test(Y, F1, F2)["p"] > 0.01 and gw_test(Y, F1, F2)["p"] > 0.01


def test_peor_claramente():
    F3 = Y + rng.normal(0, 60, Y.shape)
    for t in (dm_test, gw_test):
        r = t(Y, F1, F3)
        assert r["p"] < 1e-6
    assert dm_test(Y, F1, F3)["p_a_mejor"] < 1e-6
    assert gw_test(Y, F1, F3)["mejor"] == "A"


def test_rmae():
    assert abs(rmae(Y, F1, F1) - 1) < 1e-12
    assert abs(mae([1, 2], [2, 4]) - 1.5) < 1e-12
