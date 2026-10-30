"""Utilidades básicas: la transformación asinh y su inversa, y el calendario de feriados."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pml.features import asinh_fit, asinh_inv, asinh_t, feriados_mx


def test_asinh_inversa():
    x = np.array([-50.0, 0.0, 500.0, 12000.0])
    assert np.allclose(asinh_inv(asinh_t(x, 540, 260), 540, 260), x)


def test_asinh_fit_robusto():
    med, mad = asinh_fit(np.array([1.0, 2.0, 3.0, 4.0, 1000.0]))
    assert med == 3.0 and abs(mad - 1.4826) < 1e-9          # un valor extremo no mueve la mediana ni la MAD


def test_feriados():
    f = feriados_mx(pd.DatetimeIndex(["2024-09-16", "2024-09-17"]))
    assert f.tolist() == [1, 0]
