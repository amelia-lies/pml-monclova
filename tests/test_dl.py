"""La red no ve nada posterior a lo publicado: basura en MTR > d−8 y MDA > d no cambia el pronóstico del día d."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.pml import dl, features2 as f2


@pytest.fixture(scope="module")
def ctx():
    return replace(f2.cargar_ctx(), escala=0.5)


@pytest.mark.parametrize("variante,arch", [("spread", "mlp"), ("mda_drop", "nbeats"), ("sin_mda", "mlp")])
def test_red_sin_futuro(ctx, variante, arch):
    d = pd.Timestamp("2024-08-15")
    hp = dl.HP(arch=arch, epochs=3, seeds=1, n1=16, n2=8)
    def pred(c):
        F, T, _, meta = dl.preparar(c, variante)
        return dl._bloque(d, d, F, T, hp, meta, variante, F.index.min())[1]
    a = pred(ctx)
    b = pred(f2.perturbar(ctx, d, np.random.default_rng(1)))
    assert np.allclose(a, b, atol=1e-5)
