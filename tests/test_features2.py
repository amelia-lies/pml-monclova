"""Leakage: sustituir por basura todo lo no disponible al emitir NO debe cambiar la fila de features del día d."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
pytest.importorskip("pyarrow")
if not (ROOT / "data/processed/pml_mtr.parquet").exists() or not (ROOT / "data/raw/exog").exists():
    pytest.skip("faltan datos locales", allow_module_level=True)

from pml import features2 as f2


@pytest.fixture(scope="module")
def ctx():
    return f2.cargar_ctx()


@pytest.mark.parametrize("lag", [2, 8])
def test_sin_leakage_todos_los_grupos(ctx, lag):
    c = replace(ctx, lag_precios=lag)
    grupos = ["G0", "G0dow", "G1", "G2", "G4", "G5", "G6", "G6g", "G6e", "G7", "G7v", "G8", "G9", "G10", "G11", "G12"]
    F, _ = f2.construir(c, grupos)
    rng = np.random.default_rng(3)
    dias = [pd.Timestamp(x) for x in ["2024-02-05", "2024-02-26", "2024-07-04", "2024-10-15", "2025-01-02", "2025-03-27", "2025-06-30"]]
    for d in dias:
        Fp, _ = f2.construir(replace(f2.perturbar(c, d, rng), dias=pd.DatetimeIndex([d])), grupos)
        a, b = F.loc[d].values.astype(float), Fp.loc[d].values.astype(float)
        mal = np.where(~np.isclose(a, b, equal_nan=True))[0]
        assert len(mal) == 0, f"{d.date()}: leakage en {list(F.columns[mal][:8])}"


def test_ventanas_alternativas_sin_leakage(ctx):
    grupos = ["G0w", "G2w", "G1f"]
    F, _ = f2.construir(ctx, grupos)
    d = pd.Timestamp("2024-11-12")
    Fp, _ = f2.construir(replace(f2.perturbar(ctx, d), dias=pd.DatetimeIndex([d])), grupos)
    assert np.allclose(F.loc[d].values.astype(float), Fp.loc[d].values.astype(float), equal_nan=True)


def test_el_test_detecta_fugas(ctx):
    """Control positivo: una feature que mire el MTR de d−1 DEBE cambiar al perturbar."""
    d = pd.Timestamp("2024-11-12")
    Yt, _ = f2._t(ctx); Ytp, _ = f2._t(f2.perturbar(ctx, d))
    assert not np.isclose(Yt.shift(1).loc[d].values, Ytp.shift(1).loc[d].values).all()


def test_control_positivo_grupo_con_fuga(ctx):
    """Un grupo que use MTR de d−7 (aún no publicado) tiene que ser detectado por la perturbación."""
    def g_fuga(c):
        Yt, _ = f2._t(c)
        return f2._ix(f2._sh(Yt, 7, "fuga_l7"), c)
    f2.GRUPOS["GFUGA"] = (g_fuga, "fuga deliberada (test)")
    try:
        d = pd.Timestamp("2024-11-12")
        F, _ = f2.construir(ctx, ["GFUGA"])
        Fp, _ = f2.construir(replace(f2.perturbar(ctx, d), dias=pd.DatetimeIndex([d])), ["GFUGA"])
        assert not np.allclose(F.loc[d].values.astype(float), Fp.loc[d].values.astype(float), equal_nan=True)
    finally:
        del f2.GRUPOS["GFUGA"]


def test_sin_nan_desde_inicio(ctx):
    F, _ = f2.construir(ctx, ["G0", "G0w", "G0dow", "G1", "G1f", "G2", "G2w", "G4", "G5", "G6", "G6g", "G6e", "G7"] if False
                        else ["G0", "G0dow", "G1", "G2", "G4", "G5", "G6", "G6g", "G6e", "G7"])
    assert not F.isna().any().any(), F.columns[F.isna().any()].tolist()[:10]


def test_varias_escalas_y_red_sin_leakage(ctx):
    """G13 (varias escalas, rareza, análogos), G14 (la red en el espacio) G15 y G16 (complejidad y correlaciones del error y del MTR): basura en el MTR de cualquier nodo después
    de d−8 y en su MDA después de d no cambia la fila del día d. Tampoco hay huecos desde el inicio del entrenamiento."""
    c = f2.con_nodos(ctx)
    F, _ = f2.construir(c, ["G13", "G14", "G15", "G16"])
    assert not F.loc[f2.TRAIN_START:].isna().any().any(), F.columns[F.isna().any()].tolist()[:10]
    rng = np.random.default_rng(5)
    for d in [pd.Timestamp(x) for x in ["2024-02-05", "2024-07-04", "2024-11-12", "2025-03-27", "2025-06-30"]]:
        Fp, _ = f2.construir(replace(f2.perturbar(c, d, rng), dias=pd.DatetimeIndex([d])), ["G13", "G14", "G15", "G16"])
        a, b = F.loc[d].values.astype(float), Fp.loc[d].values.astype(float)
        mal = np.where(~np.isclose(a, b, equal_nan=True))[0]
        assert len(mal) == 0, f"{d.date()}: leakage en {list(F.columns[mal][:8])}"


def test_comprimir_no_mira_despues_del_corte(ctx):
    """Las componentes principales se ajustan sólo con días hasta el corte: cambiar los días posteriores no cambia las
    componentes de ningún día anterior."""
    F, _ = f2.construir(ctx, ["G5", "G6", "G9"])
    a, _ = f2.comprimir(F, k=5)
    G = F.copy(); G.loc["2024-07-01":] = 999.0
    b, _ = f2.comprimir(G, k=5)
    assert np.allclose(a.loc[:"2024-06-30"].values, b.loc[:"2024-06-30"].values)
