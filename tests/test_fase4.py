"""No-fuga de los modelos de los notebooks 06 a 11: picos, agregación en línea, modelo enmascarado, modelos entrenados con la
red y detectores de anomalías. El pronóstico de d no puede ver el MTR posterior a d−8 ni el MDA posterior a d, de ningún nodo."""
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

from pml import agregacion as ag, features2 as f2, picos


@pytest.fixture(scope="module")
def ctx():
    return f2.cargar_ctx()


@pytest.mark.parametrize("d", ["2024-05-20", "2024-09-10", "2025-01-21"])
def test_rasgos_de_pico_sin_fuga(ctx, d):
    d = pd.Timestamp(d)
    a = picos.rasgos(ctx).loc[d]
    b = picos.rasgos(replace(f2.perturbar(ctx, d), dias=pd.DatetimeIndex([d]))).loc[d]
    assert np.allclose(a.values.astype(float), b.values.astype(float), equal_nan=True), list(a.index[~np.isclose(a, b, equal_nan=True)])


def test_agregacion_respeta_el_retraso(ctx):
    dias = pd.date_range("2024-07-01", "2024-09-30")
    rng = np.random.default_rng(0)
    E = {"MDA": ctx.X.loc[dias], "otro": ctx.X.loc[dias] * (1 + 0.2 * rng.standard_normal(ctx.X.loc[dias].shape))}
    d = pd.Timestamp("2024-08-20")
    Yp = ctx.Y.copy()
    m = Yp.index > d - pd.Timedelta(days=8)
    Yp.loc[m] = rng.normal(5000, 2000, Yp.loc[m].shape)
    prior = {"MDA": 0.8, "otro": 0.2}
    for fn, kw in ((ag.ewa, dict(eta=2.0)), (ag.ventana_lad, dict(ventana=28))):
        a = fn(E, ctx.Y, dias, prior, **kw)[0].loc[:d]
        b = fn(E, Yp, dias, prior, **kw)[0].loc[:d]
        assert np.allclose(a.values, b.values), fn.__name__


def test_modelo_enmascarado_sin_futuro(ctx):
    """El modelo enmascarado no ve MTR posterior a d−8 ni MDA posterior a d: ni al pre-entrenar, ni al ajustar, ni al predecir."""
    from pml import enmascarado as en
    c = replace(ctx, escala=0.5)
    d = pd.Timestamp("2024-08-15")
    hp = en.HPm(epocas_pre=1, epocas_ajuste=1, seeds=1, d_model=16, capas=1, cabezas=2, contextos=(14,), fragmentos=(0, 2))
    def pred(cc):
        D, m, a = en.series(cc)
        i = list(D).index(d)
        estado = en._bloque_pre(i - 8, m, a, hp, 0, 0)
        return en._bloque_ajuste(i, i, estado, m, a, hp, 0, 0)[i][0]
    a = pred(c)
    b = pred(f2.perturbar(c, d, np.random.default_rng(1)))
    assert np.allclose(a, b, atol=1e-5)


def test_enmascarado_con_nodos_de_la_red_sin_futuro(ctx):
    """Con los nodos de la red en el pre-entrenamiento: basura en su MTR posterior a d−8 y en su MDA posterior a d no
    cambia el modelo pre-entrenado para d."""
    from pml import enmascarado as en
    c = replace(ctx, escala=0.5)
    d = pd.Timestamp("2024-08-15")
    hp = en.HPm(epocas_pre=1, seeds=1, d_model=16, capas=1, cabezas=2, contextos=(14,), nodos_red=True)
    D, m, a = en.series(c)
    i = list(D).index(d)
    extra = en.series_red(c, D)[:3]
    rng = np.random.default_rng(2)
    sucio = []
    for me, ae in extra:
        me, ae = me.copy(), ae.copy()
        me[(i - 8 + 1) * 24:] = rng.normal(5, 2, me[(i - 8 + 1) * 24:].shape)
        ae[(i + 1) * 24:] = rng.normal(5, 2, ae[(i + 1) * 24:].shape)
        sucio.append((me, ae))
    A = en._bloque_pre(i - 8, m, a, hp, 0, 0, extra)
    B = en._bloque_pre(i - 8, m, a, hp, 0, 0, sucio)
    import torch
    assert all(torch.allclose(A[k], B[k], atol=1e-6) for k in A)


def _red_sucia(monkeypatch, d, troncal, seed=3):
    """Reemplaza el MTR posterior a d−8 y el MDA posterior a d de todos los nodos de la red con basura."""
    from pml import red
    Ym, Xm = red.matrices_nodo("MTR", troncal=troncal), red.matrices_nodo("MDA", troncal=troncal)
    rng = np.random.default_rng(seed)
    def sucia(M, corte):
        M = M.copy(); m = M.index > corte
        M.loc[m] = rng.normal(5000, 2000, M.loc[m].shape)
        return M
    Ys = {n: sucia(M, d - pd.Timedelta(days=8)) for n, M in Ym.items()}
    Xs = {n: sucia(M, d) for n, M in Xm.items()}
    monkeypatch.setattr(red, "matrices_nodo", lambda p, col="pml", troncal=False: Ys if p == "MTR" else Xs)


def test_gb_con_nodos_de_la_red_sin_futuro(ctx, monkeypatch):
    """GB global (Monclova + red): basura en el MTR posterior a d−8 y en el MDA posterior a d, de Monclova y de todos los
    nodos de la red, no cambia el pronóstico de Monclova para d."""
    from pml import multinodo as mn
    c = replace(ctx, escala=0.5)
    d = pd.Timestamp("2024-09-10")
    params = dict(max_iter=50, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=50)
    L, y = mn.preparar(c)
    a = mn._bloque(d, d, L, y, [x for x in L.columns if x != "fecha"], params, 1.0)
    _red_sucia(monkeypatch, d, False)
    L, y = mn.preparar(f2.perturbar(c, d, np.random.default_rng(4)))
    b = mn._bloque(d, d, L, y, [x for x in L.columns if x != "fecha"], params, 1.0)
    assert np.allclose(a.pred.values, b.pred.values)


def _lear_red(c, d, troncal):
    from pml import multinodo as mn
    F, T = mn.apilar(c, ["G0", "G2"], True, troncal)
    D = pd.get_dummies(F.corredor.astype(pd.CategoricalDtype(range(len(mn.CORR)))), prefix="corr", dtype=float)
    F = pd.concat([F.drop(columns="corredor"), D.set_axis(F.index)], axis=1)
    cols = [x for x in F.columns if x != "nodo"]
    r = mn._dias_lear([d], F[cols].values.astype(float), T.values.astype(float), F.index.values, (F.nodo == "monclova").values,
                      {"8w": 56}, np.array([0.03]), F.index.min(), 0.01)
    return r[0][1]["8w"]


def test_lear_con_la_troncal_sin_futuro(ctx, monkeypatch):
    """LEAR global con los 64 nodos de la red: la basura posterior a d−8 (MTR) y a d (MDA) en Monclova y la red no cambia d."""
    c = replace(ctx, escala=0.5)
    d = pd.Timestamp("2024-09-10")
    a = _lear_red(c, d, True)
    _red_sucia(monkeypatch, d, True)
    b = _lear_red(f2.perturbar(c, d, np.random.default_rng(4)), d, True)
    assert np.allclose(a, b)


def test_redes_con_la_troncal_sin_futuro(ctx, monkeypatch):
    """TiDE entrenada con los 64 nodos de la red (la red más compleja; MLP y NBEATSx comparten la misma preparación de datos)."""
    from pml import dl
    c = replace(ctx, escala=0.5)
    d = pd.Timestamp("2024-09-10")
    hp = replace(dl.busqueda_reg("tide")[0], seeds=1, batch=256, epochs=3)
    def pred(cc):
        F, T, meta = dl.preparar_red(cc, "spread", True)
        return dl._bloque_red(d, d, F.values.astype(float), T.values.astype(float), F.index.values, F.columns.tolist(), hp, meta, F.index.min())[1]
    a = pred(c)
    _red_sucia(monkeypatch, d, True)
    b = pred(f2.perturbar(c, d, np.random.default_rng(4)))
    assert np.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("metodo", ["isolation", "pca"])
def test_detector_de_anomalias_sin_futuro(ctx, metodo):
    """El puntaje de rareza del día d (notebook 11) no cambia si se llena de basura el MTR posterior a d−8 y el MDA
    posterior a d, de Monclova y de todos los nodos de la red."""
    from pml import anomalias as an
    c = f2.con_nodos(ctx)
    d = pd.Timestamp("2024-10-15")
    a = an.puntaje_exante(c, [d], metodo, n_jobs=1)
    b = an.puntaje_exante(f2.perturbar(c, d, np.random.default_rng(2)), [d], metodo, n_jobs=1)
    assert np.isclose(a.loc[d], b.loc[d]), (a.loc[d], b.loc[d])
