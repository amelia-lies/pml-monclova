"""Entrenar con más nodos y pronosticar sólo Monclova (notebook 06).

Las mismas features de MTR/MDA se construyen para cada nodo de la red con su propio MTR y MDA (en la escala de Monclova),
se apilan con las de Monclova y se entrena un solo modelo, con el corredor del nodo como variable. Las predicciones y la
evaluación son sólo de la zona de Monclova. Los demás nodos comparten la energía con Monclova, así que lo que aportan son
más ejemplos de cómo se separa el MTR del MDA según la congestión de cada punto de la red. El objetivo es siempre el
spread (MTR − MDA): en nivel, los nodos repetirían la misma serie de energía.

  - GB global (`preparar`, `correr`): formato largo con hora y corredor como features.
  - LEAR global (`lear`): un LASSO por hora con las filas de todos los nodos, el mismo protocolo que `backtest`
    (recalibración diaria, tres ventanas, rejilla de alpha), más una variable indicadora por corredor.
Red: los 31 nodos de `red.NODOS_RED`; con `troncal`, también el resto de la troncal de 400 kV (64 nodos con datos).
"""
import hashlib
import pickle
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler

from . import features2 as f2, gb2, red
from .backtest import ALPHAS, VENTANAS

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/processed/exp_cache"
LAG_MTR = 8
GRUPOS = ["G0", "G2", "G4"]
CORR = ["monclova"] + list(red.GRUPOS_RED)                          # código 0 = Monclova, 1… = corredores


def contextos(c: f2.Ctx, con_red=True, troncal=False, nodos=None, ident="corredor"):
    """[(nodo, código, contexto)] con Monclova primero (código 0). Cada contexto lleva el MTR y el MDA de su nodo.
    `nodos`: subconjunto de la red (por omisión, todos). `ident`: el código es el corredor del nodo o el nodo mismo."""
    out = [("monclova", 0, c)]
    if con_red:
        Ym, Xm = red.matrices_nodo("MTR", troncal=troncal), red.matrices_nodo("MDA", troncal=troncal)
        cor = red.corredores(troncal)
        lista = [n for n in Ym if nodos is None or n in nodos]
        cod = (lambda i, n: CORR.index(cor[n])) if ident == "corredor" else (lambda i, n: i + 1)
        out += [(n, cod(i, n), replace(c, Y=Ym[n].reindex(c.Y.index), X=Xm[n].reindex(c.X.index))) for i, n in enumerate(lista)]
    return out


def parecidos(c: f2.Ctx, k: int, troncal=True, hasta="2024-06-30"):
    """Los k nodos de la red cuyo MTR − MDA diario se parece más al de Monclova (correlación de Spearman), medido sólo
    con días anteriores al desarrollo, para que la elección no use información del periodo evaluado."""
    Ym, Xm = red.matrices_nodo("MTR", troncal=troncal), red.matrices_nodo("MDA", troncal=troncal)
    s0 = (c.Y - c.X).loc[:hasta].mean(axis=1)
    rho = pd.Series({n: s0.corr((Ym[n] - Xm[n]).loc[:hasta].mean(axis=1).reindex(s0.index), method="spearman") for n in Ym})
    return rho.sort_values(ascending=False), rho.nlargest(k).index.tolist()


def apilar(c: f2.Ctx, grupos, con_red=True, troncal=False):
    """Features (formato ancho, una fila por nodo y día) y spread transformado, apilados. Columnas extra: nodo, corredor."""
    F, T = [], []
    for n, cod, cn in contextos(c, con_red, troncal):
        Fn, _ = f2.construir(cn, grupos)
        Yt, Xt = f2._t(cn)
        F.append(Fn.assign(nodo=n, corredor=cod)); T.append((Yt - Xt).reindex(Fn.index))
    return pd.concat(F), pd.concat(T)


# -------------------------------------------------------------------------------------------- gradient boosting
def _largo_nodo(cn: f2.Ctx, codigo: int, grupos=GRUPOS, F=None):
    """Variables de un nodo en formato largo y su objetivo (error del MDA de ese nodo), con el código del nodo. `F`: las
    variables ya construidas (por ejemplo, con componentes principales), en lugar de construirlas de `grupos`."""
    if F is None:
        F, _ = f2.construir(cn, list(grupos))
    L = gb2.a_largo(F)
    Yt, Xt = f2._t(cn)
    S = (Yt - Xt).reindex(F.index)
    y = S[list(range(1, 25))].values.T.ravel()                         # a_largo apila hora por hora
    L["nodo"] = codigo
    return L, y


def preparar(c: f2.Ctx, con_red=True, troncal=False, nodos=None, ident="corredor", grupos=GRUPOS):
    """Formato largo con Monclova (nodo 0) y, si `con_red`, los nodos de la red (código = corredor o nodo)."""
    partes = [_largo_nodo(cn, cod, grupos) for _, cod, cn in contextos(c, con_red, troncal, nodos, ident)]
    L = pd.concat([p[0] for p in partes], ignore_index=True)
    y = np.concatenate([p[1] for p in partes])
    return L, y


def _bloque(d0, d1, L, y, cols, params, peso_mon, loss="absolute_error", quantile=None, afinar=0):
    """GB de una semana: entrena con las filas de todos los nodos de días <= d0 − 8 (Monclova con peso `peso_mon`) y
    pronostica sólo las filas de Monclova de [d0, d1]. Con `afinar`, agrega árboles entrenados sólo con Monclova."""
    fin = d0 - pd.Timedelta(days=LAG_MTR)
    tr = (L.fecha <= fin).values & ~np.isnan(y)
    te = ((L.fecha >= d0) & (L.fecha <= d1) & (L.nodo == 0)).values
    w = np.where(L.nodo.values[tr] == 0, peso_mon, 1.0)
    kw = dict(quantile=quantile) if loss == "quantile" else {}
    cat = [cols.index("nodo")] if "nodo" in cols else None
    m = HistGradientBoostingRegressor(loss=loss, l2_regularization=1.0, random_state=0, early_stopping=False,
                                      categorical_features=cat, warm_start=afinar > 0, **kw, **params)
    m.fit(L.loc[tr, cols], y[tr], sample_weight=w)
    if afinar:                                     # afinar: más árboles, ajustados sólo con las filas de Monclova
        mon = tr & (L.nodo.values == 0)
        m.set_params(max_iter=m.max_iter + afinar).fit(L.loc[mon, cols], y[mon])
    return L.loc[te, ["fecha", "hora"]].assign(pred=m.predict(L.loc[te, cols]))


def correr(L, y, dias, params=None, peso_mon=1.0, n_jobs=-1, quantile=None, sin_ident=False, afinar=0, quitar=()):
    """Spread pronosticado (escala del proyecto) de Monclova, días × 24. Con `quantile`, ese cuantil (pérdida pinball).
    `sin_ident`: el modelo no sabe de qué nodo viene cada fila. `afinar`: árboles extra entrenados sólo con Monclova.
    `quitar`: columnas que el modelo no ve (para medir cuánto pierde sin ellas)."""
    params = params or gb2.GRID[0]
    cols = [x for x in L.columns if x not in ("fecha",) and not (sin_ident and x == "nodo") and x not in set(quitar)]
    dias = pd.DatetimeIndex(dias)
    bl = [(dias[i], dias[min(i + 6, len(dias) - 1)]) for i in range(0, len(dias), 7)]
    loss = "quantile" if quantile is not None else "absolute_error"
    res = Parallel(n_jobs=n_jobs)(delayed(_bloque)(a, b, L, y, cols, params, peso_mon, loss, quantile, afinar) for a, b in bl)
    return pd.concat(res).pivot(index="fecha", columns="hora", values="pred")


# ------------------------------------------------------------------------------------------------- LEAR global
def _dias_lear(ds, Xv, Tv, fecha, mon, ventanas, alphas, inicio, winsor):
    """LEAR global para una tanda de días: un LASSO por hora con las filas de todos los nodos, en cada ventana y alpha."""
    warnings.filterwarnings("ignore")
    res = []
    for d in ds:
        fin = d - pd.Timedelta(days=LAG_MTR)
        x_d = Xv[(fecha == np.datetime64(d)) & mon]
        out = {}
        for nom, Lw in ventanas.items():
            ini = inicio if Lw is None else max(inicio, fin - pd.Timedelta(days=Lw - 1))
            m = (fecha >= np.datetime64(ini)) & (fecha <= np.datetime64(fin))
            Xtr, Ytr = Xv[m], Tv[m]
            ok = ~np.isnan(Xtr).any(axis=1) & ~np.isnan(Ytr).any(axis=1)
            Xtr, Ytr = Xtr[ok], Ytr[ok]
            if winsor:
                lo, hi = np.quantile(Ytr, [winsor, 1 - winsor]); Ytr = np.clip(Ytr, lo, hi)
            sc = StandardScaler().fit(Xtr); Xs = sc.transform(Xtr); x = sc.transform(x_d)[0]
            ym = Ytr.mean(axis=0)
            pred = np.empty((len(alphas), Ytr.shape[1]))
            for h in range(Ytr.shape[1]):
                _, cf, _ = lasso_path(Xs, Ytr[:, h] - ym[h], alphas=alphas, max_iter=2000)
                b = np.median((Ytr[:, h] - ym[h])[:, None] - Xs @ cf, axis=0)        # intercepto mediano, como LEAR base
                pred[:, h] = x @ cf + ym[h] + b
            out[nom] = pred
        res.append((d, out))
    return res


def lear(c: f2.Ctx, dias, grupos=("G0", "G2"), troncal=False, winsor=0.01, ventanas=VENTANAS, alphas=ALPHAS, n_jobs=-1, nombre="lear_red",
         comp=None, k=20):
    """LEAR entrenado con Monclova y la red. Devuelve {(ventana, alpha): días × 24 de spread transformado de Monclova},
    el mismo formato que `exp.correr_config`, para elegir alpha en desarrollo con `exp.pronostico`.
    `comp="nucleo"`: rezagos, MDA y calendario tal cual; el resto de `grupos` resumido en `k` componentes principales,
    ajustadas con las filas de todos los nodos apiladas (sólo días anteriores al desarrollo)."""
    from .todo import CALENDARIO, NUCLEO
    dias = pd.DatetimeIndex(dias)
    extra = ("" if c.modo == "asinh" else f"|{c.modo}") + (f"|comp:{comp}{k}" if comp else "")   # vacío con los valores de siempre: la clave no cambia
    clave = hashlib.md5(f"{sorted(grupos)}|{troncal}|{winsor}|{c.escala}|{c.med:.4f}|{dias[0]}|{dias[-1]}{extra}".encode()).hexdigest()[:10]
    f = CACHE / f"{nombre}_{clave}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    F, T = apilar(c, list(grupos), True, troncal)
    if comp == "nucleo":
        fijas = set(f2.construir(c, [g for g in grupos if g in NUCLEO + CALENDARIO])[0].columns)
        resto = [x for x in F.columns if x not in fijas and x not in ("nodo", "corredor")]
        pc = f2.comprimir(F[resto], k)[0]                                # F ya trae las filas apiladas: se ajusta con todas
        F = F.drop(columns=resto); F[pc.columns] = pc.values
    D = pd.get_dummies(F.corredor.astype(pd.CategoricalDtype(range(len(CORR)))), prefix="corr", dtype=float)
    F = pd.concat([F.drop(columns="corredor"), D.set_axis(F.index)], axis=1)
    cols = [x for x in F.columns if x != "nodo"]
    Xv, Tv = F[cols].values.astype(float), T.values.astype(float)
    fecha, mon = F.index.values, (F.nodo == "monclova").values
    tandas = [dias[i:i + 7] for i in range(0, len(dias), 7)]
    res = Parallel(n_jobs=n_jobs)(delayed(_dias_lear)(ds, Xv, Tv, fecha, mon, ventanas, alphas, F.index.min(), winsor) for ds in tandas)
    res = [r for rr in res for r in rr]
    preds = {(nom, float(a)): pd.DataFrame(np.stack([r[1][nom][i] for r in res]), index=[r[0] for r in res], columns=T.columns)
             for nom in ventanas for i, a in enumerate(alphas)}
    f.write_bytes(pickle.dumps(preds))
    return preds


def preparar_pc(c: f2.Ctx, nucleo, resto, modo, k=20, troncal=True):
    """Como `preparar`, pero con las variables resumidas en `k` componentes principales (notebook 07, calculado con `servidor.py`).
    `modo` = 'nucleo': los grupos `nucleo` tal cual y los `resto` resumidos; 'todo': `nucleo` + `resto` resumidos juntos.
    Las componentes se ajustan con las filas de todos los nodos apiladas (sólo días anteriores al desarrollo), así cada
    nodo se resume con la misma transformación."""
    ctxs = contextos(c, True, troncal)
    grupos = {"nucleo": list(nucleo), "todo": list(nucleo) + list(resto)}
    Fr = [f2.construir(cn, list(resto) if modo == "nucleo" else grupos["todo"])[0] for _, _, cn in ctxs]
    apilado = pd.concat(Fr)
    partes = []
    for (_, cod, cn), fr in zip(ctxs, Fr):
        pc = f2.comprimir(fr, k, ajustar_con=apilado)[0]
        F = pd.concat([f2.construir(cn, list(nucleo))[0], pc], axis=1) if modo == "nucleo" else pc
        partes.append(_largo_nodo(cn, cod, F=F))
    L = pd.concat([p[0] for p in partes], ignore_index=True)
    return L, np.concatenate([p[1] for p in partes])
