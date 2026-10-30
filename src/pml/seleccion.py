"""Selección de exógenas sin ablación grupo por grupo.

Idea: en vez de correr un backtest por cada grupo, se ajusta UNA vez un modelo con la base + todas las candidatas y se
pregunta al propio modelo qué usa:
  1. frecuencia de selección: en cuántos de los ajustes (bloque semanal × hora) el LASSO deja un coeficiente distinto de 0;
  2. importancia por permutación: cuánto empeora el MAE de desarrollo si se barajan los días de un grupo (rompe su relación
     con el objetivo sin cambiar su distribución). Si barajar un grupo no empeora nada, el modelo no le saca provecho.
Se hace con LEAR (lineal) y con gradient boosting (no lineal), con recalibración semanal y alpha elegido por validación
cruzada temporal DENTRO del entrenamiento (no con el desarrollo), así que la selección no mira el periodo que evalúa.
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit

from . import features2 as f2, gb2

LAG_MTR = 8


def _bloques(dias):
    """Divide los días en semanas: cada semana se pronostica con un modelo entrenado con días <= inicio − 8."""
    dias = pd.DatetimeIndex(dias)
    return [(dias[i], dias[min(i + 6, len(dias) - 1)]) for i in range(0, len(dias), 7)]


def _lear_bloque(d0, d1, F, T, winsor):
    """LEAR de una semana: un LASSO por hora con alpha elegido por validación cruzada temporal. Devuelve la estandarización y
    los coeficientes de cada hora (de ahí sale qué columnas quedaron activas)."""
    warnings.filterwarnings("ignore")
    fin = d0 - pd.Timedelta(days=LAG_MTR)
    tr = F.index[F.index <= fin]
    Xtr, Ytr = F.loc[tr].values, T.loc[tr].values
    lo, hi = np.quantile(Ytr, [winsor, 1 - winsor]); Ytr = np.clip(Ytr, lo, hi)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xs = (Xtr - mu) / sd
    coefs, inter = np.zeros((24, F.shape[1])), np.zeros(24)
    for h in range(24):
        m = LassoCV(alphas=15, cv=TimeSeriesSplit(4), max_iter=3000).fit(Xs, Ytr[:, h])
        coefs[h] = m.coef_
        inter[h] = np.median(Ytr[:, h] - Xs @ m.coef_)            # intercepto mediano (la métrica es MAE)
    return d0, d1, mu, sd, coefs, inter


def lear_estabilidad(c, grupos, dias, winsor=0.01, n_jobs=-1):
    """Ajuste semanal de LEAR con todos los grupos. Devuelve un dict con los bloques ajustados y metadatos."""
    F, G = f2.construir(c, grupos)
    Yt, Xt = f2._t(c)
    T = (Yt - Xt).loc[F.index]
    res = Parallel(n_jobs=n_jobs)(delayed(_lear_bloque)(a, b, F, T, winsor) for a, b in _bloques(dias))
    return dict(F=F, G=G, bloques=res, Xt=Xt, c=c)


def _lear_pred(E, F=None):
    """Pronóstico de LEAR en MXN/MWh con las variables `F` (las originales o con un grupo barajado)."""
    F = E["F"] if F is None else F
    out = []
    for d0, d1, mu, sd, coefs, inter in E["bloques"]:
        ix = F.index[(F.index >= d0) & (F.index <= d1)]
        Z = ((F.loc[ix].values - mu) / sd) @ coefs.T + inter
        out.append(pd.DataFrame(Z, index=ix, columns=range(1, 25)))
    P = pd.concat(out)
    return f2.inv(P + E["Xt"].loc[P.index].values, E["c"])


def frecuencia_seleccion(E):
    """Por columna: fracción de ajustes (bloque × hora) con coeficiente ≠ 0, y el grupo al que pertenece."""
    nz = np.mean([np.abs(b[4]) > 1e-10 for b in E["bloques"]], axis=(0, 1))
    return pd.DataFrame({"frecuencia": nz, "grupo": E["G"].reindex(E["F"].columns).values}, index=E["F"].columns)


def _permutar(F, cols, dias, rng):
    """Baraja, dentro de `dias`, los valores de las columnas `cols` entre días (la misma permutación para todas)."""
    Fp = F.copy()
    ix = F.index[F.index.isin(dias)]
    Fp.loc[ix, cols] = F.loc[ix, cols].values[rng.permutation(len(ix))]
    return Fp


def importancia_permutacion(pred_fn, F, G, Y, dias, n_rep=20, seed=0):
    """ΔMAE (MXN/MWh) al barajar cada grupo; media y percentiles 5–95 sobre `n_rep` permutaciones."""
    rng = np.random.default_rng(seed)
    y = Y.loc[dias].values
    base = np.mean(np.abs(y - pred_fn(F).loc[dias].values))
    filas = {}
    for g in G.unique():
        cols = G.index[G == g].tolist()
        d = [np.mean(np.abs(y - pred_fn(_permutar(F, cols, dias, rng)).loc[dias].values)) - base for _ in range(n_rep)]
        filas[g] = dict(dMAE=np.mean(d), p5=np.percentile(d, 5), p95=np.percentile(d, 95), columnas=len(cols))
    return base, pd.DataFrame(filas).T.sort_values("dMAE", ascending=False)


# ---------------------------------------------------------------------------------------------- gradient boosting
def _gb_bloque(d0, L, y, cols, params):
    """GB de una semana, con las variables en formato largo."""
    fin = d0 - pd.Timedelta(days=LAG_MTR)
    tr = L.fecha <= fin
    m = HistGradientBoostingRegressor(loss="absolute_error", l2_regularization=1.0, random_state=0, early_stopping=False, **params)
    return d0, m.fit(L.loc[tr, cols], y[tr])


def gb_estabilidad(c, grupos, dias, params=None, n_jobs=-1):
    """GB con todas las variables, reentrenado cada semana del desarrollo (se guarda para la permutación)."""
    params = params or gb2.GRID[0]
    F, G = f2.construir(c, grupos)
    Yt, Xt = f2._t(c)
    sp = (Yt - Xt).stack(); sp.index.names = ["fecha", "hora"]
    L = gb2.a_largo(F)
    y = pd.Series(sp.reindex(pd.MultiIndex.from_frame(L[["fecha", "hora"]])).values, index=L.index)
    cols = [x for x in L.columns if x != "fecha"]
    bl = _bloques(dias)
    res = Parallel(n_jobs=n_jobs)(delayed(_gb_bloque)(a, L, y, cols, params) for a, _ in bl)
    return dict(F=F, G=G, modelos={d0: m for d0, m in res}, bloques=bl, cols=cols, Xt=Xt, c=c)


def _gb_pred(E, F=None):
    """Pronóstico del GB en MXN/MWh con las variables `F` (las originales o con un grupo barajado)."""
    F = E["F"] if F is None else F
    L = gb2.a_largo(F)
    out = []
    for d0, d1 in E["bloques"]:
        te = (L.fecha >= d0) & (L.fecha <= d1)
        out.append(L.loc[te, ["fecha", "hora"]].assign(pred=E["modelos"][d0].predict(L.loc[te, E["cols"]])))
    P = pd.concat(out).pivot(index="fecha", columns="hora", values="pred")
    return f2.inv(P + E["Xt"].loc[P.index].values, E["c"])
