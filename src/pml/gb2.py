"""Gradient boosting global (hora como feature) para cualquier conjunto de grupos de features2.

Formato largo (día, hora): las columnas `base_h{h}` se convierten en una sola columna `base` con el valor de esa hora;
las diarias se replican. Pérdida L1 (MAE) o cuantílica; recalibración semanal con ventana expansiva (días <= d0−8).
"""
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor

LAG_MTR = 8
HRE = re.compile(r"^(.*)_h(\d+)$")
GRID = [dict(max_leaf_nodes=7, max_iter=200, learning_rate=0.05, min_samples_leaf=100),
        dict(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, min_samples_leaf=50)]


def a_largo(F: pd.DataFrame) -> pd.DataFrame:
    """Del formato ancho (un renglón por día, columnas por hora) al largo (un renglón por día y hora). Las columnas
    `{base}_h{h}` se vuelven una sola columna `base` con el valor de la hora h; las diarias se repiten; se agrega `hora`."""
    horarias, diarias = {}, []
    for col in F.columns:
        m = HRE.match(col)
        if m:
            horarias.setdefault(m.group(1), {})[int(m.group(2))] = col
        else:
            diarias.append(col)
    filas = []
    for h in range(1, 25):
        d = F[diarias].copy()
        for base, mp in horarias.items():
            if h in mp:
                d[base] = F[mp[h]].values
        d["hora"] = h
        d["fecha"] = F.index
        filas.append(d.reset_index(drop=True))
    return pd.concat(filas, ignore_index=True)


def _bloque(d0, d1, L, y, cols, params, inicio, loss, q):
    """GB entrenado con las filas de días <= d0 − 8 que pronostica la semana [d0, d1]."""
    fin = d0 - pd.Timedelta(days=LAG_MTR)
    tr = (L.fecha >= inicio) & (L.fecha <= fin)
    te = (L.fecha >= d0) & (L.fecha <= d1)
    kw = dict(loss=loss, l2_regularization=1.0, random_state=0, early_stopping=False, **params)
    if loss == "quantile":
        kw["quantile"] = q
    m = HistGradientBoostingRegressor(**kw).fit(L.loc[tr, cols], y[tr])
    return L.loc[te, ["fecha", "hora"]].assign(pred=m.predict(L.loc[te, cols]))


def correr(L, y, cols, dias, params, inicio, loss="absolute_error", q=0.5, n_jobs=-1):
    """Backtest del GB: una semana a la vez, cada una con su propio modelo. Devuelve días × 24 (escala del proyecto)."""
    bloques = [(dias[i], dias[min(i + 6, len(dias) - 1)]) for i in range(0, len(dias), 7)]
    res = Parallel(n_jobs=n_jobs)(delayed(_bloque)(a, b, L, y, cols, params, inicio, loss, q) for a, b in bloques)
    return pd.concat(res).pivot(index="fecha", columns="hora", values="pred")


def pinball(y, f, q):
    """Pérdida de cuantiles (pinball) del cuantil q: castiga distinto quedarse corto que pasarse."""
    d = np.asarray(y) - np.asarray(f)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))
