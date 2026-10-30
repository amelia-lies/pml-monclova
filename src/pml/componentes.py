"""Separar el pronóstico en energía y congestión (experimento descartado; ver el README).

El PML = energía + pérdidas + congestión. La energía es la misma en todos los nodos del SIN y explica casi todo el error
del MDA en Monclova; la congestión es local, el MDA la exagera y el MTR la encoge. Son dos fenómenos con causas distintas,
así que los modelamos por separado y sumamos:
  - energía: LEAR (el mismo procedimiento del notebook 04) con el MTR y el MDA de energía en lugar del PML total;
  - congestión + pérdidas: regresión de mediana por hora sobre la congestión que trae el MDA de d, su error reciente
    (<= d−8) y la congestión de los corredores de la red. Pocos parámetros, porque es la parte chica.
Recalibración semanal, entrenamiento sólo con días <= d0 − 8.
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import QuantileRegressor

from . import features2 as f2

LAG_MTR = 8


def _local(c: f2.Ctx, proceso: str) -> pd.DataFrame:
    """Congestión + pérdidas de Monclova (la parte que no es energía), días × 24."""
    return c.comp[proceso]["cng"] + c.comp[proceso]["per"]


def rasgos_congestion(c: f2.Ctx) -> tuple:
    """(dict hora -> DataFrame de rasgos, objetivo días × 24 = congestión + pérdidas del MTR)."""
    G = f2.g10(c)
    diarias = [x for x in G.columns if not x.startswith("cngmda_h")]
    L = _local(c, "MDA").reindex(c.dias)
    R = {h: pd.concat([L[h].rename("local_mda_h"), L.mean(axis=1).rename("local_mda_dia"), G[diarias]], axis=1) for h in f2.HORAS}
    return R, _local(c, "MTR")


def _bloque(d0, d1, R, y, alpha):
    """Regresión de mediana por hora para la semana [d0, d1], entrenada con días <= d0 − 8."""
    fin = d0 - pd.Timedelta(days=LAG_MTR)
    out = {}
    for h, F in R.items():
        tr = F.index[(F.index <= fin) & F.index.isin(y.dropna().index)]
        te = F.index[(F.index >= d0) & (F.index <= d1)]
        med = F.loc[tr].median(); sd = F.loc[tr].std().replace(0, 1).fillna(1)
        Z = lambda ix: ((F.loc[ix] - med) / sd).fillna(0).values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs").fit(Z(tr), y.loc[tr, h].values)
        out[h] = pd.Series(m.predict(Z(te)), index=te)
    return pd.DataFrame(out)


def congestion(c: f2.Ctx, dias, alpha=0.05, n_jobs=-1) -> pd.DataFrame:
    """Pronóstico de congestión + pérdidas del MTR (MXN/MWh), días × 24. Ningún notebook ni prueba la usa hoy."""
    R, y = rasgos_congestion(c)
    dias = pd.DatetimeIndex(dias)
    bl = [(dias[i], dias[min(i + 6, len(dias) - 1)]) for i in range(0, len(dias), 7)]
    res = Parallel(n_jobs=n_jobs)(delayed(_bloque)(a, b, R, y, alpha) for a, b in bl)
    return pd.concat(res).sort_index()
