"""Modelo aparte para los días de pico (MTR > 3000 MXN/MWh en alguna hora de la zona).

Dos piezas, con las mismas reglas de tiempo que el resto (se entrena con días <= d0−8, recalibración semanal):
  1. clasificador diario P(pico en d): logístico regularizado y gradient boosting con monotonicidad en el nivel del MDA;
     se compara con dos referencias simples, la climatología por mes y un logístico que sólo ve el máximo del MDA;
  2. cuantiles con rasgos de pico: el modelo cuantílico del notebook 09 más las features diarias del clasificador.

Con MAE el pronóstico óptimo es la mediana, así que un día con P(pico) < 0.5 no debería mover el punto. El valor
esperado de esta pieza está en la alerta y en la cola alta de los intervalos.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import features2 as f2

LAG_MTR = 8
UMBRAL = 3000.0
EVE = [19, 20, 21, 22]


def etiqueta(c: f2.Ctx, umbral=UMBRAL) -> pd.Series:
    """1 si el MTR de la zona supera el umbral en alguna hora del día."""
    return (c.Y.max(axis=1) > umbral).astype(int)


def rasgos(c: f2.Ctx, umbral=UMBRAL) -> pd.DataFrame:
    """Features diarias conocidas al emitir (d−1): forma del MDA de d, historia de picos hasta d−8, calor pronosticado,
    estados operativos, mes. En precios (MXN/MWh), sin transformar, para que el logístico sea legible."""
    X, Y = c.X.asfreq("D"), c.Y.asfreq("D")
    pico = (Y.max(axis=1) > umbral).astype(float)
    r = pd.DataFrame(index=X.index)
    r["mda_max"] = X.max(axis=1)
    r["mda_eve"] = X[EVE].mean(axis=1)
    r["mda_rampa"] = X[EVE].mean(axis=1) - X[[12, 13, 14, 15]].mean(axis=1)
    r["mda_max_d1"] = r["mda_max"] - r["mda_max"].shift(1)
    r["mda_max_rel28"] = r["mda_max"] / r["mda_max"].shift(1).rolling(28, min_periods=7).median()
    r["picos_28d"] = pico.shift(LAG_MTR).rolling(28, min_periods=7).mean()        # sólo MTR publicado (<= d−8)
    r["mtr_max_7d"] = Y.max(axis=1).shift(LAG_MTR).rolling(7, min_periods=3).max()
    r["mtr_vs_mda_7d"] = (Y.max(axis=1) / X.max(axis=1)).shift(LAG_MTR).rolling(7, min_periods=3).mean()
    W = c.W["temperature_2m"].asfreq("D")
    r["tmax_zona"] = W.max(axis=1)
    if c.Wext:
        r["tmax_mty"] = c.Wext["Monterrey"]["temperature_2m"].asfreq("D").max(axis=1)
    eo = (c.eo if c.eo is not None else pd.Series(dtype=float)).reindex(r.index).fillna(0)
    r["eo_7d"] = (eo > 0).astype(float).shift(2).rolling(7, min_periods=1).sum()
    doy = r.index.dayofyear.values / 365.25
    r["sin1"], r["cos1"] = np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy)
    return r.loc[c.dias]


SOLO_MDA = ["mda_max"]


def _ajusta(kind, Xtr, ytr, cols):
    """Entrena un clasificador (logístico estandarizado o gradient boosting monótono en el MDA) y devuelve su función
    de probabilidad."""
    if kind == "logit" or kind == "solo_mda":
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        m = LogisticRegression(C=0.3, max_iter=2000).fit((Xtr - mu) / sd, ytr)
        return lambda X: m.predict_proba((X - mu) / sd)[:, 1]
    mono = [1 if col in ("mda_max", "mda_eve") else 0 for col in cols]
    m = HistGradientBoostingClassifier(max_leaf_nodes=7, max_iter=150, learning_rate=0.05, min_samples_leaf=15,
                                       l2_regularization=1.0, monotonic_cst=mono, random_state=0).fit(Xtr, ytr)
    return lambda X: m.predict_proba(X)[:, 1]


def _bloque(d0, d1, R, y, kind, cols):
    """Probabilidad de pico de cada día de la semana [d0, d1], con un clasificador entrenado con días <= d0 − 8."""
    fin = d0 - pd.Timedelta(days=LAG_MTR)
    tr = R.index[(R.index <= fin)]
    te = R.index[(R.index >= d0) & (R.index <= d1)]
    Xtr = R.loc[tr, cols].fillna(R.loc[tr, cols].median()).values
    med = R.loc[tr, cols].median()
    if kind == "clima":                                            # climatología: tasa de picos por mes, suavizada
        yy = y.loc[tr]
        tasa_m = yy.groupby(yy.index.month).agg(["sum", "count"])
        glob = yy.mean()
        p = [(tasa_m.loc[m, "sum"] + 5 * glob) / (tasa_m.loc[m, "count"] + 5) if m in tasa_m.index else glob for m in te.month]
        return pd.Series(p, index=te)
    f = _ajusta(kind, Xtr, y.loc[tr].values, cols)
    return pd.Series(f(R.loc[te, cols].fillna(med).values), index=te)


def clasificar(c: f2.Ctx, dias, kind="logit", n_jobs=-1) -> pd.Series:
    """P(pico) fuera de muestra para `dias`, con recalibración semanal. kind: logit | gb | solo_mda | clima."""
    R, y = rasgos(c), etiqueta(c)
    y = y.loc[R.index]
    R = R.loc[R.index >= f2.TRAIN_START + pd.Timedelta(days=28)]           # descarta el arranque de las ventanas móviles
    y = y.loc[R.index]
    cols = SOLO_MDA if kind == "solo_mda" else R.columns.tolist()
    dias = pd.DatetimeIndex(dias)
    bl = [(dias[i], dias[min(i + 6, len(dias) - 1)]) for i in range(0, len(dias), 7)]
    res = Parallel(n_jobs=n_jobs)(delayed(_bloque)(a, b, R, y, kind, cols) for a, b in bl)
    return pd.concat(res)


def evaluar(p: pd.Series, y: pd.Series) -> dict:
    """AUC, Brier (y el de usar la tasa real del periodo), picos y cuántos marca con P > 0.5, y falsas alarmas."""
    y = y.loc[p.index]
    clima = np.full(len(y), y.mean())
    return dict(AUC=roc_auc_score(y, p) if y.nunique() > 1 else np.nan, Brier=brier_score_loss(y, p),
                Brier_ref_tasa=brier_score_loss(y, clima), picos=int(y.sum()), dias=len(y),
                aciertos_p50=int(((p > 0.5) & (y == 1)).sum()), falsas_p50=int(((p > 0.5) & (y == 0)).sum()))


def conformal(lo: pd.DataFrame, hi: pd.DataFrame, Y: pd.DataFrame, dias, ventana=56, nivel=0.8, min_dias=21):
    """Corrección conformal en línea de un intervalo (CQR, Romano et al. 2019) con el retraso del MTR.
    Puntaje = max(lo − y, y − hi). Para el día d se usa el cuantil `nivel` de los puntajes de los últimos `ventana`
    días conocidos (<= d−8), por separado para la tarde-noche (18–23 h) y el resto, y se ensancha (o estrecha) el
    intervalo en esa cantidad. Los primeros días sin historia suficiente se dejan sin corregir."""
    dias = pd.DatetimeIndex(dias)
    S = np.maximum(lo.loc[dias] - Y.loc[dias], Y.loc[dias] - hi.loc[dias])
    tarde = [h for h in S.columns if 18 <= h <= 23]
    resto = [h for h in S.columns if h not in tarde]
    adj = pd.DataFrame(0.0, index=dias, columns=S.columns)
    for d in dias:
        ok = S.index[(S.index <= d - pd.Timedelta(days=LAG_MTR))][-ventana:]
        if len(ok) < min_dias:
            continue
        for grupo in (tarde, resto):
            adj.loc[d, grupo] = np.nanquantile(S.loc[ok, grupo].values, nivel)
    return lo.loc[dias] - adj, hi.loc[dias] + adj
