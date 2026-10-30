"""Backtest rolling-origin con LASSO (LEAR simplificado): recalibración diaria, 3 ventanas, rejilla de alpha.

Para el día d sólo se entrena con filas cuyo MTR ya era conocido: días <= d−8.
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler

LAG_MTR = 8
ALPHAS = np.array([0.3, 0.1, 0.03, 0.01, 0.003])
VENTANAS = {"8w": 56, "12w": 84, "max": None}


def _dia(d, F, Yt, cols, ventanas, alphas, inicio, winsor=None, centro="media"):
    """Pronóstico de un día: para cada ventana de entrenamiento (8 semanas, 12 semanas, todo) ajusta un LASSO por hora
    con días <= d − 8 y lo evalúa en toda la rejilla de alphas."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*converge.*")
    fin = d - pd.Timedelta(days=LAG_MTR)
    out = {}
    for nom, L in ventanas.items():
        ini = inicio if L is None else max(inicio, fin - pd.Timedelta(days=L - 1))
        tr = F.index[(F.index >= ini) & (F.index <= fin)]
        Xtr = F.loc[tr, cols].values; Ytr = Yt.loc[tr].values
        if winsor:                                   # recorte del objetivo SÓLO en entrenamiento (cuantiles de la ventana)
            lo, hi = np.quantile(Ytr, [winsor, 1 - winsor]); Ytr = np.clip(Ytr, lo, hi)
        sc = StandardScaler().fit(Xtr); Xs = sc.transform(Xtr)
        ym = Ytr.mean(axis=0)
        x = sc.transform(F.loc[[d], cols].values)[0]
        pred = np.empty((len(alphas), Ytr.shape[1]))
        for h in range(Ytr.shape[1]):                            # un LASSO independiente por hora (como LEAR)
            _, c, _ = lasso_path(Xs, Ytr[:, h] - ym[h], alphas=alphas, max_iter=2000)  # (p, nA)
            b = np.median((Ytr[:, h] - ym[h])[:, None] - Xs @ c, axis=0) if centro == "mediana" else 0.0
            pred[:, h] = x @ c + ym[h] + b     # centro="mediana": intercepto robusto (la métrica es MAE, no error cuadrático)
        out[nom] = pred
    return d, out


def correr(F, Yt, cols, dias, ventanas=VENTANAS, alphas=ALPHAS, inicio=None, n_jobs=-1, winsor=None, centro="media"):
    """Devuelve {(ventana, alpha): DataFrame días × 24 en espacio transformado}."""
    inicio = inicio or F.index.min()
    res = Parallel(n_jobs=n_jobs)(delayed(_dia)(d, F, Yt, cols, ventanas, alphas, inicio, winsor, centro) for d in dias)
    preds = {}
    for nom in ventanas:
        for i, a in enumerate(alphas):
            preds[(nom, float(a))] = pd.DataFrame(np.stack([r[1][nom][i] for r in res]), index=[r[0] for r in res], columns=Yt.columns)
    return preds
