"""LEAR sobre las componentes principales de la curva diaria del objetivo (en vez de 24 modelos horarios independientes).

PCA ajustado SÓLO con las filas de entrenamiento de cada ventana (días <= d−8). Intercepto robusto: la reconstrucción en
muestra se corrige con la mediana horaria de los residuos.
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler

LAG_MTR = 8
ALPHAS = np.array([0.3, 0.1, 0.03, 0.01, 0.003])
VENTANAS = {"8w": 56, "12w": 84, "max": None}


def _dia(d, F, Yt, cols, ventanas, alphas, inicio, k, winsor):
    """Pronóstico de un día: PCA del perfil y un LASSO por componente, en cada ventana y para cada alpha."""
    warnings.filterwarnings("ignore")
    fin = d - pd.Timedelta(days=LAG_MTR)
    out = {}
    for nom, L in ventanas.items():
        ini = inicio if L is None else max(inicio, fin - pd.Timedelta(days=L - 1))
        tr = F.index[(F.index >= ini) & (F.index <= fin)]
        Xtr = F.loc[tr, cols].values; Ytr = Yt.loc[tr].values
        if winsor:
            lo, hi = np.quantile(Ytr, [winsor, 1 - winsor]); Ytr = np.clip(Ytr, lo, hi)
        sc = StandardScaler().fit(Xtr); Xs = sc.transform(Xtr); x = sc.transform(F.loc[[d], cols].values)[0]
        pca = PCA(n_components=k).fit(Ytr)
        S = pca.transform(Ytr)                                        # scores de entrenamiento (n × k)
        sm = S.mean(axis=0)
        coefs = [lasso_path(Xs, S[:, j] - sm[j], alphas=alphas, max_iter=2000)[1] for j in range(k)]   # k × (p, nA)
        pred = np.empty((len(alphas), Ytr.shape[1]))
        for a in range(len(alphas)):
            B = np.stack([coefs[j][:, a] for j in range(k)], axis=1)  # (p, k)
            Str = Xs @ B + sm                                         # scores ajustados en entrenamiento
            res = Ytr - pca.inverse_transform(Str)                    # residuo por hora
            b = np.median(res, axis=0)                                # intercepto horario robusto
            pred[a] = pca.inverse_transform((x @ B + sm)[None, :])[0] + b
        out[nom] = pred
    return d, out


def correr(F, Yt, cols, dias, k=5, ventanas=VENTANAS, alphas=ALPHAS, inicio=None, winsor=None, n_jobs=-1):
    """Como `backtest.correr`, pero cada LASSO pronostica una de las `k` componentes principales del perfil diario
    en vez de una hora. Devuelve {(ventana, alpha): días × 24 en la escala del proyecto}."""
    inicio = inicio or F.index.min()
    res = Parallel(n_jobs=n_jobs)(delayed(_dia)(d, F, Yt, cols, ventanas, alphas, inicio, k, winsor) for d in dias)
    preds = {}
    for nom in ventanas:
        for i, a in enumerate(alphas):
            preds[(nom, float(a))] = pd.DataFrame(np.stack([r[1][nom][i] for r in res]), index=[r[0] for r in res], columns=Yt.columns)
    return preds
