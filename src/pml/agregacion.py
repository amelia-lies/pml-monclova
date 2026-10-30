"""Agregación en línea de expertos: pesos que cambian con el tiempo en lugar del 0.8·MDA + 0.2·modelo fijo.

El pronóstico del día d sólo puede usar el error de días <= d−8, porque antes no se conoce el MTR. Todos los
algoritmos respetan ese retraso. Los pesos parten de un prior (por defecto el 0.8 / 0.2 elegido en desarrollo) y se
mueven con la evidencia.

  - `ventana_lad`: pesos convexos que minimizan el MAE de las últimas `ventana` jornadas conocidas (programación lineal),
    encogidos hacia el prior con peso `encoge`.
  - `ewa`: promedio con pesos exponenciales (Hedge) sobre el gradiente linealizado de la pérdida absoluta, como en
    BOA / ML-Poly (Berrisch y Ziel); `eta` controla qué tan rápido reacciona.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

LAG_MTR = 8


def _alinea(E: dict, Y: pd.DataFrame, dias):
    """Pone los pronósticos de los expertos en un arreglo días × 24 × expertos, junto al MTR real."""
    dias = pd.DatetimeIndex(dias)
    nombres = list(E)
    P = np.stack([E[k].loc[dias].values for k in nombres], axis=-1)          # días × 24 × K
    return dias, nombres, P, Y.loc[dias].values


def _lad_convexo(F, y):
    """min Σ|y − F w| con w >= 0, Σ w = 1 (F: n × K)."""
    n, K = F.shape
    c = np.r_[np.zeros(K), np.ones(n)]
    A_ub = np.block([[F, -np.eye(n)], [-F, -np.eye(n)]])
    b_ub = np.r_[y, -y]
    A_eq = np.r_[np.ones(K), np.zeros(n)][None]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * (K + n), method="highs")
    return r.x[:K] if r.success else np.full(K, 1.0 / K)


def ventana_lad(E: dict, Y, dias, prior: dict, ventana=28, encoge=0.5, cada=7):
    """Pesos re-estimados cada `cada` días con las `ventana` jornadas más recientes conocidas (todas las horas juntas)."""
    dias, nombres, P, y = _alinea(E, Y, dias)
    w0 = np.array([prior.get(k, 0.0) for k in nombres]); w0 = w0 / w0.sum()
    W = np.zeros((len(dias), len(nombres)))
    w = w0.copy()
    for t, d in enumerate(dias):
        if t % cada == 0:
            conocidos = np.where(dias <= d - pd.Timedelta(days=LAG_MTR))[0][-ventana:]
            if len(conocidos) >= 14:
                Fk = P[conocidos].reshape(-1, len(nombres)); yk = y[conocidos].ravel()
                ok = ~np.isnan(yk) & ~np.isnan(Fk).any(1)
                w = encoge * w0 + (1 - encoge) * _lad_convexo(Fk[ok], yk[ok])
        W[t] = w
    pron = pd.DataFrame(np.einsum("thk,tk->th", P, W), index=dias, columns=Y.columns)
    return pron, pd.DataFrame(W, index=dias, columns=nombres)


def ewa(E: dict, Y, dias, prior: dict, eta=2.0, por_hora=False):
    """Hedge con gradiente linealizado y retraso de 8 días. Los pesos de d usan las pérdidas de días <= d−8.
    `eta` se aplica sobre pérdidas normalizadas por la escala típica del error (MAE del prior), así que es adimensional."""
    dias, nombres, P, y = _alinea(E, Y, dias)
    K = len(nombres)
    logw0 = np.log(np.array([max(prior.get(k, 0.0), 1e-3) for k in nombres]))
    H = 24 if por_hora else 1
    acum = np.zeros((H, K))
    W = np.zeros((len(dias), 24, K))
    escala = None
    for t, d in enumerate(dias):
        lw = logw0[None] - eta * acum
        w = np.exp(lw - lw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        W[t] = w if por_hora else np.repeat(w, 24, 0)
        s = t - LAG_MTR + 1                                     # el día que se vuelve conocido para el pronóstico de d+1
        if s >= 0 and not np.isnan(y[s]).any():
            pf = np.einsum("hk,hk->h", P[s], W[s])
            g = np.sign(pf - y[s])[:, None] * P[s]               # gradiente de |y − Σ w f| respecto a w
            if escala is None:
                escala = np.mean(np.abs(pf - y[s])) + 1e-9
            gl = g / escala
            acum += gl if por_hora else gl.mean(0, keepdims=True)
    pron = pd.DataFrame(np.einsum("thk,thk->th", P, W), index=dias, columns=Y.columns)
    return pron, pd.DataFrame(W.mean(1), index=dias, columns=nombres)
