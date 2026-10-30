"""Cómo medimos los pronósticos.

Todos los pronósticos son matrices días × 24 horas (en MXN/MWh). La medida principal del reto es el error absoluto
medio (MAE). Para comparar dos pronósticos usamos además:

  - en cuántos meses uno le gana al otro (`mae_por_bloque`), que dice si la ventaja es pareja o viene de unos días;
  - el peor mes (`peor_mes`): el mayor cambio mensual del error contra el MDA. Es el criterio para elegir entre versiones,
    mezclas y modelo final, porque castiga a los modelos que pierden mucho en algún mes;
  - la prueba de Giacomini–White (`gw_test`), que responde: si en realidad los dos fueran igual de buenos, ¿qué tan
    probable sería ver una diferencia así de consistente día a día? Un valor menor a 0.05 indica que es difícil que
    la diferencia sea suerte. Diebold–Mariano (`dm_test`) es la versión clásica de la misma pregunta.

Las pruebas trabajan con el error de cada día (promedio de sus 24 horas), porque las horas de un mismo día no son
independientes entre sí.
"""
import numpy as np
import pandas as pd
from scipy import stats


def mae(y, f):
    """Error absoluto medio. Ignora huecos (NaN)."""
    return float(np.nanmean(np.abs(np.asarray(y) - np.asarray(f))))


def rmse(y, f):
    """Raíz del error cuadrático medio: castiga más los errores grandes (los picos)."""
    return float(np.sqrt(np.nanmean((np.asarray(y) - np.asarray(f)) ** 2)))


def rmae(y, f, naive):
    """MAE relativo a un pronóstico ingenuo (por ejemplo, repetir el MTR de hace dos semanas). Menor a 1 = mejor."""
    return mae(y, f) / mae(y, naive)


def _dif_diaria(y, fa, fb, norma=1):
    """Diferencia de error de cada día entre A y B (negativa = A se equivocó menos ese día).
    norma=1: error absoluto medio de las 24 horas; norma=2: raíz del error cuadrático medio."""
    ea, eb = np.asarray(y) - np.asarray(fa), np.asarray(y) - np.asarray(fb)
    if norma == 1:
        return np.mean(np.abs(ea), axis=1) - np.mean(np.abs(eb), axis=1)
    return np.sqrt(np.mean(ea ** 2, axis=1)) - np.sqrt(np.mean(eb ** 2, axis=1))


def dm_test(y, fa, fb, norma=1):
    """Diebold–Mariano: ¿la diferencia media diaria es distinta de cero? Devuelve el p bilateral y el p de que A sea
    mejor (unilateral)."""
    d = _dif_diaria(y, fa, fb, norma); n = len(d)
    t = np.sqrt(n) * d.mean() / d.std(ddof=1)
    return dict(stat=float(t), p=float(2 * stats.norm.sf(abs(t))), p_a_mejor=float(stats.norm.cdf(t)))


def gw_test(y, fa, fb, norma=1):
    """Giacomini–White: como Diebold–Mariano, pero además pregunta si la diferencia de ayer anticipa la de hoy (una
    ventaja que va y viene cuenta menos que una persistente). Estadístico de Wald con 2 grados de libertad."""
    d = _dif_diaria(y, fa, fb, norma)
    X = np.column_stack([np.ones(len(d) - 1), d[:-1]]); z = d[1:]       # instrumentos: constante y diferencia de ayer
    Zt = X * z[:, None]
    zbar = Zt.mean(axis=0); S = np.cov(Zt.T, ddof=0)
    n = len(z)
    w = float(n * zbar @ np.linalg.solve(S, zbar))
    p = float(stats.chi2.sf(w, 2))
    return dict(stat=w, p=p, mejor='A' if d.mean() < 0 else 'B')


def mae_por_bloque(y: pd.DataFrame, f: pd.DataFrame, idx, regla="M") -> pd.Series:
    """MAE de cada mes (o del bloque que diga `regla`) sobre los días `idx`."""
    e = (y.loc[idx] - f.loc[idx]).abs().mean(axis=1)
    return e.groupby(e.index.to_period(regla)).mean()


def mae_sin_picos(y: pd.DataFrame, f: pd.DataFrame, idx, umbral=3000.0) -> float:
    """MAE sin los días en que algún MTR de la zona superó `umbral` (los días con pico dominan el promedio)."""
    ok = y.loc[idx].max(axis=1) <= umbral
    return float((y.loc[idx][ok] - f.loc[idx][ok]).abs().values.mean())


def cambios_mensuales(Y, X, P, idx):
    """Cambio (%) del MAE mensual de `P` contra el del MDA `X`, un valor por mes de `idx`. `Y`, `X` y `P` son días × 24."""
    idx = pd.DatetimeIndex(idx); mes = idx.strftime("%Y-%m")
    mensual = lambda M: pd.Series(np.abs(Y.loc[idx].values - M.loc[idx].values).mean(axis=1), index=idx).groupby(mes).mean()
    return 100 * (mensual(P) / mensual(X) - 1)


def peor_mes(Y, X, P, idx):
    """Criterio de elección: el mayor cambio (%) del MAE mensual de `P` contra el del MDA `X` entre los meses de `idx`. Menos
    es mejor. A diferencia del error medio y de la mediana, castiga al modelo que pierde mucho en algún mes."""
    return float(cambios_mensuales(Y, X, P, idx).max())
    return float(cambios_mensuales(Y, X, P, idx).median())
