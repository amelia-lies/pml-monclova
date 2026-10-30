"""Detección de anomalías (notebook 09).

Dos preguntas distintas:

  1. Después del hecho: ¿qué días el MTR se separó del MDA de forma anormal? Se mide con una puntuación robusta
     (cuántas "desviaciones típicas robustas" se aleja el error de cada hora de lo normal en las cuatro semanas
     anteriores). Sirve para describir las anomalías: cuántas hay, cuándo, cuánto error cargan y si fueron de toda la
     red (energía) o sólo de Monclova (congestión).
  2. Antes del hecho, en d−1: ¿algo de lo que se sabe al emitir el pronóstico anticipa esos días? Se prueban detectores
     no supervisados (no necesitan ejemplos de anomalías): qué tan raro es el día según el MDA y la historia publicada,
     con un Isolation Forest y con el error de reconstrucción de un PCA. Se comparan con el clasificador de picos.

Reglas de tiempo de los detectores: se reentrenan cada semana con días <= d−8 y sólo usan variables disponibles en d−1.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from . import features2 as f2

LAG_MTR = 8
Z_ANOMALIA = 3.0          # un día es anómalo si su error se aleja más de 3 desviaciones robustas de lo normal
VENTANA = 56              # "lo normal" = las 8 semanas anteriores


# ------------------------------------------------------------------------------------------- 1. después del hecho
def puntaje_expost(Y: pd.DataFrame, X: pd.DataFrame, ventana=VENTANA) -> pd.Series:
    """Puntuación robusta diaria: (error medio del día − mediana) / (1.4826 · MAD), con la mediana y la MAD de los
    errores diarios de las `ventana` jornadas anteriores. Error = MTR − MDA, promedio de las 24 horas, en MXN/MWh.
    Positiva: el MTR quedó muy por encima del MDA. Es descriptiva: usa el MTR del propio día, no sirve para pronosticar."""
    e = (Y - X).asfreq("D").mean(axis=1)
    med = e.shift(1).rolling(ventana, min_periods=14).median()
    mad = (e.shift(1) - med).abs().rolling(ventana, min_periods=14).median() * 1.4826
    return (e - med) / mad


def dias_anomalos(z: pd.Series, umbral=Z_ANOMALIA) -> pd.DataFrame:
    """Etiqueta diaria: anómalo hacia arriba (MTR muy por encima del MDA), hacia abajo, o normal."""
    tipo = np.where(z > umbral, "arriba", np.where(z < -umbral, "abajo", "normal"))
    return pd.DataFrame({"z": z, "tipo": tipo, "anomalo": np.abs(z) > umbral}, index=z.index)


def alcance_en_la_red(Ym: dict, Xm: dict, dias, umbral=Z_ANOMALIA) -> pd.Series:
    """Para cada día, la fracción de nodos de la red que también tuvieron un día anómalo. Cerca de 1: la anomalía fue
    de todo el sistema (energía); cerca de 0: fue local (congestión)."""
    marcas = pd.DataFrame({n: dias_anomalos(puntaje_expost(Ym[n], Xm[n]), umbral)["anomalo"].reindex(dias) for n in Ym})
    return marcas.mean(axis=1)


# ------------------------------------------------------------------------------------------- 2. antes del hecho
def rasgos_dia(c: f2.Ctx) -> pd.DataFrame:
    """Una fila por día con lo que se sabe en d−1: la forma y el nivel del MDA de d, las estadísticas móviles del
    error ya publicado (G4) y las de varias escalas y de la red (G13, G14). Sólo variables diarias."""
    grupos = ["G4", "G13"] + (["G14"] if c.nodos else [])
    F, _ = f2.construir(c, grupos)
    diarias = [x for x in F.columns if "_h" not in x]
    _, Xt = f2._t(c)
    perfil = Xt.reindex(F.index).sub(Xt.reindex(F.index).mean(axis=1), axis=0)
    perfil.columns = [f"mda_forma_{h}" for h in perfil.columns]
    return pd.concat([F[diarias], perfil], axis=1)


def _bloque(d0, d1, R, metodo, semilla):
    """Puntaje de rareza de la semana [d0, d1] con un detector entrenado con días <= d0 − 8 (estandarizado con esos
    mismos días)."""
    fin = d0 - pd.Timedelta(days=LAG_MTR)
    tr = R.loc[:fin].dropna()
    te = R.loc[d0:d1]
    sc = StandardScaler().fit(tr.values)
    A, B = sc.transform(tr.values), np.nan_to_num(sc.transform(te.fillna(tr.median()).values))
    if metodo == "isolation":                       # cuántos cortes al azar hacen falta para aislar el día
        m = IsolationForest(n_estimators=300, random_state=semilla).fit(A)
        s = -m.score_samples(B)
    elif metodo == "pca":                           # qué tanto del día no explican los patrones habituales
        p = PCA(n_components=0.9, random_state=semilla).fit(A)
        s = ((B - p.inverse_transform(p.transform(B))) ** 2).mean(axis=1)
    else:
        raise ValueError(metodo)
    return pd.Series(s, index=te.index)


def puntaje_exante(c: f2.Ctx, dias, metodo="isolation", n_jobs=-1, semilla=0) -> pd.Series:
    """Puntuación de "día raro" fuera de muestra, con reentrenamiento semanal (días <= d−8). Más alto = más raro."""
    R = rasgos_dia(replace_dias(c))
    dias = pd.DatetimeIndex(dias)
    bl = [(dias[i], dias[min(i + 6, len(dias) - 1)]) for i in range(0, len(dias), 7)]
    res = Parallel(n_jobs=n_jobs)(delayed(_bloque)(a, b, R, metodo, semilla) for a, b in bl)
    return pd.concat(res)


def replace_dias(c: f2.Ctx) -> f2.Ctx:
    """El contexto con todos los días desde el inicio del entrenamiento (los detectores aprenden de toda la historia)."""
    from dataclasses import replace
    return replace(c, dias=pd.date_range(f2.TRAIN_START, f2.TEST_FIN))


def separacion(puntaje: pd.Series, etiquetas: dict, error: pd.Series, nivel: pd.Series) -> dict:
    """Qué tan bien ordena un puntaje los días:
      - AUC contra cada etiqueta 0/1 (0.5 = azar, 1 = perfecto);
      - correlación de rangos con el error absoluto diario del MDA, y la misma descontando el nivel del MDA (correlación
        parcial): un puntaje que sólo repite "el precio está alto" no agrega nada que los modelos no sepan;
      - cuántas veces más error tiene el 10% de días con puntaje más alto que el resto."""
    from sklearn.metrics import roc_auc_score
    ix = puntaje.dropna().index
    out = {f"AUC {k}": roc_auc_score(v.reindex(ix), puntaje.loc[ix]) for k, v in etiquetas.items()}
    r = pd.DataFrame({"s": puntaje, "e": error, "l": nivel}).loc[ix].rank()
    rs, rl, re_ = r.s.corr(r.e), r.s.corr(r.l), r.e.corr(r.l)
    out["correlación con el error"] = rs
    out["la misma sin el nivel del MDA"] = (rs - rl * re_) / np.sqrt((1 - rl ** 2) * (1 - re_ ** 2))
    top = puntaje.loc[ix] >= puntaje.loc[ix].quantile(0.9)
    out["error del 10% más riesgoso / resto"] = error.loc[ix][top].mean() / error.loc[ix][~top].mean()
    return out


def percentil_historico(puntaje: pd.Series, historia: pd.Series, ventana=91) -> pd.Series:
    """Percentil de cada día dentro de los puntajes de los `ventana` días anteriores (sólo pasado): 0.9 = más raro que
    el 90% de los últimos tres meses. Se compara con lo reciente y no con toda la historia porque la primavera de 2024
    fue tan extrema que, contra ella, casi ningún día posterior parece raro."""
    h = historia.sort_index()
    return pd.Series({d: (h.loc[d - pd.Timedelta(days=ventana):d - pd.Timedelta(days=1)] < v).mean() for d, v in puntaje.items()})


def intervalo_por_riesgo(lo: pd.DataFrame, hi: pd.DataFrame, Y: pd.DataFrame, alto: pd.Series, dias, ventana=120, nivel=0.8,
                         min_dias=14):
    """Corrección conformal del intervalo [lo, hi] separada por nivel de riesgo. Para cada día d se toman los días ya
    publicados (<= d−8) del mismo grupo de riesgo que d (alto o normal), los últimos `ventana`, y se mide cuánto se
    salió el MTR del intervalo (puntaje = max(lo − y, y − hi)). El intervalo de d se ensancha (o se estrecha) en el
    cuantil `nivel` de esos puntajes. Así los días riesgosos pueden recibir intervalos más anchos que los normales.
    Ningún notebook ni prueba la usa hoy."""
    dias = pd.DatetimeIndex(dias)
    S = np.maximum(lo - Y, Y - hi).dropna(how="all")
    alto = alto.reindex(S.index.union(dias)).fillna(False).astype(bool)
    adj = pd.Series(0.0, index=dias)
    for d in dias:
        ok = S.index[(S.index <= d - pd.Timedelta(days=LAG_MTR)) & (alto.reindex(S.index).values == alto.loc[d])][-ventana:]
        if len(ok) >= min_dias:
            adj.loc[d] = np.nanquantile(S.loc[ok].values, nivel)
    return lo.loc[dias].sub(adj, axis=0), hi.loc[dias].add(adj, axis=0)
