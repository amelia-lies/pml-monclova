"""Mezclas de 2 y 3 modelos entre todas las versiones de la escalera (notebooks 10 y 11).

Cada candidata combina 1, 2 o 3 pronósticos (el MDA o cualquier versión de cualquier modelo de la escalera) con pesos múltiplos
de `paso` que suman 1, por ejemplo 0.3·TiDE con MDA + 0.7·GB con todas las variables, o 0.5·red sin MDA con PCA + 0.2·GB con
MDA + 0.3·LEAR con PCA. Todo se calcula con pronósticos ya hechos, sin reentrenar nada. Con muchas candidatas el cálculo se
corre en el servidor (`servidor.py mezclas`) y queda en caché.
"""
import hashlib
import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

CACHE = Path(__file__).resolve().parents[2] / "data/processed/exp_cache"


def candidatas(n, paso=0.1):
    """(índices, pesos) de todas las combinaciones de 1, 2 o 3 de `n` pronósticos, con pesos múltiplos de `paso`, cada uno de al
    menos `paso` y que suman 1."""
    k = round(1 / paso)
    c = [((i,), (1.0,)) for i in range(n)]
    c += [((i, j), (a / k, (k - a) / k)) for i, j in itertools.combinations(range(n), 2) for a in range(1, k)]
    c += [((i, j, l), (a / k, b / k, (k - a - b) / k)) for i, j, l in itertools.combinations(range(n), 3)
          for a in range(1, k - 1) for b in range(1, k - a)]
    return c


def nombre(nombres, cand):
    """Nombre legible de una candidata, p. ej. '0.5·LEAR en proporción · + nodos + 0.5·GB · + modelado estadístico'."""
    return " + ".join(f"{w:g}·{nombres[i]}" for i, w in zip(*cand))


def pronostico(arr, cand):
    """Pronóstico de una candidata: `arr` es pronósticos × días × 24."""
    return sum(w * arr[i] for i, w in zip(*cand))


def _bloque(yv, arr, cands, e_x, grupos):
    """Cambio (%) del MAE contra el MDA, por grupo de días (meses, desarrollo, test), para un bloque de candidatas."""
    out = np.empty((len(cands), len(grupos)))
    for r, cand in enumerate(cands):
        e = np.abs(yv - pronostico(arr, cand)).mean(axis=1)
        out[r] = [100 * (e[g].mean() / e_x[g].mean() - 1) for g in grupos]
    return out


def evaluar(Y, X, F, dev, test, paso=0.1, n_jobs=-1):
    """Califica todas las candidatas con los pronósticos `F` ({nombre: días × 24}). Devuelve un dict con los nombres de los
    pronósticos, las candidatas, el cambio contra el MDA (%) de cada candidata en cada mes (DataFrame) y en todo el desarrollo y
    todo el test (Series). Queda en caché."""
    nombres = list(F); dias = pd.DatetimeIndex(dev).append(pd.DatetimeIndex(test))
    clave = hashlib.md5(f"{nombres}|{paso}|{dias[0]}|{dias[-1]}".encode()).hexdigest()[:10]
    f = CACHE / f"mezclas_{clave}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    arr = np.stack([F[k].loc[dias].values for k in nombres]); yv = Y.loc[dias].values
    e_x = np.abs(yv - X.loc[dias].values).mean(axis=1)
    mes = np.asarray(dias.strftime("%Y-%m")); meses = sorted(set(mes)); es_dev = np.arange(len(dias)) < len(dev)
    grupos = [mes == m for m in meses] + [es_dev, ~es_dev]
    cands = candidatas(len(nombres), paso)
    trozos = [cands[i:i + 2000] for i in range(0, len(cands), 2000)]
    res = np.vstack(Parallel(n_jobs=n_jobs)(delayed(_bloque)(yv, arr, t, e_x, grupos) for t in trozos))
    out = dict(nombres=nombres, cands=cands, mensual=pd.DataFrame(res[:, :len(meses)], columns=meses),
               desarrollo=pd.Series(res[:, -2]), test=pd.Series(res[:, -1]))
    f.write_bytes(pickle.dumps(out))
    return out


def calificaciones(mensual_dev, desarrollo):
    """Calificación de cada candidata con cada criterio (menos es mejor), usando solo los meses de desarrollo."""
    d = mensual_dev.values; x = np.arange(d.shape[1])
    return pd.DataFrame({"error medio": desarrollo.values, "mediana de los meses": np.median(d, axis=1), "peor mes": d.max(axis=1),
                         "tendencia": np.polyval(np.polyfit(x, d.T, 1), len(x))})
