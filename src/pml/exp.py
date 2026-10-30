"""Ejecución y evaluación de configuraciones de LEAR (un LASSO por hora, rejilla de alpha, promedio de tres ventanas).

Protocolo: la selección de alpha y de variables usa SÓLO el periodo de desarrollo (2024-07→2024-12). Las predicciones se
calculan también para el test (2025-01→06) y se guardan, pero sólo el notebook 11 las evalúa.
"""
import hashlib
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import features2 as f2
from .backtest import ALPHAS, VENTANAS, correr
from .metrics import mae

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/processed/exp_cache"
REGISTRO = ROOT / "reports/experimentos.csv"


def dias_eval(c):
    """Días que se pronostican: desarrollo y test (del 2024-07-01 al 2025-06-30)."""
    return pd.date_range(f2.DEV_INI, f2.TEST_FIN)


def idx_dev():
    """Días de desarrollo (jul–dic 2024): aquí se decide todo."""
    return pd.date_range(f2.DEV_INI, f2.DEV_FIN)


def idx_test():
    """Días del test (ene–jun 2025): sólo los evalúa el notebook 11."""
    return pd.date_range(f2.TEST_INI, f2.TEST_FIN)


def _clave(nombre, grupos, objetivo, c, winsor, centro="media", quitar=()):
    """Nombre del archivo de caché de una configuración (cambia si cambia cualquier opción que afecta el resultado)."""
    extra = ("" if centro == "media" else "|" + centro) + ("|sin:" + ",".join(sorted(quitar)) if quitar else "")
    h = hashlib.md5(f"{nombre}|{sorted(grupos)}|{objetivo}|{c.modo}|{c.escala}|{c.lag_precios}|{winsor}|{c.med:.4f}{extra}".encode()).hexdigest()[:10]
    return f"{nombre.replace(' ', '_').replace('/', '-')}_{h}"


def correr_config(c, nombre, grupos, objetivo="spread", winsor=None, ventanas=VENTANAS, alphas=ALPHAS, F_extra=None, centro="mediana",
                  quitar=()):
    """Ejecuta (o recupera de caché) el backtest de una configuración. Devuelve {(ventana, alpha): días × 24 en espacio transformado}.
    `quitar`: prefijos de columnas que se eliminan (para medir cuánto pierde el modelo sin ellas)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (_clave(nombre, grupos, objetivo, c, winsor, centro, quitar) + ".pkl")
    if f.exists():
        return pickle.loads(f.read_bytes())
    if grupos:
        F, G = f2.construir(c, grupos)
        if quitar:
            F = F[[x for x in F.columns if not any(x.startswith(q) for q in quitar)]]
        if F_extra is not None:
            F = pd.concat([F, F_extra.loc[F.index]], axis=1)
    else:                                                  # sin grupos: el modelo sólo ve `F_extra` (por ejemplo, componentes de todo)
        F = F_extra
    Yt, Xt = f2._t(c)
    tgt = (Yt - Xt) if objetivo == "spread" else Yt
    preds = correr(F, tgt.loc[F.index], F.columns.tolist(), dias_eval(c), ventanas=ventanas, alphas=alphas,
                   inicio=F.index.min(), winsor=winsor, centro=centro)
    f.write_bytes(pickle.dumps(preds))
    return preds


def a_precio(P, c, objetivo):
    """Del pronóstico en escala del proyecto al precio en MXN/MWh (sumando el MDA si el objetivo es el error del MDA)."""
    Z = P + (f2._t(c)[1].loc[P.index] if objetivo == "spread" else 0)
    return f2.inv(Z, c)


def pronostico(c, preds, objetivo="spread", ventanas=VENTANAS):
    """Elige alpha por ventana con el MAE de DESARROLLO y promedia las ventanas (ensemble aritmético)."""
    dev = idx_dev()
    out, elegido = [], {}
    for w in ventanas:
        cand = {a: mae(c.Y.loc[dev].values, a_precio(preds[(w, a)], c, objetivo).loc[dev].values) for (ww, a) in preds if ww == w}
        a_best = min(cand, key=cand.get); elegido[w] = a_best
        out.append(a_precio(preds[(w, a_best)], c, objetivo))
    return sum(out) / len(out), elegido


def registrar(nombre, grupos, objetivo, n_feat, mae_dev, extra=""):
    """Anota cada configuración probada (sólo métricas de desarrollo) para contar multiplicidad."""
    REGISTRO.parent.mkdir(exist_ok=True)
    fila = pd.DataFrame([dict(nombre=nombre, grupos="+".join(grupos), objetivo=objetivo, n_features=n_feat, mae_dev=round(mae_dev, 3), nota=extra)])
    if REGISTRO.exists():
        old = pd.read_csv(REGISTRO)
        old = old[~((old.nombre == nombre) & (old.objetivo == objetivo) & (old.grupos == "+".join(grupos)))]
        fila = pd.concat([old, fila], ignore_index=True)
    fila.to_csv(REGISTRO, index=False)


def correr_y_evaluar(c, nombre, grupos, objetivo="spread", winsor=None, nota="", **kw):
    """Atajo: backtest + selección de alpha en dev + registro. Devuelve (pronóstico en precios, alphas elegidos)."""
    preds = correr_config(c, nombre, grupos, objetivo, winsor, **kw)
    fc, elegido = pronostico(c, preds, objetivo)
    n = f2.construir(replace(c, dias=c.dias[:30]), grupos)[0].shape[1] if grupos else kw["F_extra"].shape[1]
    registrar(nombre, grupos, objetivo, n, mae(c.Y.loc[idx_dev()].values, fc.loc[idx_dev()].values), nota)
    return fc, elegido


def tabla_dev(c, FC, base, nombres, refs: dict, DEV=None):
    """Tabla de desarrollo de los notebooks 04 y 05: para cada pronóstico de `FC`, su MAE, su MAE sin días de pico, el
    cambio contra el MDA y, contra la configuración `base`, el cambio, la p de Giacomini–White y en cuántos meses le
    gana. `legible` la deja lista para mostrar."""
    from .metrics import gw_test, mae_por_bloque, mae_sin_picos
    DEV = idx_dev() if DEV is None else DEV
    f = lambda d: d.loc[DEV].values
    mb = mae_por_bloque(c.Y, FC[base], DEV)
    rows = {}
    for k in [base] + nombres:
        mk = mae_por_bloque(c.Y, FC[k], DEV)
        r = dict(MAE=mae(f(c.Y), f(FC[k])), MAE_sin_picos=mae_sin_picos(c.Y, FC[k], DEV),
                 vs_MDA_pct=100 * (mae(f(c.Y), f(FC[k])) / refs["MDA"] - 1))
        if k != base:
            r.update(dMAE_pct=100 * (mae(f(c.Y), f(FC[k])) / mae(f(c.Y), f(FC[base])) - 1),
                     p_GW=gw_test(f(c.Y), f(FC[k]), f(FC[base]))["p"], meses_mejor=int((mk < mb).sum()), de_meses=len(mk))
        rows[k] = r
    return pd.DataFrame(rows).T


NOMBRES_LEGIBLES = {"MAE": "MAE", "MAE_sin_picos": "MAE sin días de pico", "vs_MDA_pct": "vs MDA (%)", "dMAE_pct": "vs referencia (%)",
                    "p_GW": "p", "meses_mejor": "meses mejor", "de_meses": "de"}


def legible(T, decimales=2):
    """Tabla con nombres de columna legibles y redondeo, para mostrar o guardar en reports/. Hay una sola prueba (p):
    no hace falta corregir por comparaciones múltiples porque el modelo final se fija en desarrollo antes de abrir el
    test (notebook 11), y en el test sólo se le hace una pregunta: ¿le gana al MDA?"""
    T = T.rename(columns=NOMBRES_LEGIBLES)
    for col in ("meses mejor", "de"):
        if col in T:
            T[col] = T[col].astype("Int64")
    pcols = [x for x in T.columns if x == "p" or x.startswith("p ")]
    num = [x for x in T.select_dtypes("number").columns if x not in pcols]
    T[num] = T[num].round(decimales)
    T[pcols] = T[pcols].astype(float).round(3)
    return T


def comparar(Y, filas, X, idx=None, registrar_como=None):
    """Tabla para leer de un vistazo. `filas`: [(grupo, nombre, pronóstico, nombre de su referencia o None)].
    Columnas: MAE; cambio contra el MDA; cambio contra su referencia; en cuántos meses le gana a su referencia; y p, la
    probabilidad de ver una diferencia así de consistente día a día si en realidad los dos fueran igual de buenos
    (prueba de Giacomini–White). Si `registrar_como`, anota cada fila con referencia en el registro de experimentos."""
    from .metrics import gw_test, mae_por_bloque
    idx = idx_dev() if idx is None else idx
    f = lambda d: d.loc[idx].values
    m = lambda P: mae(f(Y), f(P))
    P_ = {n: P for _, n, P, _ in filas}
    out = []
    for g, n, P, r in filas:
        d = {"grupo": g, "modelo": n, "MAE": m(P), "vs MDA (%)": 100 * (m(P) / m(X) - 1)}
        if r:
            d.update({"referencia": r, "vs referencia (%)": 100 * (m(P) / m(P_[r]) - 1),
                      "meses mejor": f"{int((mae_por_bloque(Y, P, idx) < mae_por_bloque(Y, P_[r], idx)).sum())} de {len(mae_por_bloque(Y, P, idx))}",
                      "p": gw_test(f(Y), f(P), f(P_[r]))["p"]})
            if registrar_como:
                registrar(n, [registrar_como], g, 0, m(P), registrar_como)
        out.append(d)
    return pd.DataFrame(out).set_index(["grupo", "modelo"]).round({"MAE": 1, "vs MDA (%)": 1, "vs referencia (%)": 1, "p": 3}).fillna("")
